import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import torch.nn as nn
from datasets import load_dataset
import torch
import deepspeed
import argparse
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers.models.jamba.modeling_jamba import (
    JambaForSequenceClassification,
    JambaMambaDecoderLayer,
    JambaRMSNorm,
    JambaAttentionDecoderLayer
)
import os
import time
import json
import wandb
import psutil
import numpy as np
from torch.autograd import Function
from typing import Dict, List, Tuple, Optional
import pynvml
import gc
from contextlib import contextmanager
from torch.utils.data import IterableDataset
from data import get_imdb_dataset

# Initialize NVML for GPU monitoring
pynvml.nvmlInit()

class GPUMonitor:
    """Monitor GPU usage and memory during training"""
    def __init__(self, log_dir=None):
        self.log_dir = log_dir or f'./logs/gpu_stats_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.log_dir, exist_ok=True)
        self.device_count = torch.cuda.device_count()
        self.device_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]
        self.stats = []
        
    def get_gpu_stats(self):
        stats = []
        for i, handle in enumerate(self.device_handles):
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats.append({
                'device': i,
                'memory_used': memory_info.used / (1024**2),  # MB
                'memory_total': memory_info.total / (1024**2),  # MB
                'memory_percent': memory_info.used / memory_info.total * 100,
                'gpu_utilization': utilization.gpu
            })
        return stats
    
    def log_step(self, step):
        stats = self.get_gpu_stats()
        self.stats.append({'step': step, 'gpu_stats': stats})
        
    def save_stats(self):
        with open(os.path.join(self.log_dir, 'gpu_stats.json'), 'w') as f:
            json.dump(self.stats, f)

@contextmanager
def timer(name, rank, step, stats_dict):
    """Context manager for timing operations"""
    torch.cuda.synchronize()
    start = time.time()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    
    if name not in stats_dict:
        stats_dict[name] = []
    
    stats_dict[name].append({
        'step': step,
        'rank': rank,
        'time': elapsed_time
    })
class CustomProfiler:
    """Custom profiler that tracks detailed metrics for each pipeline stage"""
    def __init__(self, pipe_model, rank, world_size, log_dir=None):
        self.pipe_model = pipe_model
        self.rank = rank
        self.world_size = world_size
        self.log_dir = log_dir or f'./logs/custom_profile_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Metrics to track
        self.timing_stats = {}
        self.memory_stats = []
        self.layer_distribution = self._get_layer_distribution()
        
    def _get_layer_distribution(self):
        """Analyze which layers are assigned to each pipeline stage"""
        if not hasattr(self.pipe_model, '_topo'):
            return {}
        
        # Get the current pipeline stage for this rank
        try:
            current_stage = None
            if hasattr(self.pipe_model, '_grid'):
                # In newer DeepSpeed versions
                current_stage = self.pipe_model._grid.get_stage_id()
            elif hasattr(self.pipe_model, '_topo') and hasattr(self.pipe_model._topo, 'get_coord'):
                # Alternative way to get stage
                coords = self.pipe_model._topo.get_coord(self.rank)
                if coords and len(coords) > 0:
                    current_stage = coords[0]  # First coordinate is typically the stage
            else:
                # Fallback to calculate stage based on rank and pipeline size
                stages_per_rank = self.world_size // self.pipe_model.num_stages
                if stages_per_rank > 0:
                    current_stage = self.rank // stages_per_rank
                else:
                    current_stage = self.rank % self.pipe_model.num_stages
        except Exception as e:
            print(f"Error determining stage: {e}")
            current_stage = None
            
        # Get layer info for the current stage
        layers = self._get_layer_info_for_stage(current_stage)
            
        # Gather layer info from all ranks
        all_stage_layers = {}
        for stage in range(self.pipe_model.num_stages):
            # Create a list to collect info from all ranks
            gathered_layers = [None] * self.world_size
            if dist.is_initialized():
                # Each rank only submits info if it's part of this stage
                my_layers = layers if stage == current_stage else []
                dist.all_gather_object(gathered_layers, my_layers)
                
            # Merge all non-empty lists from the gathered results
            stage_layers = []
            for rank_layers in gathered_layers:
                if rank_layers:  # Only add if the list is not empty
                    stage_layers.extend(rank_layers)
            
            if stage_layers:
                all_stage_layers[f"stage_{stage}"] = stage_layers
        
        return all_stage_layers
    
    def _get_layer_info_for_stage(self, stage):
        """Get information about layers assigned to this stage"""
        layers = []
        
        # Extract layer info from the pipeline module if we're part of a stage
        if stage is not None:
            try:
                if hasattr(self.pipe_model, 'forward_funcs'):
                    for i, layer in enumerate(self.pipe_model.forward_funcs):
                        # Try to get class name
                        if hasattr(layer, '__self__'):
                            layer_name = layer.__self__.__class__.__name__
                        elif hasattr(layer, '__class__'):
                            layer_name = layer.__class__.__name__
                        else:
                            layer_name = str(layer)
                        
                        # Add layer details
                        layer_info = {
                            "index": i,
                            "name": layer_name,
                            "type": "unknown"
                        }
                        
                        # Try to determine if it's a Mamba or Attention layer
                        if "Mamba" in layer_name:
                            layer_info["type"] = "mamba"
                        elif "Attention" in layer_name:
                            layer_info["type"] = "attention"
                        elif "Embedding" in layer_name or "embed" in str(layer).lower():
                            layer_info["type"] = "embedding"
                        elif "Norm" in layer_name or "norm" in str(layer).lower():
                            layer_info["type"] = "normalization"
                        elif "Linear" in layer_name or "classifier" in str(layer).lower():
                            layer_info["type"] = "linear"
                            
                        layers.append(layer_info)
            except Exception as e:
                print(f"Error getting layer info: {e}")
                
        return layers
    
    def log_memory(self, step):
        """Log memory usage for current rank"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
            
            self.memory_stats.append({
                'step': step,
                'rank': self.rank,
                'allocated_mb': allocated,
                'reserved_mb': reserved
            })
    
    def save_stats(self):
        """Save all collected statistics"""
        # Save timing stats
        with open(os.path.join(self.log_dir, f'timing_rank_{self.rank}.json'), 'w') as f:
            json.dump(self.timing_stats, f)
            
        # Save memory stats
        with open(os.path.join(self.log_dir, f'memory_rank_{self.rank}.json'), 'w') as f:
            json.dump(self.memory_stats, f)
            
        # Save layer distribution info
        if self.rank == 0:  # Only save once from rank 0
            with open(os.path.join(self.log_dir, 'layer_distribution.json'), 'w') as f:
                json.dump(self.layer_distribution, f)

def get_ranks_in_stage(stage, world_size, num_stages):
    ranks_per_stage = world_size // num_stages
    start_rank = stage * ranks_per_stage
    return list(range(start_rank, start_rank + ranks_per_stage))


def get_args():
    parser = argparse.ArgumentParser(description='Jamba Pipeline Parallelism')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('-s',
                        '--steps',
                        type=int,
                        default=100,
                        help='quit after this many steps')
    parser.add_argument('-p',
                        '--pipeline-parallel-size',
                        type=int,
                        default=2,
                        help='pipeline parallelism')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        help='distributed backend')
    parser.add_argument('--seed', 
                        type=int, 
                        default=1138, 
                        help='PRNG seed')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='batch size')
    parser.add_argument('--micro-batch-size',
                        type=int,
                        default=4,
                        help='micro batch size')
    parser.add_argument('--seq-length',
                        type=int,
                        default=512,
                        help='sequence length')
    parser.add_argument('--tokenizer',
                        type=str,
                        default="EleutherAI/gpt-neox-20b",
                        help='tokenizer to use')
    parser.add_argument('--use-wandb',
                        action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project',
                        type=str,
                        default="jamba-pipeline",
                        help='WandB project name')
    parser.add_argument('--wandb-run-name',
                        type=str,
                        default=None,
                        help='WandB run name')
    parser.add_argument('--profiling-freq',
                        type=int,
                        default=10,
                        help='How often to collect profiling data')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

class BlockPipe(nn.Module):
    """Wrap a Block so it only propagates `hidden_states` onward."""
    def __init__(self, block: nn.Module, block_type: str = None):
        super().__init__()
        self.block = block
        self.block_type = block_type or block.__class__.__name__

    def forward(self, x):
        # block(x) returns (hidden_states, residual)
        output = self.block(x)
        return output[0] if isinstance(output, tuple) else output

class ClassificationModel(nn.Module):
    def __init__(self, backbone, d_model, num_classes=2):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, inputs):
        # Unpack input_ids from the tuple
        if isinstance(inputs, tuple):
            input_ids = inputs[0]
        else:
            input_ids = inputs

        outputs = self.backbone(input_ids=input_ids)
        pooled_output = outputs[:, 0, :]  # Use the [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits

def analyze_model_structure(model):
    """Analyze model structure to understand which parts are Mamba vs Transformer"""
    layer_info = []
    
    # Analyze backbone layers
    for i, layer in enumerate(model.backbone.layers):
        if isinstance(layer, JambaMambaDecoderLayer):
            layer_info.append({
                'index': i,
                'type': 'Mamba',
                'params': sum(p.numel() for p in layer.parameters())
            })
        elif isinstance(layer, JambaAttentionDecoderLayer):
            layer_info.append({
                'index': i,
                'type': 'Transformer',
                'params': sum(p.numel() for p in layer.parameters())
            })
        else:
            layer_info.append({
                'index': i,
                'type': layer.__class__.__name__,
                'params': sum(p.numel() for p in layer.parameters())
            })
    
    # Calculate totals
    total_params = sum(layer['params'] for layer in layer_info)
    mamba_params = sum(layer['params'] for layer in layer_info if layer['type'] == 'Mamba')
    transformer_params = sum(layer['params'] for layer in layer_info if layer['type'] == 'Transformer')
    
    summary = {
        'total_params': total_params,
        'mamba_params': mamba_params,
        'transformer_params': transformer_params,
        'mamba_percentage': mamba_params / total_params * 100 if total_params > 0 else 0,
        'transformer_percentage': transformer_params / total_params * 100 if total_params > 0 else 0,
    }
    
    return layer_info, summary

def train_pipeline_jamba(args):
    # Initialize distributed environment
    local_rank = args.local_rank
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Create run directory
    run_dir = f'./runs/jamba_pp{args.pipeline_parallel_size}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize WandB if requested
    if args.use_wandb and rank == 0:
        wandb_run_name = args.wandb_run_name or f"jamba-pp{args.pipeline_parallel_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config={
                "pipeline_parallel_size": args.pipeline_parallel_size,
                "batch_size": args.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "steps": args.steps,
                "seed": args.seed,
                "seq_length": args.seq_length,
            }
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create dataset
    train_set, vocab_size, pad_token_id = get_imdb_dataset(
        seq_len=512, batch_size=32,
    )
    
    # Create JambaConfig
    jamba_config = {
        "vocab_size": vocab_size,
        "hidden_size": 128,
        "num_hidden_layers": 7,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "expert_layer_offset": 1,
        
        "attn_layer_period": 6,
        "attn_layer_offset": 1,
        
        "use_mamba_kernels": False, 
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "output_router_logits": False,
    }
    
    # Create model configuration
    config_path = os.path.join('.', 'jamba_config.json')
    jambaconfig = JambaConfig(**jamba_config)

    # Create base model
    model = JambaForSequenceClassification(jambaconfig)
    model = ClassificationModel(model.model, model.model.embed_tokens.embedding_dim, num_classes=2)
    
    # Analyze model structure - this is important for hybrid models like Jamba
    if rank == 0:
        layer_info, summary = analyze_model_structure(model)
        print(f"Model structure analysis:")
        print(f"Total parameters: {summary['total_params']:,}")
        print(f"Mamba parameters: {summary['mamba_params']:,} ({summary['mamba_percentage']:.2f}%)")
        print(f"Transformer parameters: {summary['transformer_params']:,} ({summary['transformer_percentage']:.2f}%)")
        
        # Save analysis to file
        with open(os.path.join(run_dir, 'model_analysis.json'), 'w') as f:
            json.dump({
                'layer_info': layer_info,
                'summary': summary
            }, f, indent=2)
    
    # Create pipeline model layers
    # For pipeline parallelism, tag each layer with its type for better monitoring
    pipeline_layers = [
        BlockPipe(model.backbone.embed_tokens, "Embeddings")
    ]
    
    for i, layer in enumerate(model.backbone.layers):
        layer_type = "MambaLayer" if isinstance(layer, JambaMambaDecoderLayer) else "TransformerLayer"
        pipeline_layers.append(BlockPipe(layer, layer_type))
    
    pipeline_layers.extend([
        BlockPipe(model.backbone.final_layernorm, "LayerNorm"),
        lambda x: x[:, 0, :],  # Extract [CLS] token
        model.classifier
    ])
    
    # Create loss function for pipeline
    def loss_fn(outputs, labels):
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss
    
    # Create pipeline model
    pipe_model = PipelineModule(
        layers=pipeline_layers,
        loss_fn=loss_fn,
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
        activation_checkpoint_interval=0
    )
    
    # Print pipeline configuration
    if rank == 0:
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Micro batch size: {args.micro_batch_size}")
        print(f"Number of pipeline stages: {pipe_model.num_stages}")
    
    # Initialize monitoring tools
    gpu_monitor = GPUMonitor(log_dir=os.path.join(run_dir, 'gpu_stats'))
    custom_profiler = CustomProfiler(
        pipe_model=pipe_model,
        rank=rank,
        world_size=world_size,
        log_dir=os.path.join(run_dir, 'custom_profile')
    )
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=pipe_model.parameters(),
        training_data=train_set,
    )
    
    # Setup PyTorch Profiler
    profiler_log_dir = os.path.join(run_dir, 'profiler')
    os.makedirs(profiler_log_dir, exist_ok=True)
    
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    
    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=4, 
                                         warmup=4,
                                         active=30,
                                         repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            profiler_log_dir, 
            worker_name=f"worker_{rank}"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    
    # Setup TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tensorboard'))
    
    # Save pipeline structure information
    if rank == 0:
        stage_info = {}
        for stage in range(pipe_model.num_stages):
            stage_ranks = get_ranks_in_stage(stage, world_size, pipe_model.num_stages)
            stage_info[f"stage_{stage}"] = {
                "ranks": stage_ranks,
                "num_ranks": len(stage_ranks)
            }
        
        with open(os.path.join(run_dir, 'pipeline_structure.json'), 'w') as f:
            json.dump(stage_info, f, indent=2)
    
    # Training loop
    prof.start()
    for step in range(args.steps):
        # Time the forward+backward pass
        with timer("train_step", rank, step, custom_profiler.timing_stats):
            loss = engine.train_batch()
        
        prof.step()
        
        # Log metrics
        if step % args.profiling_freq == 0:
            # Log GPU metrics
            gpu_monitor.log_step(step)
            
            # Log memory usage
            custom_profiler.log_memory(step)
            
            # Collect and log timings
            if rank == 0 and writer:
                writer.add_scalar('Loss/train', loss.item() if isinstance(loss, torch.Tensor) else loss, step)
                
                # Log to WandB if enabled
                if args.use_wandb:
                    wandb_metrics = {
                        'train/loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
                        'train/step': step,
                    }
                    
                    # Add GPU metrics if we can gather them
                    if hasattr(gpu_monitor, 'stats') and len(gpu_monitor.stats) > 0:
                        latest_stats = gpu_monitor.stats[-1]['gpu_stats']
                        for i, gpu_stat in enumerate(latest_stats):
                            wandb_metrics[f'gpu/{i}/util_percent'] = gpu_stat['gpu_utilization']
                            wandb_metrics[f'gpu/{i}/mem_percent'] = gpu_stat['memory_percent']
                    
                    wandb.log(wandb_metrics, step=step)
        
        # Print progress
        if rank == 0 and step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
    
    # Stop profiling and save statistics
    prof.stop()
    gpu_monitor.save_stats()
    custom_profiler.save_stats()
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    # Close WandB
    if args.use_wandb and rank == 0:
        wandb.finish()

if __name__ == '__main__':
    args = get_args()
    deepspeed.init_distributed(dist_backend=args.backend)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(args.local_rank)
    train_pipeline_jamba(args)
    