from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
from datetime import datetime
import deepspeed
import argparse


import os
import argparse
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers import AutoTokenizer
from transformers.models.jamba.configuration_jamba import JambaConfig
from transformers.models.jamba.modeling_jamba import (
    JambaForSequenceClassification,
    JambaMambaDecoderLayer,
    JambaRMSNorm,
    JambaAttentionDecoderLayer
)
import torch.nn.functional as F

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
                        default=4,
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
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

class DummyTextDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, seq_length, size=10000, num_classes=2):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.size = size
        self.num_classes = num_classes
        self.vocab_size = len(tokenizer.vocab)
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Generate random token ids
        input_ids = torch.randint(1, self.vocab_size, (self.seq_length,))
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {'input_ids': input_ids, 'labels': label}

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return input_ids, labels

class JambaPipelineWrapper(torch.nn.Module):
    """
    Wrapper class for Jamba layers to adapt them for DeepSpeed pipeline parallelism
    """
    def __init__(self, layer_type, layer_name, model):
        super().__init__()
        self.layer_type = layer_type
        self.layer_name = layer_name
        
        # Get the actual layer from the model using layer_name as a path
        parts = layer_name.split('.')
        curr_module = model
        for part in parts:
            if part.isdigit():
                curr_module = curr_module[int(part)]
            else:
                curr_module = getattr(curr_module, part)
        
        self.layer = curr_module
        
    def forward(self, inputs):
        if self.layer_type == "embed":
            return self.layer(inputs)
        elif self.layer_type == "mamba_layer" or self.layer_type == "attention_layer":
            return self.layer(inputs)
        elif self.layer_type == "layernorm":
            return self.layer(inputs)
        elif self.layer_type == "head":
            # For the classification head
            sequence_output = inputs
            pooled_output = sequence_output[:, -1]  # Use last token
            return self.layer(pooled_output)
        else:
            raise ValueError(f"Unsupported layer type: {self.layer_type}")

def create_pipeline_specs(model):
    """Create layer specs for pipeline parallelism based on model architecture"""
    specs = [
        {"tied": None, "type": "embed", "name": "model.embed_tokens"},
    ]
    
    # Add layer specs for each layer
    for i in range(len(model.model.layers)):
        if isinstance(model.model.layers[i], JambaMambaDecoderLayer):
            specs.append({"tied": None, "type": "mamba_layer", "name": f"model.layers.{i}"})
        elif isinstance(model.model.layers[i], JambaAttentionDecoderLayer):
            specs.append({"tied": None, "type": "attention_layer", "name": f"model.layers.{i}"})
        else:
            specs.append({"tied": None, "type": "layer", "name": f"model.layers.{i}"})
    
    # Add final layernorm and classification head
    specs.append({"tied": None, "type": "layernorm", "name": "model.final_layernorm"})
    specs.append({"tied": None, "type": "head", "name": "score"})
    
    return specs

def create_pipeline_model(model, specs, args):
    """Create a PipelineModule from model and specs"""
    layers = []
    
    for spec in specs:
        layer_type = spec["type"]
        layer_name = spec["name"]
        
        # Create a wrapper for each layer
        layer = JambaPipelineWrapper(layer_type, layer_name, model)
        layers.append(layer)
    
    # Create PipelineModule
    loss_fn = lambda x, y: F.cross_entropy(x, y)
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=loss_fn,
        num_stages=args.pipeline_parallel_size,
        partition_method='uniform',
        activation_checkpoint_interval=0
    )
    
    return pipe_model

def train_pipeline_jamba(args):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create JambaConfig
    jamba_config = {
        "vocab_size": len(tokenizer.vocab),
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "expert_layer_offset": 1,
        
        "attn_layer_period": 1,
        "attn_layer_offset": 1,
        
        "use_mamba_kernels": False, 
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    }
    
    # Create model configuration
    jambaconfig = JambaConfig(**jamba_config)
    
    # Create base model (not for training, just for extracting structure)
    model = JambaForSequenceClassification(jambaconfig)
    
    # Create specs for pipeline parallelism
    specs = create_pipeline_specs(model)

    print("Pipeline specs:", specs)
    
    # Create pipeline model
    pipe_model = create_pipeline_model(model, specs, args)
    
    # Create dataset
    trainset = DummyTextDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length
    )
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=[p for p in pipe_model.parameters() if p.requires_grad],
        training_data=trainset
    )
    
    # Print pipeline configuration
    if dist.get_rank() == 0:
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Micro batch size: {engine.train_micro_batch_size_per_gpu()}")
        print(f"Number of pipeline stages: {pipe_model.num_stages}")
        
    # Training loop
    for step in range(args.steps):
        loss = engine.train_batch()
        
        # Print progress
        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")

class RepeatingLoader:
    """Iterator wrapper that keeps reusing the iterator for multiple epochs."""
    def __init__(self, loader):
        self.loader = loader
        self.iterator = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)

def train_base(args):
    """Training without pipeline parallelism for comparison"""
    torch.manual_seed(args.seed)
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create JambaConfig
    jamba_config = {
        "vocab_size": len(tokenizer.vocab),
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "expert_layer_offset": 1,
        
        "attn_layer_period": 1,
        "attn_layer_offset": 1,
        
        "use_mamba_kernels": False, 
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    }
    
    # Create model configuration
    jambaconfig = JambaConfig(**jamba_config)
    
    # Create model
    model = JambaForSequenceClassification(jambaconfig)
    
    # Create dataset
    trainset = DummyTextDataset(
        tokenizer=tokenizer,
        seq_length=args.seq_length
    )
    
    # Initialize DeepSpeed
    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=trainset
    )
    
    # Create repeating loader
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
    imdb = load_dataset("imdb")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=700, padding="max_length")

    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=700)



    train_dataloader = DataLoader(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=args.batch,
        collate_fn=data_collator
    )
    data_iter = iter(train_dataloader)
    
    # Training loop
    rank = dist.get_rank()
    print(f"Rank {rank} starting training...")
  
    total_steps = args.steps * engine.gradient_accumulation_steps()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    step = 0
    for micro_step in range(total_steps):
        batch = next(data_iter)
        print('batch:', batch)
        inputs = batch[0].to(engine.device)
        labels = batch[1].to(engine.device)
        
        outputs = engine(inputs).logits
        loss = criterion(outputs, labels)
        engine.backward(loss)
        engine.step()
        
        if micro_step % engine.gradient_accumulation_steps() == 0:
            step += 1
            if rank == 0 and (step % 10 == 0):
                print(f'step: {step:3d} / {args.steps:3d} loss: {loss.item()}')
                
if __name__ == '__main__':
    args = get_args()
    
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(args.local_rank)
    
    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipeline_jamba(args)