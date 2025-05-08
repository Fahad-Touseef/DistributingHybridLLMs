from transformers import MistralConfig, MistralForSequenceClassification, AutoTokenizer
from deepspeed.pipe import PipelineModule
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
import deepspeed
import torch
import torch.distributed as dist
from datetime import datetime
from data import get_imdb_dataset
import argparse
import os



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
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

class BlockPipe(nn.Module):
    """Wrap a Block so it only propagates `hidden_states` onward."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        # block(x) returns (hidden_states, residual)
        output = self.block(x)
        return output[0] if isinstance(output, tuple) else output

def mistral_pipeline_layers(model):
    return [
        model.model.embed_tokens,                   # Embedding
        *[BlockPipe(layer) for layer in model.model.layers],  # Transformer layers
        model.model.norm,                           # Final layernorm
        lambda x: x[:, 0, :],                        # Pooling (CLS or first token)
        model.score                                  # Classification head
    ]

def train_pipeline_mistral(args):
    # Load tokenizer (you may change to mistralai/Mistral-7B-v0.1 if needed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load dataset
    train_set, vocab_size, pad_token_id = get_imdb_dataset(seq_len=512, batch_size=32)

    # Custom Mistral config
    config = MistralConfig(
        vocab_size=vocab_size,
        hidden_size=512,  # can tune
        intermediate_size=2048,
        num_hidden_layers=6,  # customize number of layers
        num_attention_heads=8,
        max_position_embeddings=4096,
        num_labels=2,
        pad_token_id=pad_token_id
    )

    # Build model from config
    model = MistralForSequenceClassification(config)
    print(model)

    # Set model in train mode
    model.train()

    temp = [BlockPipe(layer) for layer in model.model.layers]

    layers = [
        model.model.embed_tokens,
        *temp,
        model.model.norm,
        lambda x: x[:, 0, :],  # simple pooling (adjust if needed)
        model.score
    ]

    # Loss function
    def loss_fn(outputs, labels):
        return nn.CrossEntropyLoss()(outputs, labels)

    # PipelineModule
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=loss_fn,
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
        activation_checkpoint_interval=0,
    )

    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=pipe_model.parameters(),
        training_data=train_set,
    )

    if dist.get_rank() == 0:
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Micro batch size: {engine.train_micro_batch_size_per_gpu()}")
        print(f"Number of pipeline stages: {pipe_model.num_stages}")

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=4, warmup=4, active=30, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f'./logs/mistral/profiler_{datetime.now().strftime("%Y%m%d-%H%M%S")}', worker_name="worker"),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )

    prof.start()
    writer = SummaryWriter(log_dir='./logs/mistral-pipeline')

    # Training loop
    for step in range(args.steps):
        loss = engine.train_batch()
        prof.step()
        writer.add_scalar('Loss/train', loss.item(), step)

        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
    prof.stop()
    writer.close()
if __name__ == '__main__':
    args = get_args()
    
    deepspeed.init_distributed(dist_backend=args.backend)
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(args.local_rank)
    train_pipeline_mistral(args)