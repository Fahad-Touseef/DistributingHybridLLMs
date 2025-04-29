import torch.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
from datasets import load_dataset
import torch
import deepspeed
import argparse
import torch
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers.models.jamba.modeling_jamba import (
    JambaForSequenceClassification,
    JambaMambaDecoderLayer,
    JambaRMSNorm,
    JambaAttentionDecoderLayer
)
import torchvision
from torchvision import transforms
import os
from torch.utils.data import IterableDataset


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
        return (input_ids, label)

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return input_ids, labels

class PipelineCompatibleDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __iter__(self):
        for batch in self.dataset:
            # Convert dict of tensors to a list of inputs for pipeline
            # For most transformer models, we need [input_ids, attention_mask, labels]
            yield [batch["input_ids"], batch["attention_mask"], batch["labels"]]

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
            mask, input = inputs
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
# one possiblity for passing the data as input 
class JambaPipelineModule(JambaForSequenceClassification):
    def forward(self, inputs):
        hidden, mask = inputs
        output = super().forward(hidden, mask)
        return output
    

def join_jamba_layers(jamba_model):
    '''
    layer partitioning for pipeline parallelism
    '''
    print("join init layer", jamba_model)
    layers = []

    # Embedding layer first
    layers.append(jamba_model.model.embed_tokens)

    # Then the transformer/mamba decoder layers
    for layer in jamba_model.model.layers:
        layers.append(layer)

    # Final normalization
    layers.append(jamba_model.model.final_layernorm)

    # Final classification head (score projection)
    layers.append(jamba_model.score)

    return layers

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
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
        # partition_method='uniform',
        activation_checkpoint_interval=0
    )
    
    return pipe_model

def dummy_trainset(local_rank, dl_path='/tmp/cifar10-data'):
    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = DummyTextDataset(
        tokenizer=AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b"),
        seq_length=512)
    if local_rank == 0:
        dist.barrier()
    return trainset

def train_pipeline_jamba(args):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    print("World size", torch.distributed.get_world_size())
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
   
    jamabalayers = join_jamba_layers(model)
    print("Jamba layers============", jamabalayers)
    # Create specs for pipeline parallelism
    # specs = create_pipeline_specs(model)
    # print("Pipeline specs============", specs)
    
    # Create pipeline model
    #pipe_model = create_pipeline_model(model, specs, args)
    pipe_model = PipelineModule(
        layers=jamabalayers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
        # partition_method='uniform',
        activation_checkpoint_interval=0
    )

    print("Pipeline model============", pipe_model)
    
    # Create dataset
    # trainset = DummyTextDataset(
    #     tokenizer=tokenizer,
    #     seq_length=args.seq_length
    # )

    #trainset = cifar_trainset(args.local_rank)
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
        batch_size=16,
        collate_fn=data_collator
    )

    pipeline_dataset = PipelineCompatibleDataset(train_dataloader)
    train_iter = deepspeed.utils.RepeatingLoader(pipeline_dataset)
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=[p for p in pipe_model.parameters() if p.requires_grad]

    )
    
    # Print pipeline configuration
    if dist.get_rank() == 0:
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Micro batch size: {engine.train_micro_batch_size_per_gpu()}")
        print(f"Number of pipeline stages: {pipe_model.num_stages}")
        
    # Training loop
    for step in range(args.steps):
        loss = engine.train_batch(data_iter=train_iter)
        
        # Print progress
        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
class BlockPipe(nn.Module):
    """Wrap a Block so it only propagates `hidden_states` onward."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        # block(x) returns (hidden_states, residual)
        hidden_states = self.block(x[0],x[1])
        return hidden_states

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
        batch_size=16,
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
    
    deepspeed.init_distributed()
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(args.local_rank)
    
    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipeline_jamba(args)