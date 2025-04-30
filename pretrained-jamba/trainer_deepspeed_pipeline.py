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
from datetime import datetime
# import torchvision
# from torchvision import transforms
import os
from torch.utils.data import IterableDataset
from data import get_imdb_dataset



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






# one possiblity for passing the data as input 
class JambaPipelineModule(JambaForSequenceClassification):
    def forward(self, inputs):
        hidden, mask = inputs
        output = super().forward(hidden, mask)
        return output



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

class BlockPipe(nn.Module):
    """Wrap a Block so it only propagates `hidden_states` onward."""
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x):
        # block(x) returns (hidden_states, residual)
        print("BlockPipe x:", type(x))
        output = self.block(x)
        print("BlockPipe output:", type(output))
        return output[0] if isinstance(output, tuple) else output



def train_pipeline_jamba(args):
    torch.manual_seed(args.seed)
    deepspeed.runtime.utils.set_random_seed(args.seed)
    print("World size", torch.distributed.get_world_size())
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Create JambaConfig
    train_set, vocab_size, pad_token_id = get_imdb_dataset(
        seq_len=512, batch_size = 16,
    ) 
    jamba_config = {
        "vocab_size": vocab_size,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "expert_layer_offset": 1,
        
        "attn_layer_period": 2,
        "attn_layer_offset": 1,
        
        "use_mamba_kernels": False, 
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "output_router_logits":False,
    }
#     jamba_config = {
#     "vocab_size": vocab_size,
#     "hidden_size": 128,
#     "num_hidden_layers": 4,
#     "num_attention_heads": 4,
#     "num_key_value_heads": 2,

#     "num_experts_per_tok": 1,
#     "num_experts": 2,
#     "expert_layer_offset": 1,

#     "attn_layer_period": 2,
#     "attn_layer_offset": 1,

#     "use_mamba_kernels": False,
#     "mamba_d_state": 16,
#     "mamba_d_conv": 4,
#     "mamba_expand": 2,
# }
    
    # Create model configuration
    jambaconfig = JambaConfig(**jamba_config)

    # Create base model (not for training, just for extracting structure)
    model = JambaForSequenceClassification(jambaconfig)

    model = ClassificationModel(model.model, model.model.embed_tokens.embedding_dim, num_classes=2)
   
    #jamabalayers = join_jamba_layers(model)
    #print("Jamba layers============", jamabalayers)
    # Create specs for pipeline parallelism
    # specs = create_pipeline_specs(model)
    # print("Pipeline specs============", specs)
   
    temp = [ BlockPipe(b) for b in model.backbone.layers]
  
    layers = [
        model.backbone.embed_tokens,
        *temp,
        model.backbone.final_layernorm,
        lambda x: x[:, 0, :],
        model.classifier
       
    ]
    # Create pipeline model
    #pipe_model = create_pipeline_model(model, specs, args)
    def loss_fn(outputs, labels):
        #outputs = outputs.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss
    
    pipe_model = PipelineModule(
        layers=layers,
        loss_fn=loss_fn,
        num_stages=args.pipeline_parallel_size,
        partition_method='parameters',
        # partition_method='uniform',
        activation_checkpoint_interval=0
    )
    
    print("Pipeline model============", pipe_model)
    
    # Initialize DeepSpeed
    engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=pipe_model,
        model_parameters=pipe_model.parameters(),
        training_data=train_set,
    )
    
    # Print pipeline configuration
    if dist.get_rank() == 0:
        print(f"Pipeline parallel size: {args.pipeline_parallel_size}")
        print(f"Micro batch size: {engine.train_micro_batch_size_per_gpu()}")
        print(f"Number of pipeline stages: {pipe_model.num_stages}")
        
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    # Training loop
    # start profiling after epoch 1
    prof = torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=2, # #during the first 2 epochs profile is not active
                                            warmup=2, # during this phase profiler starts tracing, but the results are discarded
                                            active = 5, # actively record the next 6 steps 
                                            repeat = 1), # specififes an uppper boun on the  number of cycles
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/jamba/profiler_{datetime.now().strftime("%Y%m%d-%H%M%S")}', worker_name="worker"),
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    )
    prof.start()

    # Training loop
    for step in range(args.steps):
        loss = engine.train_batch()
        prof.step()
        print(f"Step {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")
        # Print progress
        if dist.get_rank() == 0 and step % 10 == 0:
            print(f"Step: {step}, Loss: {loss.item() if isinstance(loss, torch.Tensor) else loss}")

    prof.stop()

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
    
    deepspeed.init_distributed(dist_backend=args.backend)
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(args.local_rank)
    
    if args.pipeline_parallel_size == 0:
        train_base(args)
    else:
        train_pipeline_jamba(args)