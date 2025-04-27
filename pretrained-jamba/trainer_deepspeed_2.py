from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec


# ------------- ARGUMENTS -------------
class Args:
    batch = 16
    seed = 42
    local_rank = int(os.getenv('LOCAL_RANK', -1))

args = Args()
deepspeed.init_distributed()

torch.manual_seed(args.seed)
if args.local_rank != -1:
    torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank if args.local_rank != -1 else 0)

# ------------- DATASET -------------
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

# ------------- MODEL -------------
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
jambaconfig = JambaConfig(**jamba_config)
model = JambaForSequenceClassification(jambaconfig)

# ------------- PIPELINE WRAPPER -------------
class Stage0(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layer0 = model.model.layers[0]

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embed_tokens(input_ids)
        x = self.layer0(x)
        return x, labels

class Stage1(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layer1 = model.model.layers[1]
        self.final_layernorm = model.model.final_layernorm
        self.score = model.score

    def forward(self, hidden_states, labels=None):
        x = self.layer1(hidden_states)
        x = self.final_layernorm(x)
        logits = self.score(x[:, 0, :])  # [CLS] token
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss
        return logits

pipeline_layers = [
    LayerSpec(Stage0, model),
    LayerSpec(Stage1, model),
]

pipeline_model = PipelineModule(
    layers=pipeline_layers,
    loss_fn=None,
    num_stages=2,
    partition_method="parameters",
)

# ------------- DEEPSPEED CONFIG -------------
ds_config = {
    "train_micro_batch_size_per_gpu": args.batch // 4,
    "gradient_accumulation_steps": 1,
    "fp16": {
        "enabled": True
    },
    "pipeline": {
        "enabled": True,
        "micro_batches": 4,
        "partitions": 2
    },
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
}

# ------------- DEEPSPEED INITIALIZATION -------------
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=pipeline_model,
    config=ds_config,
    model_parameters=pipeline_model.parameters()
)

# ------------- PROFILING SETUP -------------
activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
prof = profile(
    activities=activities,
    schedule=torch.profiler.schedule(wait=2, warmup=2, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/jamba/profiler_{datetime.now().strftime("%Y%m%d-%H%M%S")}', worker_name="worker"),
    record_shapes=False,
    profile_memory=True,
    with_stack=False,
)
prof.start()

print("model_engine:", model_engine)  # Check model engine

# ------------- TRAINING LOOP -------------
writer = SummaryWriter(log_dir='./logs/jamba')

epochs = 1
for epoch in range(epochs):
    total_loss = 0
    time_per_batch = []
    train_iter = iter(train_dataloader)
    batch = next(train_iter)

    # Rename the keys to match DeepSpeed expected keys
    inputs = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    # If your model needs attention mask separately, you must handle it too
    model_inputs = {
        "inputs": inputs,          # DeepSpeed expects 'inputs' key
        "attention_mask": attention_mask,
        "labels": labels           # if your loss needs labels
    }

    # Now call train_batch
    loss = model_engine.train_batch(data_iter=iter([model_inputs])) 

    # for step, batch in enumerate(train_dataloader):
    #     prof.step()
    #     start_time = time.time()
    #     if step >= 2 + 2 + 5:  # Only for profiling purpose
    #         break

    #     input_ids = batch['input_ids'].to(device)
    #     attention_mask = batch['attention_mask'].to(device)
    #     labels = batch['labels'].to(device)
        

    #     end_time = time.time()
    #     total_loss += loss.item()
    #     time_per_batch.append(end_time - start_time)

    #     writer.add_scalar('Loss/train_batch', loss.item(), epoch * len(train_dataloader) + step)
    #     print(f"[Epoch {epoch}] Step {step}: Batch Time {end_time - start_time:.2f}s, Loss {loss.item():.4f}")

    #     del input_ids, attention_mask, labels

    # avg_loss = total_loss / len(train_dataloader)
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    # print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    # writer.add_scalar('Loss/train_epoch', avg_loss, epoch)

prof.stop()
writer.close()

print("Training complete!")
