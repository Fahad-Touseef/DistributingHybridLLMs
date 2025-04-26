from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from accelerate import Accelerator
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
from datetime import datetime
import deepspeed
import argparse


def main(args):

    # Load the IMDB dataset from Hugging Face
    if args.local_rank == -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda", args.local_rank)
    torch.manual_seed(args.seed)
    
    batch_size = args.batch
    imdb = load_dataset("imdb")
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length= 700)  # padding left to collator
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True,max_length=700)
    #outputs = model.generate(input_ids, max_new_tokens=216)

    train_dataloader = DataLoader(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
    )

    ds_config = {
        "train_batch_size": batch_size,  # Adjust to your needs
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2,  # Stage 2 is usually a good balance
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            }
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
        "gradient_accumulation_steps": 1,
        "wall_clock_breakdown": False
    }




    jamba_config = {
        "vocab_size": len(tokenizer.vocab),
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,

        # Expert / MoE (optional)
        "num_experts_per_tok": 1,
        "num_experts": 2,
        "expert_layer_offset": 1,

        # Attention layer config
        "attn_layer_period": 1,
        "attn_layer_offset": 1,

        # Mamba-specific config
        "use_mamba_kernels": False,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
    }
    # config = JambaConfig(
    #     vocab_size=50257,
    #     dim=512,
    #     num_hidden_layers=8, # number of hidden layers in transformer block

    #     hidden_size=256,
    #     attn_layer_period = 8, # every 4 layer there is a attention layer
    #     attention_offset = 0, # offset for attention layer
    #     num_attention_heads=8,
    #     num_key_value_heads = 8, # if equal with num_attention_heads, then use MultiheadAttention

    #     d_conv=4,
    #     d_state=256,
    #     num_experts_per_tok = 2,  # a router choosing 2 experts per token
    #     num_experts=2, # total number of experts
    #     expert_layer_period =4, # every 4 layer there is a expert layer
    # )
    jambaconfig = JambaConfig(**jamba_config)
    model  = JambaForSequenceClassification(jambaconfig)
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    #summary writer
    # TensorBoard writer setup
    writer = SummaryWriter(log_dir='./logs/jamba')

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("model.txt", "w") as f:
        f.write(str(model))
    #model.to(device)
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
    epochs = 1
    for epoch in range(epochs):
        #model.train() 
        total_loss = 0
        
        time_per_batch = []
        for step, batch in enumerate(train_dataloader):
            prof.step()
            start_time = time.time()
            if step >= 2 + 2+ 5:
                break
            #optimizer.zero_grad()  # Zero the gradients # deepspeed handles this
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            #batch = batch.to(device)
            #print("Data_size", get_tensor_size_in_mb(inputs)+ get_tensor_size_in_mb(attention_mask) + get_tensor_size_in_mb(targets))
            # Forward pass
            #with profile(activities=activities, profile_memory=True) as p:
            #p.export_chrome_trace(f"epoch_{epoch+1}_step_{step} trace.json")
            with record_function("model_forward"):
                outputs = model_engine(inputs, attention_mask=attention_mask, labels=targets) 
            loss = outputs.loss
            # Backward pass and optimize
            #loss.backward()
            model_engine.backward(loss)
            model_engine.step()
            end_time = time.time()
            total_loss += loss
            # Optional: log batch loss too
            time_per_batch.append(end_time - start_time)
            writer.add_scalar('Loss/train_batch', loss, epoch * len(train_dataloader) + step)
            print(f"Time taken for batch: {end_time - start_time:.2f}", )

            del inputs
            del attention_mask
            del targets
            del outputs
            # torch.cuda.empty_cache()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    prof.stop()
    writer.close()
    print("Training complete!")

    # add model evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed Training Example')
    parser.add_argument('--seed', type=int, default=42, metavar='S')
  
    # Add this line to accept local_rank from DeepSpeed
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='local rank passed from distributed launcher')
    parser.add_argument('--batch', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()
    
    main(args)


    
        



