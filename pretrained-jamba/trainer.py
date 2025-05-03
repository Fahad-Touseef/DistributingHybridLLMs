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
import argparse

def main(args):

    # Load the IMDB dataset from Hugging Face
    
    torch.manual_seed(args.seed)
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
        batch_size=16,
        collate_fn=data_collator
    )

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
        "attn_layer_period": 2,
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

    #summary writer
    # TensorBoard writer setup
    writer = SummaryWriter(log_dir='./logs/jamba')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    epochs = 3
    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        # start profiling after epoch 1
        if epoch == 1:
            prof = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=2, warmup=2, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/jamba/profiler{epoch}'),
                record_shapes=False,
                profile_memory=True,
                with_stack=False
            )
            prof.start()
        time_per_batch = []
        for step, batch in enumerate(train_dataloader):
            if epoch ==1:
                prof.step()
            start_time = time.time()
            optimizer.zero_grad()  # Zero the gradients
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            #print("Data_size", get_tensor_size_in_mb(inputs)+ get_tensor_size_in_mb(attention_mask) + get_tensor_size_in_mb(targets))
            # Forward pass
            #with profile(activities=activities, profile_memory=True) as p:
            outputs = model(input_ids = inputs, attention_mask=attention_mask,labels = targets) 
            #p.export_chrome_trace(f"epoch_{epoch+1}_step_{step} trace.json")
            loss = outputs.loss
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
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
            torch.cuda.empty_cache()
        if epoch == 1:
            prof.stop()
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
    writer.close()
    print("Training complete!")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--seed', type=int, default=42, metavar='S')
    args = parser.parse_args()
    main(args)


    
        



