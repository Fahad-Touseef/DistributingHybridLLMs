from transformers import AutoModelForCausalLM, AutoTokenizer, JambaForSequenceClassification, JambaConfig
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time

def main():

    # Load the IMDB dataset from Hugging Face
    imdb = load_dataset("imdb")
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)  # padding left to collator
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    tokenized_imdb = tokenized_imdb.remove_columns(["text"])
    tokenized_imdb.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    #outputs = model.generate(input_ids, max_new_tokens=216)

    train_dataloader = DataLoader(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator
    )

    jamba_config = {
        "vocab_size": len(tokenizer.vocab),
        "hidden_size": 256,
        "intermediate_size": 14336,
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

    prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/jamba'),
            #record_shapes=True,
            #profile_memory=True,
            #with_stack=True,
            #with_flops=True,
            #with_modules=True
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    epochs = 5
    prof.start()
    for epoch in range(epochs):
        model.train() 
        total_loss = 0
        for batch in train_dataloader:
            start_time = time.time()
            optimizer.zero_grad()  # Zero the gradients
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)
            #print("Data_size", get_tensor_size_in_mb(inputs)+ get_tensor_size_in_mb(attention_mask) + get_tensor_size_in_mb(targets))
            # Forward pass
            outputs = model(input_ids = inputs, attention_mask=attention_mask,labels = targets) 
            loss = outputs.loss
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            end_time = time.time()
            total_loss += loss
            print("Time taken for batch: ", end_time - start_time)
        prof.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader.dataset)}")
    prof.stop()
    print("Training complete!")

if __name__ == "__main__":
    main()


    
        



