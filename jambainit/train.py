import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import sys
sys.path.append('/jamba')
from jamba.model import JambaSequenceClassfication
from jamba.model import  JambaConfig
from jamba.model import Jamba

import pynvml

from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda import memory_allocated, memory_reserved, memory_summary


def get_gpu_info(current_gpu):
    handle = pynvml.nvmlDeviceGetHandleByIndex(current_gpu)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_mem = mem_info.free
    print(f"GPU {current_gpu}: Free memory = {free_mem / 1024**2:.2f} MB")
    return free_mem/ 1024**2




# DataLoader
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    # Create targets by shifting input_ids one token to the left
    labels = torch.roll(input_ids, -1, dims=-1)
    return input_ids.squeeze(), labels.squeeze()



def main():
    pynvml.nvmlInit()
    # Load the dataset from Hugging Face
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=100,
            return_tensors="pt",
        )
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids"])

    dataloader = DataLoader(
    tokenized_datasets,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    )

    # Initialize the Jamba model with tokenizer's vocab size
    model = Jamba(
    dim=512,
    depth=2,
    num_tokens=tokenizer.vocab_size,
    d_state=128,
    d_conv=128,
    heads=1,
    num_experts=1,
    num_experts_per_token=2,
    )

    model.to(device)  # Move model to GPU if available
    # Move model to GPU if available
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    def estimate_model_size_in_gb(model):
        """Estimate model size in GB based on number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        bytes_per_param = 4  # assuming float32
        total_bytes = total_params * bytes_per_param
        return total_bytes / (1024 ** 3)  # GB

    print(f"Estimated model size: {estimate_model_size_in_gb(model):.2f} GB")
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    current_gpu = torch.cuda.current_device()

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            # Print GPU memory usage
            print_gpu_memory(device,"Before training step")
            # print(inputs.shape)
            # print(targets.shape)
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()  # Zero the gradients
            # Print GPU memory usage
            #print_gpu_memory(device,"After moving data to GPU")
            
            print("GPU Status:", get_gpu_info(current_gpu))
           
            print_gpu_memory(device,"Before training step")
            try:
                # Forward pass
                with profile(activities= activities, profile_memory=True) as prof:
                    with record_function("model_training"):
                        outputs = model(inputs)
                        print(f"GPU status after forward pass: {get_gpu_info(current_gpu)}")
                        loss = criterion(
                            outputs.transpose(1, 2), targets
                        )  # Adjust for cross-entropy expecting class dimension at dim=1

                        # Backward pass and optimize
                        loss.backward()
                        optimizer.step()
                prof.export_chrome_trace("trace.json")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("CUDA out of memory error. Skipping this batch.")
                    print(f"RuntimeError: {e}")
                    print(memory_summary(device))
                else:
                    raise e
            break
        break
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Training complete!")

def print_gpu_memory(device,step=""):
    """Print GPU memory usage at a specific step"""
    print(f"\n----- GPU Memory at {step} -----")
    print(f"Allocated: {memory_allocated(device) / 1e9:.2f} GB")
    print(f"Cached: {memory_reserved(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()



