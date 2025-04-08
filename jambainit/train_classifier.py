import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch.optim as optim
from transformers import DataCollatorWithPadding
import sys
from transformers import AutoTokenizer
sys.path.append('/jamba')
from jamba.model import JambaSequenceClassfication
from jamba.model import  JambaConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
  
    # Load the IMDB dataset from Hugging Face
    imdb = load_dataset("imdb")
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)  # padding left to collator
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    tokenized_imdb = tokenized_imdb.remove_columns(["text"])

    tokenized_imdb.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_imdb["test"],
        batch_size=32,
        collate_fn=data_collator
    )

    jambaconfig = JambaConfig(
        dim=512,
        depth=1,
        num_tokens=100,
        d_state=256,
        d_conv=128,
        heads=8,
        num_experts=8,
        num_experts_per_token=2,
        return_embeddings=True,
        num_classes=2,
    )
    model = JambaSequenceClassfication(jambaconfig)

    model.to(device)
    # Loss function and optimizer

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()  # Zero the gradients
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(
                outputs, targets
            )
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Training complete!")

if __name__ == "__main__":
    main()


