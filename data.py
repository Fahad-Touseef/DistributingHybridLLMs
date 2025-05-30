from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

def get_clm_dataloader(
    tokenizer_name="gpt2",
    dataset_name="wikitext",
    dataset_config="wikitext-103-v1",
    seq_len=512,
    batch_size=4,
    streaming=False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=streaming)
    dataset = dataset.select(range(100))  # Use only the first 100 samples for testing.

    def tokenize(example):
        return tokenizer(
            example["text"], 
            truncation=True, 
            max_length=seq_len, 
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    def group_texts(examples):
        # Concatenate texts and chunk into blocks of seq_len
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_len, seq_len)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    

    print(len(tokenizer), tokenizer.vocab_size) 
    lm_dataset.set_format(type="torch", columns=["input_ids", "labels"])
    return DataLoader(lm_dataset, batch_size=batch_size, shuffle=not streaming), len(tokenizer), tokenizer.pad_token_id

class IMDBDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        input_ids = item["input_ids"]
        labels = item["labels"]
        return input_ids, labels

def get_imdb_dataset(
    tokenizer_name="bert-base-uncased",
    seq_len=512,
    batch_size=64,
    streaming=False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = load_dataset("imdb", split="train", streaming=streaming)
    total = len(dataset)
    sel = (total // batch_size) * batch_size
    dataset = dataset.select(range(sel))

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=seq_len,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    print(len(tokenizer), tokenizer.vocab_size) 
    return IMDBDataset(tokenized_dataset), len(tokenizer), tokenizer.pad_token_id