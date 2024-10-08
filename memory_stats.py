import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=77).to(device)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

dataset = load_dataset("mteb/banking77")
dataset = dataset['test']
n = 20
dataset = concatenate_datasets([dataset] * n)

def tokenize_fn(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_dataset = dataset.map(tokenize_fn, batched=True)

def create_dataloader(batch_size):
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    print(f"Allocated GPU Memory: {allocated_memory / (1024*1024*1024):.2f} GB")
    print(f"Reserved GPU Memory: {reserved_memory / (1024*1024*1024):.2f} GB")

batch_size = 2
max_batch_size = None

torch.cuda.empty_cache()

while True:
    try:
        dataloader = create_dataloader(batch_size)
        for batch in dataloader:
            input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']], dim=1).to(device)
            attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in batch['attention_mask']], dim=1).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            print(f"Current Batch Size: {batch_size}")
            print_gpu_memory()

            batch_size = batch_size + 1
            break 
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"GPU Out of memory at batch size: {batch_size}")
            max_batch_size = batch_size - 1
            break
        else:
            raise e

print(f"Maximum batch size before GPU goes out of memory for {model_name}: {max_batch_size}")

print("\nModel Inference Speed for Batch Size of 1:")
dataloader = create_dataloader(1)
for batch in dataloader:
    input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']], dim=1).to(device)
    attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in batch['attention_mask']], dim=1).to(device)
    
    start_time = time.time()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
    
    print(f"Model Inference Speed for Batch Size of 1: {1000 * (end_time - start_time):.4f} ms")
    break

if max_batch_size:
    print(f"\nModel Inference Speed for Maximum Batch Size of ({max_batch_size}):")
    dataloader = create_dataloader(max_batch_size)
    for batch in dataloader:
        input_ids = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in batch['input_ids']], dim=1).to(device)
        attention_mask = torch.stack([torch.tensor(mask, dtype=torch.long) for mask in batch['attention_mask']], dim=1).to(device)
        
        start_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        end_time = time.time()
        
        print(f"Model Inference Speed for Maximum Batch Size of ({max_batch_size}): {1000 * (end_time - start_time):.4f} ms")
        break