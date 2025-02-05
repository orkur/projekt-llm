from transformers import GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

dataset = load_dataset("json", data_files={"train": "messages.jsonl"}, split="train")
print("Data loaded")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer set")

token_lengths = [len(tokenizer(text)["input_ids"]) for text in dataset["text"]]
print(f"90th percentile token length: {np.percentile(token_lengths, 90)}")