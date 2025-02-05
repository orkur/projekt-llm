
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU device name: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


dataset = load_dataset("json", data_files={"train": "messages.jsonl"}, split="train")
dataset = dataset.train_test_split(test_size=0.05)
print("data loaded")

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token #doczytać
print("tokenizer set")

def tokenize_function(examples):
    tokenized_input = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    tokenized_input["labels"] = tokenized_input["input_ids"].copy()
    return tokenized_input
    

tokenized_datasets = dataset.map(tokenize_function, batched=True)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=128,
    n_embd=768,
    n_layer=12,
    n_head=12,
)
model = GPT2LMHeadModel(config)

training_args = TrainingArguments(
    output_dir="./scratch-model-gpt2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_steps=100,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    warmup_steps=200,
    learning_rate=3e-4,
    weight_decay=0.01,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

trainer.train()
print("Training complete")