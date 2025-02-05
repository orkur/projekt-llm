
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU device name: {torch.cuda.get_device_name(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


dataset = load_dataset("json", data_files={"train": "messages.jsonl"}, split="train")
print("data loaded")

model_name = "flax-community/papuGaPT2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
print("tokenizer set")

def tokenize_function(examples):
    tokenized_input = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
    tokenized_input["labels"] = tokenized_input["input_ids"].copy()
    return tokenized_input

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.float16)
print("Model loaded")
    
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=2,
    lora_alpha=8,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
print("PEFT model prepared")

training_args = TrainingArguments(
    output_dir = "./papuGaPT2-finetune",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_steps=100,
    fp16=True,
    warmup_steps=500,
    resume_from_checkpoint=False 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer
)

trainer.train()
print("Training complete")