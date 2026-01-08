import os, math, sacrebleu, numpy as np

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig
from trl import SFTTrainer

# This is the winner: EuroLLM LoRA fine-tuned BLEU score: 86.76

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Basic Parameters
load_dotenv()
MODEL_ID = os.getenv("EUROLLM_MODEL_PATH")
DATA_PATH = os.getenv("DATASET_PATH")
OUT_DIR = "./checkpoint"
ADAPTER_DIR = os.getenv("EUROLLM_LORA_ADAPTER")

MAX_LEN = 512
BATCH = 2
GRAD_ACC = 4
EPOCHS = 3
LR = 2e-5
WARMUP = 0.05
EVAL_EVERY_STEPS = 500

#Marker
MARKER = "### Answer:"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"MARKER raw: {repr(MARKER)}")
print("MARKER tokens:", tok.encode(MARKER, add_special_tokens=False))
print("MARKER decoded:", tok.decode(tok.encode(MARKER, add_special_tokens=False)))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "mps"},
    low_cpu_mem_usage=False,
    torch_dtype=torch.bfloat16
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# LoRa config
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Dataset split
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def merge_cols(ex):
    prompt = ex["prompt"].strip()
    completion = ex["completion"].strip()
    ex["text"] = f"{prompt}\n{MARKER}\n{completion}"
    return ex

ds = ds.map(merge_cols, remove_columns=("prompt", "completion"))

# Filter too long samples
def filter_length(example):
    tokens = tok.encode(example["text"], add_special_tokens=True)
    return len(tokens) <= MAX_LEN

print(f"Dataset size before filtering: {len(ds)}")
ds = ds.filter(filter_length)
print(f"Dataset size after filtering: {len(ds)}")

# Split train/eval
split = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

print("\n--- Testing data format ---")
sample_text = train_ds[0]["text"]
print("Sample text:", sample_text[:200] + "...")

# Counts token
tokens = tok.encode(sample_text, add_special_tokens=True)
print(f"Token count: {len(tokens)}")

# Verify that marker is present in the text
marker_in_text = MARKER in sample_text
print(f"Marker found in text: {marker_in_text}")

if marker_in_text:
    print("Data format is correct")
else:
    print("Marker not found in text - check formatting")
    print("First few examples:")
    for i in range(min(3, len(train_ds))):
        print(f"Example {i}: {train_ds[i]['text'][:100]}...")

# TrainingArguments
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=EVAL_EVERY_STEPS//5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    save_total_limit=1,
    fp16=False,
    bf16=False,
    dataloader_drop_last=False,
    remove_unused_columns=False,  #Important for SFTTrainer
)

# SFTTrainer
print("\n--- Initializing SFTTrainer ---")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_cfg,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("SFTTrainer initialized successfully")

# Training
print("\n--- Starting training ---")
trainer.train()

# Save LoRa Adapter
trainer.model.save_pretrained(ADAPTER_DIR)
print(f"Fine-tuning completed — LoRa adapter stored in {ADAPTER_DIR}")

# Statistics
effective_batch = BATCH * GRAD_ACC
steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
total_steps = steps_per_epoch * EPOCHS
print(f"  ~{steps_per_epoch} step/epoch → {total_steps} step in totale")

# Fast inference test
print("\n--- Quick inference test ---")
model.eval()

test_prompt = "Translate the following English text to IT: 'Hello world'"
test_input = f"{test_prompt}\n{MARKER}\n"

# Tokenization
inputs = tok.encode(test_input, return_tensors="pt", add_special_tokens=True)
print(f"Test input: {test_input}")
print(f"Input tokens: {inputs.shape}")

# Generating response
with torch.no_grad():
    outputs = model.generate(
        inputs.to(model.device),
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id
    )

# Output
full_text = tok.decode(outputs[0], skip_special_tokens=True)
generated_text = full_text[len(test_input):].strip()

print(f"Generated: {generated_text}")
print(f"Full output: {full_text}")

print("\n Training completed successfully!")