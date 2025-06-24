#!/usr/bin/env python
# fine_tune_mistral_chat_lora.py
# ---------------------------------------------------------
# Fine-tuning LoRA su Mistral-Small-Chat con dataset JSONL
# prompt/completion → template chat <s>[INST] … [/INST] …
# ---------------------------------------------------------

import os
from dotenv import load_dotenv

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ------------------------------------------------------------------
# 1. Impostazioni di base
# ------------------------------------------------------------------
load_dotenv()                                     # carica .env
MODEL_PATH   = os.getenv("MODEL_PATH")            # es.: mistralai/Mistral-7B-Instruct-v0.2
DATASET_PATH = os.getenv("DATASET_PATH")          # es.: /path/to/train.jsonl
OUTPUT_DIR   = "./mistral_finetuned"
MAX_LENGTH   = 256                                # token totali (prompt+risposta)
BATCH_SIZE   = 2                                  # su M-series Apple ≈ 15 GB di vRAM
GRAD_ACCUM   = 4                                  # batch effettivo = 8
NUM_EPOCHS   = 2
LR           = 1e-4                               # tipico per LoRA
WARMUP_RATIO = 0.05

# ------------------------------------------------------------------
# 2. Modello e tokenizer
# ------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,                    # MPS = fp16
    device_map="auto",
)
model.gradient_checkpointing_enable()            # meno RAM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token       = tokenizer.eos_token   # pad = eos
model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------------------------------------------------
# 3. LoRA
# ------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------------------------------------------
# 4. Dataset
# ------------------------------------------------------------------
raw_ds = load_dataset("json", data_files=DATASET_PATH, split="train")

train_size   = int(0.8 * len(raw_ds))
train_ds     = raw_ds.select(range(train_size))
eval_ds      = raw_ds.select(range(train_size, len(raw_ds)))

# ------------------------------------------------------------------
# 5. Tokenizzazione con maschera delle label
# ------------------------------------------------------------------
def tokenize_function(examples):
    # costruzione del template chat
    user_parts      = [f"<s>[INST] {p} [/INST]" for p in examples["prompt"]]
    assistant_parts = [f" {c}</s>" for c in examples["completion"]]

    # lunghezza del prompt (senza padding) per mascherare le label
    user_encodings = tokenizer(
        user_parts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # tokenizzazione del testo completo con padding
    full_encodings = tokenizer(
        [u + a for u, a in zip(user_parts, assistant_parts)],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    input_ids      = full_encodings["input_ids"]
    attention_mask = full_encodings["attention_mask"]

    labels = []
    for ids, user_ids in zip(input_ids, user_encodings["input_ids"]):
        user_len   = len(user_ids)                        # token da mascherare
        label_row  = [-100] * user_len + ids[user_len:]   # -100 = ignora loss
        label_row += [-100] * (MAX_LENGTH - len(label_row))
        labels.append(label_row[:MAX_LENGTH])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_ds = train_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=train_ds.column_names,
)
eval_ds = eval_ds.map(
    tokenize_function,
    batched=True,
    remove_columns=eval_ds.column_names,
)

# ------------------------------------------------------------------
# 6. Data collator (LM, non-MLM)
# ------------------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ------------------------------------------------------------------
# 7. TrainingArguments
# ------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="tensorboard",
)

# ------------------------------------------------------------------
# 8. Trainer
# ------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ------------------------------------------------------------------
# 9. Salvataggio degli adapter LoRA
# ------------------------------------------------------------------
model.save_pretrained("/Users/dmarcoaldi/LLM_Models/Mistral-Small-Finetuned-GPT")
print("Fine-tuning completato!")