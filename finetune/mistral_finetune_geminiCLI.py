#!/usr/bin/env python
# mistral_finetune_geminiCLI.py
# ---------------------------------------------------------------------------------
# Versione ottimizzata per Apple Silicon (M-series) con memoria unificata elevata.
#
# Modifiche principali:
# 1. Batch Size Aumentato: Sfrutta l'ampia memoria unificata (128GB) per
#    accelerare il training. BATCH_SIZE aumentato, GRAD_ACCUM ridotto.
# 2. Parametri LoRA Migliorati: Aumentati 'r' e 'lora_alpha' per migliorare
#    la capacità di apprendimento del modello, potenziando la qualità finale.
# 3. Training Esteso: Aumentato il numero di epoche per una migliore convergenza.
# 4. Attention Ottimizzata: Aggiunto 'attn_implementation="sdpa"' per usare
#    in modo esplicito l'implementazione di attention più efficiente su MPS.
#
# NOTA: Flash Attention 2 e la quantizzazione a 4-bit (QLoRA con BitsAndBytes)
# sono tecnologie basate su CUDA e non compatibili con l'hardware Apple.
# Le ottimizzazioni scelte sono specifiche per massimizzare le prestazioni su MPS.
# ---------------------------------------------------------------------------------

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
# 1. Impostazioni di base (ottimizzate per M4 Pro 128GB)
# ------------------------------------------------------------------
load_dotenv()
MODEL_PATH   = os.getenv("MODEL_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_DIR   = "./mistral_finetuned_geminiCLI"      # Directory di output separata
MAX_LENGTH   = 256
BATCH_SIZE   = 16                                 # Aumentato per sfruttare 128GB di RAM
GRAD_ACCUM   = 2                                  # Ridotto (batch effettivo = 32)
NUM_EPOCHS   = 3                                  # Aumentato per una migliore convergenza
LR           = 2e-5
WARMUP_RATIO = 0.05
# ---------------------------------

# ------------------------------------------------------------------
# 2. Modello e tokenizer
# ------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",                   # Ottimizzazione per MPS
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token       = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ------------------------------------------------------------------
# 3. LoRA (parametri potenziati)
# ------------------------------------------------------------------
lora_config = LoraConfig(
    # --- Ottimizzazioni Gemini CLI ---
    r=8,                                # Aumentato per maggiore capacità
    lora_alpha=8,                     # Adattato a 'r' (solitamente 2*r)
    # ---------------------------------
    lora_dropout=0.1,
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

# Suddivisione 80/20 per training e valutazione
train_size   = int(0.8 * len(raw_ds))
train_ds     = raw_ds.select(range(train_size))
eval_ds      = raw_ds.select(range(train_size, len(raw_ds)))

# ------------------------------------------------------------------
# 5. Tokenizzazione con maschera delle label
# ------------------------------------------------------------------
def tokenize_function(examples):
    user_parts      = [f"<s>[INST] {p} [/INST]" for p in examples["prompt"]]
    assistant_parts = [f" {c}</s>" for c in examples["completion"]]

    user_encodings = tokenizer(
        user_parts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )

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
        user_len   = len(user_ids)
        label_row  = [-100] * user_len + ids[user_len:]
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
# 6. Data collator
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
    logging_dir=f"{OUTPUT_DIR}/logs",
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
# Salva in una nuova cartella per non sovrascrivere i modelli precedenti
save_path = "/Users/dmarcoaldi/LLM_Models/Mistral-Small-Finetuned-GeminiCLI"
model.save_pretrained(save_path)
print(f"Fine-tuning completato! Modello salvato in: {save_path}")
