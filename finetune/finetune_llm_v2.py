
import os, math, sacrebleu, numpy as np

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, EarlyStoppingCallback)
from peft import LoraConfig
from trl  import (
    SFTTrainer, DataCollatorForCompletionOnlyLM
)

# --- 0. parametri base --------------------------------------------
load_dotenv()
MODEL_ID  = os.getenv("EUROLLM_MODEL_PATH")
DATA_PATH = os.getenv("DATASET_PATH")
OUT_DIR   = "./checkpoint"
ADAPTER_DIR = os.getenv("EUROLLM_LORA_ADAPTER")

MAX_LEN   = 256
BATCH     = 2
GRAD_ACC  = 4
EPOCHS    = 3
LR        = 2e-5
WARMUP    = 0.05
EVAL_EVERY_STEPS = 500
MARKER = "###"  # separatore tra prompt e completion


# --- 1. modello & tokenizer ---------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.pad_token = tok.eos_token

print(f"MARKER raw: {repr(MARKER)}")
print("MARKER decoded:", tok.decode(tok(MARKER)["input_ids"]))
print("MARKER tokens:", tok(MARKER)["input_ids"])

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": "mps"},
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# --- 2. LoRA -------------------------------------------------------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

# --- 3. dataset + split 90/10 -------------------------------------
ds = load_dataset("json", data_files=DATA_PATH, split="train")

def merge_cols(ex):
    prompt = ex["prompt"].strip()
    completion = ex["completion"].strip()
    ex["text"] = f"{prompt} {MARKER} {completion}"
    return ex

ds = ds.map(merge_cols, remove_columns=("prompt", "completion"))
split = ds.train_test_split(test_size=0.1, seed=42)
train_ds, eval_ds = split["train"], split["test"]

collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tok,
    response_template=MARKER,
    mlm=False)

# --- 4. TrainingArguments -----------------------------------------
args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP,
    save_strategy="epoch",
    eval_strategy="epoch",
    eval_steps=EVAL_EVERY_STEPS,
    logging_steps=EVAL_EVERY_STEPS//5,
    save_steps=EVAL_EVERY_STEPS,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="tensorboard",
    save_total_limit=1,
    eval_accumulation_steps=1,
    fp16=False,
    bf16=False
)

# --- 6. SFTTrainer -------------------------------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    peft_config=lora_cfg,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

trainer.model.save_pretrained(ADAPTER_DIR)
print(f"Fine-tuning completed — LoRa adapter stored in {ADAPTER_DIR}")

# --------- Total steps ----------------------------------
effective_batch = BATCH * GRAD_ACC
steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
total_steps     = steps_per_epoch * EPOCHS
print(f"ℹ️  ~{steps_per_epoch} step/epoch → {total_steps} step in totale")