import os, math
import torch
import torch.distributed as dist
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, EarlyStoppingCallback,
                          BitsAndBytesConfig)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# Mistral Large 2 Instruct (123B) — QLoRA + DDP (true data parallelism)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# DDP setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
is_main = local_rank == 0

if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if is_main:
        print("CUDA available")
        print(f"World size: {world_size}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    device = torch.device("cpu")
    print("CUDA NOT available")

# Basic Parameters
load_dotenv()

MODEL_ID = os.getenv("MISTRAL_MODEL_PATH")
TRAIN_PATH = os.getenv("TRAIN_DATASET_PATH")
EVAL_PATH = os.getenv("EVAL_DATASET_PATH")
OUT_DIR = "./checkpoint-mistral"
ADAPTER_DIR = os.getenv("MISTRAL_LORA_ADAPTER")

MAX_LEN = 512
BATCH = 2  # per GPU, with 4 GPUs = 8 total per step
GRAD_ACC = 4  # effective batch = 8 * 4 = 32
EPOCHS = 3
LR = 1e-5
WARMUP = 0.05
EVAL_EVERY_STEPS = 500

# Tokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# 4-bit quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Attention implementation
try:
    import flash_attn  # noqa: F401
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"

if is_main:
    print(f"Attention implementation: {attn_impl}")
    print(f"Loading model on device: {device}")

# Model — load on specific GPU (not device_map="auto")
# Each DDP process loads the full 4-bit model on its own GPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": local_rank},  # load entire model on this GPU
    low_cpu_mem_usage=True,
    local_files_only=True,
    attn_implementation=attn_impl,
)

# Prepare model for QLoRA training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

if is_main:
    print(f"Model loaded. Memory used: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

# LoRA config — r=32, alpha=64 for larger model (more capacity)
lora_cfg = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

# Load datasets
train_ds = load_dataset("json", data_files=TRAIN_PATH, split="train")
eval_ds = load_dataset("json", data_files=EVAL_PATH, split="train")


# Format using Mistral chat template: [INST] prompt [/INST] completion
def merge_cols(ex):
    prompt = ex["prompt"].strip()
    completion = ex["completion"].strip()
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    ex["text"] = tok.apply_chat_template(messages, tokenize=False)
    return ex


train_ds = train_ds.map(merge_cols, remove_columns=("prompt", "completion"))
eval_ds = eval_ds.map(merge_cols, remove_columns=("prompt", "completion"))


# Filter too long samples
def filter_length(example):
    tokens = tok.encode(example["text"], add_special_tokens=True)
    return len(tokens) <= MAX_LEN


if is_main:
    print(f"Train dataset size before filtering: {len(train_ds)}")
train_ds = train_ds.filter(filter_length)
if is_main:
    print(f"Train dataset size after filtering: {len(train_ds)}")
    print(f"Eval dataset size before filtering: {len(eval_ds)}")
eval_ds = eval_ds.filter(filter_length)
if is_main:
    print(f"Eval dataset size after filtering: {len(eval_ds)}")

    print("\n--- Testing data format ---")
    sample_text = train_ds[0]["text"]
    print("Sample text:", sample_text[:300] + "...")
    tokens = tok.encode(sample_text, add_special_tokens=True)
    print(f"Token count: {len(tokens)}")

# TrainingArguments with DDP
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
    logging_steps=EVAL_EVERY_STEPS // 5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    save_total_limit=1,
    fp16=False,
    bf16=True,
    dataloader_drop_last=False,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=4,
)

# SFTTrainer
if is_main:
    print("\n--- Initializing SFTTrainer ---")

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=lora_cfg,
    args=args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

if is_main:
    print("SFTTrainer initialized successfully")

# Training (resume from checkpoint if available)
if is_main:
    print("\n--- Starting training ---")
trainer.train()

# Save LoRA Adapter (only on main process)
if is_main:
    trainer.model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"Fine-tuning completed — LoRA adapter stored in {ADAPTER_DIR}")

    # Statistics
    effective_batch = BATCH * GRAD_ACC * world_size
    steps_per_epoch = math.ceil(len(train_ds) / effective_batch)
    total_steps = steps_per_epoch * EPOCHS
    print(f"  ~{steps_per_epoch} step/epoch -> {total_steps} step totali")
    print(f"  Effective batch size: {effective_batch}")

# Clean DDP shutdown
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()

if is_main:
    print("\n Training completed successfully!")
