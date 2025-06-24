import torch
import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")

# --- 1. Caricamento Modello e Tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,  # bfloat16 sarebbe preferibile se supportato
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# Aggiunta del pad token (pratica comune per i modelli decoder-only)
tokenizer.pad_token = tokenizer.eos_token
# È necessario anche per la configurazione del modello per evitare warning
model.config.pad_token_id = tokenizer.eos_token_id

# --- 2. Configurazione LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # Aggiungi i moduli target per Mistral. I più comuni sono questi:
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Utile per verificare cosa stai addestrando

# --- 3. Caricamento e Preparazione Dataset ---
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Suddivisione in train e eval
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# --- 4. FUNZIONE DI TOKENIZZAZIONE CORRETTA ---
# Definiamo una lunghezza massima per coerenza
MAX_LENGTH = 256


def create_prompt(prompt):
    # Usiamo il formato di prompt ufficiale di Mistral Instruct
    return f"<s>[INST] {prompt} [/INST]"


def tokenize_function(examples):
    # Crea il prompt formattato
    prompts_formatted = [create_prompt(p) for p in examples["prompt"]]

    # Crea il testo completo (prompt + completion + eos)
    full_text = [p + " " + c + tokenizer.eos_token for p, c in zip(prompts_formatted, examples["completion"])]

    # Tokenizza il testo completo
    model_inputs = tokenizer(
        full_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    # Crea i labels, inizialmente una copia degli input_ids
    labels = model_inputs["input_ids"].copy()

    # Tokenizza i prompt formattati da soli per sapere la loro lunghezza
    prompts_tokenized = tokenizer(
        prompts_formatted,
        max_length=MAX_LENGTH,
        truncation=True
    )

    # Maschera i token del prompt nei labels impostandoli a -100
    for i in range(len(labels)):
        prompt_len = len(prompts_tokenized["input_ids"][i])
        labels[i][:prompt_len] = [-100] * prompt_len

    model_inputs["labels"] = labels
    return model_inputs


# Applica la funzione ai dataset
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

# --- 5. Training ---
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="tensorboard",
    bf16=True,  # Corretto per macOS con MPS
    learning_rate=2e-5,  # Aggiunto un learning rate più conservativo
    lr_scheduler_type="cosine",
    max_grad_norm=1.0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train()

# --- 6. Salvataggio ---
# Salva solo gli adapter LoRA, non il modello intero
output_dir = "/Users/dmarcoaldi/LLM_Models/Mistral-Small-Finetuned-Lora"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuning completato! Adapter salvati in {output_dir}")