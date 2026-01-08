import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

load_dotenv()

# --- SETTINGS ---
EVAL_BASE = os.getenv("EVAL_BASE", "true").strip().lower() in ("1", "true", "yes", "y")

VAL_FILE = os.getenv("VAL_FILE", "resources/validation_set.jsonl")
PRED_BASE_FILE = os.getenv("PRED_BASE_FILE", "resources/predictions_base_final.jsonl")
PRED_LORA_FILE = os.getenv("PRED_LORA_FILE", "resources/predictions_lora_final.jsonl")

BASE_MODEL_PATH = os.getenv("EUROLLM_MODEL_PATH")
LORA_ADAPTER_PATH = os.getenv("EUROLLM_LORA_ADAPTER")

MARKER = "### Answer:"


def _assert_paths():
    if not BASE_MODEL_PATH:
        raise ValueError("EUROLLM_MODEL_PATH is not set in .env")
    if not os.path.isdir(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model path not found: {BASE_MODEL_PATH}")
    if not os.path.isfile(VAL_FILE):
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")
    if LORA_ADAPTER_PATH and not os.path.isdir(LORA_ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapter path not found: {LORA_ADAPTER_PATH}")


def load_model_and_tokenizer(base_model_path: str, peft_path: str | None = None):
    """Load tokenizer and model (base or base+LoRA). HPC/CUDA friendly."""
    tok = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,     # good default for H200
        device_map="auto",              # IMPORTANT: string, not dict/set
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    if peft_path:
        print(f"Applying LoRA adapter from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)

    model.eval()

    # Minimal diagnostics (shows up in .out)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device 0:", torch.cuda.get_device_name(0))
        # model.device may be "meta" with device_map; this is still fine.
        # The real placement is handled by Accelerate/Transformers internally.

    return tok, model


def generate_translation(model, tokenizer, src: str, lang: str) -> str:
    instruction = f"Translate the following English text to {lang}: '{src}'"
    prompt = f"{instruction}\n{MARKER}\n"

    # Put tensors on the same device as the first model parameter (safe with device_map="auto")
    # If model is sharded, this still works because generate() handles dispatching.
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # greedy for evaluation
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_output = tokenizer.decode(ids[0], skip_special_tokens=True)
    prediction = full_output[len(prompt):].strip()
    return prediction


def main():
    _assert_paths()

    # Load evaluation data
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    print("VAL_FILE:", VAL_FILE)
    print("PRED_LORA_FILE:", PRED_LORA_FILE)
    print("PRED_BASE_FILE:", PRED_BASE_FILE)
    print("Loading Finetuned (LoRA) Model...")
    tokenizer_lora, model_lora = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    model_base = None
    if EVAL_BASE:
        print("Loading Base Model...")
        _, model_base = load_model_and_tokenizer(BASE_MODEL_PATH, peft_path=None)

    preds_lora, refs_lora = [], []
    preds_base, refs_base = [], []

    os.makedirs(os.path.dirname(PRED_LORA_FILE) or ".", exist_ok=True)
    if EVAL_BASE:
        os.makedirs(os.path.dirname(PRED_BASE_FILE) or ".", exist_ok=True)

    with open(PRED_LORA_FILE, "w", encoding="utf-8") as f_lora, \
         (open(PRED_BASE_FILE, "w", encoding="utf-8") if EVAL_BASE else open(os.devnull, "w")) as f_base:

        for example in tqdm(val_data, desc="Generating Translations"):
            src, tgt, lang = example["src"], example["tgt"], example["lang"]

            pred_lora = generate_translation(model_lora, tokenizer_lora, src, lang)
            f_lora.write(json.dumps({**example, "prediction": pred_lora}, ensure_ascii=False) + "\n")
            preds_lora.append(pred_lora)
            refs_lora.append([tgt])

            if EVAL_BASE and model_base is not None:
                pred_base = generate_translation(model_base, tokenizer_lora, src, lang)
                f_base.write(json.dumps({**example, "prediction": pred_base}, ensure_ascii=False) + "\n")
                preds_base.append(pred_base)
                refs_base.append([tgt])

    print("\nBLEU Score Evaluation:")
    bleu_lora = sacrebleu.corpus_bleu(preds_lora, refs_lora).score
    print(f"LoRA fine-tuned BLEU score: {bleu_lora:.2f}")

    if EVAL_BASE:
        bleu_base = sacrebleu.corpus_bleu(preds_base, refs_base).score
        print(f"Base model BLEU score: {bleu_base:.2f}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
