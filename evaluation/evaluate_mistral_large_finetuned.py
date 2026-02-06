import json
import os
import torch
from collections import defaultdict
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

# Evaluate Mistral Large 2 Instruct (123B) FINE-TUNED model on translation task.
# Loads base model in bf16, applies LoRA adapter, merges weights, then evaluates.
# Uses device_map="auto" to shard across all available GPUs.

load_dotenv()

# --- SETTINGS ---
VAL_FILE = os.getenv("VAL_FILE", "resources/validation_set.jsonl")
PRED_FT_FILE = os.getenv("PRED_MISTRAL_FT_FILE", "resources/predictions_mistral_finetuned.jsonl")

BASE_MODEL_PATH = os.getenv("MISTRAL_MODEL_PATH")
ADAPTER_PATH = os.getenv("MISTRAL_LORA_ADAPTER")

# Performance knobs
BATCH_EVAL = int(os.getenv("BATCH_EVAL", "8"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Language code to full name mapping
LANG_NAMES = {
    "BG": "Bulgarian",
    "CS": "Czech",
    "DA": "Danish",
    "DE": "German",
    "EL": "Greek",
    "EN": "English",
    "ES": "Spanish",
    "ET": "Estonian",
    "FI": "Finnish",
    "FR": "French",
    "GA": "Irish",
    "HR": "Croatian",
    "HU": "Hungarian",
    "IT": "Italian",
    "LT": "Lithuanian",
    "LV": "Latvian",
    "MT": "Maltese",
    "NL": "Dutch",
    "PL": "Polish",
    "PT": "Portuguese",
    "RO": "Romanian",
    "SK": "Slovak",
    "SL": "Slovenian",
    "SV": "Swedish",
}


def _assert_paths():
    if not BASE_MODEL_PATH:
        raise ValueError("MISTRAL_MODEL_PATH is not set in .env")
    if not os.path.isdir(BASE_MODEL_PATH):
        raise FileNotFoundError(f"Base model path not found: {BASE_MODEL_PATH}")
    if not ADAPTER_PATH:
        raise ValueError("MISTRAL_LORA_ADAPTER is not set in .env")
    if not os.path.isdir(ADAPTER_PATH):
        raise FileNotFoundError(f"LoRA adapter path not found: {ADAPTER_PATH}")
    if not os.path.isfile(VAL_FILE):
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")


def load_model_and_tokenizer(base_model_path: str, adapter_path: str):
    tok = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this node/job. Did you request a GPU in Slurm?")

    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Attention implementation: {attn_impl}")

    print("Loading base model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    model.eval()

    print("BATCH_EVAL:", BATCH_EVAL, "MAX_NEW_TOKENS:", MAX_NEW_TOKENS)

    return tok, model


def build_prompt(tok, src: str, lang: str) -> str:
    """Build prompt using Mistral chat template with generation prompt."""
    lang_name = LANG_NAMES.get(lang, lang)
    instruction = (
        f"Translate the following English text to {lang_name}. "
        f"Output ONLY the translation, nothing else. No explanations, no breakdowns, no preamble.\n\n"
        f"{src}"
    )
    messages = [{"role": "user", "content": instruction}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_batch(model, tok, prompts):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    input_len = enc["input_ids"].shape[1]

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    preds = []
    for i in range(out.shape[0]):
        gen_ids = out[i, input_len:]
        preds.append(tok.decode(gen_ids, skip_special_tokens=True).strip())
    return preds


def main():
    _assert_paths()

    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    print("VAL_FILE:", VAL_FILE)
    print("Items:", len(val_data))
    print("PRED_FT_FILE:", PRED_FT_FILE)

    print("\nLoading Mistral Large 2 Instruct (123B) + LoRA adapter...")
    tok, model = load_model_and_tokenizer(BASE_MODEL_PATH, ADAPTER_PATH)

    os.makedirs(os.path.dirname(PRED_FT_FILE) or ".", exist_ok=True)

    preds_all, refs = [], []
    by_lang = defaultdict(lambda: {"preds": [], "refs": []})

    with open(PRED_FT_FILE, "w", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(val_data), BATCH_EVAL), desc="Generating (batched)"):
            batch = val_data[start:start + BATCH_EVAL]
            prompts = [build_prompt(tok, ex["src"], ex["lang"]) for ex in batch]

            batch_preds = generate_batch(model, tok, prompts)
            for ex, pred in zip(batch, batch_preds):
                f_out.write(json.dumps({**ex, "prediction": pred}, ensure_ascii=False) + "\n")
                preds_all.append(pred)
                refs.append(ex["tgt"])
                by_lang[ex["lang"]]["preds"].append(pred)
                by_lang[ex["lang"]]["refs"].append(ex["tgt"])

    # === EVALUATION METRICS ===
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS — Mistral Large 2 Instruct (fine-tuned LoRA)")
    print("=" * 60)

    print("\n[OVERALL]")
    bleu = sacrebleu.corpus_bleu(preds_all, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds_all, [refs]).score
    ter = sacrebleu.corpus_ter(preds_all, [refs]).score
    print(f"  BLEU:  {bleu:.2f}")
    print(f"  chrF:  {chrf:.2f}")
    print(f"  TER:   {ter:.2f}")

    print("\n[PER LANGUAGE]")
    print(f"  {'Lang':<6} {'BLEU':>8} {'chrF':>8} {'TER':>8} {'Count':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for lang in sorted(by_lang.keys()):
        data = by_lang[lang]
        b = sacrebleu.corpus_bleu(data["preds"], [data["refs"]]).score
        c = sacrebleu.corpus_chrf(data["preds"], [data["refs"]]).score
        t = sacrebleu.corpus_ter(data["preds"], [data["refs"]]).score
        print(f"  {lang:<6} {b:>8.2f} {c:>8.2f} {t:>8.2f} {len(data['refs']):>8}")

    print("\n" + "=" * 60)
    print(f"Predictions saved to: {PRED_FT_FILE}")


if __name__ == "__main__":
    main()
