import json
import os
import torch
from collections import defaultdict
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

# V2 batched version of the original script for evaluation.
# Distinct lang evaluation added (Bleu, ChrF, Ter)

load_dotenv()

# --- SETTINGS ---
EVAL_BASE = os.getenv("EVAL_BASE", "true").strip().lower() in ("1", "true", "yes", "y")

VAL_FILE = os.getenv("VAL_FILE", "resources/validation_set.jsonl")
PRED_BASE_FILE = os.getenv("PRED_BASE_FILE", "resources/predictions_base_final.jsonl")
PRED_LORA_FILE = os.getenv("PRED_LORA_FILE", "resources/predictions_lora_final.jsonl")

BASE_MODEL_PATH = os.getenv("EUROLLM_MODEL_PATH")
LORA_ADAPTER_PATH = os.getenv("EUROLLM_LORA_ADAPTER")

# Performance knobs
BATCH_EVAL = int(os.getenv("BATCH_EVAL", "32"))  # H200 friendly
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))

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
    tok = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # Required for correct batched generation

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this node/job. Did you request a GPU in Slurm?")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,  # H200 friendly
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)

    if peft_path:
        print(f"Applying LoRA adapter from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path).to(device)

    model.eval()

    print("CUDA device:", torch.cuda.get_device_name(0))
    print("BATCH_EVAL:", BATCH_EVAL, "MAX_NEW_TOKENS:", MAX_NEW_TOKENS)

    return tok, model


def build_prompt(src: str, lang: str) -> str:
    instruction = f"Translate the following English text to {lang}: '{src}'"
    return f"{instruction}\n{MARKER}\n"


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
    print("PRED_LORA_FILE:", PRED_LORA_FILE)
    print("PRED_BASE_FILE:", PRED_BASE_FILE)

    print("Loading Finetuned (LoRA) Model...")
    tok, model_lora = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    model_base = None
    if EVAL_BASE:
        print("Loading Base Model...")
        _, model_base = load_model_and_tokenizer(BASE_MODEL_PATH, peft_path=None)

    os.makedirs(os.path.dirname(PRED_LORA_FILE) or ".", exist_ok=True)
    if EVAL_BASE:
        os.makedirs(os.path.dirname(PRED_BASE_FILE) or ".", exist_ok=True)

    preds_lora, refs = [], []
    preds_base = []

    # Per-language tracking
    by_lang_lora = defaultdict(lambda: {"preds": [], "refs": []})
    by_lang_base = defaultdict(lambda: {"preds": [], "refs": []})

    with open(PRED_LORA_FILE, "w", encoding="utf-8") as f_lora, \
         (open(PRED_BASE_FILE, "w", encoding="utf-8") if EVAL_BASE else open(os.devnull, "w")) as f_base:

        # Batch loop
        for start in tqdm(range(0, len(val_data), BATCH_EVAL), desc="Generating (batched)"):
            batch = val_data[start:start + BATCH_EVAL]
            prompts = [build_prompt(ex["src"], ex["lang"]) for ex in batch]

            # LoRA preds
            lora_preds = generate_batch(model_lora, tok, prompts)
            for ex, pred in zip(batch, lora_preds):
                f_lora.write(json.dumps({**ex, "prediction": pred}, ensure_ascii=False) + "\n")
                preds_lora.append(pred)
                refs.append(ex["tgt"])
                # Per-language tracking
                by_lang_lora[ex["lang"]]["preds"].append(pred)
                by_lang_lora[ex["lang"]]["refs"].append(ex["tgt"])

            # Base preds
            if EVAL_BASE and model_base is not None:
                base_preds = generate_batch(model_base, tok, prompts)
                for ex, pred in zip(batch, base_preds):
                    f_base.write(json.dumps({**ex, "prediction": pred}, ensure_ascii=False) + "\n")
                    preds_base.append(pred)
                    # Per-language tracking
                    by_lang_base[ex["lang"]]["preds"].append(pred)
                    by_lang_base[ex["lang"]]["refs"].append(ex["tgt"])

    # === EVALUATION METRICS ===
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # --- LoRA Model ---
    print("\n--- LoRA Fine-tuned Model ---")
    print("\n[OVERALL]")
    bleu_lora = sacrebleu.corpus_bleu(preds_lora, [refs]).score
    chrf_lora = sacrebleu.corpus_chrf(preds_lora, [refs]).score
    ter_lora = sacrebleu.corpus_ter(preds_lora, [refs]).score
    print(f"  BLEU:  {bleu_lora:.2f}")
    print(f"  chrF:  {chrf_lora:.2f}")
    print(f"  TER:   {ter_lora:.2f}")

    print("\n[PER LANGUAGE]")
    print(f"  {'Lang':<6} {'BLEU':>8} {'chrF':>8} {'TER':>8} {'Count':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for lang in sorted(by_lang_lora.keys()):
        data = by_lang_lora[lang]
        bleu = sacrebleu.corpus_bleu(data["preds"], [data["refs"]]).score
        chrf = sacrebleu.corpus_chrf(data["preds"], [data["refs"]]).score
        ter = sacrebleu.corpus_ter(data["preds"], [data["refs"]]).score
        print(f"  {lang:<6} {bleu:>8.2f} {chrf:>8.2f} {ter:>8.2f} {len(data['refs']):>8}")

    # --- Base Model ---
    if EVAL_BASE and preds_base:
        print("\n--- Base Model ---")
        print("\n[OVERALL]")
        bleu_base = sacrebleu.corpus_bleu(preds_base, [refs]).score
        chrf_base = sacrebleu.corpus_chrf(preds_base, [refs]).score
        ter_base = sacrebleu.corpus_ter(preds_base, [refs]).score
        print(f"  BLEU:  {bleu_base:.2f}")
        print(f"  chrF:  {chrf_base:.2f}")
        print(f"  TER:   {ter_base:.2f}")

        print("\n[PER LANGUAGE]")
        print(f"  {'Lang':<6} {'BLEU':>8} {'chrF':>8} {'TER':>8} {'Count':>8}")
        print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for lang in sorted(by_lang_base.keys()):
            data = by_lang_base[lang]
            bleu = sacrebleu.corpus_bleu(data["preds"], [data["refs"]]).score
            chrf = sacrebleu.corpus_chrf(data["preds"], [data["refs"]]).score
            ter = sacrebleu.corpus_ter(data["preds"], [data["refs"]]).score
            print(f"  {lang:<6} {bleu:>8.2f} {chrf:>8.2f} {ter:>8.2f} {len(data['refs']):>8}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()