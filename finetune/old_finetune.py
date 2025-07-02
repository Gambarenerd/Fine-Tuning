#!/usr/bin/env python
# eval_translation_bleu.py
import json, os, torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

load_dotenv()

VAL_FILE       = "resources/validation_set.jsonl"
PRED_BASE_FILE = "resources/predictions_base.jsonl"
PRED_LORA_FILE = "resources/predictions_lora.jsonl"

BASE_MODEL_PATH      = os.getenv("MODEL_PATH")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_GPT_MODEL_PATH")

# ──────────────────────────────────────────────────────────────
# 1. normalizza i codici lingua a 2 lettere (GA-IE → GA, ES-ES → ES …)
ISO2 = lambda c: c[:2].upper()

# ──────────────────────────────────────────────────────────────
def load_model_and_tokenizer(base_path, peft_path=None):
    tok  = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map={"": "mps"},
    )
    if peft_path:
        base = PeftModel.from_pretrained(base, peft_path)
    base.eval()
    return tok, base

def generate(model, tok, src, lang):
    prompt = f"<s>[INST] Translate to {ISO2(lang)}:\n{src} [/INST]"
    ids = model.generate(
        **tok(prompt, return_tensors="pt").to(model.device),
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    full = tok.decode(ids[0], skip_special_tokens=True)
    return full.split("[/INST]")[-1].strip()

# ──────────────────────────────────────────────────────────────
def main():
    with open(VAL_FILE, encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    print("🔄 Carico modelli …")
    tok_base,  model_base  = load_model_and_tokenizer(BASE_MODEL_PATH)
    tok_lora,  model_lora  = load_model_and_tokenizer(BASE_MODEL_PATH, FINETUNED_MODEL_PATH)

    preds_b, refs_b, preds_l, refs_l = [], [], [], []

    with open(PRED_BASE_FILE, "w", encoding="utf-8") as fwb, \
         open(PRED_LORA_FILE, "w", encoding="utf-8") as fwl:

        for ex in tqdm(val_data, desc="🔍 Traduco"):
            src, tgt, lang = ex["src"], ex["tgt"], ex["lang"]

            pb = generate(model_base, tok_base, src, lang)
            pl = generate(model_lora, tok_lora, src, lang)

            fwb.write(json.dumps({**ex, "prediction": pb}, ensure_ascii=False) + "\n")
            fwl.write(json.dumps({**ex, "prediction": pl}, ensure_ascii=False) + "\n")

            refs_b.append([tgt]); preds_b.append(pb)
            refs_l.append([tgt]); preds_l.append(pl)

    print("\n📊 BLEU:")
    print(f"🌐 Base      : {sacrebleu.corpus_bleu(preds_b, refs_b).score:.2f}")
    print(f"🧠 LoRA fine : {sacrebleu.corpus_bleu(preds_l, refs_l).score:.2f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)   # disabilita gradienti globalmente
    main()