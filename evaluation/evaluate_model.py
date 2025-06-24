import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

load_dotenv()

VAL_FILE = "resources/validation_set.jsonl"
PRED_BASE_FILE = "resources/predictions_base.jsonl"
PRED_LORA_FILE = "resources/predictions_lora.jsonl"

# Percorsi dei modelli
BASE_MODEL_PATH = os.getenv("MODEL_PATH")
FINETUNED_MODEL_PATH = os.getenv("FINETUNED_GPT_MODEL_PATH")

def load_model_and_tokenizer(base_model_path, peft_path=None):
    tok   = AutoTokenizer.from_pretrained(base_model_path)
    base  = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map={"": "mps"},
            )
    if peft_path:
        base = PeftModel.from_pretrained(base, peft_path)
    base.eval()                       # 🤫 niente grad
    return tok, base

def generate_translation(model, tokenizer, src, lang):
    prompt = f"<s>[INST] Translate to {lang}:\n{src} [/INST]"
    ids = model.generate(
        **tokenizer(prompt, return_tensors="pt").to(model.device),
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(ids[0], skip_special_tokens=True)
    return full.split("[/INST]")[-1].strip()

def main():
    # Carica dataset di valutazione
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    print("🔄 Carico modelli...")
    tokenizer_base, model_base = load_model_and_tokenizer(BASE_MODEL_PATH)
    tokenizer_lora, model_lora = load_model_and_tokenizer(BASE_MODEL_PATH, FINETUNED_MODEL_PATH)

    preds_base, refs_base = [], []
    preds_lora, refs_lora = [], []

    with open(PRED_BASE_FILE, "w", encoding="utf-8") as fw_base, \
         open(PRED_LORA_FILE, "w", encoding="utf-8") as fw_lora:

        for example in tqdm(val_data, desc="🔍 Traduco e confronto"):
            src, tgt, lang = example["src"], example["tgt"], example["lang"]

            pred_base = generate_translation(model_base, tokenizer_base, src, lang)
            pred_lora = generate_translation(model_lora, tokenizer_lora, src, lang)

            # Salva le predizioni
            fw_base.write(json.dumps({"src": src, "tgt": tgt, "lang": lang, "prediction": pred_base}, ensure_ascii=False) + "\n")
            fw_lora.write(json.dumps({"src": src, "tgt": tgt, "lang": lang, "prediction": pred_lora}, ensure_ascii=False) + "\n")

            # Per sacreBLEU
            refs_base.append([tgt])
            preds_base.append(pred_base)

            refs_lora.append([tgt])
            preds_lora.append(pred_lora)

    print("\n📊 Valutazione BLEU:")
    bleu_base = sacrebleu.corpus_bleu(preds_base, refs_base).score
    bleu_lora = sacrebleu.corpus_bleu(preds_lora, refs_lora).score

    print(f"🌐 Base model BLEU score:  {bleu_base:.2f}")
    print(f"🧠 LoRA fine-tuned BLEU score: {bleu_lora:.2f}")
    print("✅ Valutazione completata!")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()