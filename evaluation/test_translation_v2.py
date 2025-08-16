# evaluation/test_base_readme_style.py

import os, re, json, torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

load_dotenv()
MODEL_PATH = os.getenv("EUROLLM_MODEL_PATH")      # es. /Users/…/EuroLLM
DATA_PATH  = os.getenv("DATASET_PATH")            # jsonl con {"prompt","completion"}

# ——— device/dtype: MPS su Mac, altrimenti CPU ———
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "mps" else torch.float32

# ——— mappa ISO2 → nome lingua (minima ma estendibile) ———
LANG_MAP = {
    "IT":"Italian", "DE":"German", "ES":"Spanish", "FR":"French", "PT":"Portuguese",
    "PL":"Polish", "NL":"Dutch", "TR":"Turkish", "SV":"Swedish", "CS":"Czech",
    "EL":"Greek", "HU":"Hungarian", "RO":"Romanian", "FI":"Finnish", "UK":"Ukrainian",
    "SL":"Slovenian", "SK":"Slovak", "DA":"Danish", "LT":"Lithuanian", "LV":"Latvian",
    "ET":"Estonian", "BG":"Bulgarian", "NO":"Norwegian", "CA":"Catalan", "HR":"Croatian",
    "GA":"Irish", "MT":"Maltese", "GL":"Galician", "ZH":"Chinese", "RU":"Russian",
    "KO":"Korean", "JA":"Japanese", "AR":"Arabic", "HI":"Hindi",
}

def parse_dataset_prompt(p: str):
    """
    Atteso formato: "Translate the following English text to XX: '...'"
    Ritorna (src_text, target_lang_code, target_lang_name)
    """
    m = re.search(r"to\s+([A-Z]{2}):\s*'(.*)'\s*$", p)
    if not m:
        raise ValueError(f"Prompt non riconosciuto: {p}")
    code = m.group(1).upper()
    src  = m.group(2)
    name = LANG_MAP.get(code, code)   # fallback: usa il codice se non mappato
    return src, code, name

def main():
    print(f"🔧 MODEL_PATH = {MODEL_PATH}")
    print(f"🔧 DATA_PATH  = {DATA_PATH}")
    print(f"🔧 DEVICE     = {DEVICE}  (dtype={DTYPE})")

    # 1) Carica un esempio dal dataset
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ex = ds[0]  # prendi il primo per semplicità
    src_text, tgt_code, tgt_name = parse_dataset_prompt(ex["prompt"])
    gold = ex["completion"]

    # 2) Carica tokenizer & modello base
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map={"": DEVICE} if DEVICE == "mps" else None,
    ).eval()

    # 3) Prompt stile README (completion-style, NON istruzioni)
    #    Esempio dalla card: "English: ... Portuguese:"
    prompt = f"English: {src_text} {tgt_name}:"

    # 4) Generazione (sampling leggero come da stile README)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    print("\n================= RISULTATI =================")
    print("📎 Prompt usato (stile README):")
    print(repr(prompt))
    print("\n🟢 Output modello base:")
    print(decoded)
    print("\n🎯 Gold dal dataset:")
    print(gold)
    print("=============================================\n")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()