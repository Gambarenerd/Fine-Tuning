import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from tqdm import tqdm

# Carica le variabili d'ambiente (percorsi dei modelli)
load_dotenv()

# --- IMPOSTAZIONI ---
# Imposta a True se vuoi valutare anche il modello base per confronto
EVAL_BASE = True
# File di validazione e output
VAL_FILE = "resources/validation_set.jsonl"
PRED_BASE_FILE = "resources/predictions_base_final.jsonl"
PRED_LORA_FILE = "resources/predictions_lora_final.jsonl"

# Percorsi dei modelli dal file .env
BASE_MODEL_PATH = os.getenv("EUROLLM_MODEL_PATH")
FINETUNED_MODEL_PATH = os.getenv("EUROLLM_LORA_ADAPTER")

# Marcatori usati durante il training
MARKER = "### Answer:"


def load_model_and_tokenizer(base_model_path, peft_path=None):
    """Carica il tokenizer e il modello (base o con adapter LoRA)."""
    tok = AutoTokenizer.from_pretrained(base_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        # ✅ MODIFICA 1: Coerenza con il training
        torch_dtype=torch.bfloat16,
        device_map={"": "mps"},
    )

    if peft_path:
        print(f"Applying LoRA adapter from: {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)

    model.eval()
    return tok, model


def generate_translation(model, tokenizer, src, lang):
    """Genera la traduzione usando il formato del prompt corretto."""

    # Costruisci l'istruzione come nel dataset di training
    instruction = f"Translate the following English text to {lang}: '{src}'"

    # ✅ MODIFICA 2: Formato del Prompt
    # Questo deve corrispondere ESATTAMENTE al formato usato in training
    prompt = f"{instruction}\n{MARKER}\n"

    # Tokenizza l'input e sposta su device
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Genera l'output
    ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # Usiamo greedy decoding per la valutazione
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decodifica l'output completo
    full_output = tokenizer.decode(ids[0], skip_special_tokens=True)

    # ✅ MODIFICA 3: Logica di Estrazione della Risposta
    # Rimuoviamo il prompt iniziale per isolare solo il testo generato
    # Questo metodo è più robusto di uno split
    prediction = full_output[len(prompt):].strip()

    return prediction


def main():
    # Carica il dataset di valutazione
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    print("Loading Finetuned Model...")
    tokenizer_lora, model_lora = load_model_and_tokenizer(BASE_MODEL_PATH, FINETUNED_MODEL_PATH)

    model_base = None
    if EVAL_BASE:
        print("Loading Base Model...")
        # Il tokenizer è lo stesso, non serve ricaricarlo
        _, model_base = load_model_and_tokenizer(BASE_MODEL_PATH)

    preds_lora, refs_lora = [], []
    preds_base, refs_base = [], []

    # Apre i file di output
    with open(PRED_LORA_FILE, "w", encoding="utf-8") as f_lora, \
            (open(PRED_BASE_FILE, "w", encoding="utf-8") if EVAL_BASE else open(os.devnull, "w")) as f_base:

        for example in tqdm(val_data, desc="🔍 Generating Translations"):
            src, tgt, lang = example["src"], example["tgt"], example["lang"]

            # Genera e salva la predizione del modello fine-tuned
            pred_lora = generate_translation(model_lora, tokenizer_lora, src, lang)
            f_lora.write(json.dumps({**example, "prediction": pred_lora}, ensure_ascii=False) + "\n")
            preds_lora.append(pred_lora)
            refs_lora.append([tgt])  # sacrebleu richiede una lista di reference

            # Se richiesto, fa lo stesso per il modello base
            if EVAL_BASE:
                pred_base = generate_translation(model_base, tokenizer_lora, src, lang)
                f_base.write(json.dumps({**example, "prediction": pred_base}, ensure_ascii=False) + "\n")
                preds_base.append(pred_base)
                refs_base.append([tgt])

    # Calcola e stampa i punteggi BLEU
    print("\n📊 BLEU Score Evaluation:")
    bleu_lora = sacrebleu.corpus_bleu(preds_lora, refs_lora).score
    print(f"🧠 LoRA fine-tuned BLEU score: {bleu_lora:.2f}")

    if EVAL_BASE:
        bleu_base = sacrebleu.corpus_bleu(preds_base, refs_base).score
        print(f"🌐 Base model BLEU score: {bleu_base:.2f}")


if __name__ == "__main__":
    # Disabilita il calcolo dei gradienti per l'inferenza per risparmiare memoria
    with torch.no_grad():
        main()