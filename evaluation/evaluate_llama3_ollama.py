# eval_ollama_llama3.py
import json, os, sacrebleu
from tqdm import tqdm
from dotenv import load_dotenv
import ollama

load_dotenv()

VAL_FILE   = "resources/validation_set.jsonl"
PRED_FILE  = "resources/predictions_llama3_ollama.jsonl"
OLLAMA_URL = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
MODEL_ID   = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")

client = ollama.Client(host=OLLAMA_URL)

def translate(src: str, lang: str) -> str:
    prompt = (
        "You are a translation engine. Output ONLY the translation.\n\n"
        f"Translate the following English text to {lang}:\n{src}"
    )
    rsp = client.generate(model=MODEL_ID,
                          prompt=prompt,
                          options={"temperature": 0, "stop": ["\n\n"]})
    # `stop` tronca non appena il modello va a capo due volte
    return rsp["response"].splitlines()[0].strip()

def main():
    preds, refs = [], []

    with open(PRED_FILE, "w", encoding="utf-8") as fout, \
         open(VAL_FILE,  encoding="utf-8") as fin:

        for line in tqdm(fin, desc="🔍 Evaluating Ollama Llama3"):
            ex  = json.loads(line)
            pred = translate(ex["src"], ex["lang"])
            fout.write(json.dumps({**ex, "prediction": pred},
                                  ensure_ascii=False) + "\n")
            preds.append(pred); refs.append([ex["tgt"]])

    bleu = sacrebleu.corpus_bleu(preds, refs).score
    print(f"\n📊 BLEU (gpt-oss:20b via Ollama): {bleu:.2f}   – results in {PRED_FILE}")

if __name__ == "__main__":
    main()