import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
VAL_FILE = "resources/validation_set.jsonl"
PRED_FILE = "resources/predictions_base_model_improved.jsonl"
BASE_MODEL_PATH = os.getenv("MODEL_PATH_EUROLLM")

def load_base_model_and_tokenizer(base_model_path):
    """
    Loads the base model and tokenizer from the given path.
    """
    print(f"Loading base model from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Use 'auto' for device mapping
    )
    model.eval()
    return tokenizer, model

def generate_translation(model, tok, src, lang):
   # prompt = f"Translate the following English text to {lang}:\n{src}"

    prompt = ("You are a translation engine. Output ONLY the translation.\n\n"
              f"Translate the following English text to {lang}: {src}"
              )

    inputs = tok(prompt, return_tensors="pt",add_special_tokens=False).to(model.device)

    gen = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )

    n_input = inputs["input_ids"].shape[-1]
    gen_tokens = gen[0, n_input:]
    return tok.decode(gen_tokens, skip_special_tokens=True).strip()

def main():
    """
    Main function to run the evaluation on the base model.
    """
    # Load the validation dataset
    with open(VAL_FILE, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    # Load the model and tokenizer
    tokenizer, model = load_base_model_and_tokenizer(BASE_MODEL_PATH)

    predictions = []
    references = []

    # Open the output file for writing predictions
    with open(PRED_FILE, "w", encoding="utf-8") as f_out:
        # Iterate over the validation data with a progress bar
        for example in tqdm(val_data, desc="🔍 Evaluating Base Model"):
            src, tgt, lang = example["src"], example["tgt"], example["lang"]

            # Generate the translation
            prediction = generate_translation(model, tokenizer, src, lang)

            # Save the prediction to the output file
            f_out.write(json.dumps({**example, "prediction": prediction}, ensure_ascii=False) + "\n")

            # Append results for BLEU score calculation
            predictions.append(prediction)
            references.append([tgt])

    # Calculate and print the BLEU score
    print("\n📊 BLEU Score Evaluation:")
    bleu_score = sacrebleu.corpus_bleu(predictions, references).score
    print(f"🌐 Base model BLEU score: {bleu_score:.2f}")
    print(f"✅ Evaluation complete! Predictions saved to {PRED_FILE}")

if __name__ == "__main__":
    # Disable gradient calculations for inference
    torch.set_grad_enabled(False)
    main()
