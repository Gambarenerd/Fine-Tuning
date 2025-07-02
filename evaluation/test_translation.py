# test_translation_chat.py
import os, torch, json
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

load_dotenv()
BASE_MODEL   = os.getenv("MODEL_PATH")
LORA_WEIGHTS = os.getenv("FINETUNED_MODEL_GPT")

# 1️⃣  tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 2️⃣  backbone (float16 su MPS)
base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map={"": "mps"},
)

print("🔎  Path adapter:", LORA_WEIGHTS)
print("🔎  File presenti:", os.listdir(LORA_WEIGHTS))

cfg = os.path.join(LORA_WEIGHTS, "adapter_config.json")
print("🔎  adapter_config exists?", os.path.isfile(cfg))
if os.path.isfile(cfg):
    print(json.loads(open(cfg).read())["target_modules"][:8])

# 3️⃣  carico l'adapter LoRA **senza fondere** (mi serve per i test)
lora = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map={"": "mps"},
    ),
    LORA_WEIGHTS,
)
# ---------- VERIFICA 1: parametri LoRA presenti -------------------
print("\n🔍 Trainable parameters in LoRA:")
lora.print_trainable_parameters()         # deve mostrare ~0.3-0.5 % dei parametri

# ---------- VERIFICA 2: differenza logits -------------------------
probe_prompt = "<s>[INST] Translate to Irish Gaelic:\nHello [/INST]"
enc = tokenizer(probe_prompt, return_tensors="pt").to("mps")

with torch.no_grad():
    logits_base = base(**enc).logits
    logits_lora = lora(**enc).logits

max_delta = (logits_base - logits_lora).abs().max().item()
print(f"🔍 max |Δ logit| base vs LoRA : {max_delta:.6f}")   # deve essere > 0.000001
# ------------------------------------------------------------------

# 4️⃣  ora fuse + unload (opzionale, ma serve per l’inference “pulita”)
lora_model = lora.merge_and_unload()   # modello unico, senza layer LoRA
lora_model.eval()

# ------------------------------------------------
def translate(model, text, tgt_lang="IT"):
    user_prompt = f"Translate the following English text to {tgt_lang}: '{text}'"
    chat_prompt = f"<s>[INST] {user_prompt} [/INST]"

    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(ids[0], skip_special_tokens=True)
    return full_text.split("[/INST]")[-1].strip()

TEXT = ("Decision No 1/2015 of the Joint Veterinary Committee created by the Agreement between the European Community and the Swiss Confederation on trade in agricultural products")

print("\n📌 BASE:")
print(translate(base, TEXT))

print("\n📌 LoRA fine-tuned:")
print(translate(lora_model, TEXT))