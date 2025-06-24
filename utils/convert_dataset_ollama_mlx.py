import json

# Percorsi dei file
input_train_file = "/Users/dmarcoaldi/Dataset/train.jsonl"
input_valid_file = "/Users/dmarcoaldi/Dataset/valid.jsonl"

output_train_file = "/Users/dmarcoaldi/Dataset/train_mlx.jsonl"
output_valid_file = "/Users/dmarcoaldi/Dataset/valid_mlx.jsonl"

def convert_dataset(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            new_text = f"{data['prompt']} Output: {data['completion']}"
            json.dump({"text": new_text}, f_out, ensure_ascii=False)
            f_out.write("\n")

    print(f"✅ File convertito e salvato in: {output_file}")

# Converti train e valid
convert_dataset(input_train_file, output_train_file)
convert_dataset(input_valid_file, output_valid_file)

print("🚀 Conversione completata con successo!")