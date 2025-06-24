import json
import random
import os

input_file = 'resources/dataset_ollama.jsonl'  # Nome del file di input
output_dir = 'resources/'  # Percorso in cui salvare i file di output (cartella resources)

# Nome dei file di output
train_file = os.path.join(output_dir, 'train.jsonl')
valid_file = os.path.join(output_dir, 'valid.jsonl')
train_ratio = 0.8             # Proporzione di dati da utilizzare per il training (80%)

# Carica il dataset
with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]

# Mescola i dati
random.shuffle(data)

# Calcola il numero di esempi per il training set
train_size = int(len(data) * train_ratio)

# Suddividi i dati
train_data = data[:train_size]
valid_data = data[train_size:]

# Salva il training set
with open(train_file, 'w') as f:
    for entry in train_data:
        f.write(json.dumps(entry) + '\n')

# Salva il validation set
with open(valid_file, 'w') as f:
    for entry in valid_data:
        f.write(json.dumps(entry) + '\n')

print(f'Dataset suddiviso in {len(train_data)} esempi per il training e {len(valid_data)} esempi per la validazione.')