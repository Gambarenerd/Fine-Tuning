import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from dotenv import load_dotenv
from datasets import load_dataset

#Carichiamo le variabili d'ambiente
load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH")
DATASET_PATH = os.getenv("DATASET_PATH")

#Carichiamo il modello MLX
print("Caricamento modello in MLX...")
model, tokenizer = load(MODEL_PATH)

#Assegniamo un token di padding se non esiste
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Usiamo il token di fine frase
    tokenizer.pad_token_id = tokenizer.eos_token_id  # Assegniamo anche l'ID corretto

#Verifica del tokenizer MLX
if tokenizer.chat_template is not None:
    print("Tokenizer utilizza un chat template compatibile.")

#Carichiamo il dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

#Dividiamo il dataset in training (80%) e validation (20%)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

dataset = dataset.filter(lambda x: x["prompt"] is not None and x["completion"] is not None and x["prompt"].strip() != "" and x["completion"].strip() != "")

#Tokenizziamo il dataset
def tokenize_function(examples):
    max_length = 128  # Lunghezza massima della sequenza

    # Tokenizziamo i testi e forziamo che restituiscano sempre una lista valida
    inputs = [tokenizer.encode(text) if text else [tokenizer.pad_token_id] for text in examples["prompt"]]
    targets = [tokenizer.encode(text) if text else [tokenizer.pad_token_id] for text in examples["completion"]]

    # Assicuriamoci che il padding sia applicato correttamente
    inputs = [seq[:max_length] + [tokenizer.pad_token_id] * (max_length - len(seq)) for seq in inputs]
    targets = [seq[:max_length] + [-100] * (max_length - len(seq)) for seq in targets]  # -100 evita il calcolo della loss sul padding

    # Debug: Stampiamo alcune righe per verificare che i None siano stati eliminati
    for i, seq in enumerate(inputs):
        if None in seq:
            print(f"ERRORE: input_ids alla riga {i} contiene None: {seq}")# -100 evita che il loss venga calcolato sul padding

    return {"input_ids": inputs, "labels": targets}

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

print(train_dataset[0])  # Controlla se input_ids e labels sono stati aggiunti
print(eval_dataset[0])

print("Adapting dataset in mlx tensor...")

#Convertiamo il dataset in tensori MLX
def convert_to_mlx(dataset):
    for i, ex in enumerate(dataset):
        if ex["input_ids"] is None or ex["labels"] is None:
            print(f"Riga {i} ha ancora None → input_ids={ex['input_ids']}, labels={ex['labels']}")

    inputs = mx.array([ex["input_ids"] for ex in dataset], dtype=mx.int32)
    labels = mx.array([ex["labels"] for ex in dataset], dtype=mx.int32)
    return inputs, labels

train_inputs, train_labels = convert_to_mlx(train_dataset)
eval_inputs, eval_labels = convert_to_mlx(eval_dataset)

#Configuriamo l’ottimizzatore MLX
optimizer = optim.AdamW(5e-5, model.parameters())

# Definiamo la funzione di perdita per MLX
def loss_function(model, batch_inputs, batch_labels):
    logits = model(batch_inputs)  # Forward pass
    loss = nn.losses.cross_entropy(logits, batch_labels)  # Calcoliamo la perdita
    return loss.mean()

#Parametri di training
num_epochs = 3
batch_size = 8  # Aumentato per sfruttare i 128GB di RAM
num_batches = len(train_inputs) // batch_size


def compute_loss(inputs, labels):
    logits = model(inputs)  # Forward pass senza params
    loss = nn.losses.cross_entropy(logits, labels)
    return loss.mean()

#Ciclo di training manuale (MLX non supporta Hugging Face Trainer)
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    for batch_idx in range(num_batches):
        batch_inputs = train_inputs[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch_labels = train_labels[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        # Ora calcoliamo i gradienti rispetto ai parametri del modello
        grads = mx.grad(compute_loss)(batch_inputs, batch_labels)

        # Aggiorniamo i parametri del modello
        updates, optimizer.state = optimizer.update(grads, model.trainable_parameters())

        # Applica gli aggiornamenti al modello
        model = model.apply_updates(updates)

        if batch_idx % 10 == 0:
            loss_value = loss_function(batch_inputs, batch_labels)
            print(f"Batch {batch_idx}/{num_batches} - Loss: {loss_value.item()}")

        # Valutazione su dataset di validazione
    eval_logits = model(eval_inputs)
    eval_loss = nn.losses.cross_entropy(eval_logits, eval_labels)
    print(f"Fine Epoch {epoch + 1}, Loss: {loss_value.item()}, Eval Loss: {eval_loss.item()}")

# Salviamo il modello fine-tunato
model.save_pretrained("/Users/dmarcoaldi/LLM_Models/Mistral-Small-Finetuned-MLX")
print("Fine-tuning completato con MLX!")