# FineTunig LLM
The purpose of the project is to use **tmx files** to fine-tune a small model like **mistrall-7b-instruct**
to improve its performance for multi-lingual translation.

For evaluation we compare the **sacreBLEU** score of the fine-tuned and base model computed
against the gold-standard translations extracted from TMXs files.

## Performance
The script mistral_finetune_chatGPT.py with evaluate_model.py 
shows a (sacre)BLEU score of **52.94 for the finetuned model** against **20.09 for the base one**.

Using the following hyperperameters decreased the BLEU Score for fine-tune model to 12.12 (weird)\
MAX_LENGTH   = 256\
BATCH_SIZE   = 16\
GRAD_ACCUM   = 2\
NUM_EPOCHS   = 3\
LR           = 2e-5\
WARMUP_RATIO = 0.05\
LoRa Rank = 32\
LoRa Alpha = 32\
LoRa Dropout = 0.05