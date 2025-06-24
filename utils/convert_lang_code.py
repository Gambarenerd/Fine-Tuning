#!/usr/bin/env python
# fix_lang_codes.py
# -----------------------------------------------------------
# Converte i codici lingua da "XX-YY" a "XX" (ISO-639-1)
# nel file JSON-Lines di validazione.
# -----------------------------------------------------------

import json
import pathlib
import sys

# ------------------------------------------------------------------
# 1. Percorsi (puoi passarli come argomenti da CLI se preferisci)
# ------------------------------------------------------------------
INFILE  = pathlib.Path("resources/validation_setOLD.jsonl")
OUTFILE = pathlib.Path("resources/validation_set.jsonl")

if not INFILE.exists():
    sys.exit(f"File di input non trovato: {INFILE}")

# ------------------------------------------------------------------
# 2. Conversione riga per riga
# ------------------------------------------------------------------
n_lines = 0
with INFILE.open("r", encoding="utf-8") as fin, \
     OUTFILE.open("w", encoding="utf-8") as fout:

    for line in fin:
        if not line.strip():
            continue                          # salta righe vuote

        record = json.loads(line)
        lang   = record.get("lang", "").upper()

        # Mantiene solo le prime due lettere
        record["lang"] = lang[:2]

        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        n_lines += 1

print(f"✅  Processate {n_lines} righe — output salvato in {OUTFILE}")