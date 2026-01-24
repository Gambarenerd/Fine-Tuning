#!/usr/bin/env python3
"""
Converts instruction_pairs JSON files to JSONL format compatible with:
- finetune_eurollm.py (format: train)
- evaluate_eurollm.py (format: eval)

Input format (JSON array):
[
  {
    "instruction": "Translate the following EU legislative text from English to HU-HU...",
    "input": "text to translate",
    "output": "translated text",
    "metadata": {
      "target_lang": "HU-HU",
      ...
    }
  }
]

Output formats:
- train: {"prompt": "instruction\n\nInput: input_text", "completion": "output_text"}
- eval:  {"src": "input_text", "tgt": "output_text", "lang": "HU"}
"""

import json
import argparse
import re
from pathlib import Path


# Character normalization map
CHAR_NORMALIZE_MAP = {
    '\u201c': '"',  # " left double quotation mark
    '\u201d': '"',  # " right double quotation mark
    '\u201e': '"',  # „ double low-9 quotation mark (German/Polish)
    '\u201f': '"',  # ‟ double high-reversed-9 quotation mark
    '\u2018': "'",  # ' left single quotation mark
    '\u2019': "'",  # ' right single quotation mark
    '\u201a': "'",  # ‚ single low-9 quotation mark
    '\u201b': "'",  # ‛ single high-reversed-9 quotation mark
    '\u2013': '-',  # – en dash
    '\u2014': '-',  # — em dash
    '\u2026': '...', # … ellipsis
    '\u00a0': ' ',  # non-breaking space
}


def normalize_text(text: str) -> str:
    """Normalize special Unicode characters to ASCII equivalents."""
    for char, replacement in CHAR_NORMALIZE_MAP.items():
        text = text.replace(char, replacement)
    return text


def extract_lang_code(item: dict) -> str:
    """Extract language code from metadata or instruction."""
    # Try metadata first
    if "metadata" in item and "target_lang" in item["metadata"]:
        # "HU-HU" -> "HU"
        return item["metadata"]["target_lang"].split("-")[0]

    # Fallback: parse from instruction
    # "Translate the following EU legislative text from English to HU-HU..."
    instruction = item.get("instruction", "")
    match = re.search(r"to ([A-Z]{2})-[A-Z]{2}", instruction)
    if match:
        return match.group(1)

    # Last resort: try to find any 2-letter code
    match = re.search(r"to ([A-Z]{2})", instruction)
    if match:
        return match.group(1)

    return "UNK"


def convert_file(input_path: str, output_path: str, output_format: str) -> int:
    """
    Convert a single JSON file to JSONL format.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file
        output_format: 'train' for finetune or 'eval' for evaluate

    Returns:
        Number of records converted
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            instruction = normalize_text(item.get("instruction", "").strip())
            input_text = normalize_text(item.get("input", "").strip())
            output_text = normalize_text(item.get("output", "").strip())

            if not output_text:
                continue

            if output_format == "train":
                if not input_text:
                    continue
                # Extract short language code (MT-MT -> MT)
                lang = extract_lang_code(item)
                # Remove wrapping quotes from input using regex
                # Handles: 'text', "text", 'text';, 'text';', 'text';'', etc.
                clean_input = re.sub(r"^['\"]", "", input_text)  # Remove leading quote
                clean_input = re.sub(r"['\";]+$", "", clean_input)  # Remove trailing quotes/semicolons
                # Keep completion as-is (only normalize special chars, don't remove quotes)
                # Quotes in legal text are meaningful (term definitions, citations)

                # Use the same format as migrated_tmx1.jsonl
                prompt = f"Translate the following English text to {lang}: '{clean_input}'"

                record = {
                    "prompt": prompt,
                    "completion": output_text
                }

            elif output_format == "eval":
                if not input_text:
                    continue
                lang = extract_lang_code(item)
                record = {
                    "src": input_text,
                    "tgt": output_text,
                    "lang": lang
                }

            else:
                raise ValueError(f"Unknown format: {output_format}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Convert instruction_pairs JSON to JSONL format for fine-tuning or evaluation"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSON file(s) to convert"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["train", "eval"],
        default="train",
        help="Output format: 'train' for finetune_eurollm.py, 'eval' for evaluate_eurollm.py (default: train)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help="Suffix to add to output filename (default: _train or _eval based on format)"
    )

    args = parser.parse_args()

    suffix = args.suffix if args.suffix else f"_{args.format}"

    for input_file in args.input_files:
        input_path = Path(input_file)

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path.parent

        output_filename = f"{input_path.stem}{suffix}.jsonl"
        output_path = output_dir / output_filename

        print(f"Converting: {input_path} (format: {args.format})")
        count = convert_file(str(input_path), str(output_path), args.format)
        print(f"  -> {output_path} ({count} records)")

    print("\nConversion completed!")


if __name__ == "__main__":
    main()
