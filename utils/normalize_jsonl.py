#!/usr/bin/env python3
"""
Normalize special Unicode characters in JSONL files to ASCII equivalents.
Useful for cleaning up text before training.
"""

import json
import argparse
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


def normalize_value(value):
    """Recursively normalize strings in a data structure."""
    if isinstance(value, str):
        return normalize_text(value)
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


def normalize_file(input_path: str, output_path: str) -> tuple[int, int]:
    """
    Normalize a JSONL file.

    Returns:
        Tuple of (total_records, modified_records)
    """
    total = 0
    modified = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            total += 1
            original = line
            data = json.loads(line)
            normalized_data = normalize_value(data)
            normalized_line = json.dumps(normalized_data, ensure_ascii=False) + "\n"

            if normalized_line != original:
                modified += 1

            f_out.write(normalized_line)

    return total, modified


def main():
    parser = argparse.ArgumentParser(
        description="Normalize special Unicode characters in JSONL files"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input JSONL file(s) to normalize"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--suffix",
        default="_normalized",
        help="Suffix for output filename (default: _normalized)"
    )
    parser.add_argument(
        "--inplace", "-i",
        action="store_true",
        help="Modify files in place (overwrites original)"
    )

    args = parser.parse_args()

    for input_file in args.input_files:
        input_path = Path(input_file)

        if args.inplace:
            # Write to temp file, then replace
            temp_path = input_path.with_suffix('.tmp')
            output_path = temp_path
        elif args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{input_path.stem}{args.suffix}.jsonl"
        else:
            output_path = input_path.parent / f"{input_path.stem}{args.suffix}.jsonl"

        print(f"Normalizing: {input_path}")
        total, modified = normalize_file(str(input_path), str(output_path))

        if args.inplace:
            output_path.replace(input_path)
            print(f"  -> {input_path} (in-place, {modified}/{total} records modified)")
        else:
            print(f"  -> {output_path} ({modified}/{total} records modified)")

    print("\nNormalization completed!")


if __name__ == "__main__":
    main()
