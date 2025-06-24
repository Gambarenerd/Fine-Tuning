import xml.etree.ElementTree as ET
import io
import json
import sys

def create_validation_set(tmx_path, output_jsonl, max_tus=20, src_lang="EN-GB"):
    print(f"📂 Parsing file: {tmx_path}")  # 👈 LOG 1

    # Leggi il file in UTF-16LE e sostituisci l'encoding nel tag XML
    try:
        with open(tmx_path, "r", encoding="utf-16le") as f:
            xml_content = f.read()
        xml_content = xml_content.replace('encoding="UTF-16LE"', 'encoding="utf-8"')
    except Exception as e:
        print(f"❌ Errore durante la lettura del file: {e}")
        return

    try:
        xml_file = io.StringIO(xml_content)
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        print(f"❌ Errore nel parsing XML: {e}")
        return

    output = []
    tus = root.findall(".//tu")
    print(f"📦 Trovate {len(tus)} TU, ne userò al massimo {max_tus}")

    for tu in tus[:max_tus]:
        segs = {}
        for tuv in tu.findall("tuv"):
            lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            seg = tuv.find("seg")
            if lang and seg is not None and seg.text:
                segs[lang] = seg.text.strip()

        src = segs.get(src_lang)
        if not src:
            continue
        for lang, tgt in segs.items():
            if lang == src_lang:
                continue
            output.append({"src": src, "tgt": tgt, "lang": lang})

    with open(output_jsonl, "w", encoding="utf-8") as fw:
        for entry in output:
            fw.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Creato file '{output_jsonl}' con {len(output)} coppie.")

# Esecuzione da terminale
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Uso corretto: python create_valutation_set.py input.tmx output.jsonl")
    else:
        create_validation_set(sys.argv[1], sys.argv[2])