import xml.etree.ElementTree as ET
import json
import io
import sys

def parse_tmx_utf16_to_json(tmx_file, output_file, source_lang="EN-GB"):
    # Legge il file in UTF-16LE e converte in UTF-8 in memoria
    with open(tmx_file, "r", encoding="utf-16le") as f:
        xml_content = f.read()
    xml_content = xml_content.replace('encoding="UTF-16LE"', 'encoding="utf-8"')
    xml_file = io.StringIO(xml_content)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    body = root.find("body")
    if body is None:
        body = root.find("{*}body")

    with open(output_file, "w", encoding="utf-8") as out_f:
        for tu in body.findall("tu"):
            segs = {}
            for tuv in tu.findall("tuv"):
                lang = tuv.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
                seg = tuv.find("seg")
                if lang and seg is not None:
                    segs[lang] = seg.text.strip()

            source_text = segs.get(source_lang)
            if not source_text:
                continue

            for lang_code, target_text in segs.items():
                if lang_code == source_lang or not target_text:
                    continue

                lang_name = lang_code.split("-")[0].upper()
                prompt = f"Translate the following English text to {lang_name}: '{source_text}'"
                completion = target_text

                json_line = {"prompt": prompt, "completion": completion}
                out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"✅ Conversione completata. File salvato in: {output_file}")

if __name__ == "__main__":
    parse_tmx_utf16_to_json(sys.argv[1], sys.argv[2])