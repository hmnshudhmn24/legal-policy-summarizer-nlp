import argparse, csv, json
from pathlib import Path
from pdfminer.high_level import extract_text

def pdf_to_text(pdf_path: str) -> str:
    return extract_text(pdf_path)

def build_prompt(mode: str, text: str) -> str:
    return f"mode: {mode.strip().lower()}\ntext: {text}"

def csv_to_jsonl(input_csv: str, output_jsonl: str):
    p_in = Path(input_csv)
    p_out = Path(output_jsonl)
    p_out.parent.mkdir(parents=True, exist_ok=True)

    with p_in.open(encoding="utf-8") as fin, p_out.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        for row in reader:
            text = row.get("text", "").strip()
            mode = row.get("mode", "").strip().lower()
            target = row.get("target", "").strip()
            if not text or not mode or not target:
                continue
            prompt = build_prompt(mode, text)
            fout.write(json.dumps({"input": prompt, "target": target}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    csv_to_jsonl(args.input, args.output)
