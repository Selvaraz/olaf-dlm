import pdfplumber
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_pdf_to_text(pdf_path, output_path):
    with pdfplumber.open(pdf_path) as pdf:
        final_text = []

        for page in pdf.pages:
            # ✅ Extract tables first
            tables = page.extract_tables()

            # ✅ Extract text with layout info
            extracted_text = page.extract_text(layout=True)
            lines = extracted_text.split("\n") if extracted_text else []

            for line in lines:
                stripped = line.strip()

                # ✅ Detect Headings
                if stripped.isupper() or re.match(r"^[A-Z].+:$", stripped):
                    final_text.append(f"# {stripped}\n")

                # ✅ Detect Subheadings
                elif re.match(r"^[A-Z][a-zA-Z0-9\s-]+$", stripped) and len(stripped.split()) <= 7:
                    final_text.append(f"## {stripped}\n")

                # ✅ Detect Bullet Points
                elif re.match(r"^[-•●]\s+", stripped) or re.match(r"^\d+\.\s+", stripped):
                    final_text.append(f"- {stripped.lstrip('-•● ')}\n")

                else:
                    final_text.append(stripped + "\n")

            # ✅ Append tables as JSON
            if tables:
                for tbl in tables:
                    headers = tbl[0]
                    rows = tbl[1:]
                    table_json = [dict(zip(headers, row)) for row in rows]
                    final_text.append("\n```json\n")
                    final_text.append(json.dumps(table_json, indent=2))
                    final_text.append("\n```\n")

        Path(output_path).write_text("".join(final_text), encoding="utf-8")
        print(f"✅ Converted: {pdf_path.name} → {output_path.name}")


def convert_single_pdf(pdf_file, output_dir):
    output_file = Path(output_dir) / (pdf_file.stem + ".txt")
    extract_pdf_to_text(pdf_file, output_file)
    return pdf_file.name  # For logging purpose


def convert_all_pdfs(input_dir, output_dir, max_workers=4):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.rglob("*.pdf"))
    if not pdf_files:
        print("⚠️ No PDFs found in input directory.")
        return

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_single_pdf, pdf_file, output_dir): pdf_file for pdf_file in pdf_files}

        for future in as_completed(futures):
            pdf_name = futures[future].name
            try:
                result = future.result()
                print(f"🎉 Finished: {result}")
            except Exception as e:
                print(f"❌ Error converting {pdf_name}: {e}")

    print(f"✅ All PDFs converted! Output saved in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to structured text with tables in JSON.")
    parser.add_argument("--input_dir", required=True, help="Directory containing PDFs (will search recursively).")
    parser.add_argument("--output_dir", required=True, help="Directory to save converted TXT files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes (default: 4)")
    args = parser.parse_args()

    convert_all_pdfs(args.input_dir, args.output_dir, max_workers=args.workers)

# Ex: python batch_pdf_to_text_v1.py --input_dir ./pdfs --output_dir ./output --workers 6
