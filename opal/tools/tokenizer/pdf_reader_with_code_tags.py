import pdfplumber
import re
import shutil
from pathlib import Path


def extract_text_blocks(pdf_path: Path):
    """Extract headings, bullets, CLI commands, and normal text lines from a PDF."""
    text_blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    extracted_text = page.extract_text(layout=True)
                    if not extracted_text:
                        continue
                    lines = extracted_text.split("\n")

                    for line in lines:
                        line_clean = line.strip()
                        if not line_clean:
                            continue

                        # Detect headings (shorter, capitalized text)
                        if re.match(r"^[A-Z][A-Za-z0-9\s\-]{2,}$", line_clean) and len(line_clean.split()) <= 8:
                            text_blocks.append(f"<SECTION>{line_clean}</SECTION>")
                        # Detect bullets
                        elif re.match(r"^[•\-\*]\s+.*", line_clean):
                            bullet_text = re.sub(r"^[•\-\*]\s*", "", line_clean)
                            text_blocks.append(f"<BULLET>{bullet_text}</BULLET>")
                        # Detect CLI commands
                        elif re.match(r"^(Switch|Router|ping|traceroute|enable|configure|show)\b", line_clean):
                            text_blocks.append(f"<CODE>{line_clean}</CODE>")
                        else:
                            text_blocks.append(line_clean)
                except Exception as e:
                    print(f"⚠️ Skipping page error in {pdf_path.name}: {e}")
    except Exception as e:
        print(f"❌ Could not open {pdf_path.name}: {e}")
    return text_blocks


def extract_tables(pdf_path: Path):
    """Extract tables and format them into <TABLE> blocks."""
    table_blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            formatted_rows = []
                            for row in table:
                                if not row:
                                    continue
                                cleaned_row = [cell if cell is not None else "" for cell in row]
                                formatted_rows.append(" | ".join(cleaned_row).strip())
                            if formatted_rows:
                                table_text = "<TABLE>\n" + "\n".join(formatted_rows) + "\n</TABLE>"
                                table_blocks.append(table_text)
                except Exception as e:
                    print(f"⚠️ Skipping table error in {pdf_path.name}: {e}")
    except Exception as e:
        print(f"❌ Could not extract tables from {pdf_path.name}: {e}")
    return table_blocks


def process_pdf_file(pdf_path: Path) -> str:
    """Process a single PDF into <DOC> format with sections, bullets, code, and tables."""
    text_blocks = extract_text_blocks(pdf_path)
    table_blocks = extract_tables(pdf_path)

    if not text_blocks and not table_blocks:
        return ""

    doc_content = f"<DOC>\n<TITLE>{pdf_path.stem}</TITLE>\n"
    doc_content += "\n".join(text_blocks)
    if table_blocks:
        doc_content += "\n\n" + "\n".join(table_blocks)
    doc_content += "\n</DOC>"
    return doc_content


def process_pdf_directory(input_dir: Path, output_dir: Path):
    """Process all PDFs in a directory, normalize, merge, and delete staging folder."""
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    merged_docs = []
    for pdf_file in input_dir.rglob("*.pdf"):
        try:
            doc_text = process_pdf_file(pdf_file)
            if not doc_text.strip():
                print(f"⚠️ No content extracted from {pdf_file.name}")
                continue

            out_file = staging_dir / (pdf_file.stem + ".txt")
            out_file.write_text(doc_text, encoding="utf-8")
            merged_docs.append(doc_text)

            print(f"✅ Processed {pdf_file.name} → {out_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")

    merged_file = output_dir / "merged_ccna_docs.txt"
    try:
        with open(merged_file, "w", encoding="utf-8") as f:
            for doc in merged_docs:
                f.write(doc + "\n\n")
        print(f"\n🎉 Merged {len(merged_docs)} PDFs → {merged_file}")
    except Exception as e:
        print(f"❌ Failed to create merged file: {e}")

    try:
        shutil.rmtree(staging_dir)
        print(f"🗑️ Deleted staging directory: {staging_dir}")
    except Exception as e:
        print(f"⚠️ Failed to delete staging directory: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize CCNA PDF files into structured <DOC> format.")
    parser.add_argument("input_dir", type=str, help="Directory containing CCNA PDF files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_pdf_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
