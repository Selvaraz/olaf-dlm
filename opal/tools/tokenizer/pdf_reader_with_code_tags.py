import fitz  # PyMuPDF
import re
import shutil
from pathlib import Path

count = 0

def extract_text_blocks(pdf_path: Path, debug=False):
    """Extract headings, bullets, CLI commands, and normal text lines from a PDF using fitz."""
    text_blocks = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            try:
                lines = page.get_text("text").split("\n")
                for line in lines:
                    line_clean = line.strip()
                    if not line_clean:
                        continue
                    # If line has just a number or is empty, skip it
                    if re.match(r"^\d+$", line_clean) or not line_clean.strip():
                        continue

                    if debug:
                        print(f"Page {page_num + 1}: Processing line: '{line_clean}'")

                    # Detect headings (shorter, capitalized text with punctuation)
                    if re.match(r"^[A-Z][A-Za-z0-9\s\-\:\/.]{2,}$", line_clean) and len(line_clean.split()) <= 8:
                        section_tag = f"<SECTION>{line_clean}</SECTION>"
                        if debug:
                            global count
                            count += 1
                            if count % 100 == 0 and debug:
                                print(f"Processed {count} sections so far...")
                                print(f"  -> SECTION: {section_tag}")
                        text_blocks.append(section_tag)
                    # Detect bullets
                    elif re.match(r"^[•\-\*]\s+.*", line_clean):
                        bullet_text = re.sub(r"^[•\-\*]\s*", "", line_clean)
                        bullet_tag = f"<BULLET>{bullet_text}</BULLET>"
                        if debug:
                            print(f"  -> BULLET: {bullet_tag}")
                        text_blocks.append(bullet_tag)
                    # Detect CLI commands
                    elif re.match(r"^(Switch|Router|ping|traceroute|enable|configure|show)\b", line_clean):
                        code_tag = f"<CODE>{line_clean}</CODE>"
                        if debug:
                            print(f"  -> CODE: {code_tag}")
                        text_blocks.append(code_tag)
                    else:
                        if debug:
                            print(f"  -> TEXT: {line_clean}")
                        text_blocks.append(line_clean)
            except Exception as e:
                print(f"⚠️ Skipping page error in {pdf_path.name}: {e}")
    except Exception as e:
        print(f"❌ Could not open {pdf_path.name}: {e}")
    return text_blocks


def extract_tables(pdf_path: Path):
    """Tables not supported in fitz like pdfplumber — fallback skipped for now."""
    return []  # Placeholder (fitz doesn’t natively extract structured tables)


def process_pdf_file(pdf_path: Path, debug=False) -> str:
    """Process a single PDF into <DOC> format with sections, bullets, code, and tables."""
    text_blocks = extract_text_blocks(pdf_path, debug)
    table_blocks = extract_tables(pdf_path)

    if not text_blocks and not table_blocks:
        return ""

    doc_content = f"<DOC>\n<TITLE>{pdf_path.stem}</TITLE>\n"
    doc_content += "\n".join(text_blocks)
    if table_blocks:
        doc_content += "\n\n" + "\n".join(table_blocks)
    doc_content += "\n</DOC>"
    return doc_content


def process_pdf_directory(input_dir: Path, output_dir: Path, debug=False):
    """Process all PDFs in a directory, normalize, merge, and delete staging folder."""
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    merged_docs = []
    for pdf_file in input_dir.rglob("*.pdf"):
        try:
            print(f"📄 Processing {pdf_file.name}")
            doc_text = process_pdf_file(pdf_file, debug)
            if not doc_text.strip():
                print(f"⚠️ No content extracted from {pdf_file.name}")
                continue

            out_file = staging_dir / (pdf_file.stem + "_pdf.txt")
            out_file.write_text(doc_text, encoding="utf-8")
            merged_docs.append(doc_text)

            print(f"✅ Processed {pdf_file.name} → {out_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")

    merged_file = output_dir / "merged_pdf_docs.txt"
    try:
        with open(merged_file, "w", encoding="utf-8") as f:
            for doc in merged_docs:
                f.write(doc + "\n\n")
        print(f"\n🎉 Merged {len(merged_docs)} PDFs → {merged_file}")
    except Exception as e:
        print(f"❌ Failed to create merged file: {e}")

    try:
        #shutil.rmtree(staging_dir)
        print(f"🗑️ Deleted staging directory: {staging_dir}")
    except Exception as e:
        print(f"⚠️ Failed to delete staging directory: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize PDF files using fitz into structured <DOC> format.")
    parser.add_argument("input_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output to see processing details")

    args = parser.parse_args()
    process_pdf_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve(), args.debug)
