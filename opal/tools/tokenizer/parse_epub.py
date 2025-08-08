import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
import shutil
import re
from secure_do_not_commit.text_normalizer import normalize_line

# # Import global normalization function
# try:
#     from secure_do_not_commit.text_normalizer import normalize_line
# except ImportError:
#     # Fallback if import fails
#     def normalize_line(text: str, apply_content_formatting: bool = False) -> str:
#         return text

MESSAGE_PATTERNS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<HEX>"),       # Pointers
    (re.compile(r"\b\d+\.\d+\b"), "<FLOAT>"),       # Floats
    (re.compile(r"\b\d+ULL\b"), "<INT>"),           # Unsigned long long
    (re.compile(r"\b\d+L\b"), "<INT>"),            # Long ints
    (re.compile(r"\b\d+\b"), "<INT>"),              # Integers
    (re.compile(r"[A-Za-z0-9_\-]+mqipc[A-Za-z0-9_\-]*"), "<STR>"),  # mqipc-like strings
]


def normalize_log_line(line: str) -> str:
    """
    Replaces timestamps and hex patterns with tokens using global normalization.
    """
    return normalize_line(line, apply_content_formatting=False)


def normalize_text(tag) -> str:
    """
    Extracts text preserving newlines in <pre>/<code> and logical structure for others.
    Removes decorative lines like '----- ------' or '==== ======'.
    """
    if tag.name in ["pre", "code"]:
        raw_text = tag.get_text()
        lines = raw_text.splitlines()

        # Match lines that are mostly decorative (repeated characters with optional spaces)
        def is_decorative(line: str) -> bool:
            # Remove spaces to normalize
            no_space = line.replace(" ", "")
            # Line must be at least 10 characters long
            if len(no_space) < 10:
                return False
            # Check if line is purely made of decoration chars (no letters/numbers)
            if re.fullmatch(r"[-=~_*]+", no_space) and not re.search(r"\w", line):
                return True
            return False


        clean_lines = [normalize_log_line(line) for line in lines if not is_decorative(line)]
        return "\n".join(clean_lines).strip()

    else:
        return tag.get_text(separator=" ").strip()



def parse_epub_file(epub_file: Path) -> str:
    """Extract structured content from a single EPUB file and return as <DOC> block."""
    book = epub.read_epub(str(epub_file))
    title = book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else epub_file.stem

    docs = []
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text_blocks = []

            for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"]):
                text = normalize_text(tag)
                if re.match(r"^[-=~_*]{10,}$", text):
                    continue  # Skip decorative lines

                # Skip empty lines and whitespace-only content
                if not text.strip():
                    continue

                if tag.name in ["h1", "h2", "h3"]:
                    text_blocks.append(f"<SECTION>{text}</SECTION>")
                elif tag.name == "p":
                    # Additional check for paragraph content to avoid empty paragraphs
                    if text.strip():
                        text_blocks.append(text)
                elif tag.name == "li":
                    text_blocks.append(f"<BULLET>{text}</BULLET>")
                elif tag.name in ["pre", "code"]:
                    text_blocks.append(f"<CODE>\n{text}\n</CODE>")  # 🔥 preserve formatting

            if text_blocks:
                doc_content = f"<DOC>\n<TITLE>{title}</TITLE>\n" + "\n\n".join(text_blocks) + "\n</DOC>"
                docs.append(doc_content)

    return "\n".join(docs)


def process_epub_directory(input_dir: Path, output_dir: Path):
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    merged_docs = []
    for epub_file in input_dir.rglob("*.epub"):
        try:
            doc_text = parse_epub_file(epub_file)
            if not doc_text.strip():
                print(f"⚠️ No content extracted from {epub_file.name}")
                continue

            # Parse each line in the doc_text to make sure there is no empty line
            doc_text = "\n".join(line.strip() for line in doc_text.splitlines() if line.strip())
            if not doc_text:
                print(f"⚠️ No valid content in {epub_file.name}")
                continue
            out_file = staging_dir / (epub_file.stem + ".txt")
            out_file.write_text(doc_text, encoding="utf-8")
            merged_docs.append(doc_text)

            print(f"✅ Processed {epub_file.name} → {out_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {epub_file.name}: {e}")

    # Merge into final file
    merged_file = output_dir / "merged_docs.txt"
    with open(merged_file, "w", encoding="utf-8") as f:
        for doc in merged_docs:
            f.write(doc + "\n\n")

    print(f"\n🎉 Merged {len(merged_docs)} EPUB files → {merged_file}")

    # Optional cleanup
    # shutil.rmtree(staging_dir)
    print(f"🗑️ Deleted staging directory: {staging_dir}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Normalize EPUB files into structured <DOC> format.")
#     parser.add_argument("input_dir", type=str, help="Directory containing EPUB files")
#     parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

#     args = parser.parse_args()
#     process_epub_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
#     # Merge into final file
#     merged_file = args.output_dir / "merged_docs.txt"
#     with open(merged_file, "w", encoding="utf-8") as f:
#         for doc in merged_docs:
#             f.write(doc + "\n\n")

#     print(f"\n🎉 Merged {len(merged_docs)} EPUB files → {merged_file}")

#     # Optional cleanup
#     # shutil.rmtree(staging_dir)
#     print(f"🗑️ Deleted staging directory: {staging_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize EPUB files into structured <DOC> format.")
    parser.add_argument("input_dir", type=str, help="Directory containing EPUB files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_epub_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
