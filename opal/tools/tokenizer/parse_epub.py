import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
import shutil


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
                text = tag.get_text().strip()
                if not text:
                    continue

                if tag.name in ["h1", "h2", "h3"]:
                    text_blocks.append(f"<SECTION>{text}</SECTION>")
                elif tag.name == "p":
                    text_blocks.append(text)
                elif tag.name == "li":
                    text_blocks.append(f"<BULLET>{text}</BULLET>")
                elif tag.name in ["pre", "code"]:
                    text_blocks.append(f"<CODE>{text}</CODE>")

            if text_blocks:
                doc_content = f"<DOC>\n<TITLE>{title}</TITLE>\n" + "\n".join(text_blocks) + "\n</DOC>"
                docs.append(doc_content)

    return "\n\n".join(docs)


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

    # Delete staging directory
    shutil.rmtree(staging_dir)
    print(f"🗑️ Deleted staging directory: {staging_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize EPUB files into structured <DOC> format.")
    parser.add_argument("input_dir", type=str, help="Directory containing EPUB files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_epub_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
