import re
import shutil
from pathlib import Path


def clean_gutenberg_headers(line: str, inside_content: bool) -> (bool, bool):
    """Detect start and end markers for Gutenberg books."""
    if not inside_content and "*** START OF" in line:
        return True, False
    if inside_content and "*** END OF" in line:
        return inside_content, True
    return inside_content, False


def process_book_streaming(book_path: Path, merged_file):
    """Process one book line by line and write directly to merged file."""
    try:
        with book_path.open("r", encoding="utf-8", errors="ignore") as f:
            inside_content = False
            merged_file.write(f"<DOC>\n<TITLE>{book_path.stem.replace('_', ' ').title()}</TITLE>\n")

            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                inside_content, end_marker = clean_gutenberg_headers(line, inside_content)
                if end_marker:
                    break
                if not inside_content:
                    continue

                # Detect chapters
                if re.match(r"^(CHAPTER|Chapter|Chap\.?)\s+[A-Za-z0-9IVXLC]+", line):
                    merged_file.write(f"<SECTION>{line}</SECTION>\n")
                # Detect dialogue
                elif line.startswith('"'):
                    merged_file.write(f"<DIALOGUE>{line}</DIALOGUE>\n")
                else:
                    merged_file.write(line + "\n")

            merged_file.write("</DOC>\n\n")

        print(f"✅ Processed {book_path.name}")
    except Exception as e:
        print(f"❌ Failed to process {book_path.name}: {e}")


def process_books_directory(input_dir: Path, output_dir: Path):
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    merged_file_path = output_dir / "merged_books.txt"
    with merged_file_path.open("w", encoding="utf-8") as merged_file:
        for book_file in input_dir.rglob("*.txt"):
            process_book_streaming(book_file, merged_file)

    try:
        shutil.rmtree(staging_dir)  # We no longer need staging
    except Exception:
        pass

    print(f"\n🎉 Merged all books → {merged_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Normalize Gutenberg books into structured <DOC> format (streaming).")
    parser.add_argument("input_dir", type=str, help="Directory containing Gutenberg .txt files")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_books_directory(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
