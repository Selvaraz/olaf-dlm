import os
from pathlib import Path
import argparse
import re

def clean_non_ascii(text: str) -> str:
    """
    Remove all non-ASCII characters (e.g., Chinese, Japanese, emojis).
    """
    return re.sub(r'[^\x00-\x7F]+', '', text)

def merge_text_files_with_dedup(input_dir: str, output_file: str):
    input_path = Path(input_dir).resolve()
    output_path = Path(output_file).resolve()

    if not input_path.exists() or not input_path.is_dir():
        raise NotADirectoryError(f"{input_path} is not a valid directory.")

    seen_lines = set()
    merged_count = 0
    skipped_files = 0

    with open(output_path, "w", encoding="utf-8") as outfile:
        for txt_file in input_path.rglob("*.txt"):
            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = clean_non_ascii(line.strip())
                        if line:  # Only write non-empty lines
                            outfile.write(line + "\n")
                merged_count += 1
                print(f"✅ Merged: {txt_file}")
            except Exception as e:
                skipped_files += 1
                print(f"⚠️ Skipped {txt_file}: {e}")

    print(f"\n🎉 Done. Merged {merged_count} files.")
    if skipped_files > 0:
        print(f"⚠️ Skipped {skipped_files} files due to errors.")
    print(f"📄 Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively merge .txt files into a single output file, removing non-ASCII characters."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing .txt files")
    parser.add_argument("output_file", type=str, help="Path for merged output file")

    args = parser.parse_args()
    merge_text_files_with_dedup(args.input_dir, args.output_file)
