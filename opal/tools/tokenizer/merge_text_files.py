import os
from pathlib import Path
import argparse


def merge_text_files_with_dedup(root_dir: str, output_file: str = "sentencepiece_input.txt"):
    root_path = Path(root_dir).resolve()

    if not root_path.exists() or not root_path.is_dir():
        raise NotADirectoryError(f"{root_path} is not a valid directory.")

    output_path = root_path / output_file
    seen_lines = set()
    merged_count = 0
    skipped_files = 0

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for txt_file in root_path.rglob("*.txt"):
            if txt_file.resolve() == output_path:
                continue  # Avoid appending the output file to itself

            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and line not in seen_lines:
                            seen_lines.add(line)
                            outfile.write(line + "\n")
                merged_count += 1
                print(f"Merged: {txt_file}")
            except Exception as e:
                skipped_files += 1
                print(f"⚠️ Skipped {txt_file}: {e}")

    print(f"\n✅ Done. Merged {merged_count} files with {len(seen_lines)} unique lines.")
    if skipped_files > 0:
        print(f"⚠️ Skipped {skipped_files} files due to errors.")
    print(f"📄 Output written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Merge and deduplicate text files for SentencePiece training.

This script recursively searches a directory for .txt files and merges them into
a single output file while removing duplicate lines. The resulting file is ideal
for training SentencePiece tokenizers as it ensures each unique sentence is
represented exactly once.

The script handles UTF-8 encoded text files and includes error handling to skip
problematic files while continuing with the rest of the processing.

Example usage:
    # Merge all .txt files in the current directory
    python3 merge_text_files.py .

    # Merge files with a custom output filename
    python3 merge_text_files.py /path/to/texts --output=merged_corpus.txt

    # Merge files from a specific subdirectory
    python3 merge_text_files.py /path/to/data/corpus
"""
    )
    parser.add_argument(
        "root_dir", 
        type=str, 
        help="""Root directory containing text files to merge. The script will recursively
             search this directory for all .txt files. The directory must exist
             and be readable."""
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="sentencepiece_input.txt",
        help="""Name of the output file that will contain all merged and deduplicated text.
             The file will be created in the root directory. If a file with the
             same name exists, it will be overwritten."""
    )

    args = parser.parse_args()
    merge_text_files_with_dedup(args.root_dir, args.output)
