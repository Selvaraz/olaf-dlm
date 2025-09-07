import os
import requests
import zipfile
import csv
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from .filter_blocked_words import filter_lines
from .merge_text_files import merge_text_files_with_dedup

BOOK_IDS = [
    "66559", "1203", "4724", "1672", "1642", "21628", "3190", "36691", "12508", "29472",
    "27830", "60186", "63292", "17667", "53513", "72010", "44381", "15017", "67645",
    "2701",   # Moby Dick (contains some dialogues)
    "1524",   # Pride and Prejudice
    "345",    # Dracula
    "84",     # Frankenstein
    "98",     # A Tale of Two Cities
    "4300",   # Ulysses
    "174",    # The Picture of Dorian Gray
    "11",     # Alice’s Adventures in Wonderland
    "55",     # The Wonderful Wizard of Oz
    "1260",   # Jane Eyre
    "768",    # Wuthering Heights
]

def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"^[A-Z]+:\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✅ Skipping download, already exists: {dest}")
        return

    response = requests.get(url, stream=True, timeout=30)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f, tqdm(
        desc=f"⬇ Downloading {dest.name}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"📦 Extracted {zip_path.name} → {extract_to}")


def download_books(save_dir: Path):
    base_url = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
    for book_id in tqdm(BOOK_IDS, desc="📚 Downloading Gutenberg Books"):
        url = base_url.format(id=book_id)
        filename = save_dir / f"book_{book_id}.txt"
        # Make directoru save_dir
        os.makedirs(save_dir, exist_ok=True)
        if not filename.exists():
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                filename.write_text(r.text, encoding="utf-8")
            except Exception as e:
                print(f"❌ Failed {book_id}: {e}")


def process_cornell(cornell_zip: Path, extract_dir: Path) -> list:
    download_file(
        "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip",
        cornell_zip,
    )
    extract_zip(cornell_zip, extract_dir)

    lines = []
    with open(
        extract_dir / "cornell movie-dialogs corpus/movie_lines.txt",
        encoding="iso-8859-1",
    ) as f:
        for row in csv.reader(f, delimiter="+"):
            if len(row) == 5:
                lines.append(clean_text(row[4]))
    return lines


def process_dailydialog(dd_zip: Path, extract_dir: Path) -> list:
    download_file("http://yanran.li/files/ijcnlp_dailydialog.zip", dd_zip)
    extract_zip(dd_zip, extract_dir)

    lines = []
    with open(
        extract_dir / "ijcnlp_dailydialog/dialogues_text.txt",
        encoding="utf-8",
    ) as f:
        for line in f:
            lines.extend(
                [clean_text(x) for x in line.split("__eou__") if x.strip()]
            )
    return lines


def process_tatoeba(csv_path: Path) -> list:
    download_file("https://downloads.tatoeba.org/exports/sentences.csv", csv_path)
    lines = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row[1] == "eng":
                lines.append(clean_text(row[2]))
    return lines


def main(output_dir: str):
    output_dir = Path(output_dir)
    downloaded_dir = output_dir / "downloaded"
    input_dir = output_dir / "input"
    cleaned_final_file = output_dir / "cleaned_final_corpus.txt"

    downloaded_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Download & process datasets
    all_lines = []
    all_lines += process_cornell(downloaded_dir / "cornell.zip", input_dir / "cornell")
    all_lines += process_dailydialog(
        downloaded_dir / "dailydialog.zip", input_dir / "dailydialog"
    )
    all_lines += process_tatoeba(downloaded_dir / "sentences.csv")

    # Download Gutenberg Books
    books_dir = downloaded_dir / "gutenberg_books"
    download_books(books_dir)

    # Save intermediate merged file
    print(f"Saving merged raw corpus to: {input_dir / 'merged_raw_corpus.txt'}")
    merged_file = input_dir / "merged_raw_corpus.txt"
    merged_file.write_text("\n".join(all_lines), encoding="utf-8")

    # Filter profanity
    print(f"Filtering profanity from merged raw corpus...")
    temp_clean_file = input_dir / "conversation_clean.txt"
    filter_lines(str(merged_file), str(temp_clean_file))

    # Merge Gutenberg books & clean
    print(f"Merging Gutenberg books & cleaning...")
    merged_books_file = input_dir / "gutenberg_books.txt"
    merge_text_files_with_dedup(str(books_dir), str(merged_books_file))

    print(f"Filtering profanity from Gutenberg books...")
    cleaned_books_file = input_dir / "gutenberg_books_clean.txt"
    filter_lines(str(merged_books_file), str(cleaned_books_file))

    # Final merge
    print(f"Merging final corpus...")
    merge_text_files_with_dedup(
        str(input_dir), str(cleaned_final_file)
    )

    print(f"✅ Final cleaned corpus saved at: {cleaned_final_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, clean, and merge conversational corpora"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where all files and final corpus will be stored",
    )
    args = parser.parse_args()
    main(args.output_dir)
