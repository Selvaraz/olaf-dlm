import os
import requests

# 📚 30 conversational / dialogue-rich book IDs from Project Gutenberg
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

def download_books(save_dir="./gutenberg_books"):
    os.makedirs(save_dir, exist_ok=True)
    base_url = "https://www.gutenberg.org/files/{id}/{id}-0.txt"

    for book_id in BOOK_IDS:
        url = base_url.format(id=book_id)
        filename = os.path.join(save_dir, f"book_{book_id}.txt")

        try:
            # If the file already exist do not attempt to download
            if os.path.exists(filename):
                print(f"Skipping {book_id} as file already exists: {filename}")
                continue
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filename, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded {book_id} → {filename}")
        except Exception as e:
            print(f"Failed to download {book_id}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download conversational books from Project Gutenberg")
    parser.add_argument("output_dir", type=str, help="Output directory to store the books")
    args = parser.parse_args()
    download_books(args.output_dir)

