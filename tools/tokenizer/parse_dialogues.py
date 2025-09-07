import csv
import sys
from pathlib import Path
import shutil
import re

# ✅ Increase max field size
csv.field_size_limit(sys.maxsize)

url_pattern = re.compile(r"https?://\S+|www\.\S+")


def process_dialogues(input_file: Path, output_dir: Path):
    conversations = {}

    with input_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for row in reader:
            if len(row) < 4:
                continue  # skip invalid rows

            conv_id = row[1].strip()
            user = "<USER>"  # ✅ Always replace username with <USER>
            message = row[-1].strip()

            # ✅ Remove URLs
            message = url_pattern.sub("<URL>", message)
    
            if message:
                if conv_id not in conversations:
                    conversations[conv_id] = []
                conversations[conv_id].append(f"{user}: {message}")

    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    all_dialogues = []
    file_counter = 1

    for conv_id, messages in conversations.items():
        if not messages:
            continue

        block_lines = ["<DIALOGUE>"]
        block_lines.extend(messages)
        block_lines.append("</DIALOGUE>")

        dialogue_text = "\n".join(block_lines)

        # ✅ Use monotonic file names
        out_file = staging_dir / f"dialogue_{file_counter}.txt"
        file_counter += 1

        out_file.write_text(dialogue_text, encoding="utf-8")
        all_dialogues.append(dialogue_text)

    # ✅ Merge into one file
    merged_file = output_dir / "merged_dialogues.txt"
    with merged_file.open("w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(d + "\n\n")

    shutil.rmtree(staging_dir)
    print(f"✅ Processed {len(all_dialogues)} conversations → {merged_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Ubuntu dialogues into <DIALOGUE> format (replace usernames with <USER>).")
    parser.add_argument("input_file", type=str, help="Path to input CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save normalized output")

    args = parser.parse_args()
    process_dialogues(Path(args.input_file).resolve(), Path(args.output_dir).resolve())
