from pathlib import Path
import re

BLOCKED_FILE = Path(__file__).parent / "blocked_words.txt"
INPUT_FILE = "conversation_corpus/conversation_corpus.txt"
OUTPUT_FILE = "conversation_corpus/conversation_clean_corpus.txt"

def filter_lines(input_file, output_file):

    # Load blocked words
    with open(BLOCKED_FILE, "r", encoding="utf-8") as f:
        blocked_words = [w.strip().lower() for w in f if w.strip()]

    blocked_pattern = re.compile(r"(" + "|".join(map(re.escape, blocked_words)) + r")", re.IGNORECASE)

    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        total, kept, flagged = 0, 0, 0
        for line in infile:
            total += 1
            if blocked_pattern.search(line):
                flagged += 1
                continue
            outfile.write(line)
            kept += 1
    print(f"✅ Total lines: {total}")
    print(f"✅ Clean lines: {kept}")
    print(f"🚨 Flagged lines: {flagged}")

#filter_lines(INPUT_FILE, OUTPUT_FILE)
