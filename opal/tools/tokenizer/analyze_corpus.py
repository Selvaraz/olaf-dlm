from collections import Counter
import re
from pathlib import Path

def extract_frequent_tokens(corpus_path, top_n=100):
    text = Path(corpus_path).read_text(encoding="utf-8", errors="ignore")

    # ✅ Extract words (CLI commands, table names, etc.)
    words = re.findall(r"[a-zA-Z0-9_]+", text)

    # ✅ Ignore placeholders like <UUID>, <DATETIME>
    words = [w for w in words if not w.startswith("<") and len(w) > 2]

    counter = Counter(words)
    return counter.most_common(top_n)

def main():
    corpus_file = "/home/selvaraj/training/tokenizer_data/olaf_tokenizer_corpus.txt"  # ✅ Replace with your merged corpus path
    tokens = extract_frequent_tokens(corpus_file, top_n=200)

    print("✅ Top 50 Frequent Tokens (for possible user-defined symbols):\n")
    for word, freq in tokens[:50]:
        print(f"{word}  ({freq})")

    # ✅ Generate <TOKEN> style user-defined symbols for keywords with underscores
    user_tokens = []
    for word, freq in tokens:
        if "_" in word or word.isupper() or word.islower():
            user_tokens.append(f"<{word.upper()}>")

    print("\n📌 Suggested User-Defined Tokens to Add:\n")
    print(",".join(sorted(set(user_tokens))))

if __name__ == "__main__":
    main()
