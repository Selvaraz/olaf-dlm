import sentencepiece as spm
from collections import Counter
import argparse
import re
import json
from pathlib import Path

# 🔹 Your fixed user-defined tokens
FIXED_USER_TOKENS = [
    "<OID>",
        "<DATETIME>",
        "<INSTANCE>",
        "<UUID>",
        "<RA>",
        "<SEVERITY>",
        "<STR>",
        "<INT>",
        "<HEX>",
        "<FLOAT>",
        "<LONG>",
        "<ULL>",
        "<PTR>",
        "<CHAR>",
        "<USER>",
        "<ASSISTANT>",
        "<URL>",
        "<TABLE_DEF>",
        "<TYPE_DEF>",
        "<ENUM_DEF>",
        "<FLAG_DEF>",
        "<ACTION_DEF>",
        "<EXECUTION_CONTEXT_DEF>",
        "<QUERY_DEF>",
        "<PATH_DEF>",
        "<URI_NODE>",
        "<TABLE_INSTANCE_DEF>",
        "<DIRECTED_TRAVERSAL>",
        "<GREEN_OPERATION_DEF>",
        "<PROCESS_DEF>",
        "<DATABASE_DEF>",
        "<SERVICE_DEF>",
        "<USES_TABLE_DEF>",
        "<LOOKUP_BY_KEY>",
        "<EMBEDS>",
        "<EXTENSION>",
        "<CONSTANT_DEF>",
        "<DEFAULT_DEF>",
        "<RETURNS>",
        "<SHOW>",
        "<PORT>",
        "<PROCESS>",
        "<MODE>",
        "<TEST>",
        "<IP>",
        "<MAC>",
        "<INTERFACE>",
        "<SICON>",
        "<IPC>",
        "<DATABASE>",
        "<SUMMARY>",
        "<LOG>",
        "</LOG>",
        "<SCHEMA>",
        "</SCHEMA>",
        "<METADATA>",
        "</METADATA>",
        "<INDEX>",
        "</INDEX>",
        "<DOC>",
        "</DOC>",
        "<TITLE>",
        "</TITLE>",
        "<SECTION>",
        "</SECTION>",
        "<BULLET>",
        "</BULLET>",
        "<CODE>",
        "</CODE>",
        "<TABLE>",
        "</TABLE>",
        "<DIALOGUE>",
        "</DIALOGUE>",
]

def load_corpus_lines(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def analyze_sentencepiece(model_path, corpus_path, output_json):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    lines = load_corpus_lines(corpus_path)

    total_tokens, total_sentences, oov_count = 0, 0, 0
    piece_counter = Counter()
    split_counter = Counter()

    for line in lines:
        pieces = sp.encode(line, out_type=str)
        total_tokens += len(pieces)
        total_sentences += 1
        oov_count += pieces.count("<unk>")

        for p in pieces:
            piece_counter[p] += 1

        for word in line.split():
            word_pieces = sp.encode(word, out_type=str)
            if len(word_pieces) > 3:  # word split into too many pieces
                split_counter[word] += 1

    avg_tokens_per_sentence = total_tokens / total_sentences if total_sentences > 0 else 0
    oov_rate = oov_count / total_tokens if total_tokens > 0 else 0

    # 🔹 Find candidate reserved tokens
    candidate_tokens = [
        w for w, _ in split_counter.most_common(300)
        if re.match(r"^[A-Za-z0-9_/.-]+$", w)  # Alphanumeric words
    ]

    # 🔹 Combine fixed + suggested tokens (remove duplicates, keep order)
    all_tokens = FIXED_USER_TOKENS + [t for t in candidate_tokens if t not in FIXED_USER_TOKENS]

    # Print Summary
    print("\n📊 SentencePiece Model Analysis")
    print("=======================================")
    print(f"✅ Vocabulary Size     : {sp.get_piece_size()}")
    print(f"✅ Corpus Lines        : {len(lines)}")
    print(f"✅ Total Tokens        : {total_tokens}")
    print(f"✅ Avg Tokens/Sentence : {avg_tokens_per_sentence:.2f}")
    print(f"✅ OOV Rate            : {oov_rate:.4%}")

    print("\n🔹 Top 20 Frequent Tokens:")
    for token, count in piece_counter.most_common(20):
        print(f"  {token:20}  {count}")

    print("\n⚠️  Top 20 Frequently Split Words:")
    for word, count in split_counter.most_common(20):
        print(f"  {word:30}  {count}")

    print("\n💡 Final Suggested Reserved Tokens (first 30):")
    for t in all_tokens[:30]:
        print(f"  {t}")

    # 🔹 Save JSON file
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump({"final_reserved_tokens": all_tokens}, jf, indent=2)

    print(f"\n📄 Final reserved tokens saved to: {output_json}")
    print(f"📌 Total tokens suggested: {len(all_tokens)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SentencePiece model and suggest reserved tokens.")
    parser.add_argument("model", type=str, help="Path to SentencePiece model (.model)")
    parser.add_argument("corpus", type=str, help="Path to text corpus file")
    parser.add_argument("--output", type=str, default="final_reserved_tokens.json",
                        help="Output JSON file to save final tokens")

    args = parser.parse_args()
    analyze_sentencepiece(args.model, args.corpus, args.output)
