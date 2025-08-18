import sentencepiece as spm
import argparse
import logging
from pathlib import Path
import time
from typing import Iterable, Iterator

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - graceful fallback if tqdm missing
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []


def train_tokenizer(
    input_file: Path,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    output_dir: Path = Path("."),
    show_progress: bool = True,
    block_wrappers = sorted([
        "<BULLET>",
        "</BULLET>",
        "<CODE>",
        "</CODE>",
        "<DIALOGUE>",
        "</DIALOGUE>",
        "<DOC>",
        "</DOC>",
        "<INDEX>",
        "</INDEX>",
        "<LOG>",
        "</LOG>",
        "<METADATA>",
        "</METADATA>",
        "<SCHEMA>",
        "</SCHEMA>",
        "<SECTION>",
        "</SECTION>",
        "<TABLE>",
        "</TABLE>",
        "<TITLE>",
        "</TITLE>",
        "<RECORD>",
        "</RECORD>"
    ]),

    user_defined_symbols=sorted([
        "<SEP>",
        "<ASSISTANT>",
        "<DATABASE>",
        "<DATETIME>",
        "<EXTENSION>",
        "<FILEPATH>",
        "<FLOAT>",
        "<FUNC>",
        "green_operation_def",
        "table_def",
        "type_def",
        "<HEX>",
        "<INSTANCE>",
        "<INT>",
        "<INTERFACE>",
        "<IP>",
        "<IPC>",
        "<LINE>",
        "<MAC>",
        "<MODE>",
        "<MSG>",
        "<OID>",
        "<PID>",
        "<PORT>",
        "<PROCESS>",
        "<RA>",
        "<RETURNS>",
        "<SICON>",
        "<SEVERITY>",
        "<SHOW>",
        "<STR>",
        "<SUMMARY>",
        "<TEST>",
        "<URL>",
        "<USER>",
        "<UUID>",
        "u_int8_t",
        "<STUB>",
        "<END>",
    ]),
):
    logging.info(f"Starting tokenizer training...")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Model prefix: {model_prefix}")
    logging.info(f"Vocab size: {vocab_size}")
    logging.info(f"Model type: {model_type}")
    logging.info(f"Output directory: {output_dir}")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        logging.error(f"Input file does not exist: {input_file}")
        return

    # Merge the user_defined_symbols with block_wrappers
    user_defined_symbols = user_defined_symbols + block_wrappers
    # Build the full model_prefix path in the output directory
    full_model_prefix = str(output_dir / model_prefix)

    # Helper: iterator that yields lines and updates tqdm if enabled
    def _line_iterator(path: Path, enable_bar: bool) -> Iterator[str]:
        # We intentionally don't pre-count lines to avoid extra I/O on huge corpora
        desc = "Feeding sentences to trainer"
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            if enable_bar:
                for line in tqdm(f, desc=desc, unit="lines", leave=True):
                    yield line
            else:
                for line in f:
                    yield line

    start_time = time.time()
    # Use kwargs-based API so we can pass a sentence iterator to show progress.
    # All parameters mirror the CLI flags used previously.
    spm.SentencePieceTrainer.train(
        sentence_iterator=_line_iterator(input_file, show_progress),
        model_prefix=full_model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type=model_type,
        unk_id=3,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        byte_fallback=True,
        hard_vocab_limit=False,
        train_extremely_large_corpus=True,
        user_defined_symbols=user_defined_symbols,
        add_dummy_prefix=True,  # Always add dummy prefix for consistency
    )
    duration = time.time() - start_time

    model_file = output_dir / f"{model_prefix}.model"
    vocab_file = output_dir / f"{model_prefix}.vocab"

    if model_file.exists() and vocab_file.exists():
        logging.info(f"✅ Tokenizer training complete in {duration:.2f} seconds.")
        logging.info(f"Model saved as: {model_file}")
        logging.info(f"Vocab saved as: {vocab_file}")
    else:
        logging.error("❌ Tokenizer training failed. Output files not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="""Train a SentencePiece tokenizer for code tokenization.

This script trains a SentencePiece tokenizer on a given input text file and generates
a model file (.model) and vocabulary file (.vocab) that can be used for tokenizing
source code or other text data.

The tokenizer supports different model types (BPE, Unigram, Char, Word) and allows
for custom vocabulary sizes. It also includes support for user-defined symbols that
are commonly used in source code (e.g., STR, INT, HEX).

Example usage:
    # Train with default settings
    python3 tokenizer_trainer.py tokenized_trace_logs.txt

    # Train with custom settings and model prefix
    python3 tokenizer_trainer.py tokenized_trace_logs.txt \
        --model_prefix=custom_tokenizer \
        --vocab_size=8000 \
        --model_type=bpe
"""
)
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input text file containing the training data. This file should contain one sentence per line."
    )
    parser.add_argument(
        "--model_prefix", 
        type=str, 
        default="opal_tokenizer",
        help="Prefix for output model and vocab files. The tokenizer will generate two files: \
             <prefix>.model and <prefix>.vocab"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=8000,
        help="Size of the vocabulary to generate. Larger vocab sizes can capture more unique tokens but may increase memory usage."
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="""Type of tokenization model to train:
            - bpe: Byte Pair Encoding (recommended for code)
            - unigram: Unigram language model
            - char: Character-level tokenization
            - word: Word-level tokenization"""
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where the trained model and vocabulary will be saved. Defaults to current directory."
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar while feeding sentences to the trainer.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    train_tokenizer(
        input_file=Path(args.input_file),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
    output_dir=Path(args.output_dir),
    show_progress=not args.no_progress,
    )


# python3 train_sentencepiece_tokenizer.py tokenized_trace_logs.txt --model_prefix=opal_tokenizer --vocab_size=8000
