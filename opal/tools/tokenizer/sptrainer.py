import sentencepiece as spm
import argparse
import logging
from pathlib import Path
import time


def train_tokenizer(
    input_file: Path,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    output_dir: Path = Path("."),
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
    ]),

    user_defined_symbols=sorted([
        "<ACTION_DEF>",
        "<ASSISTANT>",
        "<CONSTANT_DEF>",
        "<DATABASE>",
        "<DATABASE_DEF>",
        "<DATETIME>",
        "<DEFAULT_DEF>",
        "<DIRECTED_TRAVERSAL>",
        "<EMBEDS>",
        "<ENUM_DEF>",
        "<EXECUTION_CONTEXT_DEF>",
        "<EXTENSION>",
        "<FILEPATH>",
        "<FLAG_DEF>",
        "<FLOAT>",
        "<FUNC>",
        "<GREEN_OPERATION_DEF>",
        "<HEX>",
        "<INDEX>",
        "<INSTANCE>",
        "<INT>",
        "<INTERFACE>",
        "<IP>",
        "<IPC>",
        "<LINE>",
        "<LOG>",
        "<LOOKUP_BY_KEY>",
        "<MAC>",
        "<MODE>",
        "<MSG>",
        "<OID>",
        "<PATH_DEF>",
        "<PID>",
        "<PORT>",
        "<PROCESS>",
        "<PROCESS_DEF>",
        "<QUERY_DEF>",
        "<RA>",
        "<RETURNS>",
        "<SICON>",
        "<SEVERITY>",
        "<SHOW>",
        "<STR>",
        "<SUMMARY>",
        "<TABLE_DEF>",
        "<TABLE_INSTANCE_DEF>",
        "<TEST>",
        "<TYPE_DEF>",
        "<URI_NODE>",
        "<URL>",
        "<USER>",
        "<USES_TABLE_DEF>",
        "<UUID>",
        "u_int8",
        "uses_table_instance",
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

    cmd = (
        f"--input={input_file} "
        f"--model_prefix={full_model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--character_coverage=1.0 "
        f"--model_type={model_type} "
        f"--unk_id={3} "
        f"--pad_id={0} "
        f"--bos_id={1} "
        f"--eos_id={2} "
        f"--user_defined_symbols={','.join(user_defined_symbols)}"
    )

    logging.info("Training command:")
    logging.info(cmd)

    start_time = time.time()
    spm.SentencePieceTrainer.Train(cmd)
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
        output_dir=Path(args.output_dir)
    )


# python3 train_sentencepiece_tokenizer.py tokenized_trace_logs.txt --model_prefix=opal_tokenizer --vocab_size=8000
