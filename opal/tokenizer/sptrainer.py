#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_sentencepiece_tokenizer.py
# author: Selvaraj Mani
# date: 09/15/2025
# Purpose: Train a SentencePiece tokenizer for code tokenization.
# Note: This script requires the `sentencepiece` library, which can be installed via pip:
# ```pip install sentencepiece```

import sentencepiece as spm
import argparse
import logging
from pathlib import Path
import time
from typing import Iterable, Iterator

### Normalization rules to apply (For the Cisco Community Forums)
# - Normalize whitespace: collapse multiple spaces/tabs/newlines into a single space.
# - Keep **one** boundary marker: `<|doc|>` at the start of each thread.
# - Use fenced blocks for structured text:
#   - ```cisco for CLI/VSAs/config
#   - ```log for time-stamped logs
#   - ```output for tabular command output
# - Replace volatiles/PII:
#   - `<IP>`, `<MAC>`, `<TS>`, `<HEX>`, `<URL>`, `<EMAIL>`, `<USER>`, `<ORG>`
# - Strip forum cruft: signatures, “Solved/Kudos” badges, page chrome, trackers.
# - Flatten deep quoting if noisy (keep only minimal quoted lines that add technical value).
# - Keep a lightweight header (Title/Topic/Date) as plain text—no custom XML tags.

# This keeps your pretraining data consistent with the rest of the corpus (configs + logs + prose), and you can later convert a subset of threads into instruction-tuning samples without changing the raw data.


try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - graceful fallback if tqdm missing
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable if iterable is not None else []

# Best practices of using the tokens in the pretrain corpus
# 1. Use <|doc|> to delimit documents
# 2. Use consistent casing and formatting for all tokens
# 3. Explicit whitespace tokens (\n, \t) prevent hex encoding
# 4. User-defined symbols preserve domain-specific terms

# The tokenizer will add the 
# --bos_piece='<s>' --eos_piece='</s>', unk_piece='<unk>', pad_piece='<pad>'
# All the router/switch/.. prompts will be replaced with device

# ✅ COMPLETED IMPROVEMENTS:
#   ✅ Added explicit newline "\n" and tab "\t" tokens
#   ✅ Added <|reasoning|> and <|end-reasoning|> tokens  
#   ✅ Added <|summary|> and <|end-summary|> tokens
#   ✅ Disabled byte_fallback to prevent hex encoding
#   ✅ Set character_coverage=1.0 for complete character inclusion


user_defined_symbols = [
        "\n",  # Explicit newline token
        "\t",  # Explicit tab token
        "```",
    "```cisco-config",
    "```cisco-exec",
    "```cisco-output",
    "```cisco-syntax",
    "```language",
    "```log",
    "```tcl",
    "<|answer|>",
    "<|command|>",
    "<|end-command|>",
    "<|doc|>",
    "<|end-answer|>",
    "<|end-question|>",
    "<|end-system|>",
    "<|interface|>",
    "<|ip|>",
    "<|mac|>",
    "<|question|>",
    "<|reasoning|>",
    "<|end-reasoning|>",
    "<|ssid|>",
    "<|ssid-name|>",
    "<|summary|>",
    "<|end-summary|>",
    "<|system|>",
    "<|timestamp|>",
    "<|url|>",
    "<|user|>",
    "<|vlan-id|>",
    "802.1X",
    "QoS",
    "STP",
    "aaa",
    "access-list",
    "access-session",
    "accounting",
    "acl",
    "action",
    "args",
    "authenticate",
    "authentication",
    "authorize",
    "authorization",
    "bay",
    "bgp",
    "binos",
    "brief",
    "bshell",
    "btdecode",
    "btrace",
    "callhome",
    "capture",
    "certificate",
    "chassis",
    "class-map",
    "clear",
    "cisco",
    "cmand",
    "coa",
    "command",
    "configure",
    "cpu",
    "crimson",
    "crypto",
    "cts",
    "database",
    "dbm",
    "debug",
    "description",
    "details",
    "dhcp",
    "disable",
    "dns",
    "dot1x",
    "enable",
    "encryption",
    "ethernet",
    "evpn",
    "event",
    "execute",
    "explanation",
    "fman_rp",
    "free",
    "fru",
    "function",
    "gre",
    "hardware",
    "http",
    "icmp",
    "identity",
    "igmp",
    "interface",
    "internal",
    "iosd",
    "ip",
    "ipv6",
    "lacp",
    "license",
    "logging",
    "login",
    "logs",
    "mac",
    "mab",
    "malloc",
    "maroon",
    "memeory",
    "metadata",
    "monitor",
    "mpls",
    "mqipc",
    "node",
    "olaf",
    "password",
    "pcap",
    "platform",
    "polcy-map",
    "policy",
    "port",
    "process",
    "prompt",
    "radius",
    "rbacl",
    "reasoning",
    "redundancy",
    "request",
    "response",
    "routing",
    "server",
    "service",
    "service-policy",
    "shell",
    "shell-manager",
    "show",
    "sicon",
    "sipcap",
    "slot",
    "smart",
    "software",
    "spanning-tree",
    "ssl",
    "standby",
    "status",
    "step_conf",
    "step_db",
    "step_exec",
    "step_explain",
    "step_function",
    "step_internal",
    "step_summary",
    "step_trace",
    "step_type",
    "steps",
    "subscription",
    "summary",
    "switch",
    "sxp",
    "syslog",
    "table",
    "tcp",
    "telemetry",
    "timeout",
    "traffic-shape",
    "troubleshoot",
    "trunk",
    "type",
    "udp",
    "using",
    "vlan",
    "vty",
    "vrf",
    "webauth",
    "wireless",
    "wlan",
    ]

def train_tokenizer(
    input_file: Path,
    model_prefix: str,
    vocab_size: int = 8000,
    model_type: str = "bpe",
    output_dir: Path = Path("."),
    show_progress: bool = True):
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
    # Optimized settings for small language model with user-defined symbols
    spm.SentencePieceTrainer.train(
        sentence_iterator=_line_iterator(input_file, show_progress),
        model_prefix=full_model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,  # Include ALL characters including whitespace
        model_type=model_type,
        unk_id=3,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        byte_fallback=False,  # Disable byte fallback to prevent hex encoding
        hard_vocab_limit=False,  # Enforce strict vocab limit for small models
        train_extremely_large_corpus=True,  # Better for smaller datasets/models
        user_defined_symbols=user_defined_symbols,
        input_sentence_size=70_000_000,  # Use a large sample of sentences
        add_dummy_prefix=False,  # Keep disabled to prevent symbol splitting
        treat_whitespace_as_suffix=False,  # Preserve whitespace as tokens
        allow_whitespace_only_pieces=True,  # Allow whitespace tokens
        split_digits=False,  # Keep numeric content intact (chapter names, versions, IPs, etc.)
        normalization_rule_name="nmt_nfkc",  # Preserve whitespace structure
        remove_extra_whitespaces=False,  # Keep all whitespace
        shuffle_input_sentence=True,  # Better training diversity
        seed_sentencepiece_size=2000000,  # Reasonable seed size
        shrinking_factor=0.75,  # Help with vocabulary pruning
        num_threads=24,  # Utilize multiple cores
        max_sentencepiece_length=20,  # Increased to accommodate longer user symbols
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
