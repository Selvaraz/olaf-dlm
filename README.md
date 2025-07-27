# olaf-dlm
Olaf Domain Specific Language Model

## Setup
# Create the token model

Olaf O-PAL uses the sentencepiece as a tokenizer with 8000 tokens. We are using the BPE (Byte Pair Encoding) Tokenizer. 

Collect all the corpus text that will be used to train the tokenizer, the tokenizer uses this corpus to create the vocabulary. 

Run the following command to create the tokenizer:

```
python3 -m opal.tools.tokenizer.tokenizer_trainer -h
usage: tokenizer_trainer.py [-h] [--model_prefix MODEL_PREFIX] [--vocab_size VOCAB_SIZE] [--model_type {bpe,unigram,char,word}] input_file

Train a SentencePiece tokenizer for code tokenization. This script trains a SentencePiece tokenizer on a given input text file and generates a model
file (.model) and vocabulary file (.vocab) that can be used for tokenizing source code or other text data. The tokenizer supports different model types
(BPE, Unigram, Char, Word) and allows for custom vocabulary sizes. It also includes support for user-defined symbols that are commonly used in source
code (e.g., STR, INT, HEX). Example usage: # Train with default settings python3 tokenizer_trainer.py tokenized_trace_logs.txt # Train with custom
settings and model prefix python3 tokenizer_trainer.py tokenized_trace_logs.txt --model_prefix=custom_tokenizer --vocab_size=8000 --model_type=bpe

positional arguments:
  input_file            Path to the input text file containing the training data. This file should contain one sentence per line.

options:
  -h, --help            show this help message and exit
  --model_prefix MODEL_PREFIX
                        Prefix for output model and vocab files. The tokenizer will generate two files: <prefix>.model and <prefix>.vocab
  --vocab_size VOCAB_SIZE
                        Size of the vocabulary to generate. Larger vocab sizes can capture more unique tokens but may increase memory usage.
  --model_type {bpe,unigram,char,word}
                        Type of tokenization model to train: - bpe: Byte Pair Encoding (recommended for code) - unigram: Unigram language model - char:
                        Character-level tokenization - word: Word-level tokenization
```

This will create a tokenizer model in the opal/tools/tokenizer directory.

# Tools

The tools directory contains the tokenizer and the trainer scripts.

## PDF To Text Converter

The pdf_to_text_converter.py script converts a PDF file to a text file. Give the input directory where 
the pdf files are stored the output converted text file is stored in the txt_converted directory.


```
usage: batch_pdf_to_text.py [-h] [--threads THREADS] dir_path

Multithreaded PDF to text converter

positional arguments:
  dir_path           Directory containing PDF files.

options:
  -h, --help         show this help message and exit
  --threads THREADS  Number of threads to use (default: number of CPU cores)

Example: python3 -m opal.tools.tokenizer.batch_pdf_to_text /home/selvaraj/sample_data/pdf 

```

## Merge Text Files

The merge_text_files.py script merges and deduplicates text files for SentencePiece training. This script recursively searches a directory for .txt files and merges them into a single output file while removing duplicate lines. The resulting file is ideal for training SentencePiece tokenizers as it ensures each unique sentence is represented exactly once. The script handles UTF-8 encoded text files and includes error handling to skip problematic files while continuing with the rest of the processing.

```
python3 -m opal.tools.tokenizer.merge_text_files -h
usage: merge_text_files.py [-h] [--output OUTPUT] root_dir

Merge and deduplicate text files for SentencePiece training. This script recursively searches a directory for .txt files and merges them into a single
output file while removing duplicate lines. The resulting file is ideal for training SentencePiece tokenizers as it ensures each unique sentence is
represented exactly once. The script handles UTF-8 encoded text files and includes error handling to skip problematic files while continuing with the
rest of the processing. Example usage: # Merge all .txt files in the current directory python3 merge_text_files.py . # Merge files with a custom output
filename python3 merge_text_files.py /path/to/texts --output=merged_corpus.txt # Merge files from a specific subdirectory python3 merge_text_files.py
/path/to/data/corpus

positional arguments:
  root_dir         Root directory containing text files to merge. The script will recursively search this directory for all .txt files. The directory
                   must exist and be readable.

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Name of the output file that will contain all merged and deduplicated text. The file will be created in the root directory. If a
                   file with the same name exists, it will be overwritten.
```


# Export to ONNX

```
python opal/export/export_onnx.py \
  --checkpoint checkpoints/opal_checkpoint_epoch10.pt \
  --output checkpoints/opal_model.onnx \
  --device cpu

```
  
# Quantize ONNX model

```
python opal/export/quantize_onnx.py \
  --input checkpoints/opal_model.onnx \
  --output checkpoints/opal_model_int8.onnx
```


# Logging and Monitoring dashboards

Wandb is used to log the training and validation loss, and the model checkpoints. Tensorboard is used to log the training and validation loss, and the model checkpoints. We can use either of them to monitor the training process. Follow the below steps.

## Tensorboard

The tensorboard logging is stored in the runtime directory. To start the tensorboard server run the following command:

```
tensorboard --logdir {RUNTIME_ROOT_PATH}/runs
```


