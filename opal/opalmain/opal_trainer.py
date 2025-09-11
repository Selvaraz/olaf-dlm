import json
import random
import tempfile
import atexit
import random
import glob
from datetime import datetime
import multiprocessing
import json
import time
import os
import shutil
import psutil
import tempfile
import atexit
from opal.dataloader.OpalFileDataSet import OpalFileDataset
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List
from ..dataloader.OpalDataSet import OpalDataset
from ..dataloader.OpalFineTuneDataSet import OpalFinetuneDataset
from torch.utils.data import Dataset, DataLoader
from ..utils.opal_constants import OpalConstants
from ..export.export_onnx import export_and_quantize_model
from opal.config.opal_config import TRAINING_CONFIG, get_gpu_memory_allocated_size, get_scaler
import sentencepiece as spm
from tqdm import tqdm
from ..export.opal_evaluator import evaluate_pytorch, evaluate_onnx
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

#For TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
# For weights and biases logging
import wandb


class Opal:
    def __init__(self, config, tokenizer=None, 
                start_fresh=False, is_finetune=False,
                finetune_data_path=None):
        self.config = config
        self.tokenizer = tokenizer
        self.start_fresh = start_fresh
        self.is_finetune = is_finetune
        self.finetune_data_path = finetune_data_path
    
    def collate_finetune(self, batch):
        """
        Dynamically pads a batch of fine-tuning samples to the longest sequence.
        This function is used by the DataLoader to pad input and label tensors
        to a uniform size for the current batch.
        """
        # CRITICAL FIX: Ensure pad_id is within vocab bounds to prevent CUDA index errors
        vocab_size = self.tokenizer.get_piece_size()
        pad_id = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else self.tokenizer.unk_id()
        
        # Validate pad_id is within bounds
        if pad_id >= vocab_size:
            pad_id = 0  # Use 0 as fallback
        
        # Silent bounds checking and fixing
        for i, (input_ids, labels) in enumerate(batch):
            input_min, input_max = input_ids.min().item(), input_ids.max().item()
            labels_valid = labels[labels != -100]
            if len(labels_valid) > 0:
                labels_min, labels_max = labels_valid.min().item(), labels_valid.max().item()
            else:
                labels_min, labels_max = -100, -100
                
            # Silently clamp any out-of-bounds tokens
            if input_max >= vocab_size or input_min < 0:
                batch[i] = (torch.clamp(input_ids, 0, vocab_size - 1), labels)
                
            if labels_max >= vocab_size or (labels_min < 0 and labels_min != -100):
                valid_mask = labels != -100
                labels[valid_mask] = torch.clamp(labels[valid_mask], 0, vocab_size - 1)
                batch[i] = (batch[i][0], labels)

        # Unzip the batch into separate lists for inputs and labels
        input_ids, labels = zip(*batch)

        # Pad the input tensors
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_id
        )

        # Pad the labels tensors, using -100 to ignore loss on padded tokens
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )

        # FINETUNE_PH2: Enforce max context length to prevent memory issues
        max_context_length = self.config.get("context_length", 512)
        if padded_inputs.size(1) > max_context_length:
            padded_inputs = padded_inputs[:, :max_context_length]
            padded_labels = padded_labels[:, :max_context_length]

        # Final silent validation and fixing
        max_input_token = padded_inputs.max().item()
        min_input_token = padded_inputs.min().item()
        
        if min_input_token < 0 or max_input_token >= vocab_size:
            # Emergency clamp ALL tokens
            padded_inputs = torch.clamp(padded_inputs, 0, vocab_size - 1)
            
            # Also clamp labels (but preserve -100)
            label_mask = padded_labels != -100
            padded_labels[label_mask] = torch.clamp(padded_labels[label_mask], 0, vocab_size - 1)

        # Final silent double-check
        final_input_min, final_input_max = padded_inputs.min().item(), padded_inputs.max().item()
        final_labels_valid = padded_labels[padded_labels != -100]
        if len(final_labels_valid) > 0:
            final_labels_min, final_labels_max = final_labels_valid.min().item(), final_labels_valid.max().item()
        else:
            final_labels_min, final_labels_max = -100, -100
            
        # Absolute final silent fix
        if final_input_max >= vocab_size or final_input_min < 0:
            padded_inputs = torch.clamp(padded_inputs, 0, vocab_size - 1)
            
        if final_labels_max >= vocab_size or (final_labels_min < 0 and final_labels_min != -100):
            label_mask = padded_labels != -100
            padded_labels[label_mask] = torch.clamp(padded_labels[label_mask], 0, vocab_size - 1)

        return padded_inputs, padded_labels

    def collate_unused_finetune(self, batch):
        """Return a tuple (input_ids, labels) to match the model's forward(input, labels) signature."""
        """
            Dynamically pads a batch of fine-tuning samples to the longest sequence.
            This function is used by the DataLoader to pad input and label tensors
            to a uniform size for the current batch.
        """
        pad_id = self.tokenizer.pad_id() if self.tokenizer.pad_id() >= 0 else self.tokenizer.unk_id()
        max_len = max(len(x[0]) for x in batch)
        input_ids, labels = [], []

        for input_ids_tensor, labels_tensor in batch:
            pad_len = max_len - len(input_ids_tensor)

            padded_inputs = torch.cat(
                [input_ids_tensor, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            padded_labels = torch.cat(
                [labels_tensor, torch.full((pad_len,), -100, dtype=torch.long)]
            )

            input_ids.append(padded_inputs)
            labels.append(padded_labels)

        return torch.stack(input_ids), torch.stack(labels)

    def createOpalFinetuneDataLoader(
        self,
        data_jsonl: str,              # path to JSONL file for fine-tuning
        batch_size: int = None,
        max_length: int = 1024,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
    ):
        """
        Creates a DataLoader for fine-tuning using OpalFinetuneDataset.

        Args:
            data_jsonl (str): Path to JSONL fine-tuning dataset file.
            batch_size (int, optional): If None, uses TRAINING_CONFIG["batch_size"].
            max_length (int): Maximum token sequence length per sample.
            shuffle (bool): Whether to shuffle dataset each epoch.
            drop_last (bool): Drop last batch if incomplete.
            num_workers (int): Number of workers for DataLoader (0 for CUDA to avoid fork issue).
            device (str): Device to put data on ('cpu' or 'cuda' or 'mps').

        Returns:
            DataLoader: PyTorch DataLoader for fine-tuning.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when creating Opal instance.")
        
        # with open(data_jsonl, "r", encoding="utf-8") as f:
        #     data = [json.loads(line.strip()) for line in f]
        data = []
        with open(data_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"⚠ Skipping malformed JSONL line: {line[:50]}...")  # Review-1: Added malformed line handling

        print(f"📄 Loaded {len(data)} fine-tuning samples from {data_jsonl}")

        # Create dataset
        dataset = OpalFinetuneDataset(
            data=data,
            tokenizer=self.tokenizer,
        )

        # Set batch size
        if batch_size is None:
            batch_size = TRAINING_CONFIG.get("batch_size", 4)

        print(f"✅ Creating Fine-tune DataLoader → batch_size={batch_size}, shuffle={shuffle}, workers={num_workers}")

        # Create DataLoader with collate function for dynamic padding
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self.collate_finetune
        )

    def createOpalDataLoader(
        self,
        txt,  # Can be str or torch.Tensor 
        batch_size: int = TRAINING_CONFIG["batch_size"],
        max_length: int = 1280,
        stride: int = 256,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        device: str = None
    ):
        """
        Creates a DataLoader for the OpalDataset using optimal settings based on CPU cores.

        Args:
            txt (str or torch.Tensor): Raw input text or pretokenized tensor.
            batch_size (int, optional): If None, automatically set to 8 for >16 cores, else 4.
            max_length (int): Maximum token sequence length per sample.
            stride (int): Overlap between chunks.
            shuffle (bool): Whether to shuffle dataset each epoch.
            drop_last (bool): Drop last batch if incomplete.
            num_workers (int, optional): If None, automatically set to available CPU cores.
            device (str): Device to put data on ('cpu' or 'cuda' or 'mps').

        Returns:
            DataLoader: PyTorch DataLoader optimized for your CPU.
        """
        device = TRAINING_CONFIG["device"]

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when creating Opal instance.")

        # Print out chosen parameters for transparency
        print(f"Creating DataLoader with {num_workers} workers, batch_size={batch_size}, prefetch_factor=4")

        dataset = OpalDataset(
            txt=txt,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=stride,
            device=device
        )

        # Use persistent_workers=True and prefetch_factor=4 to reduce worker startup overhead
        return DataLoader(
            dataset,
            batch_size=batch_size,  # 🔧 FIXED: Use passed batch_size parameter, not config
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True, 
            persistent_workers=self.config["persistent_workers"],
            prefetch_factor= 4 if num_workers > 0 else None
        )
    
    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded).unsqueeze(0)

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())

    ##

    def _pretokenize_corpus(self, input_text_file, tokenizer_model, output_file):
        """
        Tokenizes the entire corpus and saves as a tensor for faster training restarts.
        STREAMING implementation: encodes line-by-line to avoid gigantic single-string encode.
        """
        print(f"✅  Using the tokenizer model at path: {tokenizer_model}")
        if not os.path.exists(tokenizer_model):
            raise ValueError(f"❌  Tokenizer model file not found: {tokenizer_model}")
        sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

        # Fail fast: tokenizer should produce something for a probe
        probe = sp.encode("hello world", out_type=int)
        if not probe:
            raise RuntimeError("Tokenizer probe returned 0 ids — check TOKENIZER_MODEL_PATH.")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"✅  Pretokenizing corpus (streaming)... from {input_text_file}")
        eos_id = sp.eos_id() if sp.eos_id() >= 0 else 0
        add_eos = True

        total_ids = 0
        total_lines = 0
        ids_chunks = []  # accumulate moderate-size chunks

        # If the user requested start_fresh=False and we see UNKs, abort early
        abort_on_unk = (not self.start_fresh)

        with open(input_text_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                total_lines += 1
                line = line.rstrip("\n")
                if not line:
                    ids = [eos_id] if add_eos else []
                else:
                    ids = sp.encode(line, out_type=int)
                    if abort_on_unk:
                        toks = sp.encode(line, out_type=str)
                        if "<unk>" in toks:
                            print(f"⚠️ New tokens found (UNK) in line {total_lines}: {line[:120]}")
                            raise ValueError(
                                "New tokens found in text. Please start with start_fresh=True "
                                "(retrain tokenizer or accept UNKs)."
                            )
                    if add_eos:
                        ids.append(eos_id)

                if ids:
                    ids_chunks.extend(ids)
                    total_ids += len(ids)

                if total_lines % 100_000 == 0:
                    print(f"  ...lines={total_lines:,}, ids={total_ids:,}")

        if total_ids == 0:
            raise RuntimeError("Pretokenization produced 0 token ids. Check filtering/encoding logic.")

        # Save as torch.long to match downstream expectations
        token_ids = torch.tensor(ids_chunks, dtype=torch.long)
        torch.save(token_ids, output_file)
        print(f"-- Pretokenized dataset saved to {output_file} (length={token_ids.numel():,})\n")


    ##
    def __pretokenize_corpus(self, input_text_file, tokenizer_model, output_file):
        """
        Tokenizes the entire corpus once and saves as a tensor for faster training restarts.
        """

        print(f"✅  Using the tokenizer model at path: {tokenizer_model}")
        if not os.path.exists(tokenizer_model):
            raise ValueError(f"❌  Tokenizer model file not found: {tokenizer_model}")
        
        sp = spm.SentencePieceProcessor(model_file=tokenizer_model)

        print(f" Loading the corpus text ...")
        with open(input_text_file, "r") as f:
            text = f.read()
        
        if self.check_new_tokens(text) and not self.start_fresh:
            print(f"🚨🚨🚨 New tokens found in text. Pretokenizing corpus...")
            raise ValueError("New tokens found in text. Please start training from scratch with start_fresh=True")
        else:
            print(f"✅  No new tokens found in text.")

        # If the directory to the output file is not found, create it
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        print(f"✅  Pretokenizing corpus... size of corpus: {len(text)}")
        token_ids = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
        torch.save(token_ids, output_file)

        print(f"-- Pretokenized dataset saved to {output_file} (length={len(token_ids)})")

    def loadTrainingData(self, token_model):
        """
        Loads training data from a text file.

        Reads the content of the file "the-verdict.txt" located in the "data" directory
        relative to the current file's directory. The contents of the file are returned
        as a single string.

        Returns:
            str: The content of the text file as a string.
        """
        txt = None
    
        # Get the directory of the current file        # Construct path
        #file_path = parent_dir / "data" / "tokenizer_text" / "network_tokenizer_text_v1.txt"
        #file_path = parent_dir / "sample_data"  / "the-verdict.txt"
        file_path = OpalConstants.PRETRAIN_DATA_PATH
        pretokenized_path = OpalConstants.PRETOKENIZED_DATA_PATH

        print(f"✅  Using the corpus text at path: {file_path}")

        if not os.path.exists(file_path):
            raise ValueError(f"❌  Corpus text file not found: {file_path}")

        if os.path.exists(pretokenized_path):
            print(f"-- Loading pre-tokenized dataset: {pretokenized_path}")
            return torch.load(pretokenized_path)
        else:
            print(f"❌ Pre-tokenized dataset not found: {pretokenized_path}")
            print(f"Loading raw text from: {file_path}")
            self._pretokenize_corpus(file_path, token_model, pretokenized_path)
            print(f"✅ Pre-tokenized dataset saved to {pretokenized_path}")
            return torch.load(pretokenized_path)

    # Ex: top_p=0.9 and optionally temperature > 0.7
    def generate(
    self,
    model,
    idx: torch.Tensor,                  # [B, T] token ids (usually B=1 for sampling)
    max_new_tokens: int,
    context_size: int,                  # model's max context length (e.g., pos_emb size)
    top_k: int | None = None,
    top_p: float | None = None,         # nucleus sampling (0<top_p<=1)
    temperature: float = 1.0,           # 0 => greedy
    eos_id: int | None = None,
    repetition_penalty: float = 1.0,    # >1.0 penalizes repeats
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with top-k / top-p (nucleus) filtering,
        temperature scaling, and an improved repetition penalty that weighs recent occurrences.

        This function is hardened against advanced-indexing mistakes that can
        cause CUDA device-side asserts (IndexKernel). In particular, top-p filtering
        uses scatter() / masked_fill() per-batch instead of 2D tensor fancy indexing.
        """
        model.eval()
        device = idx.device

        B = idx.size(0)
        assert B >= 1, "idx must have batch dimension >=1 (e.g., [1, T])"

        with torch.no_grad():
            for _ in range(max_new_tokens):

                # Keep the last `context_size` tokens to stay within the positional embedding range
                idx_cond = idx[:, -context_size:]

                # Forward pass
                model_output = model(idx_cond)
                # ✅ Take only last token logits
                logits = model_output["logits"] if isinstance(model_output, dict) else model_output
                logits = logits[:, -1, :]  # [B, V] — logits for the next-token position only
                V = logits.size(-1)

                # ---------------------------------------------------------------------
                # Improved repetition penalty - penalize recently repeated tokens more heavily
                # We apply a stronger penalty for tokens that appear multiple times
                # in the *recent* context (last 20 tokens), with extra weight for
                # very recent (last 3) positions.
                # ---------------------------------------------------------------------
                if repetition_penalty and repetition_penalty > 1.0:
                    # Consider only the last 20 tokens per sequence
                    window = min(20, idx_cond.size(1))
                    recent = idx_cond[:, -window:]  # [B, W]
                    # Clamp recent tokens to valid vocab just in case (prevents indexing surprises)
                    recent = recent.clamp_(0, V - 1)

                    # For B==1 (typical sampling) we can do a fast path; otherwise do a loop
                    for b in range(B):
                        recent_b = recent[b]  # [W]
                        # Count occurrences and remember positions
                        # Build a map: token_id -> positions (0..W-1)
                        # (Python dict is fine at tiny W; this keeps code clear.)
                        token_positions = {}
                        # Enumerate from left to right; positions grow with recency
                        for i_pos, tok in enumerate(recent_b.tolist()):
                            token_positions.setdefault(tok, []).append(i_pos)

                        # Apply penalties
                        # We only touch tokens that appear more than once to avoid over-penalizing
                        # legitimate single occurrences.
                        for tok, positions in token_positions.items():
                            if len(positions) <= 1:
                                continue
                            # Base frequency penalty scales with count
                            freq_pen = repetition_penalty ** len(positions)

                            # Extra penalty for *very* recent repetition (appeared in last 3 tokens)
                            most_recent_pos = positions[-1]
                            if most_recent_pos >= window - 3:
                                freq_pen *= 1.5

                            # Apply to the corresponding logit (batched-safe)
                            # Guard for safety in case tok somehow falls outside [0, V-1]
                            if 0 <= tok < V:
                                # If logit is positive we divide; if negative we multiply (GPT-2 style)
                                # to reduce the probability of repeated tokens while preserving sign.
                                if logits[b, tok] > 0:
                                    logits[b, tok] = logits[b, tok] / freq_pen
                                else:
                                    logits[b, tok] = logits[b, tok] * freq_pen

                # ---------------------------------------------------------------------
                #  Apply temperature scaling (makes probabilities sharper or smoother)
                # ---------------------------------------------------------------------
                if temperature is not None and temperature > 0.0:
                    logits = logits / temperature
                    # (If temperature == 0.0 we will fall back to greedy after filtering.)
                # else: keep original logits for greedy

                # ---------------------------------------------------------------------
                # Top-k filtering (keep only the k highest logit tokens per row)
                # ---------------------------------------------------------------------
                if top_k is not None and top_k > 0 and top_k < V:
                    # min value per row to keep
                    topk_vals, _ = logits.topk(top_k, dim=-1)                 # [B, k]
                    kth = topk_vals[..., -1, None]                            # [B, 1]
                    # Mask out everything below the kth value
                    logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)

                # ---------------------------------------------------------------------
                # Top-p (nucleus) filtering (keep the smallest set whose cumulative prob >= p)
                # IMPORTANT: do all masking per-batch via scatter()/masked_fill() to avoid
                #            out-of-bounds advanced indexing on [B, V] tensors.
                # ---------------------------------------------------------------------
                if top_p is not None and 0.0 < top_p < 1.0:
                    # Work on softmax probabilities in *sorted* order
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # [B, V]
                    probs = torch.softmax(sorted_logits, dim=-1)                                  # [B, V]
                    cumulative_probs = probs.cumsum(dim=-1)                                       # [B, V]

                    # Tokens to remove: those after the nucleus
                    sorted_indices_to_remove = cumulative_probs > top_p                           # [B, V]
                    # Shift to always keep at least the first (highest-prob) token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False

                    # ✅ Scatter the "remove" mask back to original indices
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)                # [B, V]
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

                    # ✅ Mask them out in-place (batched safe)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # ---------------------------------------------------------------------
                # Sample or greedy-pick the next token
                # ---------------------------------------------------------------------
                if temperature is not None and temperature <= 0.0:
                    # Greedy (argmax) if temperature==0
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]
                else:
                    # Convert to proper probabilities after all filtering
                    probs = torch.softmax(logits, dim=-1)                    # [B, V]

                    # Safety: if a row becomes all -inf (rare with top-k/p), fallback to argmax
                    # (softmax(-inf) -> NaN); detect and recover.
                    nan_rows = torch.isnan(probs).any(dim=-1)                # [B]
                    if nan_rows.any():
                        fallback = torch.argmax(logits[nan_rows], dim=-1, keepdim=True)
                        # initialize next_token with multinomial then patch fallback rows
                        next_token = torch.multinomial(
                            torch.softmax(torch.where(nan_rows[:, None], torch.zeros_like(logits), logits), dim=-1),
                            num_samples=1
                        )
                        next_token[nan_rows] = fallback
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

                # Append sampled token
                idx = torch.cat([idx, next_token], dim=1)  # [B, T+1]

                # Early stop if EOS reached for all sequences
                if eos_id is not None:
                    if torch.all(next_token.squeeze(-1) == eos_id):
                        break

        model.train()  # restore training mode for caller, if needed
        return idx


    def generate_v0(self, model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

         
        # The following loop generates one token at a time, for a total
        # of max_new_tokens iterations. At each iteration, the model
        # is fed the current sequence (idx) and generates a new token.
        # The new token is then appended to the current sequence, and
        # the loop continues until max_new_tokens tokens have been
        # generated.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                model_output = model(idx_cond)
            logits = model_output["logits"][0, -1, :]

            #  Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            #  Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx

    def train_model_simple(self, model, train_loader, val_loader, 
                        optimizer, scheduler, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer,
                        writer=None, log_to_wandb=False):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        best_val_loss = float("inf")
        epochs_no_improve = 0  # Track epochs without improvement
        early_stopping_patience = self.config["early_stopping_patience"]

        # FINETUNE_PH2: Get configuration values for both pretraining and fine-tuning
        use_mixed_precision = TRAINING_CONFIG.get("mixed_precision", False)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        
        # 🚨 CRITICAL FIX: Use LOWER gradient accumulation for fine-tuning to prevent CUDA errors
        if self.is_finetune:
            default_accumulation = 1  # 🚨 REDUCED: Lower for fine-tuning stability
        else:
            default_accumulation = 4  # Standard for pretraining
            
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", default_accumulation)

        # 🚨 CRITICAL FIX: Force disable mixed precision for fine-tuning to prevent CUDA errors
        if self.is_finetune:
            use_mixed_precision = False
            print(f"🔧 Mixed precision FORCED OFF for fine-tuning stability")
        
        scaler = get_scaler() if use_mixed_precision else None

        # FINETUNE_PH2: Adaptive Warmup - both pretraining and fine-tuning benefit from warmup
        # Adjust total steps for gradient accumulation
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        total_steps = num_epochs * steps_per_epoch
        if self.is_finetune:
            # Fine-tuning: lighter warmup (2% of total steps or configured warmup_steps)
            #warmup_steps = min(self.config.get("warmup_steps", int(total_steps * 0.02)), int(total_steps * 0.1))
            warmup_steps = 0
        else:
            # Pretraining: standard warmup (5% of total steps)
            warmup_steps = int(total_steps * 0.05)
        
        print(f"🚀 === TRAINING PIPELINE INITIALIZATION ===")
        print(f"📊 Mode: {'FINE-TUNING' if self.is_finetune else 'PRETRAINING'}")
        print(f"📊 Training setup: {total_steps:,} total steps, {warmup_steps:,} warmup steps")
        print(f"📊 Epochs: {num_epochs}, Batches per epoch: {len(train_loader):,}")
        print(f"📊 Effective batches per epoch (with accumulation): {steps_per_epoch:,}")
        print(f"📊 Batch size: {len(train_loader.dataset) // len(train_loader)}")
        print(f"📊 Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"📊 Effective batch size: {(len(train_loader.dataset) // len(train_loader)) * gradient_accumulation_steps}")
        print(f"📊 Evaluation frequency: every {eval_freq} steps, {eval_iter} batches per eval")
        print(f"📊 Early stopping patience: {early_stopping_patience} epochs")
        print(f"📊 Mixed precision: {use_mixed_precision}")
        print(f"📊 Max gradient norm: {max_grad_norm}")
        print(f"📊 Device: {device}")
        print(f"🚀 ==========================================")
        
        # Create warmup scheduler
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # The main scheduler (cosine) is passed from train_and_save_model

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_start_time = time.time()

            print(f"\n🔄 === EPOCH {epoch+1}/{num_epochs} STARTING ===")
            print(f"📊 Best validation loss so far: {best_val_loss:.6f}")
            print(f"📊 Epochs without improvement: {epochs_no_improve}")

            # Create a progress bar for the training data
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()

            epoch_best_val_loss = best_val_loss  # Track best val loss for this epoch

            # Initialize gradient accumulation for this epoch
            optimizer.zero_grad(set_to_none=True)

            # Training loop with gradient accumulation
            accumulated_loss = 0.0
            for batch_idx, (input_ids, targets) in enumerate(pbar):
                # Move input and target tensors to the specified device
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Calculate loss for this batch
                loss = self.calc_loss_batch(input_ids, targets, model, device)
                
                # Scale loss by gradient accumulation steps to get the average
                loss = loss / gradient_accumulation_steps
                accumulated_loss += loss.item()

                # Safety check for NaN or infinite loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"🚨 WARNING: Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {global_step+1}!")
                    print(f"🚨 Skipping this batch and continuing training...")
                    continue

                # Backpropagation with mixed precision if enabled
                if use_mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Only update weights every gradient_accumulation_steps
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                is_last_batch = batch_idx == len(train_loader) - 1
                
                if is_accumulation_step or is_last_batch:
                    total_norm = 0.0  # For gradient norm calculation
                    
                    if use_mixed_precision:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        
                        # Calculate gradient norm for logging
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Calculate gradient norm for logging
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        # Clip gradients and step optimizer
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                        optimizer.step()

                    # Zero gradients after weight update
                    optimizer.zero_grad(set_to_none=True)

                    # Update learning rate and global step only after actual weight updates
                    if global_step < warmup_steps:
                        warmup_scheduler.step()
                    elif global_step == warmup_steps:
                        current_lr = optimizer.param_groups[0]["lr"]
                        print(f"\n🔥 WARMUP COMPLETED! Transitioning to cosine annealing at step {global_step+1}")
                        print(f"🔥 Learning rate at warmup completion: {current_lr:.2e}")
                        if scheduler:
                            scheduler.step()
                    else:
                        if scheduler:
                            scheduler.step()

                    global_step += 1
                    
                    # Update progress bar with accumulated loss
                    if hasattr(loss, 'item'):
                        pbar.set_postfix({
                            'acc_loss': f'{accumulated_loss:.4f}',
                            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                            'tokens': f'{tokens_seen:,}',
                            'acc_step': f'{(batch_idx + 1) // gradient_accumulation_steps + 1}'
                        })
                    
                    # Reset accumulated loss
                    accumulated_loss = 0.0

                # Update tokens seen for every batch (not just accumulation steps)
                tokens_seen += input_ids.numel()

                # Print a sample text after each epoch
                # self.generate_and_print_sample(
                #     model, tokenizer, device, start_context
                # )

                # Evaluation - only check on actual weight update steps
                if is_accumulation_step or is_last_batch:
                    # Adaptive evaluation frequency for pretraining vs fine-tuning
                    eval_frequency = eval_freq if not self.is_finetune else max(eval_freq * 4, 100)
                    if global_step % eval_frequency == 0 and global_step > 0:
                        train_loss, val_loss = self.evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        
                        # Calculate perplexity from losses
                        train_perplexity = torch.exp(torch.tensor(train_loss)).item()
                        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                        
                        # Get current learning rate for logging
                        current_lr = optimizer.param_groups[0]["lr"]
                        warmup_progress = min(global_step / warmup_steps, 1.0) if warmup_steps > 0 else 1.0
                        
                        # Early Stopping Logic (best val loss updated here)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(f"🔥 New best val_loss {val_loss:.6f}! Saving checkpoint...")
                            self.save_model_checkpoint(
                                self.config, model, optimizer, scheduler,
                                epoch, train_losses, val_losses,
                                tokenizer_model=OpalConstants.TOKENIZER_MODEL_PATH
                            )
                        else:
                            print(f"⚠️ No improvement (current: {val_loss:.6f}, best: {best_val_loss:.6f})")

                        # Calculate tokens/sec
                        elapsed = time.time() - start_time
                        tokens_per_sec = tokens_seen / max(elapsed, 1e-6)

                        # Get memory usage
                        cpu_mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                        gpu_mem_mb = get_gpu_memory_allocated_size() / (1024 * 1024) if get_gpu_memory_allocated_size() > 0 else 0
                        
                        # Enhanced logging with warmup info and perplexity
                        warmup_status = f"Warmup {warmup_progress:.1%}" if global_step < warmup_steps else "Post-warmup"
                        mode_prefix = "FT" if self.is_finetune else "PT"
                        print(f"{mode_prefix} Ep {epoch+1} (Step {global_step:06d}/{total_steps:06d}) {warmup_status}: "
                            f"Train loss {train_loss:.6f} (PPL {train_perplexity:.2f}), "
                            f"Val loss {val_loss:.6f} (PPL {val_perplexity:.2f}), "
                            f"LR {current_lr:.2e}, "
                            f"CPU mem {cpu_mem_mb:.2f} MB, GPU mem {gpu_mem_mb:.2f} MB, "
                            f"Tokens/sec {tokens_per_sec:.2f}")

                        # Log metrics to TensorBoard
                        if writer:
                            writer.add_scalar("Loss/train", train_loss, global_step)
                            writer.add_scalar("Loss/val", val_loss, global_step)
                            writer.add_scalar("Perplexity/train", train_perplexity, global_step)
                            writer.add_scalar("Perplexity/val", val_perplexity, global_step)
                            writer.add_scalar("LearningRate", current_lr, global_step)
                            writer.add_scalar("WarmupProgress", warmup_progress, global_step)
                            writer.add_scalar("GradNorm", total_norm, global_step)
                            writer.add_scalar("Tokens/sec", tokens_per_sec, global_step)
                            writer.add_scalar("CPU_Memory_MB", cpu_mem_mb, global_step)
                            if gpu_mem_mb > 0:
                                writer.add_scalar("GPU_Memory_MB", gpu_mem_mb, global_step)

                        # Log metrics to Weights & Biases
                        if log_to_wandb:
                            wandb.log({
                                "train_loss": train_loss,
                                "val_loss": val_loss,
                                "train_perplexity": train_perplexity,
                                "val_perplexity": val_perplexity,
                                "lr": current_lr,
                                "warmup_progress": warmup_progress,
                                "grad_norm": total_norm,
                                "tokens_per_sec": tokens_per_sec,
                                "cpu_memory_mb": cpu_mem_mb,
                                "gpu_memory_mb": gpu_mem_mb,
                                "step": global_step,
                                "mode": "finetune" if self.is_finetune else "pretrain"
                            })

            # ✅ After each epoch, check if val_loss improved in this epoch
            epoch_duration = time.time() - epoch_start_time
            
            if best_val_loss < epoch_best_val_loss:
                epochs_no_improve = 0
                improvement_msg = f"✅ Validation loss improved this epoch!"
            else:
                epochs_no_improve += 1
                improvement_msg = f"⚠️ No improvement for {epochs_no_improve} epochs"
            
            print(f"\n🏁 === EPOCH {epoch+1}/{num_epochs} COMPLETED ===")
            print(f"⏱️ Epoch duration: {epoch_duration:.2f} seconds")
            print(f"📊 {improvement_msg}")
            print(f"📊 Current best validation loss: {best_val_loss:.6f}")

            if epochs_no_improve >= early_stopping_patience:
                print(f"\n⛔ === EARLY STOPPING TRIGGERED ===")
                print(f"⛔ No improvement for {early_stopping_patience} consecutive epochs!")
                print(f"⛔ Final best validation loss: {best_val_loss:.6f}")
                print(f"⛔ Training stopped at epoch {epoch+1}/{num_epochs}")
                
                # Export to ONNX even when early stopping
                self._export_to_onnx(device, val_loader, writer, log_to_wandb)
                
                return train_losses, val_losses, track_tokens_seen

            # Print a sample text after each epoch
            # self.generate_and_print_sample(
            #     model, tokenizer, device, start_context
            # )

            if self.is_finetune:
                self.generate_for_finetune(
                    model, tokenizer, device, start_context
                )
                # Every few epochs, test generation diversity
                if (epoch + 1) % 3 == 0:  # Every 3rd epoch
                    self.improve_generation_diversity(
                        model, tokenizer, device, start_context
                    )
            else:
                self.generate_with_topk(
                    model, tokenizer, device, start_context, top_k=50
                )

        print(f"\n🎉 === TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"🎉 All {num_epochs} epochs completed!")
        print(f"🎉 Final best validation loss: {best_val_loss:.6f}")
        print(f"🎉 Total training steps: {global_step+1:,}")
        print(f"🎉 Total tokens processed: {tokens_seen:,}")
        print(f"🎉 ========================================")

        # Export to ONNX and quantized ONNX after training completion
        self._export_to_onnx(device, val_loader, writer, log_to_wandb)

        return train_losses, val_losses, track_tokens_seen

    def _export_to_onnx(self, device, val_loader=None, writer=None, log_to_wandb=False):
        """
        Helper method to export the trained model to ONNX and quantized ONNX formats.
        
        Args:
            device (str): Device used for training
            val_loader: Validation DataLoader for evaluation (optional)
            writer: TensorBoard writer (optional)
            log_to_wandb (bool): Whether to log to Weights & Biases
        """
        try:
            print(f"\n🔄 === EXPORTING TO ONNX ===")
            
            # Get the latest checkpoint path
            if not self.is_finetune:
                latest_checkpoint = os.path.join(OpalConstants.CHECKPOINT_DIR, "checkpoint-latest.pt")
            else:
                latest_checkpoint = os.path.join(OpalConstants.CHECKPOINT_DIR, "finetune-latest.pt")
            
            if os.path.exists(latest_checkpoint):
                # Resolve symlink to get actual checkpoint path
                final_checkpoint_path = os.path.realpath(latest_checkpoint)
                checkpoint_dir = os.path.dirname(final_checkpoint_path)
                checkpoint_filename = os.path.splitext(os.path.basename(final_checkpoint_path))[0]
                
                # Define ONNX output paths
                onnx_path = os.path.join(checkpoint_dir, f"{checkpoint_filename}.onnx")
                quantized_path = os.path.join(checkpoint_dir, f"{checkpoint_filename}_quantized.onnx")
                
                print(f"📦 Exporting PyTorch model to ONNX...")
                print(f"📦 Checkpoint: {final_checkpoint_path}")
                print(f"📦 ONNX output: {onnx_path}")
                print(f"📦 Quantized output: {quantized_path}")
                
                # Export and quantize
                from ..export.export_onnx import export_and_quantize_model
                export_and_quantize_model(
                    config=self.config,
                    checkpoint_path=final_checkpoint_path,
                    onnx_output_path=onnx_path,
                    quantized_output_path=quantized_path,
                    device=device
                )
                
                print(f"✅ ONNX export completed successfully!")
                print(f"✅ Standard ONNX model: {onnx_path}")
                print(f"✅ Quantized ONNX model: {quantized_path}")
                
                # Optionally evaluate the exported models for comparison
                if val_loader is not None:
                    print(f"\n🔍 === EVALUATING EXPORTED MODELS ===")
                    try:
                        # Evaluate PyTorch model
                        pytorch_loss, pytorch_ppl = evaluate_pytorch(final_checkpoint_path, val_loader, device)
                        print(f"📊 PyTorch Model - Loss: {pytorch_loss:.4f}, Perplexity: {pytorch_ppl:.4f}")
                        
                        # Evaluate Quantized ONNX model
                        onnx_loss, onnx_ppl = evaluate_onnx(quantized_path, val_loader, device)
                        print(f"📊 Quantized ONNX - Loss: {onnx_loss:.4f}, Perplexity: {onnx_ppl:.4f}")
                        
                        # Log to TensorBoard if available
                        if writer:
                            writer.add_scalar("Final_Eval/Loss_PyTorch", pytorch_loss)
                            writer.add_scalar("Final_Eval/Perplexity_PyTorch", pytorch_ppl)
                            writer.add_scalar("Final_Eval/Loss_ONNX", onnx_loss)
                            writer.add_scalar("Final_Eval/Perplexity_ONNX", onnx_ppl)
                        
                        # Log to W&B if available
                        if log_to_wandb:
                            wandb.log({
                                "final_loss_pytorch": pytorch_loss,
                                "final_ppl_pytorch": pytorch_ppl,
                                "final_loss_onnx": onnx_loss,
                                "final_ppl_onnx": onnx_ppl
                            })
                        
                    except Exception as eval_error:
                        print(f"⚠️ Model evaluation failed: {eval_error}")
                        print(f"⚠️ ONNX models exported successfully but evaluation skipped")
                
            else:
                print(f"⚠️ No checkpoint found at {latest_checkpoint}, skipping ONNX export")
                
        except Exception as export_error:
            print(f"❌ ONNX export failed: {export_error}")
            print(f"❌ Training completed but ONNX export encountered an error")


    def evaluate_model(self, model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            print(f"   [ ✅ Evaluating... eval_iter={eval_iter}, val_loader batches={len(val_loader)} ]")
            train_loss = self.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
            if torch.isnan(torch.tensor(val_loss)):
                print("⚠️ WARNING: val_loss became NaN!")   
        model.train()
        return train_loss, val_loss


    def generate_with_topk(self, model, tokenizer, device, start_context, top_k):
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate(model=model, idx=encoded, 
                                      context_size=context_size, 
                                      top_k=40,  # Reduced top_k for less randomness
                                      top_p=0.85,  # Reduced nucleus sampling for more focus
                                      temperature=0.7,  # Lower temperature for less randomness
                                      max_new_tokens=30,  # Shorter outputs to prevent repetition
                                      eos_id=tokenizer.eos_id(),
                                      repetition_penalty=3.0)  # Higher repetition penalty
            decoded_text = self.token_ids_to_text(token_ids)
            print("==========================================")
            print(decoded_text.replace("\n", " "))  # Compact print format
            print("==========================================")
        model.train()

    def generate_for_finetune(self, model, tokenizer, device, start_context):
        """
        Specialized generation method for fine-tuning with more conservative settings
        to avoid repetitive outputs.
        """
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate(model=model, idx=encoded, 
                                      context_size=context_size, 
                                      top_k=30,  # More focused top-k
                                      top_p=0.85,  # Slightly more conservative nucleus sampling
                                      temperature=0.8,  # Lower temperature for more deterministic output
                                      max_new_tokens=40,  # Slightly fewer tokens
                                      eos_id=tokenizer.eos_id(),
                                      repetition_penalty=3.0)  # Strong repetition penalty
            decoded_text = self.token_ids_to_text(token_ids)
            print("\n")
            print("========== FINE-TUNE GENERATION ==========")
            print(decoded_text.replace("\n", " "))  # Compact print format
            print("===========================================")
            print("\n")
        model.train()

    def analyze_finetune_data_quality(self, jsonl_file, sample_size=5):
        """
        Analyze fine-tuning data for potential issues that could cause repetitive generation.
        """
        print("🔍 === ANALYZING FINE-TUNING DATA QUALITY ===")
        
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
            
        print(f"📊 Total examples: {len(lines)}")
        
        # Sample some examples for analysis
        sample_lines = lines[:sample_size] if len(lines) >= sample_size else lines
        
        # Track statistics
        total_prompt_len = 0
        total_response_len = 0
        json_responses = 0
        str_responses = 0
        complex_structures = 0
        
        for i, line in enumerate(sample_lines):
            item = json.loads(line)
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            
            total_prompt_len += len(prompt)
            
            # Analyze response structure
            if isinstance(response, str):
                str_responses += 1
                response_text = response
                total_response_len += len(response_text)
            else:
                json_responses += 1
                response_text = json.dumps(response, ensure_ascii=False, separators=(',', ':'))
                total_response_len += len(response_text)
                
                # Check for complex nested structures
                def count_nesting(obj, level=0):
                    if isinstance(obj, dict):
                        return max(count_nesting(v, level + 1) for v in obj.values()) if obj else level
                    elif isinstance(obj, list):
                        return max(count_nesting(item, level + 1) for item in obj) if obj else level
                    return level
                
                nesting = count_nesting(response)
                if nesting > 3:
                    complex_structures += 1
            
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"Response type: {'JSON' if isinstance(response, dict) else 'String'}")
            print(f"Response length: {len(response_text)} chars")
            print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            print(f"Response: {response_text[:100]}..." if len(response_text) > 100 else f"Response: {response_text}")
            
            # Check for repetitive patterns
            response_words = response_text.split()
            word_counts = {}
            for word in response_words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            repeated_words = {word: count for word, count in word_counts.items() if count > 3}
            if repeated_words:
                print(f"⚠️ Repeated words in response: {repeated_words}")
        
        # Print summary statistics
        print(f"\n� === DATASET SUMMARY ===")
        print(f"Average prompt length: {total_prompt_len / len(sample_lines):.1f} chars")
        print(f"Average response length: {total_response_len / len(sample_lines):.1f} chars")
        print(f"String responses: {str_responses}/{len(sample_lines)}")
        print(f"JSON responses: {json_responses}/{len(sample_lines)}")
        print(f"Complex nested structures: {complex_structures}/{json_responses if json_responses > 0 else 1}")
        
        if complex_structures > 0:
            print("⚠️ WARNING: Complex JSON structures detected. Consider simplifying or using structured tokens.")
        
        if total_response_len / len(sample_lines) > 500:
            print("⚠️ WARNING: Very long responses detected. Consider truncating or chunking.")
            
        print("�🔍 ==========================================")
        
        return {
            "avg_prompt_len": total_prompt_len / len(sample_lines),
            "avg_response_len": total_response_len / len(sample_lines),
            "json_responses": json_responses,
            "str_responses": str_responses,
            "complex_structures": complex_structures
        }

    def improve_generation_diversity(self, model, tokenizer, device, start_context, num_samples=3):
        """
        Generate multiple samples with different settings to test diversity.
        """
        print("🎯 === TESTING GENERATION DIVERSITY ===")
        
        # Test different parameter combinations
        test_configs = [
            {"top_k": 25, "top_p": 0.8, "temp": 0.7, "rep_penalty": 3.5, "name": "Conservative"},
            {"top_k": 40, "top_p": 0.9, "temp": 1.0, "rep_penalty": 2.5, "name": "Balanced"},
            {"top_k": 60, "top_p": 0.95, "temp": 1.2, "rep_penalty": 2.0, "name": "Creative"},
        ]
        
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        
        for config in test_configs:
            print(f"\n--- {config['name']} Settings ---")
            print(f"top_k={config['top_k']}, top_p={config['top_p']}, temp={config['temp']}, rep_penalty={config['rep_penalty']}")
            
            with torch.no_grad():
                token_ids = self.generate(
                    model=model, 
                    idx=encoded.clone(), 
                    context_size=context_size, 
                    top_k=config['top_k'], 
                    top_p=config['top_p'],
                    temperature=config['temp'],
                    max_new_tokens=30,
                    eos_id=tokenizer.eos_id(),
                    repetition_penalty=config['rep_penalty']
                )
                decoded_text = self.token_ids_to_text(token_ids)
                print(f"Output: {decoded_text.replace(chr(10), ' ')}")  # Replace newlines with spaces
        
        model.train()
        print("🎯 =====================================")
            

            
    def _generate_and_print_sample(self, model, tokenizer, device, start_context):
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
            decoded_text = self.token_ids_to_text(token_ids)
            print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()


    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def calc_loss_batch(self, input_batch, target_batch, model, device):
        """
        Compute the loss for a single batch during training.

        Reads the input and target tensors, moves them to the specified device,
        and computes the loss using the model's forward method.

        Args:
            input_batch (torch.Tensor): Input tensor batch (e.g., token IDs).
            target_batch (torch.Tensor): Target tensor batch (e.g., labels).
            model (torch.nn.Module): The model used for training.
            device (str): The device to perform the computation on ('cpu' or 'cuda').

        Returns:
            torch.Tensor: The computed loss value.
        """

        # Move inputs to the correct device
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        # 1️⃣ Forward pass: Let the model compute logits and loss
        # If labels are provided, the model itself computes loss (with ignore_index=-100)
        model_output = model(input_batch, labels=target_batch)

        # 2️⃣ Extract loss properly
        if isinstance(model_output, dict):
            # Preferred path – model returns {"logits": ..., "loss": ...}
            loss = model_output.get("loss", None)
            if loss is None:
                # Fallback if loss not computed in forward()
                logits = model_output["logits"]
                
                # ✅ CRITICAL DEBUG: Check logits dimensions before loss computation
                print(f"🔍 Loss computation debug:")
                print(f"   Logits shape: {logits.shape}")
                print(f"   Target shape: {target_batch.shape}")
                print(f"   Logits vocab dimension: {logits.size(-1)}")
                print(f"   Expected vocab size: {self.config.get('vocab_size', 'MISSING')}")
                print(f"   Target range: [{target_batch.min().item()}, {target_batch.max().item()}]")
                
                # Check if vocab dimensions match
                expected_vocab = self.config.get('vocab_size', 12000)
                actual_vocab = logits.size(-1)
                if actual_vocab != expected_vocab:
                    print(f"🚨 CRITICAL MISMATCH: Logits vocab={actual_vocab} != expected={expected_vocab}")
                    print(f"   This indicates model was trained with different vocab size!")
                    print(f"   🛡️  EMERGENCY: Cannot fix vocab size mismatch at runtime")
                    raise ValueError(f"Model vocab size mismatch: {actual_vocab} vs {expected_vocab}")
                
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), target_batch.view(-1))
        else:
            # If model returns only logits (legacy behavior)
            logits = model_output
            
            # ✅ CRITICAL DEBUG: Check logits dimensions before loss computation  
            print(f"🔍 Loss computation debug (legacy path):")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Target shape: {target_batch.shape}")
            print(f"   Logits vocab dimension: {logits.size(-1)}")
            print(f"   Expected vocab size: {self.config.get('vocab_size', 'MISSING')}")
            print(f"   Target range: [{target_batch.min().item()}, {target_batch.max().item()}]")
            
            # Check if vocab dimensions match
            expected_vocab = self.config.get('vocab_size', 12000)
            actual_vocab = logits.size(-1)
            if actual_vocab != expected_vocab:
                print(f"🚨 CRITICAL MISMATCH: Logits vocab={actual_vocab} != expected={expected_vocab}")
                raise ValueError(f"Model vocab size mismatch: {actual_vocab} vs {expected_vocab}")
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_batch.view(-1))

        return loss



    def calc_loss_loader(self,data_loader, model, device, num_batches=None):
        """
        Calculate total loss over a portion of a data loader.

        Args:
        - data_loader (DataLoader): A PyTorch DataLoader.
        - model (nn.Module): A PyTorch neural network model.
        - device (torch.device): The device (e.g. GPU or CPU) to use for computations.
        - num_batches (int, optional): The number of batches to use from the data loader.
            If None, use all batches in the data loader. Defaults to None.

        Returns:
        - total_loss (float): The total loss over the given number of batches.
        """
        total_loss = 0.
        if len(data_loader) == 0:
            print("⚠️ Validation loader is EMPTY! Returning NaN")
            return float("nan")
        
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def generate_text_simple(self, model, idx, max_new_tokens, context_size):
        print("Generating text with context size:", max_new_tokens)
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]
            
            # Get the predictions
            with torch.no_grad():
                model_output = model(idx_cond)
                logits = model_output["logits"]
            
            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
    
    def save_model_checkpoint(self, config, model, optimizer, scheduler, epoch, train_losses, 
                              val_losses, tokenizer_model,
                              timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")):
        """
        Saves a trained model checkpoint including model state, optimizer state,
        epoch, training history, and config.

        Args:
            model (torch.nn.Module): The trained OpalGPT model instance.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): Current epoch number (for resuming training).
            train_losses (list): Training loss history.
            val_losses (list): Validation loss history.
            config (dict): Model configuration dictionary.
            save_dir (str): Directory to save checkpoints.

        Returns:
            str: Path to the saved checkpoint file.
        """
        #os.makedirs(OpalConstants.CHECKPOINT_DIR, exist_ok=True)

        # checkpoint_path = os.path.join(OpalConstants.CHECKPOINT_DIR, 
        #                             f"opal_gpt_checkpoint_{timestamp}.pt")
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config,
            "tokenizer_model": tokenizer_model,
        }


        # Create a directory with current date and save the model inside the path
        date_dir = datetime.now().strftime("%Y%m%d")
        if not self.is_finetune:
            checkpoint_dir = os.path.join(OpalConstants.CHECKPOINT_DIR, date_dir, timestamp)
        else:
            checkpoint_dir = os.path.join(OpalConstants.CHECKPOINT_DIR, "_finetune_", date_dir, timestamp)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"opal_gpt_checkpoint_{timestamp}.pt")
        
        torch.save(checkpoint, checkpoint_path)

        # Copy the tokenizer model to the checkpoint directory
        tokenizer_model_path = os.path.join(checkpoint_dir, "opal_tokenizer.model")
        shutil.copyfile(tokenizer_model, tokenizer_model_path)

        #print(f"Model checkpoint saved to {checkpoint_path}")
        # Create a symlink to the latest checkpoint
        if not self.is_finetune:
            symlink_path = os.path.join(OpalConstants.CHECKPOINT_DIR, "checkpoint-latest.pt")
        else:
            symlink_path = os.path.join(OpalConstants.CHECKPOINT_DIR, "finetune-latest.pt")

        if os.path.exists(symlink_path):
            if os.path.islink(symlink_path) or os.path.isfile(symlink_path):
                #print(f"Removing existing symlink or file at {symlink_path}")
                os.remove(symlink_path)
            elif os.path.isdir(symlink_path):
                #print(f"Removing existing directory at {symlink_path}")
                shutil.rmtree(symlink_path)
        os.symlink(checkpoint_path, symlink_path)
        #print(f"Latest checkpoint symlink created at {symlink_path}")

        return checkpoint_path


    def load_model_checkpoint(self, model_class, checkpoint_path, device="cpu", start_fresh=False, create_new=True):
        """
        Loads a trained model checkpoint and restores model, optimizer, and training state.

        Args:
            model_class (type): The class of the model (e.g., OpalGPT).
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to load model on ('cpu' or 'cuda').
            start_fresh (bool): Whether to start training from scratch.

        Returns:
            model (torch.nn.Module): Loaded model with restored weights.
            optimizer_state_dict (dict): State dict for optimizer (can be used to resume training).
            epoch (int): Last epoch from checkpoint.
            train_losses (list): Training loss history.
            val_losses (list): Validation loss history.
            config (dict): Model configuration dictionary.
        """
        checkpoint = {}
        optimizer_state_dict= None
        scheduler_state_dict = None
        config = self.config

        if self.is_finetune and not os.path.isfile(os.path.realpath(checkpoint_path)):
            print("*** Checkpoint not found for finetuning. Please provide a valid checkpoint path")
            exit(1)
            

        if ((create_new == False) and (not os.path.isfile(os.path.realpath(checkpoint_path)))):
            raise ValueError("Checkpoint not found and create_new is False")
            
        if (not os.path.isfile(os.path.realpath(checkpoint_path))) or start_fresh:
            print(f"⚠️ Checkpoint {checkpoint_path} not found (or) start_fresh is requested. Creating new model.")
            model = model_class(self.config).to(device)
        else:
            print(f"✅ Model loaded from {checkpoint_path}")
            checkpoint = torch.load(os.path.realpath(checkpoint_path), map_location=device)
            
            # Display checkpoint training metrics
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            epoch = checkpoint.get("epoch", 0)
            
            if train_losses and val_losses:
                final_train_loss = train_losses[-1] if train_losses else "N/A"
                final_val_loss = val_losses[-1] if val_losses else "N/A"
                
                # Calculate perplexity from loss (perplexity = exp(loss))
                train_perplexity = math.exp(final_train_loss) if isinstance(final_train_loss, (int, float)) else "N/A"
                val_perplexity = math.exp(final_val_loss) if isinstance(final_val_loss, (int, float)) else "N/A"
                
                print(f"📊 Checkpoint epoch: {epoch}")
                print(f"📊 Final training loss: {final_train_loss:.6f}, perplexity: {train_perplexity:.2f}")
                print(f"📊 Final validation loss: {final_val_loss:.6f}, perplexity: {val_perplexity:.2f}")
            else:
                print("📊 No loss history found in checkpoint")
            
            # Load model with saved config to ensure same architecture
            config = checkpoint["config"]
            model = model_class(config).to(device)
            missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            
            # ✅ CRITICAL DEBUG: Check for vocab size mismatches in loaded model
            print(f"🔍 CHECKPOINT LOADING DEBUG:")
            print(f"   Missing keys: {len(missing)} - {missing[:5] if missing else 'None'}")
            print(f"   Unexpected keys: {len(unexpected)} - {unexpected[:5] if unexpected else 'None'}")
            
            # Check if embedding/output layers were properly loaded
            embedding_loaded = not any('token_embeddings' in key or 'token_emb' in key for key in missing)
            output_loaded = not any('out_head' in key or 'output' in key for key in missing)
            
            print(f"   Token embeddings loaded: {embedding_loaded}")
            print(f"   Output head loaded: {output_loaded}")
            
            if not embedding_loaded:
                print(f"🚨 CRITICAL: Token embeddings not loaded from checkpoint!")
                print(f"   This indicates vocab size mismatch between checkpoint and current config")
            if not output_loaded:
                print(f"🚨 CRITICAL: Output head not loaded from checkpoint!")
                print(f"   This indicates vocab size mismatch between checkpoint and current config")
                
            # Check actual model dimensions after loading
            actual_emb_size = model.token_embeddings.num_embeddings if hasattr(model, 'token_embeddings') else 'N/A'
            actual_out_size = model.out_head.out_features if hasattr(model, 'out_head') else 'N/A'
            config_vocab = self.config.get('vocab_size', 'N/A')
            
            print(f"   After loading - Embedding size: {actual_emb_size}")
            print(f"   After loading - Output size: {actual_out_size}")
            print(f"   Config vocab size: {config_vocab}")
            
            if actual_emb_size != config_vocab or actual_out_size != config_vocab:
                print(f"🚨 CONFIRMED VOCAB MISMATCH!")
                print(f"   This WILL cause CUDA index out of bounds errors!")
                print(f"   Solution: Train from scratch OR use matching checkpoint")
            print("❌ Missing keys:", missing)
            print("⚠️ Unexpected keys:", unexpected)
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
            scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
            model.to(device)
        print(model)
        return (
            model,
            optimizer_state_dict  if optimizer_state_dict else None,
            scheduler_state_dict if scheduler_state_dict else None,
            checkpoint["epoch"] if "epoch" in checkpoint else 0,
            checkpoint["train_losses"] if "train_losses" in checkpoint else [],
            checkpoint["val_losses"] if "val_losses" in checkpoint else [],
            config,
        )

    def _plot_and_save_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path):
        """
        Helper function to generate and save the loss plot.
        """
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add a second x-axis for tokens seen
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible for alignment
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory

    def plot_losses(self, epochs_seen, tokens_seen, train_losses, val_losses, checkpoint_path):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        
        #Get directory name from checkpoint path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Get just the filename without extension
        filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        plt.savefig(os.path.join(checkpoint_dir, f"{filename}-loss-plot.pdf"))
        plt.show()

    # Load instruction dataset from jsonl file. The format of the 
    # jsonl file should be {"prompt": "", "response": ""}
    # Ex: 
    # {"prompt": "", "response": ""}
    # {"prompt": "", "response": ""}
    # 
    def load_instruction_dataset(self, jsonl_file, max_length=768):
        all_ids = []
        # Load the jsonl file
        with open(jsonl_file, "r") as f:
            # for each line in the jsonl file, each line is 
            # {"prompt": "", "response": ""}
            for line in f:
                if self.check_new_tokens(line):
                    print("🚨 New tokens found in instruction dataset, please start training from scratch with start_fresh=True")
                    raise ValueError("New tokens found in instruction dataset, please start training from scratch with start_fresh=True")

                # Load the json line
                item = json.loads(line)
                # Tokenize the prompt and response
                prompt_ids = self.tokenizer.encode(item["prompt"], out_type=int)
                response_ids = self.tokenizer.encode(item["response"], out_type=int)
                # Combine the prompt and response
                combined = prompt_ids + response_ids
                # Truncate the combined ids to max_length
                combined = combined[:max_length]
                all_ids.extend(combined)
        return torch.tensor(all_ids, dtype=torch.long)

    def _evaluate_and_log_models(
        self,
        final_ckpt: str,
        val_loader,
        device: str,
        writer=None,
        log_to_wandb=False
    ):
        """
        Evaluates both the trained PyTorch model and its quantized ONNX version.
        Logs final metrics to console, TensorBoard, and Weights & Biases.

        Args:
            final_ckpt (str): Path to final PyTorch checkpoint (.pt file)
            val_loader (DataLoader): Validation DataLoader (already created in parent function)
            device (str): "cpu" or "cuda" or "mps"
            writer: TensorBoard SummaryWriter (optional)
            log_to_wandb (bool): Whether to log metrics to W&B
        """

        # 1. Evaluate PyTorch model
        final_loss, final_ppl = evaluate_pytorch(final_ckpt, val_loader, device)
        print(f"Final PyTorch Model Eval -> Loss={final_loss:.4f}, Perplexity={final_ppl:.4f}")

        # 2. Export to ONNX + Quantize
        onnx_path = final_ckpt.replace(".pt", ".onnx")
        quant_path = final_ckpt.replace(".pt", "_quantized.onnx")

        export_and_quantize_model(
            config=self.config,
            checkpoint_path=final_ckpt,
            onnx_output_path=onnx_path,
            quantized_output_path=quant_path,
            device=device
        )

        # 3. Evaluate Quantized ONNX model
        onnx_loss, onnx_ppl = evaluate_onnx(quant_path, val_loader, device)
        print(f"Final Quantized ONNX Eval -> Loss={onnx_loss:.4f}, Perplexity={onnx_ppl:.4f}")

        # 4. Log metrics to TensorBoard
        if writer:
            writer.add_scalar("Eval/Loss_PyTorch", final_loss)
            writer.add_scalar("Eval/Perplexity_PyTorch", final_ppl)
            writer.add_scalar("Eval/Loss_ONNX", onnx_loss)
            writer.add_scalar("Eval/Perplexity_ONNX", onnx_ppl)

        # 5. Log metrics to W&B
        if log_to_wandb:
            wandb.log({
                "final_loss_pytorch": final_loss,
                "final_ppl_pytorch": final_ppl,
                "final_loss_onnx": onnx_loss,
                "final_ppl_onnx": onnx_ppl
            })

        return {
            "pytorch_loss": final_loss,
            "pytorch_ppl": final_ppl,
            "onnx_loss": onnx_loss,
            "onnx_ppl": onnx_ppl,
            "onnx_model": onnx_path,
            "quant_model": quant_path
        }
    
    def check_new_tokens(self, text_or_list):
        """Check for unknown tokens in text or list of texts."""
        new_tokens_found = False
        
        # Handle both single text and list of texts
        texts = [text_or_list] if isinstance(text_or_list, str) else text_or_list
        
        for text in texts:
            # Process in chunks for large texts to avoid memory issues
            chunk_size = 10000  # Process 10k characters at a time
            if len(text) > chunk_size:
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    tokens = self.tokenizer.encode(chunk, out_type=str)
                    if "<unk>" in tokens:
                        print(f"⚠️ New tokens found in text chunk: {chunk[:100]}...")
                        new_tokens_found = True
                        break
            else:
                tokens = self.tokenizer.encode(text, out_type=str)
                if "<unk>" in tokens:
                    print(f"⚠️ New tokens found in text: {text[:100]}...")
                    new_tokens_found = True
        
        return new_tokens_found

    def lr_lambda(self, step):
        if step < self.config.get("warmup_steps", 0):
            return float(step) / float(max(1, self.config["warmup_steps"]))
        return 1.0

    # Train and save model, Main training loop function
    # to train the model from scratch or continue training
    # or fine-tune the model on a new dataset
    def train_and_save_model(
        self,
        model_class,
        config,
        device,
        tokenizer,
        checkpoint_path,
        corpus_text = None, # This is the pretokenized corpus text for pretraining
        num_epochs=10,
        batch_size=8,
        train_ratio=0.9,
        lr=4e-4,
        weight_decay=0.1,
        eval_freq=5,
        eval_iter=5,
        start_context="Help me find why 'show ip route' CLI fails",
        log_to_tensorboard=True,
        log_to_wandb=False,
        wandb_project="opal-training",
        start_fresh=False
    ):
        """
        Train, continue pretraining, or fine-tune a model.
        Includes:
        ✅ Fine-tuning support (JSONL loader)
        ✅ Warmup + CosineAnnealingLR scheduler
        ✅ Gradient accumulation + clipping
        """

        if corpus_text is None and not self.is_finetune:
            raise ValueError("corpus_text must be provided for pretraining")

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        torch.manual_seed(123)

        # Retrieve class-level flags for fine-tuning  #Finetune-Optional
        is_finetune = getattr(self, "is_finetune", False)
        finetune_data_path = getattr(self, "finetune_data_path", None)

        # ----------------------------------------
        # Logging Setup
        # ----------------------------------------
        writer = None
        if log_to_tensorboard:
            date_dir = datetime.now().strftime("%Y%m%d")
            run_dir = os.path.join(OpalConstants.TENSORBOARD_RUN_DIR, date_dir, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=run_dir)
            latest_symlink = os.path.join(OpalConstants.TENSORBOARD_RUN_DIR, "tensorboard_latest")
            if os.path.islink(latest_symlink):
                os.remove(latest_symlink)
            os.symlink(run_dir, latest_symlink)
            print(f"-- TensorBoard logging enabled at {run_dir}")

        if log_to_wandb:
            wandb.init(project=wandb_project, config=config)
            wandb.watch(model_class, log="all")

        # ----------------------------------------
        # Load Checkpoint if available
        # ----------------------------------------
        # During fine tune we must need the previous checkpoint
        if self.is_finetune and not os.path.exists(checkpoint_path):
            print(f"❌ Fine-tuning requires a checkpoint, but {checkpoint_path} not found!")
            return None

        try:
            print(f"Attempting to load model checkpoint from {checkpoint_path}...")
            model, optimizer_state_dict, scheduler_state_dict, epoch, train_losses, val_losses, _ = \
                self.load_model_checkpoint(model_class, checkpoint_path, device, start_fresh)
            print("✅ Successfully loaded model checkpoint!")
        except Exception as e:
            print(f"⚠️ No checkpoint found. Training from scratch: {e}")
            model = model_class(config).to(device)
            optimizer_state_dict, scheduler_state_dict = None, None
            train_losses, val_losses, epoch = [], [], 0

        # ----------------------------------------
        # Optimizer
        # ----------------------------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        # ----------------------------------------
        # Data Loading
        # ----------------------------------------
        if is_finetune:
            if not finetune_data_path:
                raise ValueError("finetune_data_path must be provided for fine-tuning")

            print("✅ Fine-tuning mode enabled, starting the finetune data pipeline")
            print(f"-- Fine-tuning with {finetune_data_path}")
            
            # 🔧 FIXED: Load data once and split PROPERLY (avoid data leakage)
            with open(finetune_data_path, "r", encoding="utf-8") as f:
                all_data = []
                for line_num, line in enumerate(f, 1):
                    try:
                        data_item = json.loads(line.strip())
                        # 🔧 Add validation for required fields
                        if "prompt" not in data_item or "response" not in data_item:
                            print(f"⚠ Skipping invalid JSONL line {line_num}: missing prompt/response")
                            continue
                        all_data.append(data_item)
                    except json.JSONDecodeError:
                        print(f"⚠ Skipping malformed JSONL line {line_num}: {line[:50]}...")
    
            if len(all_data) == 0:
                raise ValueError("No valid training data found in JSONL file")
                
            print(f"📊 Loaded {len(all_data)} valid samples from {finetune_data_path}")
            
            # 🔧 FIXED: Use deterministic split with shuffle to prevent data leakage
            random.seed(42)  # Deterministic split
            random.shuffle(all_data)  # Shuffle before split
            
            split_idx = int(train_ratio * len(all_data))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            print(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation samples")
            
            # 🔧 Ensure validation set is not too small
            if len(val_data) < 100:
                print(f"⚠️ WARNING: Validation set very small ({len(val_data)} samples). Consider larger dataset or different split ratio.")
    
            # Create temporary files for split data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as train_file:
                for item in train_data:
                    train_file.write(json.dumps(item) + '\n')
                train_file_path = train_file.name
    
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as val_file:
                for item in val_data:
                    val_file.write(json.dumps(item) + '\n')
                val_file_path = val_file.name
    
            # 🚨 CRITICAL FIX: Force safe settings for fine-tuning to prevent CUDA errors
            safe_num_workers = 0 if self.is_finetune else TRAINING_CONFIG["num_workers"]
            print(f"🔧 Using {safe_num_workers} workers for {'fine-tuning' if self.is_finetune else 'pretraining'}")
            
            training_loader = self.createOpalFinetuneDataLoader(
                data_jsonl=train_file_path,
                batch_size=batch_size,
                max_length=config["context_length"],
                shuffle=False,
                num_workers=safe_num_workers  # 🚨 Force 0 for fine-tuning
            )
            val_loader = self.createOpalFinetuneDataLoader(
                data_jsonl=val_file_path,
                batch_size=batch_size,
                max_length=config["context_length"],
                shuffle=False,
                num_workers=safe_num_workers  # 🚨 Force 0 for fine-tuning
            )
    
            # Clean up temporary files after use
            atexit.register(lambda: os.unlink(train_file_path) if os.path.exists(train_file_path) else None)
            atexit.register(lambda: os.unlink(val_file_path) if os.path.exists(val_file_path) else None)
        else:
            print("-- Pretraining with provided corpus_text")
            if not isinstance(corpus_text, torch.Tensor):
                raise ValueError(f"corpus_text must be a pre-tokenized torch.Tensor for pretraining, "
                               f"but got {type(corpus_text)}. Expected tensor from loadTrainingData().")
            
            print(f"📊 Pretraining corpus size: {len(corpus_text):,} tokens")
            
            # Handle tensor/tokenized data - deterministic split for reproducible training
            total_length = len(corpus_text)
            split_idx = int(train_ratio * total_length)
            train_data, val_data = corpus_text[:split_idx], corpus_text[split_idx:]
            
            print(f"📊 Data split: {len(train_data):,} train tokens, {len(val_data):,} val tokens")
            
            training_loader = self.createOpalDataLoader(
                txt=train_data,
                batch_size=batch_size,  # Use the passed batch_size parameter
                max_length=config["context_length"],
                stride=256,
                shuffle=False,  # Don't shuffle for deterministic training
                num_workers=TRAINING_CONFIG["num_workers"],
            )
            val_loader = self.createOpalDataLoader(
                txt=val_data,
                batch_size=batch_size,  # Use the passed batch_size parameter
                max_length=config["context_length"],
                stride=256,
                shuffle=False,  # Never shuffle validation data
                num_workers=TRAINING_CONFIG["num_workers"],
            )

        # ----------------------------------------
        # Scheduler with Warmup + CosineAnnealingLR  #Finetune-Optional
        # ----------------------------------------
        total_steps = num_epochs * len(training_loader)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        print("✅ Created learning rate scheduler")
        if scheduler_state_dict:
            cosine_scheduler.load_state_dict(scheduler_state_dict)

        # ----------------------------------------
        # Training Loop
        # ----------------------------------------
        global_step = 0
        model.train()
        train_losses, val_losses, tokens_seen = [], [], []

        print("✅ Starting training loop ^^^^^^^^^^^^ ")
        # for epoch_idx in range(num_epochs):
        #     print(f"Epoch {epoch_idx + 1}/{num_epochs}")
        #     for step, batch in enumerate(training_loader):
        #         input_ids, labels = batch
        #         model_output = model(input_ids, labels)
        #         loss = model_output["loss"] / config.get("gradient_accumulation_steps", 1)
        #         loss.backward()

        #         if (step + 1) % config.get("gradient_accumulation_steps", 1) == 0:
        #             torch.nn.utils.clip_grad_norm_(
        #                 model.parameters(), config.get("max_grad_norm", 1.0)
        #             )
        #             optimizer.step()
        #             optimizer.zero_grad()

        #             if global_step < config.get("warmup_steps", 0):
        #                 warmup_scheduler.step()
        #             else:
        #                 cosine_scheduler.step()

        #             global_step += 1

        #     # Validation step
        #     if (epoch_idx + 1) % eval_freq == 0:
        #         val_loss = self.evaluate_model(model, val_loader, device)
        #         val_losses.append(val_loss)

        print("✅ Starting training loop")
        train_losses, val_losses, track_tokens_seen = self.train_model_simple(
            model=model,
            train_loader=training_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=cosine_scheduler,
            device=device,
            num_epochs=num_epochs,
            eval_freq=eval_freq,
            eval_iter=eval_iter,
            start_context=start_context,
            tokenizer=tokenizer,
            writer=writer,
            log_to_wandb=log_to_wandb
        )
        # Save final checkpoint
        print("✅ Saving final checkpoint")
        # FINETUNE_PH2: Return training results with correct variable name
        return train_losses, val_losses, track_tokens_seen