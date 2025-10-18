import json
import random
import tempfile
import atexit
import glob
from datetime import datetime
import multiprocessing
import time
import os
import shutil
import psutil
import torch
import math
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MaxNLocator
from typing import List, Optional, Union
import sentencepiece as spm
from tqdm import tqdm
import torch.nn.functional as F
import traceback  # 🔧 CRITICAL: Add traceback for error handling
import wandb  # 🔧 CRITICAL: Add wandb import (even though commented out)

# Opal imports
from opal.dataloader.OpalFileDataSet import OpalFileDataset
from ..dataloader.OpalDataSet import OpalDataset
from ..dataloader.OpalFineTuneDataSet import OpalFinetuneDataset
from torch.utils.data import Dataset, DataLoader
from ..utils.opal_constants import OpalConstants
from ..export.export_onnx import export_and_quantize_model
from opal.config.opal_config import TRAINING_CONFIG, get_gpu_memory_allocated_size, get_scaler
from ..export.opal_evaluator import evaluate_pytorch, evaluate_onnx
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


class Opal:
    def __init__(self, config, tokenizer=None, 
                start_fresh=False, is_finetune=False,
                finetune_data_path=None,
                is_dapt=False):
        self.config = config
        self.tokenizer = tokenizer
        self.start_fresh = start_fresh
        self.is_finetune = is_finetune
        self.finetune_data_path = finetune_data_path
        self.is_dapt = is_dapt
    
    def collate_finetune(self, batch):
        from torch.nn.utils.rnn import pad_sequence
        cols = list(zip(*batch))
        if len(cols) == 3:
            inputs, labels, weights = cols
        else:
            inputs, labels = cols
            weights = [torch.tensor([1.0 if int(t) != -100 else 0.0 for t in lab], dtype=torch.float32) for lab in labels]

        pad_id = 0
        max_ctx = 512
        try:
            from ..config.opal_config import OPAL_MODEL_CONFIG as _OPAL_MODEL_CONFIG
            pad_id = int(_OPAL_MODEL_CONFIG.get("pad_id", 0))
            max_ctx = int(_OPAL_MODEL_CONFIG.get("context_length", 512))
        except Exception:
            pass

        padded_inputs  = pad_sequence(inputs,  batch_first=True, padding_value=pad_id)
        padded_labels  = pad_sequence(labels,  batch_first=True, padding_value=-100)
        padded_weights = pad_sequence(weights, batch_first=True, padding_value=0.0)

        if padded_inputs.size(1) > max_ctx:
            padded_inputs  = padded_inputs[:, :max_ctx]
            padded_labels  = padded_labels[:, :max_ctx]
            padded_weights = padded_weights[:, :max_ctx]

        return padded_inputs, padded_labels, padded_weights

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
        max_length: int = 512,
        shuffle: bool = True,
        drop_last: bool = False,
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
        # SFT: Verion_1.0 — robust batch sizing for tiny datasets
        dataset_len = len(data)
        if dataset_len == 0:
            raise RuntimeError(f"SFT dataset at {data_jsonl} is empty.")


        # Create dataset
        dataset = OpalFinetuneDataset(
            data=data,
            tokenizer=self.tokenizer,
            return_weights=True,  # Always return weights for fine-tuning
            max_length=max_length,
        )

        # Set batch size
        if batch_size is None:
            batch_size = TRAINING_CONFIG.get("batch_size", 4)

        
        # SFT: Verion_1.0 — avoid empty DataLoader when batch_size > dataset_len
        if batch_size is None:
            batch_size = TRAINING_CONFIG.get("batch_size", 4)
        if batch_size > len(dataset):
            print(f"⚠️ SFT: Adjusting batch_size {batch_size} → {len(dataset)} to avoid empty DataLoader when drop_last=True")
            batch_size = len(dataset)
            drop_last = False

        print(f"✅ Creating Fine-tune DataLoader → batch_size={batch_size}, shuffle={shuffle}, workers={num_workers}")

        # Standard DataLoader for CUDA/CPU
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
        
        # Memory optimization for very large datasets
        if isinstance(txt, torch.Tensor) and txt.numel() > 1e9:  # >1B tokens
            print(f"🔧 Large corpus detected ({txt.numel():,} tokens) - applying memory optimizations")
            # Reduce prefetch factor for very large datasets to save memory
            prefetch_factor = 2 if num_workers > 0 else None
            # Force persistent_workers=False for large datasets to prevent memory leaks
            persistent_workers_setting = False
            print(f"🔧 Adjusted settings: prefetch_factor={prefetch_factor}, persistent_workers={persistent_workers_setting}")
        else:
            prefetch_factor = 4 if num_workers > 0 else None
            persistent_workers_setting = self.config["persistent_workers"]

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
            persistent_workers=persistent_workers_setting,
            prefetch_factor=prefetch_factor
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

    # ----

    def generate(
    self,
    model,
    idx: torch.Tensor,                  # [B, T]
    max_new_tokens: int,
    context_size: int,                  # model’s max context length
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,         # (0,1]
    temperature: float = 1.0,           # 0 => greedy
    eos_id: Optional[int] = None,
    repetition_penalty: float = 1.0,    # multiplicative (GPT-2 style)
    # 🔽 Anti-repetition knobs (new)
    no_repeat_ngram_size: Optional[int] = 3,     # e.g., 3 to block tri-gram repeats
    presence_penalty: float = 0.0,            # additive: -beta if token seen in window
    frequency_penalty: float = 0.0,           # additive: -alpha * count in window
    penalty_window: int = 64,                  # window for presence/frequency penalties
    max_consecutive_repeats: int = 3,         # if the last token already occurs N times tailing, ban it
    ) -> torch.Tensor:
        """
        Decoding with top-k / top-p, temperature, improved repetition penalty,
        plus no-repeat n-gram, presence/frequency penalties, and max-consecutive guard.
        Uses scatter()/masked_fill() for top-p to avoid CUDA indexing asserts.
        
        LoRA Domain Adaptation: This method works seamlessly with LoRA models.
        LoRA adapters are automatically applied during the forward pass without
        requiring special handling in the generation logic.
        """
        # LoRA Domain Adaptation: Ensure model is in evaluation mode
        model.eval()
        device = idx.device
        B = idx.size(0)
        assert B >= 1
        
        # LoRA Domain Adaptation: Optional LoRA status logging for debugging
        if hasattr(model, 'is_lora_enabled') and model.is_lora_enabled():
            # Only log once per generation to avoid spam
            if not hasattr(self, '_lora_gen_logged'):
                self._lora_gen_logged = True
                try:
                    lora_info = model.get_lora_info()
                    print(f"🎯 LoRA Generation: {lora_info['total_lora_modules']} modules active")
                except:
                    print(f"🎯 LoRA Generation: LoRA adapters active")

        def _apply_no_repeat_ngram_block(logits_row: torch.Tensor, seq_row: torch.Tensor, n: int):
            """In-place: set logits of tokens that would create a repeated n-gram to -inf (B=1 fast path; loops are fine)."""
            if n is None or n <= 1 or seq_row.numel() < n - 1:
                return
            # Build map of (n-1)-gram -> set(next_token) from history
            history = seq_row.tolist()
            prefix_to_next = {}
            for i in range(len(history) - n + 1):
                prefix = tuple(history[i:i + n - 1])
                nxt = history[i + n - 1]
                s = prefix_to_next.get(prefix)
                if s is None:
                    s = set()
                    prefix_to_next[prefix] = s
                s.add(nxt)
            # Current prefix (last n-1)
            cur_prefix = tuple(history[-(n - 1):]) if n - 1 > 0 else tuple()
            if cur_prefix in prefix_to_next:
                bad_next = prefix_to_next[cur_prefix]
                for tok in bad_next:
                    if 0 <= tok < logits_row.numel():
                        logits_row[tok] = float("-inf")

        def _apply_presence_frequency_penalties(logits_row: torch.Tensor, recent_row: torch.Tensor):
            """Additive penalties (OpenAI-style): subtract alpha*count + beta*1{seen}."""
            if (presence_penalty <= 0.0) and (frequency_penalty <= 0.0):
                return
            # Count in a small recent window
            vals, counts = recent_row.unique(return_counts=True)
            # Only penalize valid ids
            V = logits_row.numel()
            m = (vals >= 0) & (vals < V)
            if m.any():
                vals = vals[m]
                counts = counts[m].to(logits_row.dtype)
                # logits[tok] -= frequency_penalty * count + presence_penalty * 1
                # Do it in vectorized chunks
                for tok, c in zip(vals.tolist(), counts.tolist()):
                    logits_row[tok] -= (frequency_penalty * c + presence_penalty * 1.0)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Trim to model context
                idx_cond = idx[:, -context_size:]

                model_output = model(idx_cond)
                logits = model_output["logits"] if isinstance(model_output, dict) else model_output
                logits = logits[:, -1, :]  # [B, V] next-token logits
                V = logits.size(-1)

                # === Improved repetition penalty (multiplicative, recent window 20)
                if repetition_penalty and repetition_penalty > 1.0:
                    window = min(20, idx_cond.size(1))
                    recent = idx_cond[:, -window:].clamp_(0, V - 1)
                    for b in range(B):
                        recent_b = recent[b]
                        token_positions = {}
                        for i_pos, tok in enumerate(recent_b.tolist()):
                            token_positions.setdefault(tok, []).append(i_pos)
                        for tok, positions in token_positions.items():
                            if len(positions) <= 1 or tok < 0 or tok >= V:
                                continue
                            freq_pen = repetition_penalty ** len(positions)
                            if positions[-1] >= window - 3:  # last-3 boost
                                freq_pen *= 1.5
                            if logits[b, tok] > 0:
                                logits[b, tok] = logits[b, tok] / freq_pen
                            else:
                                logits[b, tok] = logits[b, tok] * freq_pen

                # === Additive presence/frequency penalties on a larger window
                if (presence_penalty > 0.0) or (frequency_penalty > 0.0):
                    win = min(penalty_window, idx_cond.size(1))
                    recent_big = idx_cond[:, -win:].clamp_(0, V - 1)
                    for b in range(B):
                        _apply_presence_frequency_penalties(logits[b], recent_big[b])

                # === Max consecutive token guard
                if max_consecutive_repeats and max_consecutive_repeats > 0:
                    for b in range(B):
                        last = int(idx[b, -1].item())
                        # count tail run length of 'last'
                        run = 1
                        j = idx.size(1) - 2
                        while j >= 0 and int(idx[b, j].item()) == last:
                            run += 1
                            if run >= max_consecutive_repeats:
                                # Ban the last token to force diversity
                                if 0 <= last < V:
                                    logits[b, last] = float("-inf")
                                break
                            j -= 1

                # === Temperature
                if temperature is not None and temperature > 0.0:
                    logits = logits / temperature

                # === Top-k
                if top_k is not None and 0 < top_k < V:
                    topk_vals, _ = logits.topk(top_k, dim=-1)
                    kth = topk_vals[..., -1, None]
                    logits = torch.where(logits < kth, torch.full_like(logits, float('-inf')), logits)

                # === Top-p (nucleus) with scatter-back (batched-safe)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)  # [B,V]
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = probs.cumsum(dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float('-inf'))

                # === No-repeat n-gram blocking (applied *after* k/p filtering)
                if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                    # Use the same trimmed sequence we conditioned on
                    for b in range(B):
                        _apply_no_repeat_ngram_block(logits[b], idx_cond[b], no_repeat_ngram_size)

                # === Sample / Greedy
                if temperature is not None and temperature <= 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    probs = torch.softmax(logits, dim=-1)
                    nan_rows = torch.isnan(probs).any(dim=-1)
                    if nan_rows.any():
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    else:
                        next_token = torch.multinomial(probs, num_samples=1)

                # Append
                idx = torch.cat([idx, next_token], dim=1)

                # Early stop on EOS for all
                if eos_id is not None and torch.all(next_token.squeeze(-1) == eos_id):
                    break

        model.train()
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
        
        # LoRA Domain Adaptation: Detect LoRA based on config and model state
        config_has_lora = self.config.get('use_lora', False)
        model_has_lora = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
        has_lora = config_has_lora or model_has_lora
        
        # Debug logging for LoRA detection
        if self.is_dapt:
            print(f"🎯 LoRA DAPT Detection:")
            print(f"   config_lora={config_has_lora}")
            print(f"   model_lora={model_has_lora}") 
            print(f"   final_lora={has_lora}")
            
            # Verify LoRA parameters have gradients
            if has_lora:
                lora_params_with_grad = 0
                lora_params_without_grad = 0
                for name, param in model.named_parameters():
                    if 'lora_A' in name or 'lora_B' in name:
                        if param.requires_grad:
                            lora_params_with_grad += 1
                        else:
                            lora_params_without_grad += 1
                
                print(f"   LoRA params with grad: {lora_params_with_grad}")
                print(f"   LoRA params without grad: {lora_params_without_grad}")
                
                if lora_params_without_grad > 0:
                    print("❌ WARNING: Some LoRA parameters don't have gradients!")
        
        # Optional-LoRA-Finetune: Prioritize LoRA behavior when enabled
        if has_lora:
            default_accumulation = 2  # Moderate for LoRA
        elif self.is_finetune:
            default_accumulation = 1  # Lower for traditional fine-tuning stability
        else:
            default_accumulation = 4  # Standard for pretraining
            
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", default_accumulation)
        
        if has_lora:
            phase_desc = "LoRA fine-tuning" if self.is_finetune else "LoRA domain adaptation"
            print(f"🎯 {phase_desc}: Using gradient accumulation steps: {gradient_accumulation_steps}")

        # LoRA Domain Adaptation: Enhanced scaler handling
        if use_mixed_precision:
            scaler = get_scaler()
            print(f"🎯 Mixed precision enabled with scaler: {type(scaler)}")
        else:
            scaler = None
            print(f"🎯 Mixed precision disabled")

        # FINETUNE_PH2: Adaptive Warmup - both pretraining and fine-tuning benefit from warmup
        # Adjust total steps for gradient accumulation
        steps_per_epoch = max(1, len(train_loader) // max(1, gradient_accumulation_steps))  # SFT: Verion_1.0 guard
        total_steps = num_epochs * steps_per_epoch
        if self.is_finetune:
            # Fine-tuning: lighter warmup (2% of total steps or configured warmup_steps)
            #warmup_steps = min(self.config.get("warmup_steps", int(total_steps * 0.02)), int(total_steps * 0.1))
            warmup_steps = max(1, int(total_steps * 0.03))  # SFT: Verion_1.0
        else:
            # Pretraining: standard warmup (5% of total steps)
            # For very large corpora (e.g., 3B tokens, 600k+ steps), a 5% warmup (30k+ steps) may be excessive.
            # Typical recommendations for large-scale training are 1%–3% warmup.
            # You can use 0.01 (1%) or 0.02 (2%) for faster ramp-up.
            warmup_steps = max(1, int(total_steps * 0.01))  # Use 2% warmup for large datasets
        
        print(f"🚀 === TRAINING PIPELINE INITIALIZATION ===")
        print(f"📊 Mode: {'FINE-TUNING' if self.is_finetune else 'PRETRAINING'}")
        print(f"📊 Training setup: {total_steps:,} total steps, {warmup_steps:,} warmup steps")
        print(f"📊 Epochs: {num_epochs}, Batches per epoch: {len(train_loader):,}")  # SFT: Verion_1.0
        print(f"📊 Effective batches per epoch (with accumulation): {steps_per_epoch:,}")
        bs_est = (len(train_loader.dataset) // len(train_loader)) if len(train_loader) > 0 else len(train_loader.dataset)
        print(f"📊 Batch size (est): {bs_est}")  # SFT: Verion_1.0
        print(f"📊 Gradient accumulation steps: {gradient_accumulation_steps}")
        eff_bs = bs_est * max(1, gradient_accumulation_steps)
        print(f"📊 Effective batch size: {eff_bs}")  # SFT: Verion_1.0
        print(f"📊 Evaluation frequency: every {eval_freq} steps, {eval_iter} batches per eval")
        print(f"📊 Early stopping patience: {early_stopping_patience} epochs")
        print(f"📊 Mixed precision: {use_mixed_precision}")
        print(f"📊 Max gradient norm: {max_grad_norm}")
        print(f"📊 Device: {device}")
        print(f"📊 warmup_steps: {warmup_steps}")
        print(f"🚀 ==========================================")        # Create warmup scheduler
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
            accumulated_loss = 0.0

            print(f"\n🔄 === EPOCH {epoch+1}/{num_epochs} STARTING ===")
            print(f"📊 Best validation loss so far: {best_val_loss:.6f}")
            print(f"📊 Epochs without improvement: {epochs_no_improve}")
            
            # 🔍 DEBUG: Check what's actually in self.config
            if 'learning_rate' in self.config:
                print(f"🔍 DEBUG: self.config['learning_rate']: {self.config['learning_rate']}")

            # Create a progress bar for the training data
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            start_time = time.time()

            epoch_best_val_loss = best_val_loss  # Track best val loss for this epoch

            # Initialize gradient accumulation for this epoch
            optimizer.zero_grad(set_to_none=True)

            # LoRA Domain Adaptation: Verify optimizer has parameters at start of epoch
            if has_lora and epoch == 0:
                optimizer_param_count = sum(len(group['params']) for group in optimizer.param_groups)
                print(f"🎯 LoRA DAPT: Optimizer managing {optimizer_param_count} parameters at epoch start")

            # Training loop with gradient accumulation
            accumulated_loss = 0.0
            for batch_idx, (input_ids, targets, weights) in enumerate(pbar):
                # 🔍 LoRA DEBUG: Periodically check LoRA parameter state
                if has_lora and batch_idx % 100 == 0:  # Check every 100 batches
                    lora_param_with_grad = 0
                    lora_param_without_grad = 0
                    for name, param in model.named_parameters():
                        if 'lora_A' in name or 'lora_B' in name:
                            if param.requires_grad:
                                lora_param_with_grad += 1
                            else:
                                lora_param_without_grad += 1
                    
                    if lora_param_without_grad > 0:
                        print(f"🚨 WARNING: {lora_param_without_grad} LoRA parameters lost requires_grad at batch {batch_idx}!")
                        # Re-enable gradients for LoRA parameters
                        for name, param in model.named_parameters():
                            if ('lora_A' in name or 'lora_B' in name) and not param.requires_grad:
                                param.requires_grad = True
                                print(f"🔧 Re-enabled gradients for: {name}")
                
                # Move input and target tensors to the specified device
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if weights is not None:
                    weights = weights.to(device, non_blocking=True)

                # Calculate loss for this batch
                loss = self.calc_loss_batch(input_ids, targets, weights, model, device)
                
                # LoRA Domain Adaptation: Validate loss has gradients
                if has_lora and not loss.requires_grad:
                    print(f"🚨 CRITICAL: Loss does not require gradients at batch {batch_idx}!")
                    print(f"🚨 This indicates optimizer parameters are not in computation graph")
                    
                    # Check optimizer parameters
                    optimizer_param_ids = set()
                    for group in optimizer.param_groups:
                        for p in group['params']:
                            optimizer_param_ids.add(id(p))
                    
                    # Check which model parameters are in optimizer
                    model_params_in_opt = 0
                    model_params_not_in_opt = 0
                    for name, param in model.named_parameters():
                        if 'lora_A' in name or 'lora_B' in name:
                            if id(param) in optimizer_param_ids:
                                model_params_in_opt += 1
                            else:
                                model_params_not_in_opt += 1
                                print(f"   ❌ LoRA param NOT in optimizer: {name}")
                    
                    print(f"   LoRA params in optimizer: {model_params_in_opt}")
                    print(f"   LoRA params NOT in optimizer: {model_params_not_in_opt}")
                    
                    # Skip this batch to avoid crash
                    continue
                
                # Scale loss by gradient accumulation steps
                loss = loss / gradient_accumulation_steps
                accumulated_loss += loss.item()

                # Safety check for NaN or infinite loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"🚨 WARNING: Loss became {'NaN' if torch.isnan(loss) else 'infinite'} at step {global_step+1}!")
                    print(f"🚨 Skipping this batch and continuing training...")
                    continue

                # LoRA Domain Adaptation: Enhanced backpropagation with proper scaler handling
                if use_mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Only update weights every gradient_accumulation_steps
                is_accumulation_step = (batch_idx + 1) % gradient_accumulation_steps == 0
                is_last_batch = batch_idx == len(train_loader) - 1
                
                if is_accumulation_step or is_last_batch:
                    total_norm = 0.0
                    
                    # LoRA Domain Adaptation: Get optimizer parameters correctly
                    if has_lora:
                        optimizer_params = []
                        for group in optimizer.param_groups:
                            optimizer_params.extend(group['params'])
                    else:
                        optimizer_params = list(model.parameters())
                    
                    # LoRA Domain Adaptation: Enhanced gradient clipping and stepping
                    if use_mixed_precision and scaler is not None:
                        # Unscale gradients before clipping
                        scaler.unscale_(optimizer)
                        
                        # Calculate gradient norm
                        for p in optimizer_params:
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=max_grad_norm)
                        
                        # Step and update
                        scaler.step(optimizer)
                        scaler.update()
                        
                    else:
                        # Non-mixed precision
                        for p in optimizer_params:
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        # Clip and step
                        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=max_grad_norm)
                        optimizer.step()

                    # Zero gradients after weight update
                    optimizer.zero_grad(set_to_none=True)

                    # Update learning rate and global step
                    if global_step < warmup_steps:
                        warmup_scheduler.step()
                    elif global_step == warmup_steps:
                        current_lr = optimizer.param_groups[0]["lr"]
                        print(f"\n🔥 WARMUP COMPLETED at step {global_step+1}")
                        print(f"🔥 Learning rate: {current_lr:.2e}")
                        if scheduler:
                            scheduler.step()
                    else:
                        if scheduler:
                            scheduler.step()

                    global_step += 1
                    
                    # Generate sample frequency adjusted for training phase
                    # LoRA Domain Adaptation: Less frequent generation (every 9000 steps) for efficiency  
                    # Regular training: More frequent generation (every 5000 steps) for monitoring
                    sample_freq = 9000 if (has_lora and not self.is_finetune) else 5000
                    if global_step > 0 and global_step % sample_freq == 0:
                        print(f"\n🎯 === GENERATION SAMPLE AT STEP {global_step} (freq={sample_freq}) ===")
                        
                        # LoRA Domain Adaptation: Phase-specific generation logic with adaptive frequency
                        # Optional-LoRA-Finetune: Prioritize LoRA behavior when enabled
                        if has_lora and not self.is_finetune:
                            print("\n🎯 LoRA Domain Adaptation: Generating sample...")
                            self.generate_with_topk(
                                model, tokenizer, device, start_context, top_k=40
                            )
                        elif has_lora and self.is_finetune:
                            print("\n🎯 Optional-LoRA-Finetune: Generating sample...")
                            self.generate_with_topk(
                                model, tokenizer, device, start_context, top_k=35
                            )
                        elif self.is_finetune:
                            # Traditional fine-tuning without LoRA
                            self.generate_for_finetune(
                                model, tokenizer, device, start_context
                            )
                            # Every few epochs, test generation diversity
                            if (epoch + 1) % 1 == 0:  # Every epoch
                                self.improve_generation_diversity(
                                    model, tokenizer, device, start_context
                                )
                        else:
                            self.generate_with_topk(
                                model, tokenizer, device, start_context, top_k=50
                            )
                        print(f"\n🎯 ============================================\n")
                    
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

                # Update tokens seen
                tokens_seen += input_ids.numel()

                # Print a sample text after each epoch
                # self.generate_and_print_sample(
                #     model, tokenizer, device, start_context
                # )

                # Evaluation - only check on actual weight update steps
                if is_accumulation_step or is_last_batch:
                    # Adaptive evaluation frequency for pretraining vs fine-tuning
                    eval_frequency = eval_freq if not self.is_finetune else max(eval_freq, 10)
                    
                    # 🔧 H200 OPTIMIZATION: Adaptive eval_iter based on training progress
                    if self.is_dapt:
                        # Early training (first 20%): Use more batches for stability
                        # Later training: Use fewer batches for speed
                        progress = global_step / total_steps
                        if progress < 0.2:
                            adaptive_eval_iter = min(eval_iter, 1000)  # Max 1000 early
                        elif progress < 0.5:
                            adaptive_eval_iter = min(eval_iter, 500)   # 500 mid-training
                        else:
                            adaptive_eval_iter = min(eval_iter, 300)   # 300 late-training
                    else:
                        adaptive_eval_iter = eval_iter
                    
                    if global_step % eval_frequency == 0 and global_step > 0:
                        print(f"\n📊 === EVALUATION AT STEP {global_step} (using {adaptive_eval_iter} batches) ===")
                        train_loss, val_loss = self.evaluate_model(
                            model, train_loader, val_loader, device, adaptive_eval_iter)
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
                            print(f"\n🔥 New best val_loss {val_loss:.6f}! Saving checkpoint...")
                            # Try to get tokenizer model path
                            tokenizer_model_path = None
                            if hasattr(tokenizer, 'model_file') and tokenizer.model_file:
                                tokenizer_model_path = tokenizer.model_file
                            elif hasattr(tokenizer, 'model_path') and tokenizer.model_path:
                                tokenizer_model_path = tokenizer.model_path
                            else:
                                tokenizer_model_path = OpalConstants.TOKENIZER_MODEL_PATH
                            
                            self.save_model_checkpoint(
                                self.config, model, optimizer, scheduler,
                                epoch, train_losses, val_losses,
                                tokenizer_model_path
                            )
                        else:
                            print(f"\n⚠️ No improvement (current: {val_loss:.6f}, best: {best_val_loss:.6f})")

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

            # 🔧 FORCED EVALUATION: Ensure evaluation happens at least once per epoch
            print(f"\n📊 === FORCED END-OF-EPOCH EVALUATION ===")
            train_loss, val_loss = self.evaluate_model(
                model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            
            # Update best validation loss if improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"\n🎉 NEW BEST VALIDATION LOSS: {best_val_loss:.6f}")
                
                # Save checkpoint when validation improves
                # Try to get tokenizer model path from various sources
                tokenizer_model_path = None
                if hasattr(tokenizer, 'model_file') and tokenizer.model_file:
                    tokenizer_model_path = tokenizer.model_file
                elif hasattr(tokenizer, 'model_path') and tokenizer.model_path:
                    tokenizer_model_path = tokenizer.model_path
                elif hasattr(self.tokenizer, 'model_file') and self.tokenizer.model_file:
                    tokenizer_model_path = self.tokenizer.model_file
                
                checkpoint_path = self.save_model_checkpoint(
                    self.config, model, optimizer, scheduler, epoch, 
                    train_losses, val_losses, tokenizer_model_path)
                print(f"\n💾 Checkpoint saved: {checkpoint_path}")

            # ✅ After each epoch, check if val_loss improved in this epoch
            epoch_duration = time.time() - epoch_start_time
            
            if best_val_loss < epoch_best_val_loss:
                epochs_no_improve = 0
                improvement_msg = f"✅ Validation loss improved this epoch!"
            else:
                epochs_no_improve += 1
                improvement_msg = f"⚠️ No improvement for {epochs_no_improve} epochs"
            
            print(f"\n🏁 === EPOCH {epoch+1}/{num_epochs} COMPLETED ===")
            print(f"\n⏱️ Epoch duration: {epoch_duration:.2f} seconds")
            print(f"\n📊 {improvement_msg}")
            print(f"\n📊 Current best validation loss: {best_val_loss:.6f}")

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
                if (epoch + 1) % 1 == 0:  # Every epoch
                    self.improve_generation_diversity(
                        model, tokenizer, device, start_context
                    )
            else:
                self.generate_with_topk(
                    model, tokenizer, device, start_context, top_k=50
                )

        print(f"\n🎉 === TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"\n🎉 All {num_epochs} epochs completed!")
        print(f"\n🎉 Final best validation loss: {best_val_loss:.6f}")
        print(f"\n🎉 Total training steps: {global_step+1:,}")
        print(f"\n🎉 Total tokens processed: {tokens_seen:,}")
        print(f"\n🎉 ========================================")

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
        # LoRA Domain Adaptation: Ensure proper model state for generation
        model.eval()
        
        # LoRA Domain Adaptation: Check and log LoRA status for debugging
        is_lora_enabled = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
        if is_lora_enabled:
            print(f"🎯 LoRA Generation: Using LoRA-enabled model")
            # LoRA Domain Adaptation: Ensure LoRA adapters are in correct state
            try:
                lora_info = model.get_lora_info()
                print(f"   LoRA modules: {lora_info['total_lora_modules']}, params: {lora_info['total_lora_parameters']:,}")
            except Exception as e:
                print(f"   ⚠️ Could not get LoRA info: {e}")
        
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        
        with torch.no_grad():
            # LoRA Domain Adaptation: Use passed top_k parameter instead of hardcoded value
            token_ids = self.generate(model=model, idx=encoded, 
                                      context_size=context_size, 
                                      top_k=top_k,  # LoRA Domain Adaptation: Use dynamic top_k
                                      top_p=0.85,  # Reduced nucleus sampling for more focus
                                      temperature=0.7,  # Lower temperature for less randomness
                                      max_new_tokens=30,  # Shorter outputs to prevent repetition
                                      eos_id=tokenizer.eos_id(),
                                      repetition_penalty=3.0)  # Higher repetition penalty
            decoded_text = self.token_ids_to_text(token_ids)
            print("==========================================")
            print(decoded_text.replace("\n", " "))  # Compact print format
            print("==========================================")
        
        # LoRA Domain Adaptation: Ensure model returns to training state
        model.train()

    def generate_for_finetune(self, model, tokenizer, device, start_context):
        """
        Specialized generation method for fine-tuning with more conservative settings
        to avoid repetitive outputs.
        """
        # LoRA Domain Adaptation: Ensure proper model state for generation
        model.eval()
        
        # LoRA Domain Adaptation: Check LoRA status for fine-tuning
        is_lora_enabled = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
        if is_lora_enabled:
            print("🎯 LoRA Fine-tuning Generation: Using LoRA-enabled model")
        
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        
        with torch.no_grad():
            # LoRA Domain Adaptation: Adjust parameters for LoRA fine-tuning
            if is_lora_enabled:
                # LoRA fine-tuning: More conservative settings for stability
                top_k = 25
                temperature = 0.6
                repetition_penalty = 3.5
            else:
                # Traditional fine-tuning settings
                top_k = 30
                temperature = 0.8
                repetition_penalty = 3.0
                
            token_ids = self.generate(model=model, idx=encoded, 
                                      context_size=context_size, 
                                      top_k=top_k,  # LoRA Domain Adaptation: Dynamic based on LoRA status
                                      top_p=0.85,  # Slightly more conservative nucleus sampling
                                      temperature=temperature,  # LoRA Domain Adaptation: Dynamic temperature
                                      max_new_tokens=40,  # Slightly fewer tokens
                                      eos_id=tokenizer.eos_id(),
                                      repetition_penalty=repetition_penalty)  # LoRA Domain Adaptation: Dynamic penalty
            decoded_text = self.token_ids_to_text(token_ids)
            print("\n")
            print("========== FINE-TUNE GENERATION ==========")
            print(decoded_text.replace("\n", " "))  # Compact print format
            print("===========================================")
            print("\n")
        
        # LoRA Domain Adaptation: Ensure model returns to training state
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
            {"top_k": 50, "top_p": 0.92, "temp": 0.8, "rep_penalty": 2.0, "name": "Custom"},
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
    
    
    def calc_loss_batch(self, input_batch, target_batch, weights, model, device):
        """
        # SFT: Verion_1.0
        Compute a WEIGHTED cross-entropy loss for a single batch.

        • Uses ignore_index=-100 to mask prompt tokens.
        • Multiplies token losses by `weights` (e.g., CODE up-weighting).
        • Normalizes by the number of valid (non-masked) tokens to keep scale stable.
        """
        # Move inputs
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        weights = weights.to(device) if weights is not None else None

        # Forward pass
        out = model(input_batch, labels=None)  # we will compute loss manually for weights
        logits = out["logits"] if isinstance(out, dict) else out[0] if isinstance(out, (tuple, list)) else out

        # Shapes
        B, T, V = logits.size()
        logits = logits.view(B*T, V)
        targets = target_batch.view(B*T)

        # Build base per-token loss (no reduction)
        ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_tok_loss = ce(logits, targets)  # shape [B*T]

        # Apply weights to non-masked region
        if weights is None:
            # If no weights provided, default weight=1.0 on non-masked tokens
            valid = (targets != -100).float()
            loss = (per_tok_loss * valid).sum() / (valid.sum().clamp_min(1.0))
        else:
            w = weights.view(B*T).to(per_tok_loss.dtype)
            # Zero weights where labels are masked
            valid = (targets != -100).float()
            w = w * valid
            loss = (per_tok_loss * w).sum() / (w.sum().clamp_min(1.0))

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
        for i, batch in enumerate(data_loader):
            if i < num_batches:
                # Handle both 2-value and 3-value returns from dataset
                if len(batch) == 3:
                    input_batch, target_batch, weights = batch
                    # Use weights in evaluation for consistent metric calculation
                    loss = self.calc_loss_batch(input_batch, target_batch, weights, model, device)
                elif len(batch) == 2:
                    input_batch, target_batch = batch
                    loss = self.calc_loss_batch(input_batch, target_batch, None, model, device)
                else:
                    raise ValueError(f"Unexpected batch format: expected 2 or 3 values, got {len(batch)}")
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
        epoch, training history, and config. For LoRA models, saves both unified
        and adapter-only checkpoints.

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
        # LoRA Domain Adaptation: Check if model has LoRA adapters
        config_has_lora = config.get('use_lora', False)
        model_has_lora = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
        has_lora = config_has_lora or model_has_lora
        
        # Validate LoRA state before saving
        if has_lora:
            if not hasattr(model, 'lora_config'):
                print("❌ LoRA CRITICAL: Model missing lora_config attribute!")
                raise RuntimeError("LoRA Domain Adaptation: Cannot save - model missing lora_config")
            
            if not hasattr(model, 'get_lora_info'):
                print("❌ LoRA CRITICAL: Model missing get_lora_info method!")
                raise RuntimeError("LoRA Domain Adaptation: Cannot save - model missing LoRA methods")
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config,
            "tokenizer_model": tokenizer_model,
            "has_lora": has_lora,
        }

        # Create checkpoint directory
        date_dir = datetime.now().strftime("%Y%m%d")
        if not self.is_finetune:
            checkpoint_dir = os.path.join(OpalConstants.CHECKPOINT_DIR, date_dir, timestamp)
        else:
            checkpoint_dir = os.path.join(OpalConstants.CHECKPOINT_DIR, "_finetune_", date_dir, timestamp)
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"opal_gpt_checkpoint_{timestamp}.pt")
        
        # LoRA Domain Adaptation: Handle LoRA-specific checkpointing
        if has_lora:
            print("🎯 LoRA Domain Adaptation: Saving LoRA checkpoints...")
            
            # Create subdirectories
            base_dir = os.path.join(checkpoint_dir, "base")
            lora_dir = os.path.join(checkpoint_dir, "lora") 
            merged_dir = os.path.join(checkpoint_dir, "merged")
            os.makedirs(base_dir, exist_ok=True)
            os.makedirs(lora_dir, exist_ok=True)
            os.makedirs(merged_dir, exist_ok=True)
            
            # Save base model checkpoint
            base_checkpoint_path = os.path.join(base_dir, f"opal_gpt_base_{timestamp}.pt")
            torch.save(checkpoint, base_checkpoint_path)
            print(f"🎯 LoRA: Saved base checkpoint: {base_checkpoint_path}")
            
            # Get LoRA info and config
            lora_config = model.lora_config
            lora_info = model.get_lora_info()
            
            print(f"🎯 LoRA Model Info:")
            print(f"   Total LoRA modules: {lora_info['total_lora_modules']}")
            print(f"   Total LoRA parameters: {lora_info['total_lora_parameters']:,}")
            print(f"   LoRA percentage: {lora_info['lora_percentage']:.2f}%")
            
            base_model_info = {
                "checkpoint_path": base_checkpoint_path,
                "timestamp": timestamp,
                "epoch": epoch,
                "vocab_size": config.get("vocab_size", 12000),
                "emb_dim": config.get("emb_dim", 512),
                "n_layers": config.get("n_layers", 12),
            }
            
            # Save LoRA adapter weights
            from ..attention.lora_utils import save_lora_adapters
            adapter_path = os.path.join(lora_dir, f"lora_adapter_{timestamp}")
            
            try:
                lora_manifest = save_lora_adapters(
                    model=model,
                    save_path=adapter_path,
                    lora_config=lora_config,
                    base_model_info=base_model_info,
                    format=lora_config.checkpoint_format
                )
                print(f"🎯 LoRA: Saved adapters: {adapter_path}")
                print(f"🎯 LoRA Manifest: {lora_manifest['total_adapters']} adapters, {lora_manifest['total_parameters']:,} params")
            except Exception as e:
                print(f"❌ Failed to save LoRA adapters: {e}")
                import traceback
                traceback.print_exc()
            
            # Create and save merged model if configured
            if lora_config.merge_on_finalize:
                print("🎯 LoRA: Creating merged model checkpoint...")
                
                from ..attention.lora_utils import merge_lora_weights
                
                # LoRA Domain Adaptation: merge_lora_weights now handles the copy internally
                # No need to copy here - let the utility function handle it defensively
                merged_model = merge_lora_weights(model, verbose=True)
                
                # 🔧 CRITICAL FIX: Create checkpoint dict for merged model
                merged_checkpoint_dict = {
                    "model_state_dict": merged_model.state_dict(),  # <-- Use merged model's state!
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "config": config,
                    "tokenizer_model": tokenizer_model,
                    "has_lora": False,  # Merged model has no active LoRA
                    "is_merged": True,  # Flag to indicate this is a merged checkpoint
                }
                
                # Save merged checkpoint
                merged_checkpoint_path = os.path.join(merged_dir, f"opal_gpt_merged_{timestamp}.pt")
                torch.save(merged_checkpoint_dict, merged_checkpoint_path)  # <-- Save merged dict!
                print(f"🎯 LoRA: Saved merged checkpoint: {merged_checkpoint_path}")
                
                # Note: No need to clean up model_copy since merge_lora_weights manages its own copy
            
            checkpoint_path = base_checkpoint_path
            
        else:
            # Standard checkpointing
            torch.save(checkpoint, checkpoint_path)

        # Copy tokenizer model
        if tokenizer_model and os.path.exists(tokenizer_model):
            tokenizer_model_path = os.path.join(checkpoint_dir, "opal_tokenizer.model")
            try:
                shutil.copyfile(tokenizer_model, tokenizer_model_path)
                print(f"✅ Tokenizer model copied to checkpoint directory")
            except Exception as e:
                print(f"⚠️ Warning: Could not copy tokenizer model: {e}")

        # Create symlink
        if not self.is_finetune:
            symlink_path = os.path.join(OpalConstants.CHECKPOINT_DIR, "checkpoint-latest.pt")
        else:
            symlink_path = os.path.join(OpalConstants.CHECKPOINT_DIR, "finetune-latest.pt")

        # Enhanced symlink handling
        if os.path.exists(symlink_path) or os.path.islink(symlink_path):
            try:
                if os.path.islink(symlink_path):
                    os.unlink(symlink_path)
                elif os.path.isfile(symlink_path):
                    os.remove(symlink_path)
                elif os.path.isdir(symlink_path):
                    shutil.rmtree(symlink_path)
                import time
                time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Could not remove existing symlink: {e}")
        
        # Create new symlink with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if has_lora:
                    os.symlink(checkpoint_dir, symlink_path)
                else:
                    os.symlink(checkpoint_path, symlink_path)
                print(f"✅ Symlink created: {symlink_path}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.2)
                else:
                    print(f"❌ Symlink creation failed: {e}")

        if has_lora:
            print(f"🎯 LoRA: All checkpoints saved in: {checkpoint_dir}")
            return checkpoint_dir
        else:
            return checkpoint_path

    def load_model_checkpoint(self, model_class, checkpoint_path, device="cpu", start_fresh=False, create_new=True):
        """
        Loads a trained model checkpoint and restores model, optimizer, and training state.
        
        LoRA Domain Adaptation: Enhanced to properly handle loading pretrained models
        and injecting LoRA adapters for continued training.
        """
        checkpoint = {}
        optimizer_state_dict= None
        scheduler_state_dict = None
        config = self.config

        # LoRA Domain Adaptation: Validate checkpoint requirements
        if self.is_finetune and not os.path.isfile(os.path.realpath(checkpoint_path)):
            print("❌ Checkpoint not found for finetuning. Please provide a valid checkpoint path")
            exit(1)
            
        if self.is_dapt and not os.path.isfile(os.path.realpath(checkpoint_path)):
            print("❌ Checkpoint not found for DAPT. Please provide a valid pretrained checkpoint path")
            exit(1)

        if ((create_new == False) and (not os.path.isfile(os.path.realpath(checkpoint_path)))):
            raise ValueError("Checkpoint not found and create_new is False")
            
        if (not os.path.isfile(os.path.realpath(checkpoint_path))) or start_fresh:
            print(f"⚠️ Checkpoint {checkpoint_path} not found (or) start_fresh is requested. Creating new model.")
            model = model_class(self.config).to(device)
            model_config = self.config
        else:
            print(f"✅ Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(os.path.realpath(checkpoint_path), map_location=device)
            
            # Display checkpoint training metrics
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            epoch = checkpoint.get("epoch", 0)
            
            if train_losses and val_losses:
                final_train_loss = train_losses[-1] if train_losses else "N/A"
                final_val_loss = val_losses[-1] if val_losses else "N/A"
                train_perplexity = math.exp(final_train_loss) if isinstance(final_train_loss, (int, float)) else "N/A"
                val_perplexity = math.exp(final_val_loss) if isinstance(final_val_loss, (int, float)) else "N/A"
                
                print(f"📊 Checkpoint epoch: {epoch}")
                print(f"📊 Final training loss: {final_train_loss:.6f}, perplexity: {train_perplexity:.2f}")
                print(f"📊 Final validation loss: {final_val_loss:.6f}, perplexity: {val_perplexity:.2f}")
            else:
                print("📊 No loss history found in checkpoint")
            
            checkpoint_config = checkpoint["config"]
            
            # LoRA Domain Adaptation: For DAPT, create model WITH LoRA using current config
            if self.is_dapt:
                print("🎯 LoRA DAPT: Creating model with LoRA adapters for domain adaptation")
                
                # Start with checkpoint config for architecture
                model_config = checkpoint_config.copy()
                
                # Override with LoRA settings from current config
                lora_keys = ['use_lora', 'lora_rank', 'lora_alpha', 'lora_dropout', 
                            'target_modules', 'lora_include_mlp']
                for key in lora_keys:
                    if key in self.config:
                        model_config[key] = self.config[key]
                
                # Map alternative names
                if 'lora_rank' in model_config:
                    model_config['rank'] = model_config['lora_rank']
                if 'lora_alpha' in model_config:
                    model_config['alpha'] = model_config['lora_alpha']
                if 'lora_dropout' in model_config:
                    model_config['dropout'] = model_config['lora_dropout']
                
                print(f"🎯 LoRA DAPT: Model config:")
                print(f"   use_lora: {model_config.get('use_lora', False)}")
                print(f"   rank: {model_config.get('rank', 'N/A')}")
                print(f"   alpha: {model_config.get('alpha', 'N/A')}")
                print(f"   dropout: {model_config.get('dropout', 'N/A')}")
                
                # Create model with LoRA - this will trigger LoRA injection in __init__
                print(f"🎯 LoRA DAPT: Creating model (LoRA injection will happen in __init__)...")
                model = model_class(model_config).to(device)
                
                # LoRA Domain Adaptation: CRITICAL - Save LoRA module references BEFORE loading weights
                from ..attention.lora import LoRAInjectedLinear
                lora_modules_before = {}
                for name, module in model.named_modules():
                    if isinstance(module, LoRAInjectedLinear):
                        lora_modules_before[name] = id(module)
                
                print(f"✅ LoRA modules BEFORE weight loading: {len(lora_modules_before)}")
                
                # LoRA Domain Adaptation: Verify LoRA was actually injected
                lora_enabled = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
                if not lora_enabled:
                    print("❌ CRITICAL: LoRA injection failed in model __init__!")
                    raise RuntimeError("LoRA injection failed - model creation did not enable LoRA")
                
                print(f"✅ LoRA injection verified: model.is_lora_enabled() = {lora_enabled}")
                
                # LoRA Domain Adaptation: Verify LoRA modules exist
                lora_module_count = sum(1 for m in model.modules() if isinstance(m, LoRAInjectedLinear))
                print(f"✅ Found {lora_module_count} LoRAInjectedLinear modules in model")
                
                if lora_module_count == 0:
                    print("❌ CRITICAL: No LoRAInjectedLinear modules found!")
                    raise RuntimeError("LoRA injection verification failed - no LoRA modules in model")
                
                # Now load pretrained weights into the LoRA model
                print("🎯 LoRA DAPT: Loading pretrained weights into LoRA model...")
                checkpoint_state = checkpoint["model_state_dict"]
                
                # Get current model state to see what we're working with
                model_state = model.state_dict()
                
                # 🔍 CRITICAL DEBUG: Show what keys we're working with
                print("\n🔍 CRITICAL DEBUG: Checkpoint vs Model Keys")
                
                # Sample checkpoint keys
                ckpt_keys = list(checkpoint_state.keys())
                print(f"\n📋 Sample checkpoint keys ({len(ckpt_keys)} total):")
                attention_keys = [k for k in ckpt_keys if any(x in k for x in ['Wq', 'Wk', 'Wv', 'out_proj', 'attention'])]
                for key in attention_keys[:10]:  # Show first 10 attention keys
                    print(f"   {key}")
                
                # Sample model keys
                model_keys = list(model_state.keys())
                print(f"\n📋 Sample model keys ({len(model_keys)} total):")
                lora_keys = [k for k in model_keys if 'lora' in k.lower() or any(x in k for x in ['Wq', 'Wk', 'Wv', 'out_proj'])]
                for key in lora_keys[:10]:  # Show first 10 LoRA/attention keys
                    print(f"   {key}")
                
                # Map checkpoint keys to model keys
                transferred_weights = {}
                mapping_stats = {'direct': 0, 'mapped': 0, 'unmapped': 0}
                
                for ckpt_key, ckpt_value in checkpoint_state.items():
                    matched = False
                    
                    if ckpt_key in model_state:
                        # Direct match (embeddings, layer norms, etc.)
                        transferred_weights[ckpt_key] = ckpt_value
                        mapping_stats['direct'] += 1
                        matched = True
                    else:
                        # Try to map attention layers to base_linear
                        # Pattern 1: transformers_block.X.mhAttention.Wq.weight -> transformers_block.X.mhAttention.Wq.base_linear.weight
                        if any(attn_key in ckpt_key for attn_key in ['Wq.weight', 'Wk.weight', 'Wv.weight', 'out_proj.weight',
                                                                       'Wq.bias', 'Wk.bias', 'Wv.bias', 'out_proj.bias']):
                            # Replace .weight or .bias with .base_linear.weight/bias
                            if '.weight' in ckpt_key:
                                lora_key = ckpt_key.replace('.weight', '.base_linear.weight')
                            elif '.bias' in ckpt_key:
                                lora_key = ckpt_key.replace('.bias', '.base_linear.bias')
                            else:
                                lora_key = None
                            
                            if lora_key and lora_key in model_state:
                                transferred_weights[lora_key] = ckpt_value
                                mapping_stats['mapped'] += 1
                                matched = True
                                if mapping_stats['mapped'] <= 5:  # Show first 5 mappings
                                    print(f"✅ Mapped: {ckpt_key} -> {lora_key}")
                        
                        # Pattern 2: Try MLP layers if lora_include_mlp is True
                        if not matched and model_config.get('lora_include_mlp', False):
                            if any(mlp_key in ckpt_key for mlp_key in ['w1.weight', 'w2.weight', 'w3.weight',
                                                                        'w1.bias', 'w2.bias', 'w3.bias']):
                                if '.weight' in ckpt_key:
                                    lora_key = ckpt_key.replace('.weight', '.base_linear.weight')
                                elif '.bias' in ckpt_key:
                                    lora_key = ckpt_key.replace('.bias', '.base_linear.bias')
                                else:
                                    lora_key = None
                                
                                if lora_key and lora_key in model_state:
                                    transferred_weights[lora_key] = ckpt_value
                                    mapping_stats['mapped'] += 1
                                    matched = True
                    
                    if not matched:
                        # Keep original key for non-attention layers
                        transferred_weights[ckpt_key] = ckpt_value
                        mapping_stats['unmapped'] += 1
                
                print(f"\n📊 Weight Mapping Statistics:")
                print(f"   Direct matches: {mapping_stats['direct']}")
                print(f"   Mapped (attention->base_linear): {mapping_stats['mapped']}")
                print(f"   Unmapped (kept original): {mapping_stats['unmapped']}")
                print(f"   Total transferred: {len(transferred_weights)}")
                
                # Load weights with strict=False to allow LoRA parameters to be missing
                missing_keys, unexpected_keys = model.load_state_dict(transferred_weights, strict=False)
                
                print(f"\n🎯 LoRA DAPT: Weight loading summary:")
                print(f"   Transferred: {len(transferred_weights)} weights")
                print(f"   Missing: {len(missing_keys)} keys")
                print(f"   Unexpected: {len(unexpected_keys)} keys")
                
                # 🔍 CRITICAL: Show which attention weights were NOT loaded
                missing_attention = [k for k in missing_keys if 'base_linear' in k]
                if missing_attention:
                    print(f"\n🚨 WARNING: Missing base_linear weights for attention layers:")
                    for key in missing_attention[:10]:
                        print(f"   ❌ {key}")
                
                # CRITICAL: Verify LoRA modules STILL EXIST after weight loading
                lora_modules_after = {}
                for name, module in model.named_modules():
                    if isinstance(module, LoRAInjectedLinear):
                        lora_modules_after[name] = id(module)
                
                print(f"✅ LoRA modules AFTER weight loading: {len(lora_modules_after)}")
                
                # Check if module instances changed (this would be BAD)
                if len(lora_modules_before) != len(lora_modules_after):
                    print("❌ CRITICAL: Number of LoRA modules changed after weight loading!")
                    print(f"   Before: {len(lora_modules_before)}, After: {len(lora_modules_after)}")
                    raise RuntimeError("LoRA modules were replaced during weight loading!")
                
                modules_changed = []
                for name in lora_modules_before:
                    if name not in lora_modules_after:
                        modules_changed.append(name)
                    elif lora_modules_before[name] != lora_modules_after[name]:
                        modules_changed.append(f"{name} (ID changed)")
                
                if modules_changed:
                    print("❌ CRITICAL: LoRA module instances changed during weight loading!")
                    for name in modules_changed[:5]:  # Show first 5
                        print(f"   Changed: {name}")
                    raise RuntimeError("LoRA module instances were replaced during weight loading!")
                
                # Verify LoRA parameters are in missing keys (they should be since they're new)
                lora_missing = [k for k in missing_keys if 'lora_A' in k or 'lora_B' in k]
                print(f"   LoRA adapters in missing: {len(lora_missing)} (expected - they're new parameters)")
                
                # CRITICAL: Verify base_linear weights were loaded
                base_linear_loaded = any('base_linear' in k for k in transferred_weights.keys())
                print(f"   Base linear weights loaded: {base_linear_loaded}")
                
                if not base_linear_loaded:
                    print("⚠️ WARNING: No base_linear weights found in transferred weights!")
                
                # Final verification: Test forward pass to ensure gradients flow
                print("🔍 Testing gradient flow through LoRA modules...")
                model.train()
                test_input = torch.randint(0, model_config['vocab_size'], (1, 10)).to(device)
                test_output = model(test_input)
                test_loss = test_output['logits'].sum()
                
                # Check if loss requires grad
                if not test_loss.requires_grad:
                    print("❌ CRITICAL: Test loss does not require gradients!")
                    print("❌ This means LoRA parameters are not in the computation graph")
                    raise RuntimeError("LoRA parameters not in computation graph after loading")
                
                # Try backward to verify gradients
                test_loss.backward()
                
                # Check if LoRA parameters have gradients
                lora_params_with_grad = 0
                for name, param in model.named_parameters():
                    if ('lora_A' in name or 'lora_B' in name) and param.grad is not None:
                        lora_params_with_grad += 1
                
                print(f"✅ Test backward pass successful: {lora_params_with_grad} LoRA params have gradients")
                
                if lora_params_with_grad == 0:
                    print("❌ CRITICAL: No LoRA parameters received gradients in test!")
                    raise RuntimeError("LoRA parameters not receiving gradients")
                
                # Zero gradients after test
                model.zero_grad()
                
                # Final verification: Check that LoRA modules have non-zero base weights
                sample_verified = False
                for name, module in model.named_modules():
                    if isinstance(module, LoRAInjectedLinear):
                        base_weight_norm = module.base_linear.weight.norm().item()
                        lora_a_norm = module.lora_A.norm().item()
                        lora_b_norm = module.lora_B.norm().item()
                        # CRITICAL: Add LoRA contribution check
                        lora_contribution = (module.lora_B @ module.lora_A).norm().item() * module.scaling
                        
                        print(f"🔍 Sample LoRA module '{name}':")
                        print(f"   base_linear.weight norm: {base_weight_norm:.4f}")
                        print(f"   lora_A norm: {lora_a_norm:.4f}")
                        print(f"   lora_B norm: {lora_b_norm:.4f}")
                        print(f"   LoRA contribution (B@A*scaling): {lora_contribution:.6f}")  # <-- Should be ~0.0
                        print(f"   Ratio (LoRA/Base): {lora_contribution/base_weight_norm:.8f}")  # <-- Should be tiny
                        print(f"   requires_grad: base={module.base_linear.weight.requires_grad}, A={module.lora_A.requires_grad}, B={module.lora_B.requires_grad}")
                        
                        if base_weight_norm > 0:
                            sample_verified = True
                        break
                
                if not sample_verified:
                    print("❌ CRITICAL: Base linear weights appear to be zero!")
                    raise RuntimeError("Base linear weights not properly loaded")
            
            else:
                # Standard loading for fine-tuning or continued pretraining
                model_config = checkpoint_config
                model = model_class(model_config).to(device)
                missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print(f"Standard loading - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # Don't load optimizer/scheduler state for DAPT - use fresh state
            if not self.is_dapt:
                optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
                scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
            else:
                print("🎯 LoRA DAPT: Using fresh optimizer and scheduler state")
                optimizer_state_dict = None
                scheduler_state_dict = None
            
            model.to(device)
            
        print(model)
        return (
            model,
            optimizer_state_dict if optimizer_state_dict else None,
            scheduler_state_dict if scheduler_state_dict else None,
            checkpoint.get("epoch", 0) if checkpoint else 0,
            checkpoint.get("train_losses", []) if checkpoint else [],
            checkpoint.get("val_losses", []) if checkpoint else [],
            model_config,
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
    def train_and_save_model(
        self,
        model_class,
        config,
        device,
        tokenizer,
        checkpoint_path,
        corpus_text = None,
        num_epochs=10,
        batch_size=8,
        train_ratio=0.9,
        lr=4e-4,
        weight_decay=0.1,
        eval_freq=5,
        eval_iter=5,
        start_context='The Spanning Tree Protocol (STP) is primarily used for the purpose of ',
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
        # LoRA Domain Adaptation: Must load checkpoint for DAPT
        if (self.is_finetune or self.is_dapt) and not os.path.exists(checkpoint_path):
            print(f"❌ {'Fine-tuning' if self.is_finetune else 'DAPT'} requires checkpoint: {checkpoint_path} not found!")
            return None

        try:
            print(f"Loading model checkpoint from {checkpoint_path}...")
            model, optimizer_state_dict, scheduler_state_dict, epoch, train_losses, val_losses, _ = \
                self.load_model_checkpoint(model_class, checkpoint_path, device, start_fresh)
            print("✅ Successfully loaded model checkpoint!")
            
            # LoRA Domain Adaptation: Ensure LoRA adapters are on correct device
            if self.is_dapt or (self.is_finetune and config.get('use_lora', False)):
                from ..attention.lora_utils import ensure_lora_device_consistency
                ensure_lora_device_consistency(model, torch.device(device), verbose=True)
                
        except Exception as e:
            print(f"⚠️ Checkpoint loading failed: {e}")
            import traceback
            traceback.print_exc()
            
            if self.is_dapt:
                print("❌ DAPT requires valid checkpoint - cannot continue")
                return None
                
            print("⚠️ Creating new model for training from scratch")
            model = model_class(self.config).to(device)
            optimizer_state_dict, scheduler_state_dict = None, None
            train_losses, val_losses, epoch = [], [], 0

        # ----------------------------------------
        # Optimizer Setup
        # ----------------------------------------
        print(f"Creating optimizer with learning rate: {lr}")
        optimizer = None

        # LoRA Domain Adaptation: Detect LoRA state
        config_has_lora = config.get('use_lora', False)
        model_has_lora = hasattr(model, 'is_lora_enabled') and model.is_lora_enabled()
        has_lora = config_has_lora or model_has_lora
        
        print(f"🎯 LoRA Detection:")
        print(f"   Config use_lora: {config_has_lora}")
        print(f"   Model has LoRA: {model_has_lora}")
        print(f"   Final LoRA enabled: {has_lora}")
        
        if has_lora:
            print("🎯 LoRA: Creating optimizer for LoRA parameters only")
            
            # Get LoRA parameters
            lora_params = model.get_lora_parameters()
            
            if not lora_params:
                print("❌ LoRA CRITICAL: No LoRA parameters found!")
                print("❌ Model state:")
                print(f"   has is_lora_enabled: {hasattr(model, 'is_lora_enabled')}")
                if hasattr(model, 'is_lora_enabled'):
                    print(f"   is_lora_enabled(): {model.is_lora_enabled()}")
                print(f"   has lora_config: {hasattr(model, 'lora_config')}")
                print(f"   has get_lora_parameters: {hasattr(model, 'get_lora_parameters')}")
                
                # Check for LoRA modules
                lora_module_names = [name for name, module in model.named_modules() 
                                    if 'LoRA' in str(type(module))]
                print(f"   LoRA modules found: {len(lora_module_names)}")
                if lora_module_names:
                    print(f"   LoRA module names: {lora_module_names[:5]}")
                
                raise RuntimeError("LoRA: Cannot create optimizer - no LoRA parameters")
            
            print(f"🎯 LoRA: Found {len(lora_params)} LoRA parameter tensors")
            total_lora_params = sum(p.numel() for p in lora_params)
            print(f"🎯 LoRA: Total parameters: {total_lora_params:,}")
            
            # Validate gradients
            params_with_grad = sum(1 for p in lora_params if p.requires_grad)
            print(f"🎯 LoRA: Parameters with gradients: {params_with_grad}/{len(lora_params)}")
            
            if params_with_grad == 0:
                print("❌ LoRA CRITICAL: No LoRA parameters require gradients!")
                raise RuntimeError("LoRA: No trainable parameters found")
            
            # Create optimizer
            adamw_kwargs = dict(betas=(0.9, 0.95), lr=lr, weight_decay=weight_decay, eps=1e-8)
            try:
                optimizer = torch.optim.AdamW(lora_params, fused=True, **adamw_kwargs)
                print(f"🎯 LoRA: Created fused AdamW optimizer")
            except TypeError:
                optimizer = torch.optim.AdamW(lora_params, **adamw_kwargs)
                print(f"🎯 LoRA: Created standard AdamW optimizer")
            
            # Verify optimizer has parameters
            optimizer_param_count = sum(len(group['params']) for group in optimizer.param_groups)
            print(f"🎯 LoRA: Optimizer managing {optimizer_param_count} parameter groups")
            
        elif self.is_finetune:
            # Standard fine-tuning optimizer
            decay, no_decay = set(), set()
            param_dict = {n: p for n, p in model.named_parameters()}
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.endswith(".weight") and p.ndim > 1:
                    decay.add(name)
                else:
                    no_decay.add(name) 
                    
            optim_groups = [
                {"params": [param_dict[n] for n in sorted(decay)], "weight_decay": 0.1},
                {"params": [param_dict[n] for n in sorted(no_decay)], "weight_decay": 0.0},
            ]
            adamw_kwargs = dict(betas=(0.9, 0.95), lr=lr, weight_decay=0.0, eps=1e-8)
            try:
                optimizer = torch.optim.AdamW(optim_groups, fused=True, **adamw_kwargs)
            except TypeError:
                optimizer = torch.optim.AdamW(optim_groups, **adamw_kwargs)
            print("✅ Fine-tuning optimizer with weight decay")
        else:
            # Standard pretraining optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            print("✅ Pretraining optimizer created")
        
        # LoRA Domain Adaptation: Skip loading optimizer state for DAPT
        if optimizer_state_dict and not self.is_finetune and not self.is_dapt:
            print("✅ Loading optimizer state (pretraining)")
            optimizer.load_state_dict(optimizer_state_dict)
        elif self.is_dapt:
            print("🎯 LoRA DAPT: Using fresh optimizer state")
        elif self.is_finetune:
            print("🔧 Fine-tuning: Using fresh optimizer state")
        else:
            print("✅ No optimizer state to load")

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
            
            # 🔧 FIXED: Ensure minimum validation size for small datasets
            min_val_samples = 10  # Minimum validation samples
            if len(all_data) < min_val_samples * 2:  # Too small for proper split
                print(f"⚠️ WARNING: Dataset too small ({len(all_data)} samples). Using 20% for validation.")
                split_idx = max(1, int(train_ratio * len(all_data)))
            else:
                split_idx = int(train_ratio * len(all_data))
                # Ensure validation set has at least min_val_samples
                if (len(all_data) - split_idx) < min_val_samples:
                    split_idx = len(all_data) - min_val_samples
            
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]

            print(f"📊 Data split: {len(train_data)} train, {len(val_data)} validation samples")
            
            # 🔧 Enhanced validation set size check
            if len(val_data) < 5:
                print(f"❌ CRITICAL: Validation set too small ({len(val_data)} samples). Need at least 5 samples for reliable evaluation.")
                raise ValueError("Validation set too small for reliable evaluation")
    
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
            safe_num_workers = TRAINING_CONFIG["num_workers"]
            print(f"🔧 Using {safe_num_workers} workers for {'fine-tuning' if self.is_finetune else 'pretraining'}")
            
            training_loader = self.createOpalFinetuneDataLoader(
                data_jsonl=train_file_path,
                batch_size=batch_size,
                max_length=config["context_length"],
                shuffle=True,
                num_workers=safe_num_workers  # 🚨 Force 0 for fine-tuning
            )
            # SFT: Verion_1.0 — create val loader (may be None/empty for tiny sets)
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
        # Scheduler
        # ----------------------------------------
        total_steps = num_epochs * len(training_loader)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        print("✅ Created learning rate scheduler")
        
        # LoRA Domain Adaptation: Skip loading scheduler state for DAPT
        if scheduler_state_dict and not is_finetune and not self.is_dapt:
            print("✅ Loading scheduler state (pretraining)")
            cosine_scheduler.load_state_dict(scheduler_state_dict)
        elif self.is_dapt:
            print("🎯 LoRA DAPT: Using fresh scheduler state")
        elif is_finetune:
            print("🔧 Fine-tuning: Using fresh scheduler state")
        else:
            print("✅ No scheduler state to load")

        # ----------------------------------------
        # Training Loop
        # ----------------------------------------
        global_step = 0
        model.train()
        train_losses, val_losses, tokens_seen = [], [], []

        print("✅ Starting training loop ^^^^^^^^^^^^ ")
        
        # LoRA Domain Adaptation: CRITICAL - Verify LoRA is actually being used in forward pass
        if has_lora:
            print("🔍 LoRA DAPT: CRITICAL pre-training verification...")
            model.train()  # Ensure we're in training mode
            
            # Create a test batch
            test_input = torch.randint(0, config['vocab_size'], (2, 10)).to(device)
            
            # Forward pass
            test_output = model(test_input)
            test_loss = test_output['logits'].sum()
            
            print(f"   Test loss requires_grad: {test_loss.requires_grad}")
            
            if not test_loss.requires_grad:
                print("❌ CRITICAL: Forward pass test FAILED - loss has no gradients!")
                print("❌ This means LoRA modules are NOT in the computation graph")
                
                # Debug: Check which modules are actually being called
                print("🔍 Checking module call stack...")
                
                # Check if LoRA modules exist
                from ..attention.lora import LoRAInjectedLinear
                lora_module_count = 0
                for name, module in model.named_modules():
                    if isinstance(module, LoRAInjectedLinear):
                        lora_module_count += 1
                        print(f"   Found LoRA module: {name}")
                        print(f"     base_linear.weight.requires_grad: {module.base_linear.weight.requires_grad}")
                        print(f"     lora_A.requires_grad: {module.lora_A.requires_grad}")
                        print(f"     lora_B.requires_grad: {module.lora_B.requires_grad}")
                        break  # Just show first one
                
                print(f"   Total LoRA modules: {lora_module_count}")
                
                # Check model structure - are attention layers using LoRA?
                print("🔍 Checking transformer block structure...")
                if len(model.transformers_block) > 0:
                    first_block = model.transformers_block[0]
                    print(f"   First block type: {type(first_block)}")
                    if hasattr(first_block, 'mhAttention'):
                        print(f"   Has mhAttention: {type(first_block.mhAttention)}")
                        attn = first_block.mhAttention
                        if hasattr(attn, 'Wq'):
                            print(f"   Wq type: {type(attn.Wq)}")
                            print(f"   Wq is LoRAInjectedLinear: {isinstance(attn.Wq, LoRAInjectedLinear)}")
                
                raise RuntimeError("LoRA DAPT: Forward pass does not use LoRA modules!")
            
            # Try backward
            test_loss.backward()
            
            # Check if LoRA params got gradients
            lora_grads = 0
            for name, param in model.named_parameters():
                if ('lora_A' in name or 'lora_B' in name) and param.grad is not None:
                    lora_grads += 1
            
            print(f"   LoRA params with gradients after backward: {lora_grads}")
            
            if lora_grads == 0:
                print("❌ CRITICAL: No LoRA parameters received gradients!")
                raise RuntimeError("LoRA parameters not receiving gradients in training mode")
            
            # Clear test gradients
            model.zero_grad()
            print("✅ LoRA pre-training verification PASSED")
        
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
        
        # Get tokenizer model path from various sources
        tokenizer_model_path = None
        if hasattr(tokenizer, 'model_file') and tokenizer.model_file:
            tokenizer_model_path = tokenizer.model_file
        elif hasattr(tokenizer, 'model_path') and tokenizer.model_path:
            tokenizer_model_path = tokenizer.model_path
        elif hasattr(self.tokenizer, 'model_file') and self.tokenizer.model_file:
            tokenizer_model_path = self.tokenizer.model_file
        
        # Save the final checkpoint with proper scheduler reference
        final_checkpoint_path = self.save_model_checkpoint(
            self.config, model, optimizer, cosine_scheduler, num_epochs, 
            train_losses, val_losses, tokenizer_model_path
        )
        print(f"✅ Final checkpoint saved: {final_checkpoint_path}")
        
        # FINETUNE_PH2: Return training results
        return train_losses, val_losses, track_tokens_seen