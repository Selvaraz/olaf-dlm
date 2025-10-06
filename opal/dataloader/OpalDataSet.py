from opal.config.opal_config import TRAINING_CONFIG
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union
import math

class OpalDataset(Dataset):
    def __init__(
        self,
        txt: Union[str, torch.Tensor],
        tokenizer=None,
        max_length: int = 1280,
        stride: int = 256,
        device: str = None
    ):
        """
        Memory-efficient dataset for large corpora using lazy chunk generation.
        
        Args:
            txt: Raw input text (str) OR pre-tokenized IDs (torch.Tensor)
            tokenizer: SentencePieceProcessor instance (needed only if txt is str)
            max_length: Max context window (default 1280 for your model)
            stride: Overlap between consecutive chunks
            device: Device to move tensors to ('cpu' or 'cuda' or 'mps')
        """

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        # Important: keep dataset tensors on CPU. DataLoader(pin_memory=True)
        # only works with CPU tensors; we move to CUDA/MPS in the training loop.
        self.device = TRAINING_CONFIG["device"]

        # Store token IDs for lazy access instead of pre-generating all chunks
        self.token_ids = self._prepare_token_ids(txt)
        
        # Calculate total number of chunks without generating them
        self.num_chunks = max(0, (len(self.token_ids) - self.max_length) // self.stride + 1)
        
        print(f"[OpalDataset] Initialized lazy dataset with {self.num_chunks:,} potential chunks")
        print(f"[OpalDataset] Memory-efficient: chunks generated on-demand during training")

    def _prepare_token_ids(self, txt: Union[str, torch.Tensor]) -> torch.Tensor:
        """
        Prepares token IDs from input text or tensor, storing them for lazy access.
        
        If txt is a string → tokenizes using self.tokenizer.
        If txt is a torch.Tensor → uses directly.
        """
        if isinstance(txt, torch.Tensor):
            print(f"[OpalDataset] Using pre-tokenized token IDs (length={len(txt):,})")
            # Store as tensor for efficient slicing
            return txt if txt.dtype == torch.long else txt.long()
        elif isinstance(txt, str):
            assert self.tokenizer is not None, "Tokenizer must be provided when input is raw text"
            token_ids = self.tokenizer.encode(txt, out_type=int)
            print(f"[OpalDataset] Tokenized raw text into {len(token_ids):,} tokens")
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            raise ValueError("txt must be either a raw text string or a torch.Tensor of token IDs")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        """
        Lazily generate chunk on demand instead of pre-computing all chunks.
        This dramatically reduces memory usage for large corpora.
        """
        if idx >= self.num_chunks:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_chunks} chunks")
        
        # Calculate start position for this chunk
        start_idx = idx * self.stride
        
        # Extract input and target chunks on-demand
        input_chunk = self.token_ids[start_idx : start_idx + self.max_length]
        target_chunk = self.token_ids[start_idx + 1 : start_idx + self.max_length + 1]
        
        # Create weight mask (all tokens are valid for pretraining)
        weights = torch.ones_like(target_chunk, dtype=torch.float32)
        
        return input_chunk, target_chunk, weights
