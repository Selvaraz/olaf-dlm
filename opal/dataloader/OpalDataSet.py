from opal.config.opal_config import TRAINING_CONFIG
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Union

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

        # Prepare token chunks (handles both raw text and token IDs)
        self.input_ids, self.target_ids = self._prepare_data(txt)

    def _prepare_data(self, txt: Union[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prepares input and target token chunks.

        If txt is a string → tokenizes using self.tokenizer.
        If txt is a torch.Tensor → assumes already tokenized.
        """
        if isinstance(txt, torch.Tensor):
            token_ids = txt.tolist()
            print(f"[OpalDataset] Using pre-tokenized token IDs (length={len(token_ids)})")
        elif isinstance(txt, str):
            assert self.tokenizer is not None, "Tokenizer must be provided when input is raw text"
            token_ids = self.tokenizer.encode(txt, out_type=int)
            print(f"[OpalDataset] Tokenized raw text into {len(token_ids)} tokens")
        else:
            raise ValueError("txt must be either a raw text string or a torch.Tensor of token IDs")

        input_chunks = []
        target_chunks = []

        print(f"[OpalDataset] Generating chunks with max_length={self.max_length}, stride={self.stride}...")

        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i : i + self.max_length]
            target_chunk = token_ids[i + 1 : i + self.max_length + 1]

            input_chunks.append(torch.tensor(input_chunk, dtype=torch.long))
            target_chunks.append(torch.tensor(target_chunk, dtype=torch.long))

        return input_chunks, target_chunks

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.target_ids[idx]
        )
