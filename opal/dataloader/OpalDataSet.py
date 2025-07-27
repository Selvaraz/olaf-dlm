import torch
from torch.utils.data import Dataset
from typing import List, Tuple

class OpalDataset(Dataset):
    def __init__(self, txt: str, tokenizer, max_length: int = 1280, stride: int = 256, device: str = "cpu"):
        """
        Args:
            txt: Raw input text (str)
            tokenizer: SentencePieceProcessor instance (e.g., spm.SentencePieceProcessor)
            max_length: Max context window (default 1280 for your model)
            stride: Overlap between consecutive chunks
            device: Device to move tensors to ('cpu' or 'cuda')
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.device = device

        # Prepare token chunks
        self.input_ids, self.target_ids = self._prepare_data(txt)

    def _prepare_data(self, txt: str) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        token_ids = self.tokenizer.encode(txt, out_type=int)
        input_chunks = []
        target_chunks = []

        print(f"[OpalDataset] Total tokens: {len(token_ids)}")
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
            self.input_ids[idx].to(self.device),
            self.target_ids[idx].to(self.device)
        )
