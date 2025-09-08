import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from opal.config.opal_config import TRAINING_CONFIG

class OpalFileDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        max_length: int = 512,
        stride: int = 256,
    ):
        """
        Args:
            file_paths: A list of paths to your pre-tokenized PyTorch tensor files (.pt).
            max_length: Max context window.
            stride: Overlap between consecutive chunks.
        """
        self.file_paths = file_paths
        self.max_length = max_length
        self.stride = stride
        self.device = TRAINING_CONFIG["device"]
        
        # Cache for loaded files to avoid repeated I/O
        self._file_cache: Dict[str, torch.Tensor] = {}
        self._cache_size_limit = 10  # Keep max 10 files in memory
        
        # We now create a mapping from chunk index to (file_idx, start_offset)
        # based on the lengths of the pre-tokenized files.
        self._file_chunk_indices = self._create_file_chunk_mapping()

    def _create_file_chunk_mapping(self) -> List[Tuple[int, int]]:
        """
        Creates a mapping of (file_idx, start_offset) for all chunks.
        This is a critical step for file-level shuffling.
        """
        all_chunk_mappings = []
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # Load the tensor to get its length without holding it in memory
                # We need to explicitly tell it to load to CPU to avoid issues.
                token_ids = torch.load(file_path, map_location=torch.device('cpu'))
                
                # Map all possible chunks from this file
                for i in range(0, len(token_ids) - self.max_length, self.stride):
                    all_chunk_mappings.append((file_idx, i))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        print(f"[OpalFileDataset] Found {len(all_chunk_mappings)} total chunks across {len(self.file_paths)} files.")
        return all_chunk_mappings

    def __len__(self):
        # The length of the dataset is now the total number of chunks we can generate.
        return len(self._file_chunk_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a specific chunk from a pre-tokenized file with caching.
        """
        # Get the mapping for the requested chunk index
        file_idx, start_offset = self._file_chunk_indices[idx]
        file_path = self.file_paths[file_idx]

        # Try to get from cache first
        if file_path in self._file_cache:
            token_ids = self._file_cache[file_path]
        else:
            # FIXED: Always load to CPU for DataLoader pin_memory compatibility
            token_ids = torch.load(file_path, map_location='cpu')
            
            # Add to cache with size limit
            if len(self._file_cache) >= self._cache_size_limit:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._file_cache))
                del self._file_cache[oldest_key]
            
            self._file_cache[file_path] = token_ids

        # Extract the specific chunk based on the pre-calculated offsets
        input_chunk = token_ids[start_offset : start_offset + self.max_length].clone().detach()
        target_chunk = token_ids[start_offset + 1 : start_offset + self.max_length + 1].clone().detach()

        return input_chunk, target_chunk
