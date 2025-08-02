import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict
from ..config.opal_config import TRAINING_CONFIG, OPAL_MODEL_CONFIG

class OpalFinetuneDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        """
        Args:
            data: List of dicts with keys {"prompt", "response"}
            tokenizer: SentencePieceProcessor instance
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = OPAL_MODEL_CONFIG["context_length"]
        self.device = TRAINING_CONFIG["device"]

        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        for item in self.data:
            if "prompt" not in item or "response" not in item:
                raise ValueError(f"Invalid JSONL sample format: {item}")

            prompt = item["prompt"].strip()
            # Convert entire response object to a compact JSON string
            response_text = json.dumps(item["response"], ensure_ascii=False).strip()

            full_text = f"<BOS> {prompt} {response_text} <EOS>"
            #full_text = prompt.strip() + " " + response_text.strip()

            input_ids = self.tokenizer.encode(full_text, out_type=int)
            #print("OpalFineTuneDataset: Input token IDs min:", min(input_ids), "max:", max(input_ids))
            #print("UNK ID:", self.tokenizer.pad_id())

            # Truncate if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]

            # Mask prompt tokens so loss is applied only on response
            prompt_ids = self.tokenizer.encode(prompt, out_type=int)
            prompt_len = min(len(prompt_ids), len(input_ids))

            # Apply the masks for the prompt tokens, so our DLM does not 
            # Learn about the prompt :-D
            labels = [-100] * prompt_len + input_ids[prompt_len:]

            samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]
