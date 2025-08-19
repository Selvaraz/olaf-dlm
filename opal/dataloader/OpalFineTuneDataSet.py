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

            # Validate minimum length
            if len(prompt) < 5:
                print(f"⚠ Skipping too short prompt: {prompt[:50]}...")
                continue

            full_text = f"<BOS> {prompt} {response_text} <EOS>"
            #full_text = prompt.strip() + " " + response_text.strip()

            input_ids = self.tokenizer.encode(full_text, out_type=int)
            #print("OpalFineTuneDataset: Input token IDs min:", min(input_ids), "max:", max(input_ids))
            #print("UNK ID:", self.tokenizer.pad_id())

            # Mask prompt tokens so loss is applied only on response
            prompt_ids = self.tokenizer.encode(f"<BOS> {prompt}", out_type=int)
            # Truncate if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                
            prompt_len = min(len(prompt_ids), len(input_ids))

            # Apply the masks for the prompt tokens, so our DLM does not 
            # Learn about the prompt :-D
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            
            # Ensure labels are also truncated to match input_ids length
            if len(labels) > self.max_length:
                labels = labels[:self.max_length]

            samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

        print(f"📊 Fine-tune dataset prepared: {len(samples)} valid samples")
        if len(samples) > 0:
            avg_input_len = sum(len(s["input_ids"]) for s in samples) / len(samples)
            avg_label_len = sum((s["labels"] != -100).sum().item() for s in samples) / len(samples)
            print(f"   → Avg input length: {avg_input_len:.1f}, Avg response length: {avg_label_len:.1f}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]
