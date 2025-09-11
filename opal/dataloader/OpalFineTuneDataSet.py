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

        # Check if special tokens exist in tokenizer vocabulary
        vocab_pieces = [self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.get_piece_size())]
        missing_tokens = []
        for token in ["<USER>", "<ASSISTANT>"]:
            if token not in vocab_pieces:
                missing_tokens.append(token)
        
        if missing_tokens:
            print(f"⚠ Warning: Special tokens {missing_tokens} are not in the tokenizer's vocabulary.")
            print("Please ensure your tokenizer has been trained with these tokens.")

        self.samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        for item in self.data:
            if "prompt" not in item or "response" not in item:
                raise ValueError(f"Invalid JSONL sample format: {item}")

            prompt = item["prompt"].strip()
            if isinstance(item["response"], str):
                response_text = item["response"].strip()
            else:
                # Convert entire response object to a compact JSON string
                response_text = json.dumps(item["response"], ensure_ascii=False).strip()

            if len(prompt) < 5:
                print(f"⚠ Skipping too short prompt: {prompt[:50]}...")
                continue
            
            if len(response_text) < 1:
                print(f"⚠ Skipping empty response for prompt: {prompt[:50]}...")
                continue
            
            # Build the complete conversation format
            full_text = f"<USER>{prompt}<ASSISTANT>{response_text}"
            prompt_text = f"<USER>{prompt}<ASSISTANT>"
            
            # Encode both sequences (tokenizer will handle BOS/EOS automatically)
            full_text_ids = self.tokenizer.encode(full_text, out_type=int)
            prompt_ids = self.tokenizer.encode(prompt_text, out_type=int)
            
            # Get special token IDs for potential manual addition if needed
            bos_id = self.tokenizer.bos_id() if hasattr(self.tokenizer, 'bos_id') and self.tokenizer.bos_id() >= 0 else None
            eos_id = self.tokenizer.eos_id() if hasattr(self.tokenizer, 'eos_id') and self.tokenizer.eos_id() >= 0 else None
            
            # Track the original prompt length before any modifications
            original_prompt_len = len(prompt_ids)
            
            # Check if we need to manually add BOS (if tokenizer doesn't add it automatically)
            if bos_id is not None and (not full_text_ids or full_text_ids[0] != bos_id):
                full_text_ids.insert(0, bos_id)
                original_prompt_len += 1  # Account for added BOS in prompt length
            
            # Check if we need to manually add EOS (if tokenizer doesn't add it automatically)  
            if eos_id is not None and (not full_text_ids or full_text_ids[-1] != eos_id):
                full_text_ids.append(eos_id)
            
            # Truncation: Keep sequences within max_length
            if len(full_text_ids) > self.max_length:
                full_text_ids = full_text_ids[:self.max_length]
            
            # Use the tracked prompt length for proper masking
            prompt_len = min(original_prompt_len, len(full_text_ids))  # Don't exceed actual sequence length
            
            # Create labels: mask prompt tokens (-100), keep response tokens
            labels = full_text_ids.copy()  # Start with all tokens
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100  # Mask prompt tokens
            
            # Validate that we have some response tokens to learn from
            response_token_count = (torch.tensor(labels) != -100).sum().item()
            if response_token_count < 1:
                print(f"⚠ Skipping sample with no response tokens after truncation: {prompt[:50]}...")
                continue
            
            # Convert to tensors - NO PADDING (handled by collate_fn)
            samples.append((
                torch.tensor(full_text_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long)
            ))

        print(f"📊 Fine-tune dataset prepared: {len(samples)} valid samples")
        if len(samples) > 0:
            avg_input_len = sum(len(s[0]) for s in samples) / len(samples)  # s[0] is input_ids
            avg_label_len = sum((s[1] != -100).sum().item() for s in samples) / len(samples)  # s[1] is labels
            print(f"   → Avg input length: {avg_input_len:.1f}, Avg response length: {avg_label_len:.1f}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return tuple (input_ids, labels) to match collate_fn expectations
        return self.samples[idx]
