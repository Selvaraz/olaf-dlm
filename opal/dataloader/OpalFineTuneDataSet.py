import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict, Tuple
from ..config.opal_config import TRAINING_CONFIG, OPAL_MODEL_CONFIG

class OpalFinetuneDataset(Dataset):
    """
    Builds training samples from JSONL-style items { "prompt": str, "response": {...} } without
    changing your dataset format. We only wrap the example in textual tags so the model learns
    to map <USER> prompt -> <ASSISTANT> JSON response.

    Returns (input_ids, labels, weights):
      - input_ids: token IDs (with BOS/EOS if configured)
      - labels: next-token targets, masked (-100) over the user/prompt region
      - weights: per-token float weights; 1.0 for assistant tokens, boosted (e.g., x3)
                 inside any "commands": [...] JSON arrays, 0.0 for user/pad tokens
    """
    def __init__(self, data: List[Dict], tokenizer):
        """
        Args:
            data: list of dicts, each like: {"prompt": str, "response": {...}}
            tokenizer: SentencePieceProcessor (with bos_id/eos_id/pad_id configured)
        """
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer

        self.max_length: int = int(OPAL_MODEL_CONFIG.get("context_length", 512))

        # Tag tokens (keep consistent train <-> infer)
        self.user_tag_open = "<QUESTION>"
        self.user_tag_close = "</QUESTION>"
        self.asst_tag = "<ASSISTANT>"

        # Special IDs (prefer tokenizer methods if present)
        self.pad_id = self._safe_token_id("pad_id", default=OPAL_MODEL_CONFIG.get("pad_id", 0))
        self.bos_id = self._safe_token_id("bos_id", default=OPAL_MODEL_CONFIG.get("bos_id", 1))
        self.eos_id = self._safe_token_id("eos_id", default=OPAL_MODEL_CONFIG.get("eos_id", 2))

        # How strongly we upweight command tokens
        self.COMMANDS_WEIGHT = float(TRAINING_CONFIG.get("commands_weight", 3.0))

        self.samples = self._build_samples()

    # ------------------------------ utils ------------------------------

    def _safe_token_id(self, name: str, default: int) -> int:
        """Get special token id from tokenizer if exposed as a callable; otherwise fallback."""
        tid = default
        try:
            maybe = getattr(self.tokenizer, name, None)
            tid = maybe() if callable(maybe) else default
            if tid is None:
                tid = default
        except Exception:
            tid = default
        return int(tid)

    def _wrap_texts(self, prompt: str, response_obj: Dict) -> Tuple[str, str, str]:
        """
        Create the three segments:
          user_text: "<QUESTION> {prompt} </QUESTION>\n"
          asst_head: "<ASSISTANT> "
          resp_text: compact JSON string: '{"response": {...}}'
        """
        user_text = f"{self.user_tag_open} {prompt} {self.user_tag_close}\n"
        asst_head = f"{self.asst_tag} "
        # Compact JSON reduces token bloat, keep deterministic separators
        resp_text = json.dumps({"response": response_obj}, separators=(",", ":"))
        return user_text, asst_head, resp_text

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize a string to IDs; we manage BOS/EOS ourselves outside of this method."""
        return self.tokenizer.encode(text, out_type=int)

    # ----------- commands span detection (character spans in resp_text) -----------

    def _find_commands_char_spans(self, resp_text: str) -> List[Tuple[int, int]]:
        """
        Return a list of (start_char, end_char) covering each '"commands":[ ... ]' array
        (including the key). Uses bracket matching and string-state to be robust to quotes.
        """
        spans: List[Tuple[int, int]] = []
        key = '"commands":['
        i = 0
        n = len(resp_text)
        while True:
            j = resp_text.find(key, i)
            if j < 0:
                break
            arr_start = j + len(key)  # first char inside array
            # Scan forward to matching closing bracket of this array
            depth = 1
            k = arr_start
            in_str = False
            esc = False
            while k < n and depth > 0:
                ch = resp_text[k]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\\\':
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                k += 1
            spans.append((j, k))  # include key through the closing bracket
            i = k
        return spans

    def _charpos_to_token_index_lookup(self, text: str) -> List[int]:
        """
        Build a list where offset[i] = number of characters in decode(ids[:i]).
        We'll use it to map a character position to a token index by binary search.
        NOTE: This is an approximate mapping suitable for weighting; exact alignment
        would need tokenizer-provided offset mapping.
        """
        ids = self._tokenize(text)
        offsets = [0]
        accum = ""
        # To avoid quadratic behavior on large texts, we incrementally decode by appending
        # each next token to a running prefix using tokenizer.decode on slices.
        # SentencePiece decode is deterministic and stable for this purpose.
        for t in ids:
            accum = self.tokenizer.decode(self.tokenizer.encode(accum, out_type=int) + [t])
            offsets.append(len(accum))
        return ids, offsets

    @staticmethod
    def _char_to_token_index(offsets: List[int], char_pos: int) -> int:
        """Find smallest token index i where decoded chars >= char_pos."""
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if offsets[mid] < char_pos:
                lo = mid + 1
            else:
                hi = mid
        return lo

    # ------------------------------ core build ------------------------------

    def _build_samples(self):
        samples = []
        for ex in self.data:
            prompt = ex.get("prompt", "")
            response_obj = ex.get("response", {})

            user_text, asst_head, resp_text = self._wrap_texts(prompt, response_obj)

            # Tokenize parts separately
            user_ids = self._tokenize(user_text)
            asst_head_ids = self._tokenize(asst_head)
            resp_ids = self._tokenize(resp_text)

            # Compose with BOS/EOS
            input_ids: List[int] = []
            if self.bos_id >= 0:
                input_ids.append(self.bos_id)
            input_ids.extend(user_ids)
            input_ids.extend(asst_head_ids)
            assistant_start_idx = len(input_ids)  # labels start here
            input_ids.extend(resp_ids)
            if self.eos_id >= 0:
                input_ids.append(self.eos_id)

            # Truncate (prefer trimming user side; keep assistant content)
            if len(input_ids) > self.max_length:
                overflow = len(input_ids) - self.max_length
                # don't cut BOS
                cut_from = 1 if (self.bos_id >= 0) else 0
                cut_to = min(cut_from + overflow, assistant_start_idx)  # never trim into assistant
                input_ids = input_ids[:cut_from] + input_ids[cut_to:]
                assistant_start_idx = max(assistant_start_idx - (cut_to - cut_from), 0)
                input_ids = input_ids[: self.max_length]

            # Labels: mask user region
            labels = [-100] * len(input_ids)
            for i in range(assistant_start_idx, len(input_ids)):
                labels[i] = input_ids[i]

            # Weights: default 0.0 for user region, 1.0 for assistant region
            weights = [0.0] * assistant_start_idx + [1.0] * (len(input_ids) - assistant_start_idx)

            # Boost weights for "commands" spans
            try:
                # Build mapping from char->token index for resp_text only
                resp_only_ids, char_offsets = self._charpos_to_token_index_lookup(resp_text)
                # Translate resp-only token indices into full input_ids indices
                for (cs, ce) in self._find_commands_char_spans(resp_text):
                    t0 = self._char_to_token_index(char_offsets, cs)
                    t1 = self._char_to_token_index(char_offsets, ce)
                    start = assistant_start_idx + t0
                    end = assistant_start_idx + t1
                    for k in range(max(start, assistant_start_idx), min(end, len(input_ids))):
                        if labels[k] != -100:
                            weights[k] = float(self.COMMANDS_WEIGHT)
            except Exception:
                # If anything fails, keep baseline weights (still helpful)
                pass

            samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "weights": torch.tensor(weights, dtype=torch.float32),
            })

        if samples:
            sample_input = samples[0]["input_ids"]
            sample_labels = samples[0]["labels"]
            print(f"[OpalFinetuneDataset] Built {len(samples)} samples; max_length={self.max_length}")
            print(f"   → BOS={self.bos_id}, EOS={self.eos_id}, PAD={self.pad_id}")
            print(f"   → Sample input len={len(sample_input)}, labels(non-masked)={(sample_labels != -100).sum().item()}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["input_ids"], s["labels"], s["weights"]
