# SFT: Verion_1.0
# OpalFineTuneDataSet.py — complete, production-ready SFT dataset class
# -------------------------------------------------------------------
# Expectations:
# - Each JSONL record is a dict with string fields:
#       {"prompt": "<QUESTION>…</QUESTION>", "response": "<RESPONSE>…</RESPONSE>"}
# - We concatenate prompt + response to form the training text.
# - Labels: mask the prompt; learn only on the response.
# - Weights: up-weight tokens inside <CODE>…</CODE> (optionally wrapped in <CONFIG>…</CONFIG>).
# - SentencePiece tokenizer is used (no offset API); we map char→token by encoding prefixes.
# - Newline sentinel: inside <CODE>, you may use <NL>. We do not alter it; you can post-process
#   generated text to replace <NL> with '\n' for display.
#
# Notes:
# - This class does NOT json-dump the response; it consumes the strings you parsed from JSONL.
# - It is robust to short samples and respects max_length with careful trimming so the assistant
#   portion remains learnable.
#
# Usage:
#   ds = OpalFinetuneDataset(records, sp_tokenizer, max_length=1024, code_weight=3.0)
#   ids, labels, weights = ds[i]
#
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
import re

class OpalFinetuneDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        add_bos: bool = True,
        add_eos: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        code_weight: float = 3.0,
        config_wrap_bonus: float = 1.25,   # multiplies weights if <CODE> is inside <CONFIG>
        ticks_weight: float = 1.0,         # reserved for ``` blocks (not used by default)
        return_weights: bool = True,
    ) -> None:
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.code_w = float(code_weight)
        self.cfg_bonus = float(config_wrap_bonus)
        self.ticks_w = float(ticks_weight)
        self.return_weights = bool(return_weights)

        # Cache list of (full_text, assistant_char_start, response_text)
        self.recs: List[Tuple[str, int, str]] = []
        for obj in data:
            prompt = obj.get("prompt", "")
            response = obj.get("response", "")
            if not isinstance(prompt, str) or not isinstance(response, str):
                continue
            full = f"{prompt}{response}"
            asst_char_start = len(prompt)
            self.recs.append((full, asst_char_start, response))

    def __len__(self) -> int:
        return len(self.recs)

    # ----------------- helpers -----------------
    def _sp_encode(self, text: str):
        """Encode with SentencePiece -> token ids."""
        return self.tok.encode(text, out_type=int)

    def _char_to_tok(self, full: str, char_pos: int) -> int:
        """
        Map a character position in `full` to a token index by encoding the prefix full[:char_pos].
        """
        if char_pos <= 0:
            return 0
        if char_pos >= len(full):
            return len(self._sp_encode(full))
        prefix = full[:char_pos]
        return len(self._sp_encode(prefix))

    def _find_code_char_spans(self, response: str):
        """
        Find code spans in response. Returns list of (start_char, end_char, wrapped_in_config),
        positions are relative to the start of the response string.
        """
        spans = []
        # First, code blocks inside CONFIG
        for cfg in re.finditer(r"<CONFIG>(.*?)</CONFIG>", response, flags=re.DOTALL | re.IGNORECASE):
            inner = cfg.group(1)
            cfg_base = cfg.start(1)
            for m in re.finditer(r"<CODE>(.*?)</CODE>", inner, flags=re.DOTALL | re.IGNORECASE):
                s = cfg_base + m.start(1)
                e = cfg_base + m.end(1)
                spans.append((s, e, True))
        # Bare code blocks not already accounted for
        for m in re.finditer(r"<CODE>(.*?)</CODE>", response, flags=re.DOTALL | re.IGNORECASE):
            s = m.start(1); e = m.end(1)
            # Check overlap with existing
            if not any(s >= s0 and e <= e0 for (s0, e0, _) in spans):
                spans.append((s, e, False))
        return spans

    # ----------------- main item build -----------------
    def __getitem__(self, idx: int):
        full, asst_char_start, response = self.recs[idx]

        # Encode *without* BOS/EOS first so we can map char→token accurately
        ids_no_be = self._sp_encode(full)
        asst_tok_start_no_be = self._char_to_tok(full, asst_char_start)

        # Map all code spans to token spans (exclusive end), BEFORE BOS/EOS
        code_spans_tok = []  # (tok_start, tok_end, factor)
        for (rel_s, rel_e, wrapped) in self._find_code_char_spans(response):
            abs_s = asst_char_start + rel_s
            abs_e = asst_char_start + rel_e
            tok_s = self._char_to_tok(full, abs_s)
            tok_e = self._char_to_tok(full, abs_e)
            factor = self.code_w * (self.cfg_bonus if wrapped else 1.0)
            code_spans_tok.append((tok_s, tok_e, factor))

        # Now add BOS/EOS and compute final ids
        ids = ids_no_be[:]
        asst_tok_start = asst_tok_start_no_be
        bos_offset = 1 if self.add_bos else 0
        if self.add_bos:
            ids = [self.bos_id] + ids
            asst_tok_start = asst_tok_start_no_be + 1
        if self.add_eos:
            ids = ids + [self.eos_id]

        # Shift code spans by BOS offset
        code_spans_tok = [(s + bos_offset, e + bos_offset, f) for (s, e, f) in code_spans_tok]

        # Truncate with head-trim that preserves assistant
        if len(ids) > self.max_length:
            overflow = len(ids) - self.max_length
            head_keep = 1 if self.add_bos else 0
            # Trim as much as we can from prompt segment (head), but not past assistant start
            max_head_trim = max(0, asst_tok_start - head_keep)
            trim_from_head = min(overflow, max_head_trim)
            head_removed = trim_from_head
            if trim_from_head > 0:
                ids = ids[:head_keep] + ids[head_keep + trim_from_head:]
                asst_tok_start -= trim_from_head
                # Shift code spans left
                shifted_spans = []
                for (s, e, f) in code_spans_tok:
                    shifted_spans.append((max(head_keep, s - head_removed), max(head_keep, e - head_removed), f))
                code_spans_tok = shifted_spans
            # Final tail cut if still long
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]
                # Clip code spans to max_length
                code_spans_tok = [(s, min(e, self.max_length), f) for (s, e, f) in code_spans_tok]
                asst_tok_start = min(asst_tok_start, self.max_length - 1)

        # Build labels (prompt masked)
        labels = [-100] * len(ids)
        for j in range(asst_tok_start, len(ids)):
            labels[j] = ids[j]

        # Base weights: 0 for prompt, 1 for response
        weights = [0.0] * asst_tok_start + [1.0] * (len(ids) - asst_tok_start)

        # Apply code multipliers
        for (ts, te, factor) in code_spans_tok:
            ts = max(ts, asst_tok_start)
            te = min(te, len(ids))
            if ts < te:
                for k in range(ts, te):
                    weights[k] *= factor

        # Align lengths
        if len(weights) != len(ids):
            if len(weights) < len(ids):
                weights = weights + [weights[-1] if weights else 1.0] * (len(ids) - len(weights))
            else:
                weights = weights[: len(ids)]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),
        )
