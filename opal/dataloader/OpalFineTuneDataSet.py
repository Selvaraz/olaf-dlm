import json
import re
from typing import List, Dict, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset


def _sp_encode_len(tok, text: str) -> int:
    """Length in tokens for SentencePiece encoding of `text`."""
    return len(tok.encode(text, out_type=int))


class OpalFinetuneDataset(Dataset):
    """
    SFT dataset for tag-based format:
      prompt:   "<USER>... </USER>"
      response: "<ASSISTANT>\n<SUMMARY>...</SUMMARY>\n<CODE>...</CODE>\n<CONFIG><CODE>...</CODE></CONFIG>\n...</ASSISTANT>"

    Weights:
      - base_asst_weight applies to all assistant tokens
      - code_weight multiplies tokens inside any <CODE>…</CODE> (outside <CONFIG>)
      - config_weight multiplies tokens inside <CONFIG><CODE>…</CODE></CONFIG>
      - summary_weight, desc_weight multiply inside <SUMMARY>, <DESCRIPTION>
      - Optional: treat backticks (```…``` and inline `…`) as <CODE>
    """
    # Default tag set
    USER_OPEN = "<USER>"
    USER_CLOSE = "</USER>"
    ASST_OPEN = "<ASSISTANT>"
    # No explicit </ASSISTANT> required, but supported if present

    CODE_OPEN = "<CODE>"
    CODE_CLOSE = "</CODE>"
    CONFIG_OPEN = "<CONFIG>"
    CONFIG_CLOSE = "</CONFIG>"
    SUMMARY_OPEN = "<SUMMARY>"
    SUMMARY_CLOSE = "</SUMMARY>"
    DESC_OPEN = "<DESCRIPTION>"
    DESC_CLOSE = "</DESCRIPTION>"

    # Backticks regex (for optional code weighting if present in data)
    RE_TFENCE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]+)?[ \t]*\n(.*?)\n```", re.DOTALL)
    RE_INLINE_TICK = re.compile(r"`([^`\n]+)`")

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        # Labeling & sequence ends
        add_bos: bool = True,
        add_eos: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        # Weighting controls
        base_asst_weight: float = 1.0,
        code_weight: float = 2.0,
        config_weight: float = 3.0,
        summary_weight: float = 0.9,
        desc_weight: float = 0.9,
        treat_ticks_like_code: bool = True,
        # Runtime options
        lazy: bool = True,
        return_weights: bool = True,
    ) -> None:
        self.data = data
        self.tok = tokenizer
        self.max_length = int(max_length)

        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.base_asst_weight = float(base_asst_weight)
        self.code_weight = float(code_weight)
        self.config_weight = float(config_weight)
        self.summary_weight = float(summary_weight)
        self.desc_weight = float(desc_weight)
        self.treat_ticks_like_code = bool(treat_ticks_like_code)

        self.lazy = bool(lazy)
        self.return_weights = bool(return_weights)

        # Precompute merged texts
        self._pre: List[Tuple[str, str, str, int]] = []
        for rec in self.data:
            prompt = rec.get("prompt", "") or ""
            response = rec.get("response", "") or ""

            # Ensure tags are present in reasonable shape
            if not prompt.startswith(self.USER_OPEN):
                prompt = f"{self.USER_OPEN}{prompt}"
            if self.USER_CLOSE not in prompt:
                prompt = f"{prompt}{self.USER_CLOSE}"

            if not response.startswith(self.ASST_OPEN):
                response = f"{self.ASST_OPEN}{response}"

            full_text = f"{prompt}{response}"
            asst_char_start = len(prompt)  # assistant text starts where response begins

            self._pre.append((prompt, response, full_text, asst_char_start))

        self._cache = None
        if not self.lazy:
            self._cache = [self._build_one(i) for i in range(len(self._pre))]

    # -------- span finding helpers (char indices relative to response string) --------

    @staticmethod
    def _find_tag_spans(text: str, open_tag: str, close_tag: str) -> List[Tuple[int, int]]:
        """Return [ (start_idx, end_idx) ] for the INNER content between matching open/close tags."""
        spans = []
        i = 0
        L = len(text)
        while True:
            j = text.find(open_tag, i)
            if j < 0:
                break
            k = text.find(close_tag, j + len(open_tag))
            if k < 0:
                # No closing tag; take content to end of text
                start = j + len(open_tag)
                spans.append((start, L))
                break
            else:
                start = j + len(open_tag)
                end = k
                spans.append((start, end))
                i = k + len(close_tag)
        return spans

    @staticmethod
    def _subtract_contained(spans: List[Tuple[int, int]], containers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Remove spans fully contained in any container span."""
        out = []
        for s, e in spans:
            inside = False
            for cs, ce in containers:
                if s >= cs and e <= ce:
                    inside = True
                    break
            if not inside:
                out.append((s, e))
        return out

    def _find_all_weight_spans(self, response_text: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Find character spans by category within the ASSISTANT response string.
        Returns dict with keys: 'config_code', 'code', 'summary', 'desc', 'ticks'
        Note: spans are relative to response_text (not full_text).
        """
        spans = {
            "config_code": [],
            "code": [],
            "summary": [],
            "desc": [],
            "ticks": [],
        }

        # Primary tags
        cfg_spans = self._find_tag_spans(response_text, self.CONFIG_OPEN, self.CONFIG_CLOSE)
        code_spans_all = self._find_tag_spans(response_text, self.CODE_OPEN, self.CODE_CLOSE)
        sum_spans = self._find_tag_spans(response_text, self.SUMMARY_OPEN, self.SUMMARY_CLOSE)
        desc_spans = self._find_tag_spans(response_text, self.DESC_OPEN, self.DESC_CLOSE)

        # Code-inside-config (prefer weight_config)
        config_code_spans: List[Tuple[int, int]] = []
        for (cs, ce) in cfg_spans:
            # CODE spans that lie fully inside this config region
            for (s, e) in code_spans_all:
                if s >= cs and e <= ce:
                    config_code_spans.append((s, e))

            # If there is a CONFIG block without explicit CODE child,
            # treat the entire CONFIG inner body as config code.
            has_child_code = any(s >= cs and e <= ce for (s, e) in code_spans_all)
            if not has_child_code:
                config_code_spans.append((cs, ce))

        # Diagnostic code spans: CODE spans not inside CONFIG
        diag_code_spans = self._subtract_contained(code_spans_all, cfg_spans)

        spans["config_code"] = config_code_spans
        spans["code"] = diag_code_spans
        spans["summary"] = sum_spans
        spans["desc"] = desc_spans

        # Optional: treat backticks as code (if any remain in your data)
        if self.treat_ticks_like_code:
            for m in self.RE_TFENCE.finditer(response_text):
                body = m.group(2)
                s = m.start(2)
                e = s + len(body)
                spans["ticks"].append((s, e))
            for m in self.RE_INLINE_TICK.finditer(response_text):
                s = m.start(0)  # include backticks region for simplicity
                e = m.end(0)
                spans["ticks"].append((s, e))

        return spans

    # ---------------- core build ----------------

    def _build_one(self, idx: int):
        prompt_text, response_text, full_text, asst_char_start = self._pre[idx]

        # Tokenize entire string (raw; no BOS/EOS yet)
        input_ids_raw = self.tok.encode(full_text, out_type=int)

        # Assistant start in tokens via prefix length
        asst_tok_start_raw = _sp_encode_len(self.tok, full_text[:asst_char_start])

        # Truncate from prompt only (never cut into assistant)
        max_len_raw = self.max_length - (1 if self.add_bos else 0) - (1 if self.add_eos else 0)
        cut_from_raw = 0
        cut_to_raw = 0
        if len(input_ids_raw) > max_len_raw:
            overflow = len(input_ids_raw) - max_len_raw
            # We only remove from the prompt region [0, asst_tok_start_raw)
            cut_to_raw = min(overflow, asst_tok_start_raw)
            if cut_to_raw > 0:
                input_ids_raw = input_ids_raw[:cut_from_raw] + input_ids_raw[cut_to_raw:]
                asst_tok_start_raw = max(0, asst_tok_start_raw - cut_to_raw)  # cut_from_raw=0

        # Now add BOS/EOS
        input_ids = input_ids_raw
        if self.add_bos:
            input_ids = [self.bos_id] + input_ids
            asst_tok_start = asst_tok_start_raw + 1
        else:
            asst_tok_start = asst_tok_start_raw

        if self.add_eos:
            input_ids = input_ids + [self.eos_id]

        # Labels: mask prompt, learn assistant (incl. tags)
        labels = [-100] * len(input_ids)
        for j in range(asst_tok_start, len(input_ids)):
            labels[j] = input_ids[j]

        # Base weights
        weights = [0.0] * asst_tok_start + [self.base_asst_weight] * (len(input_ids) - asst_tok_start)

        # Compute weighted spans (character-level in response_text)
        spans = self._find_all_weight_spans(response_text)

        # Helper: convert response-relative char span -> token indices on the (possibly cut) sequence
        def resp_span_to_token_span(c_start: int, c_end: int) -> Tuple[int, int]:
            """
            Returns [start_tok_final, end_tok_final) indices on the final input_ids.
            """
            # Absolute char positions in full_text
            abs_start = asst_char_start + c_start
            abs_end = asst_char_start + c_end

            # Token indices BEFORE any BOS/EOS and BEFORE prompt cut:
            s_raw = _sp_encode_len(self.tok, full_text[:abs_start])
            e_raw = _sp_encode_len(self.tok, full_text[:abs_end])

            # Adjust for prompt cut (we removed tokens [0, cut_to_raw) from the very beginning)
            # Since all spans lie within assistant portion (>= original asst_tok_start_raw),
            # s_raw and e_raw are >= cut_to_raw, so subtract cut_to_raw.
            if cut_to_raw > 0:
                s_raw = max(0, s_raw - cut_to_raw)
                e_raw = max(0, e_raw - cut_to_raw)

            # Add BOS offset
            if self.add_bos:
                s_raw += 1
                e_raw += 1

            # Clamp to current sequence bounds
            s_raw = max(0, min(s_raw, len(input_ids)))
            e_raw = max(0, min(e_raw, len(input_ids)))
            if e_raw < s_raw:
                e_raw = s_raw
            return s_raw, e_raw

        # Apply multipliers
        def apply_mul(w: List[float], s: int, e: int, mult: float):
            if s >= e:
                return
            e = min(e, len(w))
            for i in range(s, e):
                w[i] *= mult

        # CONFIG code first (highest)
        for (cs, ce) in spans["config_code"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.config_weight)

        # Diagnostic code (<CODE> outside <CONFIG>)
        for (cs, ce) in spans["code"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.code_weight)

        # Backticks treated like code (optional)
        if self.treat_ticks_like_code and spans["ticks"]:
            for (cs, ce) in spans["ticks"]:
                ts, te = resp_span_to_token_span(cs, ce)
                apply_mul(weights, ts, te, self.code_weight)

        # Summary / Description (typically slightly > 1.0 or <= 1.0 depending on preference)
        for (cs, ce) in spans["summary"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.summary_weight)

        for (cs, ce) in spans["desc"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.desc_weight)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float32),  # ✅ MPS OPTIMIZATION: Use float32 for MPS compatibility
        )

    # ------------- std Dataset API -------------

    def __len__(self) -> int:
        return len(self._pre)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            x = self._cache[idx]
        else:
            x = self._build_one(idx)
        if self.return_weights:
            return x  # (input_ids, labels, weights)
        else:
            return x[0], x[1]
