import json
import re
from typing import List, Dict, Tuple, Any, Optional

import torch
from torch.utils.data import Dataset


def _sp_encode_len(tok, text: str) -> int:
    return len(tok.encode(text, out_type=int))


class OpalFinetuneDataset(Dataset):
    """
    SFT dataset for tag-based format:

      prompt (string):   "<QUESTION> ... </QUESTION>"
      response (string): "<RESPONSE>\n<SUMMARY>...</SUMMARY>\n<CODE>...</CODE>\n<CONFIG><CODE>...</CODE></CONFIG>\n...</RESPONSE>"

    Weights:
      - base_asst_weight applies to all response tokens
      - config_weight multiplies tokens inside <CONFIG> (and its inner <CODE> if present)
      - code_weight multiplies tokens inside <CODE> outside <CONFIG>
      - summary_weight, desc_weight multiply tokens inside <SUMMARY>, <DESCRIPTION>
      - Optional: backticks (```…``` and inline `…`) treated like <CODE>
    """

    # ==== Default tags: QUESTION/RESPONSE ====
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 1024,
        # BOS/EOS
        add_bos: bool = True,
        add_eos: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        # Weights
        base_asst_weight: float = 1.0,
        code_weight: float = 2.0,
        config_weight: float = 3.0,
        summary_weight: float = 0.9,
        desc_weight: float = 0.9,
        treat_ticks_like_code: bool = True,
        # Tag customization (defaults are QUESTION/RESPONSE)
        prompt_open_tag: str = "<QUESTION>",
        prompt_close_tag: str = "</QUESTION>",
        response_open_tag: str = "<RESPONSE>",
        response_close_tag: str = "</RESPONSE>",
        # Runtime
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

        # Tags
        self.P_OPEN = prompt_open_tag
        self.P_CLOSE = prompt_close_tag
        self.R_OPEN = response_open_tag
        self.R_CLOSE = response_close_tag

        # Inner structure tags
        self.CODE_OPEN, self.CODE_CLOSE = "<CODE>", "</CODE>"
        self.CONFIG_OPEN, self.CONFIG_CLOSE = "<CONFIG>", "</CONFIG>"
        self.SUMMARY_OPEN, self.SUMMARY_CLOSE = "<SUMMARY>", "</SUMMARY>"
        self.DESC_OPEN, self.DESC_CLOSE = "<DESCRIPTION>", "</DESCRIPTION>"

        # Backticks
        self.RE_TFENCE = re.compile(r"```[ \t]*([a-zA-Z0-9_-]+)?[ \t]*\n(.*?)\n```", re.DOTALL)
        self.RE_INLINE_TICK = re.compile(r"`([^`\n]+)`")

        self.lazy = bool(lazy)
        self.return_weights = bool(return_weights)

        # Precompute merged texts
        self._pre: List[Tuple[str, str, str, int]] = []
        for rec in self.data:
            prompt = (rec.get("prompt") or "").strip()
            response = (rec.get("response") or "").strip()

            # Ensure prompt has <QUESTION>...</QUESTION>
            if not prompt.startswith(self.P_OPEN):
                prompt = f"{self.P_OPEN}{prompt}"
            if self.P_CLOSE not in prompt:
                prompt = f"{prompt}{self.P_CLOSE}"

            # Ensure response starts with <RESPONSE> (and close if absent)
            if not response.startswith(self.R_OPEN):
                response = f"{self.R_OPEN}{response}"
            if self.R_CLOSE not in response:
                response = f"{response}{self.R_CLOSE}"

            full_text = f"{prompt}{response}"
            asst_char_start = len(prompt)  # start of response region

            self._pre.append((prompt, response, full_text, asst_char_start))

        self._cache = None
        if not self.lazy:
            self._cache = [self._build_one(i) for i in range(len(self._pre))]

    # ---------- span helpers ----------
    @staticmethod
    def _find_tag_spans(text: str, open_tag: str, close_tag: str) -> List[Tuple[int, int]]:
        spans = []
        i = 0
        L = len(text)
        while True:
            j = text.find(open_tag, i)
            if j < 0:
                break
            k = text.find(close_tag, j + len(open_tag))
            if k < 0:
                spans.append((j + len(open_tag), L))
                break
            spans.append((j + len(open_tag), k))
            i = k + len(close_tag)
        return spans

    @staticmethod
    def _subtract_contained(spans: List[Tuple[int, int]], containers: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        out = []
        for s, e in spans:
            inside = any(s >= cs and e <= ce for cs, ce in containers)
            if not inside:
                out.append((s, e))
        return out

    def _find_all_weight_spans(self, response_text: str):
        spans = {"config_code": [], "code": [], "summary": [], "desc": [], "ticks": []}

        cfg_spans = self._find_tag_spans(response_text, self.CONFIG_OPEN, self.CONFIG_CLOSE)
        code_spans_all = self._find_tag_spans(response_text, self.CODE_OPEN, self.CODE_CLOSE)
        sum_spans = self._find_tag_spans(response_text, self.SUMMARY_OPEN, self.SUMMARY_CLOSE)
        desc_spans = self._find_tag_spans(response_text, self.DESC_OPEN, self.DESC_CLOSE)

        # <CODE> inside <CONFIG> => config_code
        config_code_spans = []
        for cs, ce in cfg_spans:
            has_child = False
            for s, e in code_spans_all:
                if s >= cs and e <= ce:
                    has_child = True
                    config_code_spans.append((s, e))
            if not has_child:
                # treat entire CONFIG inner as config-weighted
                config_code_spans.append((cs, ce))

        diag_code_spans = self._subtract_contained(code_spans_all, cfg_spans)

        spans["config_code"] = config_code_spans
        spans["code"] = diag_code_spans
        spans["summary"] = sum_spans
        spans["desc"] = desc_spans

        if self.treat_ticks_like_code:
            for m in self.RE_TFENCE.finditer(response_text):
                s = m.start(2); e = m.end(2)
                spans["ticks"].append((s, e))
            for m in self.RE_INLINE_TICK.finditer(response_text):
                spans["ticks"].append((m.start(0), m.end(0)))

        return spans

    # ---------- core build ----------
    def _build_one(self, idx: int):
        prompt_text, response_text, full_text, asst_char_start = self._pre[idx]

        # Raw tokens (no BOS/EOS yet)
        input_ids_raw = self.tok.encode(full_text, out_type=int)
        asst_tok_start_raw = _sp_encode_len(self.tok, full_text[:asst_char_start])

        # Truncate from the prompt side only
        max_len_raw = self.max_length - (1 if self.add_bos else 0) - (1 if self.add_eos else 0)
        cut_to_raw = 0
        if len(input_ids_raw) > max_len_raw:
            overflow = len(input_ids_raw) - max_len_raw
            cut_to_raw = min(overflow, asst_tok_start_raw)  # only cut from prompt region
            if cut_to_raw > 0:
                input_ids_raw = input_ids_raw[cut_to_raw:]
                asst_tok_start_raw -= cut_to_raw
                asst_tok_start_raw = max(0, asst_tok_start_raw)

        # Add BOS/EOS
        input_ids = input_ids_raw
        if self.add_bos:
            input_ids = [self.bos_id] + input_ids
            asst_tok_start = asst_tok_start_raw + 1
        else:
            asst_tok_start = asst_tok_start_raw
        if self.add_eos:
            input_ids = input_ids + [self.eos_id]

        # Labels: mask prompt, learn response (including tags)
        labels = [-100] * len(input_ids)
        for j in range(asst_tok_start, len(input_ids)):
            labels[j] = input_ids[j]

        # Base weights
        weights = [0.0] * asst_tok_start + [self.base_asst_weight] * (len(input_ids) - asst_tok_start)

        # Find spans (char offsets relative to response_text)
        spans = self._find_all_weight_spans(response_text)

        # Convert response-relative char spans -> token spans on final sequence
        def resp_span_to_token_span(c_start: int, c_end: int) -> Tuple[int, int]:
            abs_start = asst_char_start + c_start
            abs_end = asst_char_start + c_end
            s_raw = _sp_encode_len(self.tok, full_text[:abs_start])
            e_raw = _sp_encode_len(self.tok, full_text[:abs_end])
            # Adjust for prompt cut (we chopped tokens from the very start)
            if cut_to_raw > 0:
                s_raw = max(0, s_raw - cut_to_raw)
                e_raw = max(0, e_raw - cut_to_raw)
            # +BOS
            if self.add_bos:
                s_raw += 1; e_raw += 1
            # clamp
            s_raw = max(0, min(s_raw, len(input_ids)))
            e_raw = max(0, min(e_raw, len(input_ids)))
            if e_raw < s_raw: e_raw = s_raw
            return s_raw, e_raw

        def apply_mul(w: List[float], s: int, e: int, mult: float):
            if s >= e: return
            e = min(e, len(w))
            for i in range(s, e):
                w[i] *= mult

        # Apply weights
        for cs, ce in spans["config_code"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.config_weight)

        for cs, ce in spans["code"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.code_weight)

        if self.treat_ticks_like_code:
            for cs, ce in spans["ticks"]:
                ts, te = resp_span_to_token_span(cs, ce)
                apply_mul(weights, ts, te, self.code_weight)

        for cs, ce in spans["summary"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.summary_weight)

        for cs, ce in spans["desc"]:
            ts, te = resp_span_to_token_span(cs, ce)
            apply_mul(weights, ts, te, self.desc_weight)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self._pre)

    def __getitem__(self, idx: int):
        if hasattr(self, "_cache") and self._cache is not None:
            x = self._cache[idx]
        else:
            x = self._build_one(idx)
        return x if self.return_weights else (x[0], x[1])
