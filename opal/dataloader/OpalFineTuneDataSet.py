import json
from typing import List, Dict, Tuple, Any

import torch
from torch.utils.data import Dataset


def _json_dumps_min(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

class OpalFinetuneDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        tokenizer,
        max_length: int = 1024,
        commands_weight: float = 3.0,
        lazy: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
    ) -> None:
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length
        self.commands_weight = float(commands_weight)
        self.lazy = bool(lazy)

        self.user_tag_open = "<QUESTION>"
        self.user_tag_close = "</QUESTION>"
        self.asst_tag = "<ASSISTANT>"

        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.add_bos = bool(add_bos)
        self.add_eos = bool(add_eos)

        self._pre = []
        self._has_offsets = False
        for rec in self.records:
            prompt = rec.get("prompt", "")
            response_obj = rec.get("response", {})
            user_text, resp_text, full_text, asst_char_start = self._wrap_texts(prompt, response_obj)
            has_commands = '"commands":[' in resp_text
            self._pre.append((full_text, asst_char_start, has_commands, resp_text))

        if not self.lazy:
            self._cache = [self._build_one(i) for i in range(len(self._pre))]
        else:
            self._cache = None

    def _wrap_texts(self, prompt: str, response_obj: Dict) -> Tuple[str, str, str, int]:
        user_text = f"{self.user_tag_open}{prompt}{self.user_tag_close}"
        resp_text = _json_dumps_min(response_obj)
        full_text = f"{user_text}{self.asst_tag}{resp_text}"
        asst_char_start = len(user_text) + len(self.asst_tag)
        return user_text, resp_text, full_text, asst_char_start

    @staticmethod
    def _find_commands_char_spans(resp_text: str) -> List[Tuple[int, int]]:
        spans: List[Tuple[int, int]] = []
        key = '\"commands\":['
        i = 0
        n = len(resp_text)
        while True:
            j = resp_text.find(key, i)
            if j < 0:
                break
            arr_start = j + len(key)
            depth = 1
            k = arr_start
            in_str = False
            esc = False
            while k < n and depth > 0:
                ch = resp_text[k]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == '\\':
                        esc = True
                    elif ch == '\"':
                        in_str = False
                else:
                    if ch == '\"':
                        in_str = True
                    elif ch == '[':
                        depth += 1
                    elif ch == ']':
                        depth -= 1
                k += 1
            end = min(k, n)
            spans.append((j, end))
            i = end
        return spans

    def _compute_weights_from_offsets(self, offsets, asst_char_start: int, asst_tok_start: int, resp_text: str, has_commands: bool) -> List[float]:
        L = len(offsets)
        weights = [0.0] * asst_tok_start + [1.0] * (L - asst_tok_start)
        if not has_commands:
            return weights
        spans = self._find_commands_char_spans(resp_text)
        if not spans:
            return weights
        for (cs, ce) in spans:
            abs_start = asst_char_start + cs
            abs_end = asst_char_start + ce
            for t_idx in range(asst_tok_start, L):
                s, e = offsets[t_idx]
                if s == e:
                    continue
                if e <= abs_start or s >= abs_end:
                    continue
                weights[t_idx] *= self.commands_weight
        return weights

    def _tokenize_with_offsets(self, text: str):
        try:
            out = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
            if "offset_mapping" in out:
                self._has_offsets = True
                return out["input_ids"], out["offset_mapping"]
        except TypeError:
            try:
                out = self.tok.encode_plus(text, return_offsets_mapping=True, add_special_tokens=False)
                if "offset_mapping" in out:
                    self._has_offsets = True
                    return out["input_ids"], out["offset_mapping"]
            except Exception:
                pass
        try:
            ids = self.tok.encode(text, add_special_tokens=False)
        except Exception:
            ids = self.tok(text)["input_ids"]
        return ids, [(0,0)] * len(ids)

    def _build_one(self, idx: int):
        full_text, asst_char_start, has_commands, resp_text = self._pre[idx]
        input_ids, offsets = self._tokenize_with_offsets(full_text)
        if self.add_bos:
            input_ids = [self.bos_id] + input_ids
            if self._has_offsets:
                offsets = [(0,0)] + offsets
        if self.add_eos:
            input_ids = input_ids + [self.eos_id]
            if self._has_offsets:
                offsets = offsets + [(0,0)]
        if self._has_offsets:
            asst_tok_start = 0
            for i, (s, e) in enumerate(offsets):
                if s >= asst_char_start:
                    asst_tok_start = i
                    break
        else:
            prefix = full_text[:asst_char_start]
            asst_tok_start = len(self.tok.encode(prefix, add_special_tokens=False))
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            cut_from = 1 if self.add_bos else 0
            cut_to = min(cut_from + overflow, asst_tok_start)
            input_ids = input_ids[:cut_from] + input_ids[cut_to:]
            if self._has_offsets:
                offsets = offsets[:cut_from] + offsets[cut_to:]
            asst_tok_start = max(asst_tok_start - (cut_to - cut_from), 0)
            input_ids = input_ids[: self.max_length]
            if self._has_offsets:
                offsets = offsets[: self.max_length]
        labels = [-100] * len(input_ids)
        for j in range(asst_tok_start, len(input_ids)):
            labels[j] = input_ids[j]
        if self._has_offsets:
            weights = self._compute_weights_from_offsets(offsets, asst_char_start, asst_tok_start, resp_text, has_commands)
        else:
            weights = [0.0] * asst_tok_start + [1.0] * (len(input_ids) - asst_tok_start)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float),
        )

    def __len__(self) -> int:
        return len(self._pre)

    def __getitem__(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        return self._build_one(idx)
