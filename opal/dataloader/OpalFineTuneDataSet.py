
import json
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset

class OpalFinetuneDataset(Dataset):
    """
    Patched to support QUESTION/RESPONSE schema (no JSON wrapping) and apply
    token-space weighting for <CONFIG>/<CODE>/backticks even when tags are split
    across multiple SentencePiece pieces.
    """
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
        base_asst_weight: float = 1.0,
        code_weight: float = 2.0,
        config_weight: float = 3.0,
        summary_weight: float = 0.9,
        desc_weight: float = 0.9,
        treat_ticks_like_code: bool = True,
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
        self.QO, self.QC = "<QUESTION>", "</QUESTION>"
        self.RO, self.RC = "<RESPONSE>", "</RESPONSE>"
        self.CODE_O, self.CODE_C = "<CODE>", "</CODE>"
        self.CONFIG_O, self.CONFIG_C = "<CONFIG>", "</CONFIG>"
        self.SUMMARY_O, self.SUMMARY_C = "<SUMMARY>", "</SUMMARY>"
        self.DESC_O, self.DESC_C = "<DESCRIPTION>", "</DESCRIPTION>"

        # Pre-encode tag patterns once (as id sequences; tags may be split)
        def enc(s): 
            try: return self.tok.encode(s, out_type=int)
            except Exception: return self.tok.EncodeAsIds(s)
        self._pat = {
            "CODE_O": enc(self.CODE_O), "CODE_C": enc(self.CODE_C),
            "CONFIG_O": enc(self.CONFIG_O), "CONFIG_C": enc(self.CONFIG_C),
            "SUMMARY_O": enc(self.SUMMARY_O), "SUMMARY_C": enc(self.SUMMARY_C),
            "DESC_O": enc(self.DESC_O), "DESC_C": enc(self.DESC_C),
            "RO": enc(self.RO), "RC": enc(self.RC),
        }

        # Build samples
        self.samples = []
        for i, rec in enumerate(self.data):
            prompt = (rec.get("prompt") or "").strip()
            response = rec.get("response")
            # If response is structured (dict), turn into a simple Q/R body; else assume it's already tagged
            if isinstance(response, dict):
                # Best-effort stringify
                body = json.dumps(response, ensure_ascii=False)
                response_text = f"{self.RO}\n{body}\n{self.RC}"
            else:
                response_text = (response or "").strip()
                # Ensure we have RESPONSE wrapper
                if not response_text.startswith(self.RO):
                    response_text = f"{self.RO}\n{response_text}"
                if not response_text.endswith(self.RC):
                    response_text = f"{response_text}\n{self.RC}"

            user_text = prompt
            if not user_text.startswith(self.QO):
                user_text = f"{self.QO} {user_text}"
            if not user_text.endswith(self.QC):
                user_text = f"{user_text} {self.QC}"
            user_text = user_text + "\\n"

            full_text = user_text + response_text

            # Tokenize
            try:
                ids = self.tok.encode(full_text, out_type=int)
                user_ids = self.tok.encode(user_text, out_type=int)
            except Exception:
                ids = self.tok.EncodeAsIds(full_text)
                user_ids = self.tok.EncodeAsIds(user_text)

            # Truncate from the left of prompt region only
            asst_tok_start = len(user_ids)
            max_len_raw = self.max_length - (1 if self.add_bos else 0) - (1 if self.add_eos else 0)
            if len(ids) > max_len_raw:
                overflow = len(ids) - max_len_raw
                cut = min(overflow, asst_tok_start)  # only eat prompt
                if cut > 0:
                    ids = ids[cut:]
                    asst_tok_start -= cut
                    if asst_tok_start < 0: asst_tok_start = 0

            # Add BOS/EOS
            if self.add_bos:
                ids = [self.bos_id] + ids
                asst_tok_start += 1
            if self.add_eos:
                ids = ids + [self.eos_id]

            # Labels: learn response (including <RESPONSE> tags)
            labels = [-100] * len(ids)
            for j in range(asst_tok_start, len(ids)):
                labels[j] = ids[j]

            # Base weights
            weights = [0.0] * asst_tok_start + [self.base_asst_weight] * (len(ids) - asst_tok_start)

            # Weight code/config/summary/desc using token patterns (fast, robust even if tags are split)
            def find_all(hay: List[int], needle: List[int]) -> List[int]:
                if not needle: return []
                L, N = len(hay), len(needle)
                out = []
                for k in range(L - N + 1):
                    if hay[k:k+N] == needle:
                        out.append(k)
                return out

            # response region ids (after asst start)
            resp_ids = ids[asst_tok_start:]
            Lresp = len(resp_ids)

            def spans_from_tags(open_pat_key: str, close_pat_key: str):
                os = find_all(resp_ids, self._pat[open_pat_key])
                cs = find_all(resp_ids, self._pat[close_pat_key])
                cs_iter = iter(cs)
                spans = []
                cur_c = next(cs_iter, None)
                for o in os:
                    while cur_c is not None and cur_c <= o:
                        cur_c = next(cs_iter, None)
                    if cur_c is None:
                        break
                    spans.append((o + len(self._pat[open_pat_key]), cur_c))  # inside content
                return spans

            # CODE blocks
            for s,e in spans_from_tags("CODE_O", "CODE_C"):
                s_abs, e_abs = asst_tok_start + s, asst_tok_start + e
                for t in range(s_abs, min(e_abs, len(weights))):
                    weights[t] *= self.code_weight

            # SUMMARY / DESCRIPTION (milder weight)
            for (o_key, c_key, mult) in [
                ("SUMMARY_O","SUMMARY_C", self.summary_weight),
                ("DESC_O","DESC_C", self.desc_weight),
            ]:
                for s,e in spans_from_tags(o_key, c_key):
                    s_abs, e_abs = asst_tok_start + s, asst_tok_start + e
                    for t in range(s_abs, min(e_abs, len(weights))):
                        weights[t] *= mult

            # CONFIG last (strongest; may wrap CODE)
            for s,e in spans_from_tags("CONFIG_O","CONFIG_C"):
                s_abs, e_abs = asst_tok_start + s, asst_tok_start + e
                for t in range(s_abs, min(e_abs, len(weights))):
                    weights[t] *= self.config_weight

            self.samples.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "weights": torch.tensor(weights, dtype=torch.float32),
            })

        if len(self.samples) > 0:
            print(f"[OpalFinetuneDataset PATCHED] Built {len(self.samples)} samples; max_length={self.max_length}")
            sample = self.samples[0]
            nm = int((sample["labels"] != -100).sum().item())
            print(f"   → Sample len={len(sample['input_ids'])}, learned tokens={nm}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): 
        s = self.samples[idx]
        return s["input_ids"], s["labels"], s["weights"]
