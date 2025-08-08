#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export OPAL to ONNX with explicit KV-cache I/O for a pure-C ONNX Runtime client.

It produces TWO files:

1) prefill.onnx
   Inputs:
     - input_token_ids: int64[batch, T]
   Outputs:
     - logits: float32[batch, T, vocab]
     - present_key_i:   float16[batch, H_exp, T, Dh]   for i in [0..L-1]
     - present_value_i: float16[batch, H_exp, T, Dh]   for i in [0..L-1]

2) decode.onnx
   Inputs:
     - input_token_ids: int64[batch, 1]
     - past_key_i:      float16[batch, H_exp, Tpast, Dh]
     - past_value_i:    float16[batch, H_exp, Tpast, Dh]
   Outputs:
     - logits: float32[batch, 1, vocab]
     - present_key_i:   float16[batch, H_exp, Tpast+1, Dh]
     - present_value_i: float16[batch, H_exp, Tpast+1, Dh]

     # FP32 export (recommended if you’ll quantize)
    python scripts/export_kvcache_onnx.py \
    --config-name GPT_CONFIG_OPAL_20M \
    --checkpoint path/to/weights.pt \
    --out-dir out/kv_onnx \
    --seq-len 1024 --past-len 1024 \
    --enable-new-attention \
    --opset 17 \
    --quantize-int8 --per-channel --optimize

Notes
-----
- H_exp = expanded head count seen by attention after MQA/GQA. With MQA (kv_heads=1, n_heads=8), H_exp = 8.
- These shapes/names match the C snippets I gave you earlier.
- For INT8: only weights are int8; I/O dtypes are unchanged (ids=int64, logits=float32, cache=float16).
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Make local imports work if run from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent.name == "scripts") else Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.OpalGPTModel import OpalGPT
import config.opal_config as opal_cfg

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
except Exception:
    onnx = None
    quantize_dynamic = None
    QuantType = None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", required=True, help="e.g., GPT_CONFIG_OPAL_20M")
    ap.add_argument("--checkpoint", default="", help="Optional .pt/.pth weights")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--seq-len", type=int, default=1024, help="Export-time T for prefill")
    ap.add_argument("--past-len", type=int, default=1024, help="Export-time Tpast for decode")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--enable-new-attention", action="store_true", help="Enable RoPE+MQA path if your code supports it")
    ap.add_argument("--quantize-int8", action="store_true", help="Apply ORT weight-only INT8 to exported ONNX files")
    ap.add_argument("--per-channel", action="store_true", help="Use per-channel weight quantization (recommended)")
    ap.add_argument("--optimize", action="store_true", help="Ask ORT quantizer to optimize model before quant")
    return ap.parse_args()


def load_cfg(name: str, enable_new_attention: bool):
    if not hasattr(opal_cfg, name):
        raise KeyError(f"Config '{name}' not found in config/opal_config.py")
    cfg = dict(getattr(opal_cfg, name))
    # Ensure small-model defaults
    cfg.setdefault("kv_heads", 1)
    cfg.setdefault("use_rope", True)
    cfg.setdefault("tie_embeddings", True)
    cfg.setdefault("qkv_bias", False)
    if enable_new_attention:
        cfg["enable_new_attention"] = True
    return cfg


def build_model(cfg, ckpt_path: str):
    model = OpalGPT(cfg)
    if ckpt_path:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def expanded_heads(cfg):
    H = cfg["n_heads"]
    kv_h = cfg.get("kv_heads", H)
    # After expansion, attention runs with H heads
    return H


def export_prefill(model, cfg, out_dir: Path, seq_len: int, batch: int, opset: int) -> Path:
    """Export prefill.onnx (no past -> present caches)."""
    class PrefillWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_token_ids: torch.Tensor):
            out = self.m(input_token_ids=input_token_ids, use_cache=True)
            logits = out["logits"]
            pks, pvs = [], []
            for (pk, pv) in out["present_key_values"]:
                pks.append(pk)
                pvs.append(pv)
            return (logits, *pks, *pvs)

    wrapper = PrefillWrapper(model)

    vocab = cfg["vocab_size"]
    dummy_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long)
    L = cfg["n_layers"]

    # Output names: logits + all presents
    out_names = ["logits"] + [f"present_key_{i}" for i in range(L)] + [f"present_value_{i}" for i in range(L)]

    dynamic_axes = {
        "input_token_ids": {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    }
    for i in range(L):
        dynamic_axes[f"present_key_{i}"]   = {0: "batch", 2: "seq_len"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "seq_len"}

    path = out_dir / "prefill.onnx"
    print(f"[export] prefill → {path}")
    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        f=str(path),
        input_names=["input_token_ids"],
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        opset_version=opset,
    )
    return path


def export_decode(model, cfg, out_dir: Path, past_len: int, batch: int, opset: int) -> Path:
    """Export decode.onnx (single token + past -> new present)."""
    L = cfg["n_layers"]

    class DecodeWrapper(torch.nn.Module):
        def __init__(self, m, L): super().__init__(); self.m = m; self.L = L
        def forward(self, input_token_ids: torch.Tensor, *flat_past):
            # flat_past: [past_key_0, past_value_0, ..., past_key_{L-1}, past_value_{L-1}]
            past = []
            for i in range(self.L):
                pk = flat_past[2*i]
                pv = flat_past[2*i + 1]
                past.append((pk, pv))
            out = self.m(input_token_ids=input_token_ids, past_key_values=past, use_cache=True)
            logits = out["logits"]
            pks, pvs = [], []
            for (pk, pv) in out["present_key_values"]:
                pks.append(pk)
                pvs.append(pv)
            return (logits, *pks, *pvs)

    wrapper = DecodeWrapper(model, L)

    H_exp = expanded_heads(cfg)
    Dh = cfg["emb_dim"] // cfg["n_heads"]
    vocab = cfg["vocab_size"]

    dummy_ids = torch.randint(0, vocab, (batch, 1), dtype=torch.long)

    # Build dummy past in fp16 with past_len time-steps
    past = []
    for _ in range(L):
        past_k = torch.zeros((batch, H_exp, past_len, Dh), dtype=torch.float16)
        past_v = torch.zeros((batch, H_exp, past_len, Dh), dtype=torch.float16)
        past += [past_k, past_v]

    in_names = ["input_token_ids"]
    for i in range(L):
        in_names += [f"past_key_{i}", f"past_value_{i}"]

    out_names = ["logits"] + [f"present_key_{i}" for i in range(L)] + [f"present_value_{i}" for i in range(L)]

    dynamic_axes = {
        "input_token_ids": {0: "batch", 1: "step"},
        "logits": {0: "batch", 1: "step"},
    }
    for i in range(L):
        dynamic_axes[f"past_key_{i}"]      = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"past_value_{i}"]    = {0: "batch", 2: "past_seq"}
        dynamic_axes[f"present_key_{i}"]   = {0: "batch", 2: "new_seq"}
        dynamic_axes[f"present_value_{i}"] = {0: "batch", 2: "new_seq"}

    path = out_dir / "decode.onnx"
    print(f"[export] decode → {path}")
    torch.onnx.export(
        wrapper,
        (dummy_ids, *past),
        f=str(path),
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        opset_version=opset,
    )
    return path


def maybe_quantize_int8(path: Path, per_channel: bool, optimize: bool):
    if not (onnx and quantize_dynamic):
        print(f"[quantize] onnxruntime quantization not available; skipping: {path}")
        return None
    outp = path.with_suffix(".int8.onnx")
    print(f"[quantize] INT8 weight-only (per_channel={per_channel}, optimize={optimize}) → {outp}")
    quantize_dynamic(
        model_input=str(path),
        model_output=str(outp),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        optimize_model=optimize,
    )
    return outp


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg(args.config_name, enable_new_attention=args.enable_new_attention)
    model = build_model(cfg, args.checkpoint)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    prefill_path = export_prefill(model, cfg, out_dir, seq_len=args.seq_len, batch=args.batch, opset=args.opset)
    decode_path  = export_decode(model, cfg, out_dir, past_len=args.past_len, batch=args.batch, opset=args.opset)

    if args.quantize_int8:
        maybe_quantize_int8(prefill_path, args.per_channel, args.optimize)
        maybe_quantize_int8(decode_path,  args.per_channel, args.optimize)

    print("[done] Export complete.")
    print(f"        prefill: {prefill_path}")
    print(f"        decode : {decode_path}")
