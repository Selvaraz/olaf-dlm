import torch
import onnxruntime as ort
import numpy as np
import sentencepiece as spm
from torch.nn import functional as F
from ..transformer.OpalGPTModel import OpalGPT
from ..config.opal_config import OPAL_MODEL_CONFIG
from ..dataloader.OpalDataSet import OpalDataset
from torch.utils.data import DataLoader
import argparse

def evaluate_pytorch(checkpoint_path, dataset_loader, device="cpu"):
    """
    Evaluate a PyTorch checkpoint model on a dataset and compute loss & perplexity.
    """
    model = OpalGPT(OPAL_MODEL_CONFIG).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for input_ids, targets in dataset_loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def evaluate_onnx(onnx_path, dataset_loader, device=None):
    """
    Evaluate a quantized ONNX model on a dataset and compute loss & perplexity.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
    )

    total_loss, total_tokens = 0, 0
    for input_ids, targets in dataset_loader:
        input_ids_np = input_ids.numpy()
        logits = session.run(None, {"input_ids": input_ids_np})[0]
        logits = torch.tensor(logits)  # Convert back to torch for loss calc
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item() * targets.numel()
        total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--onnx_model", required=True, help="Path to quantized .onnx model")
    parser.add_argument("--data_file", required=True, help="Path to validation text file")
    parser.add_argument("--tokenizer_model", required=True, help="Path to SentencePiece model")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_model)
    # Load dataset
    from opal.datasets.OpalDataset import OpalDataset
    dataset = OpalDataset(open(args.data_file).read(), sp, max_length=OPAL_MODEL_CONFIG["context_length"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate PyTorch Model
    pt_loss, pt_ppl = evaluate_pytorch(args.checkpoint, loader, args.device)
    print(f"PyTorch Model: Loss={pt_loss:.4f}, Perplexity={pt_ppl:.4f}")

    # Evaluate Quantized ONNX Model
    onnx_loss, onnx_ppl = evaluate_onnx(args.onnx_model, loader, args.device)
    print(f"Quantized ONNX Model: Loss={onnx_loss:.4f}, Perplexity={onnx_ppl:.4f}")


# Example RUN
# python opal/tools/evaluate_model.py \
#   --checkpoint checkpoints/opal_gpt_checkpoint_latest.pt \
#   --onnx_model checkpoints/opal_gpt_checkpoint_latest_quantized.onnx \
#   --data_file data/validation.txt \
#   --tokenizer_model /home/selvaraj/data/models/opal_tokenizer.model \
#   --device cpu

if __name__ == "__main__":
    main()
