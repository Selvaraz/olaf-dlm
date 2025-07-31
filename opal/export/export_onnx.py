import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from opal.transformer.OpalGPTModel import OpalGPT

def export_and_quantize_model(
    config: dict,
    checkpoint_path: str,
    onnx_output_path: str,
    quantized_output_path: str,
    device: str = None
):
    """
    Loads a trained checkpoint, exports it to ONNX, and creates a quantized ONNX model.

    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint (.pt file)
        onnx_output_path (str): Path to save the exported ONNX model
        quantized_output_path (str): Path to save the quantized ONNX model
        device (str): 

    Returns:
        (str, str): Paths of exported ONNX and quantized models
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model and weights
    print(f"Loading model from {checkpoint_path}...")
    model = OpalGPT(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 2. Create dummy input for tracing
    dummy_input = torch.randint(
        0, config["vocab_size"],
        (1, config["context_length"]),
        dtype=torch.long
    ).to(device)

    # 3. Export to ONNX
    print(f"Exporting to ONNX: {onnx_output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"}
        },
        opset_version=17
    )
    print(f"ONNX model saved at {onnx_output_path}")

    # 4. Quantize ONNX model
    print(f"Quantizing model...")
    quantize_dynamic(
        model_input=onnx_output_path,
        model_output=quantized_output_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized model saved at {quantized_output_path}")

    return onnx_output_path, quantized_output_path
