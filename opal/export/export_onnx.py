from opal.config.opal_config import TRAINING_CONFIG
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from opal.transformer.OpalGPTModel import OpalGPT
# LoRA Domain Adaptation: Import ONNX runtime for validation
import onnxruntime as ort
import numpy as np
import tempfile
import os

def export_and_quantize_model(
    config: dict,
    checkpoint_path: str,
    onnx_output_path: str,
    quantized_output_path: str,
    device: str = None,
    validate_with_ort: bool = True  # LoRA Domain Adaptation: Enable ONNX runtime validation
):
    """
    Loads a trained checkpoint, exports it to ONNX, and creates a quantized ONNX model.
    For LoRA models, automatically merges adapter weights before export.

    Args:
        checkpoint_path (str): Path to the PyTorch checkpoint (.pt file)
        onnx_output_path (str): Path to save the exported ONNX model
        quantized_output_path (str): Path to save the quantized ONNX model
        device (str): Device to run export on
        validate_with_ort (bool): Whether to validate exported model with ONNX Runtime

    Returns:
        (str, str): Paths of exported ONNX and quantized models
    """

    device = device or TRAINING_CONFIG["device"]

    # 1. Load model and weights
    print(f"Loading model from {checkpoint_path}...")
    model = OpalGPT(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # LoRA Domain Adaptation: Check if model has LoRA and handle merging
    has_lora = checkpoint.get("has_lora", False) or (hasattr(model, 'is_lora_enabled') and model.is_lora_enabled())
    
    if has_lora:
        print("🎯 LoRA Domain Adaptation: Detected LoRA model, merging adapters for ONNX export...")
        
        # LoRA Domain Adaptation: Merge LoRA weights before export
        model = model.merge_lora_weights(verbose=True)
        
        # LoRA Domain Adaptation: Verify merge by checking parameter counts
        lora_info = model.get_lora_info()
        if lora_info['total_lora_modules'] > 0:
            print(f"🎯 LoRA Domain Adaptation: Model has {lora_info['total_lora_modules']} LoRA modules")
            print(f"🎯 LoRA Domain Adaptation: Total LoRA parameters: {lora_info['total_lora_parameters']:,}")
        else:
            print("🎯 LoRA Domain Adaptation: No active LoRA modules found (may already be merged)")
    
    model.eval()

    # 2. Create dummy input for tracing
    batch_size = 1
    seq_length = min(64, config["context_length"])  # LoRA Domain Adaptation: Use smaller sequence for validation
    dummy_input = torch.randint(
        0, config["vocab_size"],
        (batch_size, seq_length),
        dtype=torch.long
    ).to(device)
    
    print(f"🔍 Using dummy input shape: {dummy_input.shape} for ONNX export")

    # LoRA Domain Adaptation: Get PyTorch reference output for validation
    pytorch_reference = None
    if validate_with_ort:
        print("🎯 LoRA Domain Adaptation: Generating PyTorch reference output...")
        with torch.no_grad():
            pytorch_output = model(dummy_input)
            if isinstance(pytorch_output, dict):
                pytorch_reference = pytorch_output["logits"].cpu().numpy()
            else:
                pytorch_reference = pytorch_output.cpu().numpy()
        print(f"🎯 LoRA Domain Adaptation: PyTorch output shape: {pytorch_reference.shape}")

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
        opset_version=17,  # LoRA Domain Adaptation: Use opset 17+ as requested
        do_constant_folding=True,  # LoRA Domain Adaptation: Enable optimizations
    )
    print(f"ONNX model saved at {onnx_output_path}")
    
    # LoRA Domain Adaptation: Validate ONNX model with ONNX Runtime
    if validate_with_ort and pytorch_reference is not None:
        print("🎯 LoRA Domain Adaptation: Validating ONNX model with ONNX Runtime...")
        try:
            # LoRA Domain Adaptation: Create ONNX Runtime session (CPU)
            ort_session = ort.InferenceSession(onnx_output_path, providers=['CPUExecutionProvider'])
            
            # LoRA Domain Adaptation: Run inference with ONNX Runtime
            ort_input = dummy_input.cpu().numpy()
            ort_output = ort_session.run(["logits"], {"input_ids": ort_input})[0]
            
            # LoRA Domain Adaptation: Compare outputs
            max_diff = np.abs(pytorch_reference - ort_output).max()
            mean_diff = np.abs(pytorch_reference - ort_output).mean()
            
            print(f"🎯 LoRA Domain Adaptation: ONNX validation results:")
            print(f"   Max difference: {max_diff:.2e}")
            print(f"   Mean difference: {mean_diff:.2e}")
            print(f"   PyTorch shape: {pytorch_reference.shape}")
            print(f"   ONNX shape: {ort_output.shape}")
            
            if max_diff < 1e-4:
                print("✅ ONNX model validation PASSED - outputs match PyTorch reference")
            else:
                print(f"⚠️ ONNX model validation WARNING - large differences detected")
                
        except Exception as e:
            print(f"❌ ONNX model validation FAILED: {e}")
    

    try:
        # 4. Quantize ONNX model
        print(f"Quantizing model...")
        quantize_dynamic(
            model_input=onnx_output_path,
            model_output=quantized_output_path,
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model (Int8) saved at {quantized_output_path}")
    except Exception as e:
        print(f"Error occurred while quantizing to int8: {e}")

    try:
        quantized_output_int4_path = quantized_output_path.replace(".onnx", "_int4.onnx")
        quantize_dynamic(
            model_input=onnx_output_path,
            model_output=quantized_output_int4_path,
            weight_type=QuantType.QInt4
        )
        print(f"Quantized model (int4) saved at {quantized_output_int4_path}")
    except Exception as e:
        print(f"Error occurred while quantizing to int4: {e}")
    

    return onnx_output_path, quantized_output_path
