#!/usr/bin/env python3
"""
LoRA Domain Adaptation Validation Test

This script performs an end-to-end "smoke test" of the LoRA implementation,
covering the complete workflow from model initialization through training,
checkpointing, merging, and ONNX export with validation.

Test Flow:
1. Enable LoRA configuration for domain adaptation
2. Initialize model with LoRA adapters injected
3. Run 1-2 training steps with frozen base weights
4. Save both unified and separate LoRA adapter checkpoints  
5. Create merged model checkpoint
6. Export merged model to ONNX with validation
7. Run ONNX Runtime inference to verify correctness

LoRA Domain Adaptation: End-to-end validation test for LoRA implementation
in the domain adaptation phase.
"""

import os
import sys
import torch
import tempfile
import shutil
from pathlib import Path

# LoRA Domain Adaptation: Add opal package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opal.config.opal_config import set_training_phase, OPAL_MODEL_CONFIG, TRAINING_CONFIG
from opal.transformer.OpalGPTModel import OpalGPT
from opal.attention.lora_utils import LoRAConfig
from opal.export.export_onnx import export_and_quantize_model
from opal.utils.opal_constants import OpalConstants
from opal.opalmain.opal_trainer import Opal

# LoRA Domain Adaptation: Import for creating synthetic training data
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm


class SyntheticDataset(Dataset):
    """LoRA Domain Adaptation: Synthetic dataset for testing LoRA training."""
    
    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # LoRA Domain Adaptation: Generate random token sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # LoRA Domain Adaptation: Create labels shifted by one position
        labels = torch.cat([input_ids[1:], torch.tensor([0])])  # Shift for next token prediction
        return input_ids, labels


def test_lora_domain_adaptation(use_temp_dir: bool = True, verbose: bool = True):
    """
    LoRA Domain Adaptation: Main test function for LoRA implementation.
    
    Args:
        use_temp_dir: Whether to use temporary directory for test artifacts
        verbose: Whether to print detailed test progress
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    
    if verbose:
        print("=" * 80)
        print("🎯 LoRA DOMAIN ADAPTATION VALIDATION TEST")
        print("=" * 80)
    
    # LoRA Domain Adaptation: Set up test environment
    test_passed = True
    temp_dir = None
    original_checkpoint_dir = None
    
    try:
        # LoRA Domain Adaptation: Create temporary directory for test artifacts
        if use_temp_dir:
            temp_dir = tempfile.mkdtemp(prefix="lora_test_")
            original_checkpoint_dir = OpalConstants.CHECKPOINT_DIR
            OpalConstants.CHECKPOINT_DIR = temp_dir
            if verbose:
                print(f"🔧 Using temporary directory: {temp_dir}")
        
        # LoRA Domain Adaptation: Step 1 - Configure LoRA for domain adaptation
        if verbose:
            print("\n📋 Step 1: Configuring LoRA for domain adaptation...")
            
        # LoRA Domain Adaptation: Force reload configuration to ensure fresh state
        import importlib
        from opal.config import opal_config
        importlib.reload(opal_config)
        
        set_training_phase("domain_adaptation")
        lora_config = LoRAConfig.from_dict(OPAL_MODEL_CONFIG)
        
        if verbose:
            print(f"   LoRA enabled: {lora_config.use_lora}")
            print(f"   LoRA rank: {lora_config.rank}")
            print(f"   LoRA alpha: {lora_config.alpha}")
            print(f"   Target modules: {lora_config.target_modules}")
            # LoRA Domain Adaptation: Debug configuration values
            print(f"   OPAL_MODEL_CONFIG use_lora: {OPAL_MODEL_CONFIG.get('use_lora')}")
            print(f"   _PHASE_CONFIGS domain_adaptation use_lora: {opal_config._PHASE_CONFIGS['domain_adaptation'].get('use_lora')}")
        
        if not lora_config.use_lora:
            print("❌ ERROR: LoRA not enabled in domain adaptation configuration")
            print(f"❌ DEBUG: OPAL_MODEL_CONFIG.use_lora = {OPAL_MODEL_CONFIG.get('use_lora')}")
            print(f"❌ DEBUG: lora_config.use_lora = {lora_config.use_lora}")
            return False
        
        # LoRA Domain Adaptation: Step 2 - Initialize model with LoRA
        if verbose:
            print("\n🏗️ Step 2: Initializing model with LoRA adapters...")
            
        device = torch.device("cpu")  # LoRA Domain Adaptation: Use CPU for testing
        model = OpalGPT(OPAL_MODEL_CONFIG).to(device)
        
        if not model.is_lora_enabled():
            print("❌ ERROR: Model does not have LoRA adapters enabled")
            return False
        
        lora_info = model.get_lora_info()
        if verbose:
            print(f"   LoRA modules injected: {lora_info['total_lora_modules']}")
            print(f"   LoRA parameters: {lora_info['total_lora_parameters']:,}")
            print(f"   Base parameters: {lora_info['total_base_parameters']:,}")
            print(f"   LoRA percentage: {lora_info['lora_percentage']:.2f}%")
        
        # LoRA Domain Adaptation: Verify base weights are frozen
        lora_params = model.get_lora_parameters()
        base_trainable = sum(1 for name, p in model.named_parameters() 
                            if p.requires_grad and 'lora_' not in name)
        
        if base_trainable > 0:
            print(f"❌ ERROR: {base_trainable} base parameters are still trainable")
            return False
        
        if len(lora_params) == 0:
            print("❌ ERROR: No LoRA parameters found for training")
            return False
            
        if verbose:
            print(f"   ✅ Base model frozen, {len(lora_params)} LoRA parameters trainable")
        
        # LoRA Domain Adaptation: Step 3 - Run training steps
        if verbose:
            print("\n🏃 Step 3: Running LoRA training steps...")
            
        # LoRA Domain Adaptation: Create synthetic dataset
        dataset = SyntheticDataset(
            vocab_size=OPAL_MODEL_CONFIG["vocab_size"],
            seq_len=32,  # LoRA Domain Adaptation: Small sequence length for testing
            num_samples=10
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # LoRA Domain Adaptation: Create optimizer for LoRA parameters only
        optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
        model.train()
        
        initial_lora_params = [p.clone().detach() for p in lora_params[:2]]  # LoRA Domain Adaptation: Save initial state
        
        # LoRA Domain Adaptation: Run training steps
        for step, (input_ids, labels) in enumerate(dataloader):
            if step >= 2:  # LoRA Domain Adaptation: Only run 2 steps for testing
                break
                
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
            if verbose:
                print(f"   Step {step + 1}: Loss = {loss.item():.4f}")
        
        # LoRA Domain Adaptation: Verify LoRA parameters were updated
        params_changed = False
        for initial, current in zip(initial_lora_params, lora_params[:2]):
            if not torch.allclose(initial, current, atol=1e-6):
                params_changed = True
                break
                
        if not params_changed:
            print("❌ ERROR: LoRA parameters did not change during training")
            return False
            
        if verbose:
            print("   ✅ LoRA parameters updated successfully")
        
        # LoRA Domain Adaptation: Step 4 - Save checkpoints
        if verbose:
            print("\n💾 Step 4: Saving LoRA checkpoints...")
            
        # LoRA Domain Adaptation: Create trainer for checkpoint saving
        trainer = Opal(OPAL_MODEL_CONFIG, is_finetune=False)
        
        # LoRA Domain Adaptation: Create dummy scheduler for checkpoint
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
        
        checkpoint_path = trainer.save_model_checkpoint(
            config=OPAL_MODEL_CONFIG,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=1,
            train_losses=[0.5, 0.4],
            val_losses=[0.6, 0.5],
            tokenizer_model=None
        )
        
        if verbose:
            print(f"   Checkpoint saved: {checkpoint_path}")
        
        # LoRA Domain Adaptation: Verify checkpoint structure
        if os.path.isdir(checkpoint_path):
            # LoRA Domain Adaptation: Check for expected subdirectories
            base_dir = os.path.join(checkpoint_path, "base")
            lora_dir = os.path.join(checkpoint_path, "lora")
            merged_dir = os.path.join(checkpoint_path, "merged")
            
            if not os.path.exists(base_dir):
                print(f"❌ ERROR: Base checkpoint directory not found: {base_dir}")
                return False
            if not os.path.exists(lora_dir):
                print(f"❌ ERROR: LoRA adapter directory not found: {lora_dir}")
                return False
            if not os.path.exists(merged_dir):
                print(f"❌ ERROR: Merged checkpoint directory not found: {merged_dir}")
                return False
                
            # LoRA Domain Adaptation: Check for adapter files
            lora_files = list(Path(lora_dir).glob("lora_adapter_*"))
            if not lora_files:
                print(f"❌ ERROR: No LoRA adapter files found in {lora_dir}")
                return False
                
            if verbose:
                print(f"   ✅ Found checkpoint structure with {len(lora_files)} adapter file sets")
        
        # LoRA Domain Adaptation: Step 5 - Test merged model
        if verbose:
            print("\n🔀 Step 5: Testing merged model...")
        
        # LoRA Domain Adaptation: Create test input
        test_input = torch.randint(0, OPAL_MODEL_CONFIG["vocab_size"], (1, 16)).to(device)
        
        # LoRA Domain Adaptation: Get output before merging
        model.eval()
        with torch.no_grad():
            output_before = model(test_input)["logits"]
        
        # LoRA Domain Adaptation: Merge LoRA weights
        model.merge_lora_weights(verbose=verbose)
        
        # LoRA Domain Adaptation: Get output after merging
        with torch.no_grad():
            output_after = model(test_input)["logits"]
        
        # LoRA Domain Adaptation: Verify outputs are identical
        if not torch.allclose(output_before, output_after, atol=1e-5):
            print("❌ ERROR: Model outputs differ before/after LoRA merging")
            return False
            
        if verbose:
            print("   ✅ Merged model produces identical outputs")
        
        # LoRA Domain Adaptation: Step 6 - Test ONNX export
        if verbose:
            print("\n📤 Step 6: Testing ONNX export...")
        
        # LoRA Domain Adaptation: Find merged checkpoint for export
        if os.path.isdir(checkpoint_path):
            merged_files = list(Path(merged_dir).glob("*.pt"))
            if merged_files:
                merged_checkpoint = str(merged_files[0])
            else:
                # LoRA Domain Adaptation: Use base checkpoint if merged not available
                base_files = list(Path(base_dir).glob("*.pt"))
                merged_checkpoint = str(base_files[0])
        else:
            merged_checkpoint = checkpoint_path
        
        # LoRA Domain Adaptation: Set up ONNX export paths
        onnx_path = os.path.join(temp_dir or ".", "test_model.onnx")
        quantized_path = os.path.join(temp_dir or ".", "test_model_quantized.onnx")
        
        try:
            # LoRA Domain Adaptation: Export to ONNX with validation
            export_and_quantize_model(
                config=OPAL_MODEL_CONFIG,
                checkpoint_path=merged_checkpoint,
                onnx_output_path=onnx_path,
                quantized_output_path=quantized_path,
                device="cpu",
                validate_with_ort=True
            )
            
            if not os.path.exists(onnx_path):
                print(f"❌ ERROR: ONNX file not created: {onnx_path}")
                return False
                
            if verbose:
                print(f"   ✅ ONNX export successful: {onnx_path}")
                
        except Exception as e:
            print(f"❌ ERROR: ONNX export failed: {e}")
            return False
        
        # LoRA Domain Adaptation: All tests passed
        if verbose:
            print("\n🎉 SUCCESS: All LoRA Domain Adaptation tests passed!")
            print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # LoRA Domain Adaptation: Cleanup temporary directory
        if temp_dir and use_temp_dir:
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    print(f"🧹 Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up temporary directory: {e}")
        
        # LoRA Domain Adaptation: Restore original checkpoint directory
        if original_checkpoint_dir is not None:
            OpalConstants.CHECKPOINT_DIR = original_checkpoint_dir


def main():
    """LoRA Domain Adaptation: Main function for running the validation test."""
    
    import argparse
    parser = argparse.ArgumentParser(description="LoRA Domain Adaptation Validation Test")
    parser.add_argument("--no-temp", action="store_true", 
                       help="Don't use temporary directory (keep test artifacts)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    args = parser.parse_args()
    
    success = test_lora_domain_adaptation(
        use_temp_dir=not args.no_temp,
        verbose=not args.quiet
    )
    
    if success:
        print("✅ LoRA Domain Adaptation validation test PASSED")
        sys.exit(0)
    else:
        print("❌ LoRA Domain Adaptation validation test FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
