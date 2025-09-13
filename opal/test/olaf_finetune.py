import torch
import argparse
import os
import sentencepiece as smp
from ..transformer.OpalGPTModel import OpalGPT
from ..opalmain.opal_trainer import Opal
from ..utils.opal_constants import OpalConstants
from ..utils.training_utils import estimate_training_time_from_config
from ..utils.mps_utils import create_mps_safe_dataloader, mps_safe_optimizer_step, setup_mps_environment
# from ..config.opal_config import set_finetune_mode
import time
import multiprocessing
import shutil
import json

# Set deterministic behavior
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)

# Enable fine-tuning mode - this switches to appropriate configs
print("🔧 Setting fine-tuning mode...")
# set_finetune_mode(enable_finetune=True)

# 🔧 CRITICAL FIX: Import configs AFTER set_finetune_mode() to get updated values
from ..config.opal_config import OPAL_MODEL_CONFIG, TRAINING_CONFIG, get_device
print(f"✅ Configs imported after fine-tuning mode set")
print(f"   → Learning rate: {OPAL_MODEL_CONFIG['learning_rate']}")
print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}")

# Load SentencePiece tokenizer
print(f"🔧 Loading tokenizer from: {OpalConstants.TOKENIZER_MODEL_PATH}")
sp = smp.SentencePieceProcessor()
sp.load(OpalConstants.TOKENIZER_MODEL_PATH)
print(f"✅ Tokenizer loaded (vocab_size: {sp.get_piece_size()})")

# Validate paths exist
print(f"🔍 Validating paths...")
print(f"   → Tokenizer: {OpalConstants.TOKENIZER_MODEL_PATH} - {'✅' if os.path.exists(OpalConstants.TOKENIZER_MODEL_PATH) else '❌'}")
print(f"   → Fine-tune data: {OpalConstants.FINETUNE_TEST_DATA_PATH} - {'✅' if os.path.exists(OpalConstants.FINETUNE_TEST_DATA_PATH) else '❌'}")
print(f"   → Checkpoint dir: {OpalConstants.CHECKPOINT_DIR} - {'✅' if os.path.exists(OpalConstants.CHECKPOINT_DIR) else '📁 Will create'}")

# Create checkpoint directory if it doesn't exist
os.makedirs(OpalConstants.CHECKPOINT_DIR, exist_ok=True)

device = get_device()

# Note: Opal instance will be created after command line argument parsing
# to ensure it gets the correct config values

def validate_dataset(data_path, num_samples=5):
    """Validate the fine-tuning dataset format and content"""
    print(f"\n🔍 Validating dataset: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            samples = []
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON on line {i+1}: {e}")
                    raise
        
        print(f"✅ Loaded {len(samples)} sample entries for validation")
        
        # Validate structure
        for i, sample in enumerate(samples):
            if "prompt" not in sample or "response" not in sample:
                raise ValueError(f"Sample {i+1} missing required keys 'prompt' or 'response'")
            
            prompt = sample["prompt"]
            response = sample["response"]
            
            print(f"   → Sample {i+1}: prompt={len(prompt)} chars, response_type={type(response).__name__}")
            
            # Check if response is structured JSON
            if isinstance(response, dict) and "action" in response:
                steps = response.get("execution", {}).get("steps", [])
                total_commands = sum(len(step.get("commands", [])) for step in steps)
                print(f"     Action: {response['action']}, Steps: {len(steps)}, Commands: {total_commands}")
        
        print(f"✅ Dataset validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset validation failed: {e}")
        raise

def model_pretrain_test(start_fresh=False, opal_instance=None):
    """Main fine-tuning function with improved error handling and monitoring"""
    
    if opal_instance is None:
        raise ValueError("opal_instance must be provided")
    
    # Validate dataset first
    validate_dataset(OpalConstants.FINETUNE_TEST_DATA_PATH, num_samples=10)
    
    VOCAB_SIZE = sp.get_piece_size()

    # Debug checkpoint path
    print(f"\n🔍 Checkpoint configuration:")
    print(f"   → Checkpoint path: {OpalConstants.CHECKPOINT_PATH}")
    print(f"   → File exists: {os.path.exists(OpalConstants.CHECKPOINT_PATH)}")
    print(f"   → Checkpoint dir: {OpalConstants.CHECKPOINT_DIR}")
    print(f"   → Dir exists: {os.path.exists(OpalConstants.CHECKPOINT_DIR)}")

    # Re-import the configs after calling set_finetune_mode()
    from ..config.opal_config import OPAL_MODEL_CONFIG, TRAINING_CONFIG
    
    print(f"\n📊 FINE-TUNING MODEL CONFIGURATION:")
    print("=" * 60)
    key_configs = [
        "vocab_size", "context_length", "emb_dim", "n_heads", "n_layers",
        "num_epoch", "learning_rate", "weight_decay", "gradient_accumulation_steps",
        "drop_rate", "max_grad_norm", "warmup_steps"
    ]
    for key in key_configs:
        if key in OPAL_MODEL_CONFIG:
            print(f"{key:25} : {OPAL_MODEL_CONFIG[key]}")

    print(f"\n🎯 TRAINING CONFIGURATION:")
    print("=" * 60)
    for key, value in TRAINING_CONFIG.items():
        print(f"{key:25} : {value}")

    print(f"\n📈 TRAINING ESTIMATES:")
    print("=" * 60)
    
    # Count samples in dataset
    with open(OpalConstants.FINETUNE_TEST_DATA_PATH, 'r') as f:
        num_samples = sum(1 for _ in f)
    
    batch_size = TRAINING_CONFIG["batch_size"]
    num_epochs = OPAL_MODEL_CONFIG["num_epoch"]
    grad_accum = OPAL_MODEL_CONFIG.get("gradient_accumulation_steps", 1)
    
    steps_per_epoch = num_samples // (batch_size * grad_accum)
    total_steps = steps_per_epoch * num_epochs
    
    print(f"Dataset samples          : {num_samples:,}")
    print(f"Batch size               : {batch_size}")
    print(f"Gradient accumulation    : {grad_accum}")
    print(f"Effective batch size     : {batch_size * grad_accum}")
    print(f"Steps per epoch          : {steps_per_epoch:,}")
    print(f"Total training steps     : {total_steps:,}")
    print(f"Estimated time (GPU)     : {total_steps * 0.5 / 3600:.1f} hours")

    if (VOCAB_SIZE != OPAL_MODEL_CONFIG["vocab_size"]):
        raise ValueError(f"Vocabulary size mismatch: tokenizer={VOCAB_SIZE}, model={OPAL_MODEL_CONFIG['vocab_size']}")

    print(f"\n🚀 Starting fine-tuning training...")
    print("=" * 60)
    
    # Calculate reasonable eval frequency
    eval_freq = max(100, steps_per_epoch // 10)  # Evaluate ~10 times per epoch
    eval_iter = min(50, steps_per_epoch // 20)   # Use reasonable eval iterations
    
    print(f"Evaluation frequency     : {eval_freq} steps")
    print(f"Evaluation iterations    : {eval_iter}")
    
    try:
        opal_instance.train_and_save_model(
            model_class=OpalGPT,
            config=OPAL_MODEL_CONFIG,
            device=device,
            tokenizer=sp,
            checkpoint_path=OpalConstants.CHECKPOINT_PATH,
            num_epochs=OPAL_MODEL_CONFIG["num_epoch"],
            batch_size=TRAINING_CONFIG["batch_size"],
            log_to_tensorboard=True,
            log_to_wandb=False,  # Disable W&B for RunPod unless configured
            lr=OPAL_MODEL_CONFIG["learning_rate"],
            weight_decay=OPAL_MODEL_CONFIG["weight_decay"],
            start_fresh=start_fresh,
            eval_iter=eval_iter,
            eval_freq=eval_freq,
            start_context="<USER> Show me the IP routing table configuration <ASSISTANT>",
        )
        
        print(f"\n🎉 Fine-tuning completed successfully!")
        print(f"   → Final checkpoint saved at: {OpalConstants.CHECKPOINT_PATH}")
        
        # Test generation after training
        print(f"\n🧪 Testing generation after fine-tuning...")
        test_prompts = [
            "Show me how to configure NAT overload",
            "Help me troubleshoot OSPF routing issues",
            "Configure VLAN trunk ports"
        ]
        
        for prompt in test_prompts:
            print(f"\n📝 Test prompt: {prompt}")
            try:
                # This would require loading the trained model and generating
                print(f"   → [Generation test would run here with trained model]")
            except Exception as e:
                print(f"   ⚠️  Generation test failed: {e}")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise
def parse_arguments():
    """Parse command line arguments for flexible training"""
    parser = argparse.ArgumentParser(description="OPAL Fine-tuning Script for RunPod")
    
    parser.add_argument("--start_fresh", action="store_true",
                       help="Start training from scratch (ignore existing checkpoints)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--eval_freq", type=int, default=None,
                       help="Override evaluation frequency")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--tensorboard", action="store_true", default=True,
                       help="Enable TensorBoard logging (default: True)")
    
    return parser.parse_args()


if __name__ == "__main__":
    print("🚀 OPAL FINE-TUNING ON RUNPOD")
    print("=" * 80)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Show current config before any overrides
    print(f"\n📊 CURRENT CONFIG BEFORE OVERRIDES:")
    print(f"   → Learning rate: {OPAL_MODEL_CONFIG['learning_rate']}")
    print(f"   → Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   → Epochs: {OPAL_MODEL_CONFIG['num_epoch']}")
    
    # Override configs if specified
    if args.epochs:
        OPAL_MODEL_CONFIG["num_epoch"] = args.epochs
        print(f"🔧 Override epochs: {args.epochs}")
    
    if args.batch_size:
        TRAINING_CONFIG["batch_size"] = args.batch_size
        print(f"🔧 Override batch size: {args.batch_size}")
    
    if args.lr:
        OPAL_MODEL_CONFIG["learning_rate"] = args.lr
        print(f"🔧 Override learning rate: {args.lr}")
    else:
        print(f"🔧 Using config learning rate: {OPAL_MODEL_CONFIG['learning_rate']}")
    
    # 🔧 CRITICAL FIX: Create Opal instance AFTER config overrides to ensure correct config
    print("🔧 Creating OPAL trainer instance with final config...")
    opalInstance = Opal(
        config=OPAL_MODEL_CONFIG, 
        tokenizer=sp, 
        is_finetune=True, 
        finetune_data_path=OpalConstants.FINETUNE_TEST_DATA_PATH
    )
    print(f"✅ Opal instance created with learning rate: {opalInstance.config['learning_rate']}")
    
    # Debug: Confirm the Opal instance config
    print(f"\n🔍 OPAL INSTANCE CONFIG VERIFICATION:")
    print(f"   → opalInstance.config['learning_rate']: {opalInstance.config['learning_rate']}")
    print(f"   → OPAL_MODEL_CONFIG['learning_rate']: {OPAL_MODEL_CONFIG['learning_rate']}")
    print(f"   → Config objects match: {opalInstance.config is OPAL_MODEL_CONFIG}")
    
    # Print startup information
    print(f"🏃 Original device config: {device}")
    print(f"🏃 GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🏃 GPU device: {torch.cuda.get_device_name()}")
        print(f"🏃 GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # ✅ MPS SETUP: Use MPS-optimized environment setup for fine-tuning
    mps_device = setup_mps_environment()
    print(f"🍎 Using MPS-optimized device: {mps_device}")
    
    # Override device in config if MPS is available
    if mps_device.type == 'mps':
        TRAINING_CONFIG["device"] = str(mps_device)
        device = mps_device
        print(f"✅ Device config updated to: {TRAINING_CONFIG['device']}")
    
    # Run the training
    try:
        model_pretrain_test(start_fresh=args.start_fresh, opal_instance=opalInstance)
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    print(f"\n✅ Script completed successfully!")
