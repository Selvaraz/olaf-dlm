import torch
import argparse
import sentencepiece as spm
from ..transformer.OpalGPTModel import OpalGPT
from ..opalmain.opal_trainer import Opal
from ..utils.opal_constants import OpalConstants
from ..utils.training_utils import estimate_training_time_from_config
from ..config.opal_config import OPAL_MODEL_CONFIG, TRAINING_CONFIG
import torch
import time
import multiprocessing
import shutil
import copy

# This script is for continued pretraining aka domain-adaptive pretraining (DAPT)
# on the cisco documentation and linux text books using LoRA adapters.

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(OpalConstants.TOKENIZER_MODEL_PATH)

# Create LoRA-enabled configuration for domain adaptation
DAPT_CONFIG = copy.deepcopy(OPAL_MODEL_CONFIG)

print("🎯 LoRA Domain Adaptation Configuration:")
print("=" * 50)
print(f"LoRA Enabled: {DAPT_CONFIG['use_lora']}")
print(f"LoRA Rank: {DAPT_CONFIG['lora_rank']}")
print(f"LoRA Alpha: {DAPT_CONFIG['lora_alpha']}")
print(f"LoRA Dropout: {DAPT_CONFIG['lora_dropout']}")
print(f"Target Modules: {DAPT_CONFIG['target_modules']}")
print(f"Include MLP: {DAPT_CONFIG['lora_include_mlp']}")
print(f"Learning Rate: {DAPT_CONFIG['learning_rate']}")
print(f"Epochs: {DAPT_CONFIG['num_epoch']}")
print("=" * 50)

# Create Opal instance with LoRA-enabled configuration for domain adaptation
opalInstance = Opal(config=DAPT_CONFIG, tokenizer=sp, is_dapt=True)
torch.manual_seed(123)

device = TRAINING_CONFIG["device"]

def model_dapt_train(start_fresh=False):
    VOCAB_SIZE = sp.get_piece_size()

    print("🎯 LoRA DOMAIN ADAPTATION CONFIG:")
    print("=" * 50)
    print("{:<25} {:<25}".format("Key", "Value"))
    print("-" * 50)
    for key, value in DAPT_CONFIG.items():
        print("{:<25} {:<25}".format(str(key), str(value)))

    print("\n\nOPAL TRAINING HYPER PARAMETERS:")
    print("=" * 50)
    print("{:<25} {:<25}".format("Key", "Value"))
    print("-" * 50)
    for key, value in TRAINING_CONFIG.items():
        print("{:<25} {:<25}".format(str(key), str(value)))

    print("\n\n")

    if (VOCAB_SIZE != DAPT_CONFIG["vocab_size"]):
        raise ValueError("Vocabulary size mismatch between tokenizer and model")

    print("🚀 Starting LoRA Domain Adaptation Training...")
    print(f"📊 LoRA Rank: {DAPT_CONFIG['lora_rank']}")
    print(f"📊 LoRA Alpha: {DAPT_CONFIG['lora_alpha']}")
    print(f"📊 Target Modules: {DAPT_CONFIG['target_modules']}")
    print(f"📊 Include MLP: {DAPT_CONFIG['lora_include_mlp']}")
    print(f"📊 Learning Rate: {DAPT_CONFIG['learning_rate']}")
    print(f"📊 Epochs: {DAPT_CONFIG['num_epoch']}")

    opalInstance.train_and_save_model(
        model_class=OpalGPT,
        config=DAPT_CONFIG,  # Use LoRA-enabled configuration
        device=device,
        tokenizer=sp,
        corpus_text=opalInstance.loadTrainingData(token_model=OpalConstants.TOKENIZER_MODEL_PATH),
        checkpoint_path=OpalConstants.CHECKPOINT_PATH,
        num_epochs=DAPT_CONFIG["num_epoch"],  # Use config epochs
        log_to_tensorboard=True,
        lr=DAPT_CONFIG["learning_rate"],     # Use config learning rate
        weight_decay=DAPT_CONFIG["weight_decay"],
        start_fresh=start_fresh,
        eval_iter=5,        # DEBUG: Small evaluation for fast debugging  
        eval_freq=10,       # DEBUG: Evaluate every 10 steps for rapid issue detection
        batch_size=TRAINING_CONFIG["batch_size"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA Domain Adaptation for OpalGPT model')
    parser.add_argument('--start-fresh', 
                        default=False,
                        action='store_true', 
                        help='Start LoRA domain adaptation from scratch, deleting previous checkpoint')
    args = parser.parse_args()

    print(f"🎯 LoRA Domain Adaptation - Command line arguments: {args}")

    if args.start_fresh:
        start_fresh = True
        # Delete all files in the checkpoint directory
        shutil.rmtree(OpalConstants.CHECKPOINT_DIR)
        print("🚀 Starting LoRA domain adaptation from scratch")  
    else:
        start_fresh = False
        print("🔄 Continuing LoRA domain adaptation from previous checkpoint")

    if __name__ == "__main__":
        model_dapt_train(start_fresh=start_fresh)
