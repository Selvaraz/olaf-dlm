import torch
import argparse
import sentencepiece as spm
from ..transformer.OpalGPTModel import OpalGPT
from ..opalmain.opal_trainer import Opal
from ..utils.opal_constants import OpalConstants
from ..utils.training_utils import estimate_training_time_from_config
from ..config.opal_config import OPAL_MODEL_CONFIG
import torch
import time
import multiprocessing
import shutil

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(OpalConstants.TOKENIZER_MODEL_PATH)
# Create Opal instance with tokenizer
opalInstance = Opal(config=OPAL_MODEL_CONFIG, tokenizer=sp)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_pretrain_test(prompt):
    VOCAB_SIZE = sp.get_piece_size()

    if (VOCAB_SIZE != OPAL_MODEL_CONFIG["vocab_size"]):
        raise ValueError("Vocabulary size mismatch between tokenizer and model")

    token_model=OpalConstants.TOKENIZER_MODEL_PATH
    model, optimizer_state_dict, scheduler_state_dict, epoch, train_losses, val_losses, _ = opalInstance.load_model_checkpoint(model_class=OpalGPT, 
            checkpoint_path=OpalConstants.CHECKPOINT_PATH, device="cpu")
    opalInstance.generate_with_topk(
        model=model,
        tokenizer=sp,
        device=device,
        start_context=prompt,
        top_k=OPAL_MODEL_CONFIG["top_k"] if "top_k" in OPAL_MODEL_CONFIG else 50,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train OpalGPT model')
    parser.add_argument('--prompt', 
                        type=str,
                        default="",
                        help='prompt text')
                        
    args = parser.parse_args()

    print(f"-- Command line arguments: {args}")

    if args.prompt == "":
        print("-- Provide any prompt text -- ")
        exit(1)

    model_pretrain_test(prompt=args.prompt)
