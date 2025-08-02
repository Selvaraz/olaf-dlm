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
opalInstance = Opal(config=OPAL_MODEL_CONFIG, tokenizer=sp, is_finetune=True, finetune_data_path=OpalConstants.FINETUNE_DATA_PATH)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_pretrain_test(start_fresh=False):
    VOCAB_SIZE = sp.get_piece_size()

    if (VOCAB_SIZE != OPAL_MODEL_CONFIG["vocab_size"]):
        raise ValueError("Vocabulary size mismatch between tokenizer and model")

    opalInstance.train_and_save_model(
        model_class=OpalGPT,
        config=OPAL_MODEL_CONFIG,
        device=device,
        tokenizer=sp,
        checkpoint_path=OpalConstants.CHECKPOINT_PATH,
        num_epochs=OPAL_MODEL_CONFIG["num_epoch"],
        log_to_tensorboard=True,
        lr=OPAL_MODEL_CONFIG["learning_rate"],
        weight_decay=OPAL_MODEL_CONFIG["weight_decay"],
        start_fresh=start_fresh,
        eval_iter=250,
        eval_freq=500,
    )


if __name__ == "__main__":
    model_pretrain_test(start_fresh=False)
