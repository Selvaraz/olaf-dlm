
class OpalConstants:
    """Class to hold all constants used in Opal"""
    RUNTIME_ROOT_PATH = "/home/selvaraj/runtime"
    CHECKPOINT_DIR = f"{RUNTIME_ROOT_PATH}/checkpoints"
    DATA_DIR = f"{RUNTIME_ROOT_PATH}/data"
    TENSORBOARD_RUN_DIR = f"{RUNTIME_ROOT_PATH}/runs"
    
    PRETRAIN_DATA_PATH = f"{DATA_DIR}/all-conversation-books.txt"
    TOKENIZER_MODEL_PATH = f"{DATA_DIR}/opal_tokenizer.model"
    PRETOKENIZED_DATA_PATH = f"{DATA_DIR}/pretokenized_data.pt"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint-latest.pt"
    