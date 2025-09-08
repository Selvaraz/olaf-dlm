
class OpalConstants:
    """Class to hold all constants used in Opal"""

    # RUNTIME PATHS
    #SYSTEM_ROOT_PATH = "/home/selvaraj/"
    SYSTEM_ROOT_PATH= ""
    RUNTIME_ROOT_PATH = f"{SYSTEM_ROOT_PATH}/workspace"
    CHECKPOINT_DIR = f"{RUNTIME_ROOT_PATH}/checkpoints"
    TENSORBOARD_RUN_DIR = f"{CHECKPOINT_DIR}/runs"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint-latest.pt"

    ## INPUTS
    DATA_DIR = f"{RUNTIME_ROOT_PATH}/dataset"
    PRETOKENIZED_DATA_PATH = f"{DATA_DIR}/pretokenized_unified_data/olaf_tokenizer_09062025_45M.pt"
    # PRETRAIN_DATA_PATH = f"{DATA_DIR}/consolidated_ascii_files"
    PRETRAIN_DATA_PATH = f"{DATA_DIR}/unified_data_corpus.txt"
    TOKENIZER_MODEL_PATH = f"{RUNTIME_ROOT_PATH}/tokenizer/olaf_tokenizer_09062025_45M.model"
    FINETUNE_DATA_PATH = f"{DATA_DIR}/QA_finetune_final.jsonl"

    

class OpalConstants_:
    """Class to hold all constants used in Opal"""

    # RUNTIME PATHS
    RUNTIME_ROOT_PATH = "/home/selvaraj/MyModels"
    CHECKPOINT_DIR = f"{RUNTIME_ROOT_PATH}/checkpoints"
    TENSORBOARD_RUN_DIR = f"{CHECKPOINT_DIR}/runs"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint-latest.pt"
    PRETOKENIZED_DATA_PATH = f"{CHECKPOINT_DIR}/pretokenized_data.pt"

    ## INPUTS
    DATA_DIR = f"{RUNTIME_ROOT_PATH}/dataset"
    PRETRAIN_DATA_PATH = f"{DATA_DIR}/corpus_olaf.txt"
    TOKENIZER_MODEL_PATH = f"{RUNTIME_ROOT_PATH}/checkpoints/olaf_tokenizer_073125.model"
    FINETUNE_DATA_PATH = f"{DATA_DIR}/QA_normalized.jsonl"

    