
class OpalConstants:
    """Class to hold all constants used in Opal"""

    # RUNTIME PATHS FOR LOCAL TESTING (BACKUP)
    LOCAL_CHECKPOINT_PATH = "/Users/selvmani/OLAF/checkpoints/45M/checkpoints/20250909/20250908_141411"
    LOCAL_TOKENIZER_PATH = f"{LOCAL_CHECKPOINT_PATH}/opal_tokenizer.model"
    
    # RUNTIME PATHS
    #SYSTEM_ROOT_PATH = "/Users/selvmani/Runs"
    SYSTEM_ROOT_PATH= ""
    RUNTIME_ROOT_PATH = f"{SYSTEM_ROOT_PATH}/workspace"
    CHECKPOINT_DIR = f"{RUNTIME_ROOT_PATH}/pretrain_checkpoints"
    CHECKPOINT_NAME = "checkpoint-latest-pretrain.pt"
    TENSORBOARD_RUN_DIR = f"{CHECKPOINT_DIR}/runs"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/{CHECKPOINT_NAME}"
    CHECKPOINT_NAME 
    

    ## INPUTS
    # /workspace/dataset
    DATA_DIR = f"{RUNTIME_ROOT_PATH}/dataset"
    FINETUNE_TEST_DATA_PATH = f"{DATA_DIR}/unified_finetune_corpus.jsonl"
    # /workspace/dataset/pretokenized_unified_data/olaf_tokenizer_09062025_45M.pt
    #PRETOKENIZED_DATA_PATH = f"{DATA_DIR}/pretokenized_unified_data/olaf_tokenizer_09062025_45M.pt"
    DATASET_FILE_NAME = "olaf_unified_corpus_12G_10052025"
    PRETOKENIZED_DATA_PATH = f"{DATA_DIR}/{DATASET_FILE_NAME}.pt"
    # PRETRAIN_DATA_PATH = f"{DATA_DIR}/consolidated_ascii_files"
    # /workspace/dataset/unified_data_corpus.txt
    PRETRAIN_DATA_PATH = f"{DATA_DIR}/{DATASET_FILE_NAME}.txt"
    TOKENIZER_MODEL_PATH = f"{RUNTIME_ROOT_PATH}/tokenizer/olaf_11G_unified_unigram_45M.model"
    #FINETUNE_DATA_PATH = f"{DATA_DIR}/QA_finetune_final.jsonl"

    

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

    