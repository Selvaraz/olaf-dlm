
class OpalConstants:
    """Class to hold all constants used in Opal"""

    # RUNTIME PATHS
    RUNTIME_ROOT_PATH = "/workspace"
    CHECKPOINT_DIR = f"{RUNTIME_ROOT_PATH}/checkpoints"
    TENSORBOARD_RUN_DIR = f"{CHECKPOINT_DIR}/runs"
    CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint-latest.pt"
    PRETOKENIZED_DATA_PATH = f"{CHECKPOINT_DIR}/pretokenized_data.pt"

    ## INPUTS
    DATA_DIR = f"{RUNTIME_ROOT_PATH}/dataset"
    PRETRAIN_DATA_PATH = f"{DATA_DIR}/corpus_olaf.txt"
    #PRETRAIN_DATA_PATH = "/workspace/training/tokenizer_data/corpus_olaf_cpu_150k.txt"
    #PRETRAIN_DATA_PATH = f"/home/selvaraj/olaf-dlm/opal/opalmain/data/the-verdict.txt"
    TOKENIZER_MODEL_PATH = f"{RUNTIME_ROOT_PATH}/olaf-dlm/opal/tokenizer/olaf_tokenizer.model"
    

    