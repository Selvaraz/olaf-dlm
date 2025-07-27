import torch
import sentencepiece as spm
from ..transformer.OpalGPTModel import OpalGPT
from ..opalmain.opal_trainer import Opal
from ..utils.opal_constants import OpalConstants
from ..utils.training_utils import estimate_training_time_from_config
import torch
import time
import multiprocessing

# Model config
GPT_CONFIG_OPAL_FINAL = {
    "vocab_size": 8000,       # Enough for Cisco CLI + logs
    "context_length": 768,    # Can handle moderate logs
    "emb_dim": 192,           # Rich token embeddings
    "n_heads": 6,             # Balanced attention diversity
    "n_layers": 10,           # Enough depth for summarization
    "drop_rate": 0.1,
    "transformer_drop_rate": 0.1,
    "attention_drop_rate": 0.1,
    "qkv_bias": False
}

# OPAL_MODEL_CONFIG = {
#     "vocab_size": 8000,
#     "context_length": 256,
#     "emb_dim": 128,
#     "n_heads": 8,
#     "n_layers": 10,
#     "drop_rate": 0.1,
#     "transformer_drop_rate": 0.1,
#     "attention_drop_rate": 0.1,
#     "qkv_bias": False
# }

OPAL_MODEL_CONFIG = GPT_CONFIG_OPAL_FINAL

# Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load(OpalConstants.TOKENIZER_MODEL_PATH)
# Create Opal instance with tokenizer
opalInstance = Opal(config=OPAL_MODEL_CONFIG, tokenizer=sp)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_pretrain_test():
    opalInstance.train_and_save_model(
        model_class=OpalGPT,
        config=OPAL_MODEL_CONFIG,
        device=device,
        tokenizer=sp,
        corpus_text=opalInstance.loadTrainingData(token_model=OpalConstants.TOKENIZER_MODEL_PATH),
        checkpoint_path=OpalConstants.CHECKPOINT_PATH,
        num_epochs=3,
        log_to_tensorboard=True,
    )

def model_train_test_():
    start_time = time.time()

    # Set the random seed for reproducibility
    torch.manual_seed(123)

    # Let PyTorch use all available CPU threads (important for CPU training)
    # By default, PyTorch may not use all CPU cores. Since you have 36 cores,
    # we explicitly set the number of threads to the CPU count (or 36, whichever is smaller).
    torch.set_num_threads(min(36, multiprocessing.cpu_count()))

    # Try to load existing model checkpoint if available
    try:
        print(f"Attempting to load model checkpoint from {OpalConstants.CHECKPOINT_PATH}...")
        model, optimizer_state_dict, scheduler_state_dict, epoch, train_losses, val_losses, config, tokenizer_model = \
            opalInstance.load_model_checkpoint(OpalGPT, OpalConstants.CHECKPOINT_PATH, device)
        print("Successfully loaded model checkpoint!")
    except Exception as e:
        print(f"No checkpoint found or error loading checkpoint: {str(e)}")
        print(e)
        # Exit the program
        exit(1)
    
    num_epochs = 10

    # Create an AdamW optimizer that will update the model's parameters
    # based on the gradients computed during backpropagation.
    # The learning rate is set to 0.0004 and the weight decay is set to 0.1.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Load the optimizer state if available
    if optimizer_state_dict:
        optimizer.load_state_dict(optimizer_state_dict)

    # Step 1: Load the training data (complete text corpus as a string)
    print("Loading training data...")
    total_corpus = opalInstance.loadTrainingData(token_model=OpalConstants.TOKENIZER_MODEL_PATH)

    # Step 2: Split the corpus into training and validation datasets
    train_ratio = 0.90
    split_idx = int(train_ratio * len(total_corpus))
    train_data = total_corpus[:split_idx]
    val_data = total_corpus[split_idx:]
    total_tokens = len(sp.encode(total_corpus))

    # Get the total time it will take
    training_time = estimate_training_time_from_config(OPAL_MODEL_CONFIG, total_tokens, batch_size=8)
    print("\nEstimated Training Time:")
    print("------------------------")
    print(f"  {'Parameter':<20} {'Value':<20}")
    print("  ---------------------  --------------------")
    print(f"  {'Number of Epochs':<20} {num_epochs:<20}")
    print(f"  {'Model Parameters (M)':<20} {training_time['total_params_million']:<20.2f}")
    print(f"  {'Steps per Epoch':<20} {training_time['steps_per_epoch']:<20}")
    print(f"  {'Epoch Time (minutes)':<20} {training_time['epoch_time_minutes']:<20.2f}")
    print(f"  {'Total Time (hours)':<20} {training_time['total_time_hours']:<20.2f}")
    print("")


    print("Creating training and validation loaders...")
    training_loader = opalInstance.createOpalDataLoader(
        txt=train_data,
        max_length=OPAL_MODEL_CONFIG["context_length"],
        stride=OPAL_MODEL_CONFIG["context_length"],
        drop_last=False,
        shuffle=True,          # Enabled shuffling for better training performance
    )

    # Same settings for validation loader
    print("Creating validation loader...")
    val_loader = opalInstance.createOpalDataLoader(
        txt=val_data,
        max_length=OPAL_MODEL_CONFIG["context_length"],
        stride=OPAL_MODEL_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
    )

    # Sanity check to ensure enough tokens exist for training and validation
    if total_tokens * train_ratio < OPAL_MODEL_CONFIG["context_length"]:
        print("⚠️ Not enough tokens for the training loader. "
              "Try to lower the `OPAL_MODEL_CONFIG['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < OPAL_MODEL_CONFIG["context_length"]:
        print("⚠️ Not enough tokens for the validation loader. "
              "Try to lower the `OPAL_MODEL_CONFIG['context_length']` or "
              "decrease the `training_ratio`")

    # Print the shapes of the first few batches for debugging
    for idx, (x, y) in enumerate(training_loader):
        print(f"Batch {idx}: {x.shape} {y.shape}")
        if idx > 1:  # Limit printing to first 2 batches
            break

    # Calculate total number of tokens in training and validation sets
    train_tokens = sum(input_batch.numel() for input_batch, _ in training_loader)
    val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)

    # After creating training_loader, letx calculate the number of steps, which we will use 
    # for scheduler
    total_steps = num_epochs * len(training_loader)

    # Create a scheduler to adjust learning rate during training. The scheduler will 
    # use a cosine annealing schedule to reduce the learning rate as training progresses.
    # This helps in fine-tuning the model and can lead to better performance.
    # The scheduler will use the total number of steps as the maximum number of steps.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Load the scheduler state if available    
    if scheduler_state_dict:
        scheduler.load_state_dict(scheduler_state_dict)

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # Calculate loss on training and validation sets before training
    print("Calculating loss on training and validation sets before training...")
    loss_loader_start = time.time()
    # with torch.no_grad():
    #     print("Time taken to calculate loss on training set before training:" , time.time() - loss_loader_start)
    #     train_loss = opalInstance.calc_loss_loader(training_loader, model, device)
    #     print("Time taken to calculate loss on validation set before training:", time.time() - loss_loader_start)
    #     val_loss = opalInstance.calc_loss_loader(val_loader, model, device)
    #     print("Training loss:", train_loss)
    #     print("Validation loss:", val_loss)
    
    loss_loader_end = time.time()
    print("Time taken to calculate loss on training and validation sets before training:", loss_loader_end - loss_loader_start)

    # Begin model training using the prepared DataLoaders and optimizer
    # The training process now benefits from multi-core CPU usage via DataLoader
    print("Start training...")
    train_losses, val_losses, tokens_seen = opalInstance.train_model_simple(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        tokenizer=sp,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_iter=5,
        eval_freq=5,
        start_context="how are you"
    )

    # Plot losses after training
    epochs_tensor = torch.linspace(1, num_epochs, len(train_losses))

    #Save the model
    checkpoint_path =opalInstance.save_model_checkpoint(OPAL_MODEL_CONFIG, 
                                    model, optimizer, scheduler, 
                                    num_epochs, 
                                    train_losses, val_losses, 
                                    OpalConstants.TOKENIZER_MODEL_PATH)
    opalInstance.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, checkpoint_path)




def calculate_model_size():
    model = OpalGPT(OPAL_MODEL_CONFIG)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of trainable model parameters: ", total_params)

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

def transformer_test():
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(opalInstance.text_to_token_ids(txt1).squeeze(0))
    batch.append(opalInstance.text_to_token_ids(txt2).squeeze(0))
    batch = torch.stack(batch, dim=0)

    model = OpalGPT(OPAL_MODEL_CONFIG)
    out = model(batch)
    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)


    with torch.no_grad():
        logits = model(batch)
    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    targets = batch[:, 1:]  # Shifted input for targets
    print("inputs", batch)
    print("Targets:\n", targets)
    print(f"Targets batch 1: {opalInstance.token_ids_to_text(targets[0])}")
    print(f"Outputs batch 1:"
    f" {opalInstance.token_ids_to_text(token_ids[0].flatten())}")

    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)


    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Cross-entropy loss:", loss.item())
    print(loss)

calculate_model_size()

model_pretrain_test()