from datetime import datetime
import multiprocessing
import time
import os
import shutil
import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..dataloader.OpalDataSet import OpalDataset
from torch.utils.data import Dataset, DataLoader
from ..utils.opal_constants import OpalConstants
import sentencepiece as spm
from tqdm import tqdm

class Opal:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        
    def createOpalDataLoader(
        self,
        txt: str,
        batch_size: int = None,
        max_length: int = 1280,
        stride: int = 256,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = None,
        device: str = "cpu"
    ):
        """
        Creates a DataLoader for the OpalDataset using optimal settings based on CPU cores.

        Args:
            txt (str): Raw input text.
            batch_size (int, optional): If None, automatically set to 8 for >16 cores, else 4.
            max_length (int): Maximum token sequence length per sample.
            stride (int): Overlap between chunks.
            shuffle (bool): Whether to shuffle dataset each epoch.
            drop_last (bool): Drop last batch if incomplete.
            num_workers (int, optional): If None, automatically set to available CPU cores.
            device (str): Device to put data on ('cpu' or 'cuda').

        Returns:
            DataLoader: PyTorch DataLoader optimized for your CPU.
        """

        if self.tokenizer is None:
            raise ValueError("Tokenizer must be provided when creating Opal instance.")

        # Determine number of workers based on CPU cores
        available_cores = multiprocessing.cpu_count()
        if num_workers is None:
            # Use all available cores, but cap at 16 for very large servers to avoid overhead
            num_workers = available_cores if available_cores <= 36 else 36

        # Dynamically set batch_size based on available cores if not provided
        if batch_size is None:
            if available_cores >= 32:
                batch_size = 8
            elif available_cores >= 16:
                batch_size = 6
            else:
                batch_size = 4

        # Print out chosen parameters for transparency
        print(f"Creating DataLoader with {num_workers} workers, batch_size={batch_size}, prefetch_factor=4")

        dataset = OpalDataset(
            txt=txt,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=stride,
            device=device
        )

        # Use persistent_workers=True and prefetch_factor=4 to reduce worker startup overhead
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=4 if num_workers > 0 else 2
        )
    
    def createOpalDataLoader_v0(
        self,
        txt: str,
        batch_size: int = 4,
        max_length: int = 1280,
        stride: int = 256,
        shuffle: bool = True,
        drop_last: bool = True,
        num_workers: int = 0,
        device: str = "cpu"
    ):
        """
        Creates a DataLoader for the OpalDataset using the preloaded SentencePiece tokenizer.
        """

        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Please pass a SentencePieceProcessor to the Opal constructor.")

        dataset = OpalDataset(
            txt=txt,
            tokenizer=self.tokenizer,
            max_length=max_length,
            stride=stride,
            device=device
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded).unsqueeze(0)

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())

    def loadTrainingData(self):
        """
        Loads training data from a text file.

        Reads the content of the file "the-verdict.txt" located in the "data" directory
        relative to the current file's directory. The contents of the file are returned
        as a single string.

        Returns:
            str: The content of the text file as a string.
        """
        txt = None
        import os
        import git
        from pathlib import Path

        repo = git.Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)
        repo_dir = Path(repo.git.rev_parse("--show-toplevel"))

        # Move one level up from repo top-level
        parent_dir = repo_dir.parent  

        # Construct path
        #file_path = parent_dir / "data" / "tokenizer_text" / "network_tokenizer_text_v1.txt"
        #file_path = parent_dir / "sample_data"  / "the-verdict.txt"
        file_path = OpalConstants.PRETRAIN_DATA_PATH

        with open(file_path, "r") as f:
            txt = f.read()

        return txt

    # Ex: top_p=0.9 and optionally temperature > 0.7
    def generate(self, model, idx, max_new_tokens, context_size, 
                    temperature=0.0, top_k=None, top_p=None, eos_id=None):

        # The following loop generates one token at a time, for a total
        # of max_new_tokens iterations. At each iteration, the model
        # is fed the current sequence (idx) and generates a new token.
        # The new token is then appended to the current sequence, and
        # the loop continues until max_new_tokens tokens have been
        # generated.
        for _ in range(max_new_tokens):
            # Get the last `context_size` tokens as input (context window)
            idx_cond = idx[:, -context_size:]

            # Perform inference to get logits for the next token
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]  # Take logits of the last token position

            # 🔹 New: Apply temperature scaling (makes probabilities sharper or smoother)
            if temperature > 0.0:
                logits = logits / temperature

            # 🔹 Optional: Top-k filtering (keep only the top-k highest probability tokens)
            if top_k is not None:
                # Get top-k logits
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]  # Smallest value among top-k
                # Replace logits below the kth value with -inf so they are ignored
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                )

            # 🔹 New: Top-p (nucleus) filtering
            # Instead of a fixed k, this dynamically keeps the smallest set of tokens
            # whose cumulative probability ≤ p.
            if top_p is not None:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                # Convert logits to probabilities
                probs = torch.softmax(sorted_logits, dim=-1)
                # Compute cumulative probabilities
                cumulative_probs = torch.cumsum(probs, dim=-1)

                # Identify tokens where cumulative probability > p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift mask so that the first token above p is kept
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0  # Always keep the highest probability token

                # Set logits of removed tokens to -inf
                logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

            # Choose next token
            if temperature > 0.0:
                # If temperature > 0, sample from probability distribution
                probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
                idx_next = torch.multinomial(probs, num_samples=1)  # Random sampling
            else:
                # Greedy decoding: pick the token with highest logit
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop early if EOS (end-of-sequence) token is generated
            if eos_id is not None and idx_next == eos_id:
                break

            # Append the predicted token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)  # Sequence grows by 1 token

        return idx


    def generate_v0(self, model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

         
        # The following loop generates one token at a time, for a total
        # of max_new_tokens iterations. At each iteration, the model
        # is fed the current sequence (idx) and generates a new token.
        # The new token is then appended to the current sequence, and
        # the loop continues until max_new_tokens tokens have been
        # generated.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

            if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx

    def train_model_simple(self, model, train_loader, val_loader, 
                        optimizer, scheduler, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            # Create a progress bar for the training data
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

            # This loop iterates over the training data for the specified number of epochs.
            # Since the DataLoader is set to drop the last batch if it is not full, the number of
            # iterations is equal to the total number of samples in the dataset divided by the
            # batch size, rounded down. To calculate the number of iterations, we can use the
            # following formula:
            #
            # num_iterations = math.floor(len(dataset) / batch_size)
            #
            # For example, if the dataset has 1000 samples and the batch size is 32, the number of
            # iterations is:
            #
            # num_iterations = math.floor(1000 / 32) = 31
            #
            # Therefore, the model will be trained on 31 batches of 32 samples each, for a total of
            # 992 samples (31 * 32 = 992).
            #
            # The remaining 8 samples (1000 - 992 = 8) will be dropped, since the last batch is not
            # full.
            for batch_idx, (input_ids, targets) in enumerate(pbar):
                #for input_batch, target_batch in train_loader:
                # Reset loss gradients from previous batch iteration, so we start fresh
                optimizer.zero_grad()  

                # Move input and target tensors to the specified device
                input_ids, targets = input_ids.to(device), targets.to(device)

                loss = self.calc_loss_batch(input_ids, targets, model, device)

                # Backpropagate the loss
                loss.backward()  

                # clip gradients to prevent exploding gradients, this is optional
                # But it is recommended to use it.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update model weights using loss gradients
                optimizer.step()  

                # Update learning rate
                if scheduler:
                    scheduler.step()  

                tokens_seen += input_ids.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    total_steps = len(train_loader) * num_epochs
                    print(f"Ep {epoch+1} (Step {global_step+1:06d}/{total_steps:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            # self.generate_and_print_sample(
            #     model, tokenizer, device, start_context
            # )

            self.generate_with_topk(
                model, tokenizer, device, start_context, top_k=50
            )

        return train_losses, val_losses, track_tokens_seen


    def evaluate_model(self, model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
            val_loss = self.calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
        return train_loss, val_loss


    def generate_with_topk(self, model, tokenizer, device, start_context, top_k):
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate(model, encoded, 25, 
                                      context_size, top_k=top_k, 
                                      temperature=1.5)
            decoded_text = self.token_ids_to_text(token_ids)
            print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()

            
    def _generate_and_print_sample(self, model, tokenizer, device, start_context):
        model.eval()
        context_size = model.positional_embeddings.weight.shape[0]
        encoded = self.text_to_token_ids(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generate_text_simple(
                model=model, idx=encoded,
                max_new_tokens=50, context_size=context_size
            )
            decoded_text = self.token_ids_to_text(token_ids)
            print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()


    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))
    
    def calc_loss_batch(self, input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch  )
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def calc_loss_loader(self,data_loader, model, device, num_batches=None):
        """
        Calculate total loss over a portion of a data loader.

        Args:
        - data_loader (DataLoader): A PyTorch DataLoader.
        - model (nn.Module): A PyTorch neural network model.
        - device (torch.device): The device (e.g. GPU or CPU) to use for computations.
        - num_batches (int, optional): The number of batches to use from the data loader.
            If None, use all batches in the data loader. Defaults to None.

        Returns:
        - total_loss (float): The total loss over the given number of batches.
        """
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def generate_text_simple(self, model, idx, max_new_tokens, context_size):
        print("Generating text with context size:", max_new_tokens)
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]
            
            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond)
            
            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
    
    def save_model_checkpoint(self, config, model, optimizer, scheduler, epoch, train_losses, 
                              val_losses, tokenizer_model):
        """
        Saves a trained model checkpoint including model state, optimizer state,
        epoch, training history, and config.

        Args:
            model (torch.nn.Module): The trained OpalGPT model instance.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            epoch (int): Current epoch number (for resuming training).
            train_losses (list): Training loss history.
            val_losses (list): Validation loss history.
            config (dict): Model configuration dictionary.
            save_dir (str): Directory to save checkpoints.

        Returns:
            str: Path to the saved checkpoint file.
        """
        #os.makedirs(OpalConstants.CHECKPOINT_DIR, exist_ok=True)

        # Create a unique file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # checkpoint_path = os.path.join(OpalConstants.CHECKPOINT_DIR, 
        #                             f"opal_gpt_checkpoint_{timestamp}.pt")
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": config,
            "tokenizer_model": tokenizer_model,
        }


        # Create a directory with current date and save the model inside the path
        date_dir = datetime.now().strftime("%Y%m%d")
        checkpoint_dir = os.path.join(OpalConstants.CHECKPOINT_DIR, f"opal_checkpoint_{date_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"opal_gpt_checkpoint_{timestamp}.pt")
        
        torch.save(checkpoint, checkpoint_path)

        # Copy the tokenizer model to the checkpoint directory
        tokenizer_model_path = os.path.join(checkpoint_dir, "opal_tokenizer.model")
        shutil.copyfile(tokenizer_model, tokenizer_model_path)

        print(f"Model checkpoint saved to {checkpoint_path}")
        # Create a symlink to the latest checkpoint
        symlink_path = os.path.join(OpalConstants.CHECKPOINT_DIR, "checkpoint-latest.pt")

        if os.path.exists(symlink_path):
            os.remove(symlink_path)
        os.symlink(checkpoint_path, symlink_path)
        print(f"Latest checkpoint symlink created at {symlink_path}")

        return checkpoint_path


    def load_model_checkpoint(self, model_class, checkpoint_path, device="cpu"):
        """
        Loads a trained model checkpoint and restores model, optimizer, and training state.

        Args:
            model_class (type): The class of the model (e.g., OpalGPT).
            checkpoint_path (str): Path to the checkpoint file.
            device (str): Device to load model on ('cpu' or 'cuda').

        Returns:
            model (torch.nn.Module): Loaded model with restored weights.
            optimizer_state_dict (dict): State dict for optimizer (can be used to resume training).
            epoch (int): Last epoch from checkpoint.
            train_losses (list): Training loss history.
            val_losses (list): Validation loss history.
            config (dict): Model configuration dictionary.
        """
        checkpoint = {}
        optimizer_state_dict= None
        scheduler_state_dict = None
        if not os.path.isfile(os.path.realpath(checkpoint_path)):
            print(f"Checkpoint {checkpoint_path} not found. Creating new model.")
            model = model_class(config).to(device)
        else:
            print(f"Model loaded from {checkpoint_path}")
            checkpoint = torch.load(os.path.realpath(checkpoint_path), map_location=device)
            # Load model with saved config to ensure same architecture
            config = checkpoint["config"]
            model = model_class(config).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state_dict = checkpoint.get("optimizer_state_dict", None)
            scheduler_state_dict = checkpoint.get("scheduler_state_dict", None)
            model.to(device)
        print(checkpoint)
        return (
            model,
            optimizer_state_dict  if optimizer_state_dict else None,
            scheduler_state_dict if scheduler_state_dict else None,
            checkpoint["epoch"] if "epoch" in checkpoint else 0,
            checkpoint["train_losses"] if "train_losses" in checkpoint else [],
            checkpoint["val_losses"] if "val_losses" in checkpoint else [],
            config,
        )

    def _plot_and_save_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path):
        """
        Helper function to generate and save the loss plot.
        """
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Add a second x-axis for tokens seen
        ax2 = ax1.twiny()
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible for alignment
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to free memory

    def plot_losses(self, epochs_seen, tokens_seen, train_losses, val_losses, checkpoint_path):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        
        #Get directory name from checkpoint path
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Get just the filename without extension
        filename = os.path.splitext(os.path.basename(checkpoint_path))[0]
        plt.savefig(os.path.join(checkpoint_dir, f"{filename}-loss-plot.pdf"))
        plt.show()


    def train_and_save_model(
        self,
        model_class,
        config,
        device,
        tokenizer,
        corpus_text,
        checkpoint_path,
        num_epochs=10,
        batch_size=8,
        train_ratio=0.9,
        lr=4e-4,
        weight_decay=0.1,
        eval_freq=5,
        eval_iter=5,
        start_context="how are you"
    ):
        """
        Reusable utility to train, continue pretraining, or fine-tune a model.

        - Can be used for:
            1. **Pretraining from scratch**
            2. **Continuing pretraining with new corpus (domain adaptation)**
            3. **Fine-tuning on instruction datasets**

        This function handles:
        ✅ Loading model & optimizer/scheduler states if checkpoint exists
        ✅ Creating training/validation splits and DataLoaders
        ✅ Computing total steps and creating CosineAnnealingLR scheduler
        ✅ Training for specified epochs
        ✅ Saving model checkpoints with full training state (model+optimizer+scheduler)
        ✅ Plotting loss curves after training

        Returns:
            final_ckpt (str): Path to the final saved checkpoint.
        """

        start_time = time.time()

        # ----------------------------------------
        # ✅ Set random seed for reproducibility
        # ----------------------------------------
        torch.manual_seed(123)

        # Let PyTorch use all available CPU threads (important for CPU training).
        # By default, PyTorch may not use all CPU cores. Since you have 36 cores,
        # we explicitly set the number of threads to the CPU count (or 36, whichever is smaller).
        torch.set_num_threads(min(36, multiprocessing.cpu_count()))

        # ----------------------------------------
        # ✅ Step 1: Try loading existing checkpoint
        # ----------------------------------------
        try:
            print(f"Attempting to load model checkpoint from {checkpoint_path}...")
            model, optimizer_state_dict, scheduler_state_dict, epoch, train_losses, val_losses, _ = \
                self.load_model_checkpoint(model_class, checkpoint_path, device)
            print("✅ Successfully loaded model checkpoint!")
        except Exception as e:
            print(f"⚠️ No checkpoint found. Training from scratch: {e}")
            # If no checkpoint exists, create a fresh model instance
            model = model_class(config).to(device)
            optimizer_state_dict, scheduler_state_dict = None, None
            train_losses, val_losses, epoch = [], [], 0

        # ----------------------------------------
        # ✅ Step 2: Create optimizer
        # ----------------------------------------
        # AdamW optimizer will update the model's parameters based on gradients.
        # The learning rate and weight decay values can be tuned.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # If checkpoint has optimizer state, load it to resume training seamlessly
        if optimizer_state_dict:
            optimizer.load_state_dict(optimizer_state_dict)

        # ----------------------------------------
        # ✅ Step 3: Load training data and split into train/validation sets
        # ----------------------------------------
        print("Loading training data...")

        # Split corpus into train and validation datasets based on `train_ratio`.
        split_idx = int(train_ratio * len(corpus_text))
        train_data, val_data = corpus_text[:split_idx], corpus_text[split_idx:]

        # Compute total tokens in the corpus for reporting and planning
        total_tokens = len(tokenizer.encode(corpus_text))

        # ----------------------------------------
        # ✅ Step 4: Create DataLoaders for training and validation
        # ----------------------------------------
        print("Creating training and validation loaders...")

        training_loader = self.createOpalDataLoader(
            txt=train_data,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=False,
            shuffle=True,  # Shuffling improves generalization
        )

        val_loader = self.createOpalDataLoader(
            txt=val_data,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=False,
            shuffle=False,
        )

        # ----------------------------------------
        # ✅ Step 5: Calculate total training steps for scheduler
        # ----------------------------------------
        total_steps = num_epochs * len(training_loader)

        # CosineAnnealingLR gradually reduces LR following a cosine curve.
        # T_max is set to the total number of steps so LR anneals over entire training.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        # If resuming training, load scheduler state as well
        if scheduler_state_dict:
            scheduler.load_state_dict(scheduler_state_dict)

        # ----------------------------------------
        # ✅ Optional: Print token statistics for debugging
        # ----------------------------------------
        train_tokens = sum(x.numel() for x, _ in training_loader)
        val_tokens = sum(x.numel() for x, _ in val_loader)

        print(f"Training tokens: {train_tokens}")
        print(f"Validation tokens: {val_tokens}")
        print(f"All tokens: {train_tokens + val_tokens}")

        # ----------------------------------------
        # ✅ Optional: Calculate initial losses (commented for speed)
        # ----------------------------------------
        # print("Calculating loss on training and validation sets before training...")
        # with torch.no_grad():
        #     train_loss = opal_instance.calc_loss_loader(training_loader, model, device)
        #     val_loss = opal_instance.calc_loss_loader(val_loader, model, device)
        #     print(f"Training loss: {train_loss}, Validation loss: {val_loss}")

        # ----------------------------------------
        # ✅ Step 6: Begin training
        # ----------------------------------------
        print("🚀 Start training...")
        train_losses, val_losses, tokens_seen = self.train_model_simple(
            model=model,
            train_loader=training_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            eval_iter=eval_iter,
            eval_freq=eval_freq,
            start_context=start_context,
        )

        # ----------------------------------------
        # ✅ Step 7: Save model checkpoint
        # ----------------------------------------
        # Save model, optimizer, scheduler, and training history.
        final_ckpt = self.save_model_checkpoint(
            config,
            model,
            optimizer,
            scheduler,
            epoch + num_epochs,  # Increment epoch count if resuming training
            train_losses,
            val_losses,
            tokenizer_model=OpalConstants.TOKENIZER_MODEL_PATH,
        )

        # ----------------------------------------
        # ✅ Step 8: Plot training & validation loss curves
        # ----------------------------------------
        epochs_tensor = torch.linspace(epoch + 1, epoch + num_epochs, len(train_losses))
        self.plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, final_ckpt)

        print(f"✅ Training complete! Final checkpoint saved at {final_ckpt}")
        return final_ckpt
        