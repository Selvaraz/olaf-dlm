import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..dataloader.OpalDataSet import OpalDataset
from torch.utils.data import Dataset, DataLoader

class Opal:
    def __init__(self, config, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer

    def createOpalDataLoader(self, txt, batch_size=4, max_length=256, stride=128, 
                          shuffle=True, drop_last=True, num_workers=0):
        """
        Creates a DataLoader for the OpalDataset with the specified parameters.

        Args:
            txt (str): The input text to be tokenized and used for creating the dataset.
            batch_size (int, optional): Number of samples per batch to load. Default is 4.
            max_length (int, optional): Maximum sequence length for each sample. Default is 256.
            stride (int, optional): Number of tokens to skip between sequences. Default is 128.
            shuffle (bool, optional): If True, data will be reshuffled at every epoch. Default is True.
            drop_last (bool, optional): If True, drops the last incomplete batch. Default is True.
            num_workers (int, optional): Number of subprocesses to use for data loading. Default is 0.

        Returns:
            DataLoader: A DataLoader object for the OpalDataset.
        """

        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = OpalDataset(txt, tokenizer, max_length, stride)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        
        return dataloader

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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "data", "the-verdict.txt"), "r") as f:
            txt = f.read()
        return txt

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        # The line below converts the list of token ids to a tensor and
        # unsqueezes it to shape (1, seq_len). The reason we need to unsqueeze
        # it is because the model expects a batch dimension, but we are only
        # generating text for one input at a time. So we need to create a
        # "batch" of size 1 by adding a dimension of size 1 at the beginning.
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor
    
    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())
    
    def generate(self, model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

         
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

    def train_model_simple(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                        eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
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
        logits = model(input_batch)
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
    
    def plot_losses(self, epochs_seen, tokens_seen, train_losses, val_losses):
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
        plt.savefig("loss-plot.pdf")
        plt.show()
    