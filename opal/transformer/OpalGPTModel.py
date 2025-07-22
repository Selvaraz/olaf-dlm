import torch
import torch.nn as nn
from .OpalTransformer import OpalTransformerBlock
from .OpalLayerNormalization import OpalLayerNormalization

class OpalGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Embeddings layer for tokens and positions in the input sequence
        self.token_embeddings = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Positional embeddings to encode the position of each token in the sequence
        # This helps the model understand the order of tokens in the sequence
        # without relying on the sequence order in the input data.
        # The positional embeddings are learned during training.
        self.positional_embeddings = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout layer to prevent overfitting
        # Dropout is a regularization technique that randomly sets a fraction of the input units to
        # zero during training, which helps prevent overfitting.
        self.drop_embeddings = nn.Dropout(cfg["drop_rate"])

        # Create a sequence of transformer blocks
        # Each block consists of multi-head attention and feed-forward layers
        self.transformers_block = nn.Sequential(
        *[OpalTransformerBlock(cfg)
            for _ in range(cfg["n_layers"])]
        )

        # Final layer normalization and output head
        # The final layer normalization is applied to the output of the transformer blocks
        self.final_norm = OpalLayerNormalization(cfg["emb_dim"])

        # The output head is a linear layer that maps the output of the final layer
        # normalization to the vocabulary size
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        """
        Compute the output of the GPT model given an input sequence.

        Parameters
        ----------
        in_idx : torch.LongTensor
            Input sequence of shape (batch_size, sequence_length)

        Returns
        -------
        logits : torch.FloatTensor
            Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embeddings(in_idx)
        pos_embeds = self.positional_embeddings(
        torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_embeddings(x)
        x = self.transformers_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits