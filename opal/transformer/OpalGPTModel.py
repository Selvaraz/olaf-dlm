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
        # tie embeddings
        self.out_head.weight = self.token_embeddings.weight

        self.cfg = cfg

        self.apply(self._init_weights)

    # Initialize the weights of the model, the reason for this 
    # initialization is to make the model learnable and to 
    # prevent the model from learning the same pattern in the 
    # input data.
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_token_ids, labels=None, past_key_values=None, use_cache=False):
        """
        Compute the output of the GPT model given an input sequence.

        Parameters
        ----------
        in_idx : torch.LongTensor
            Input sequence of shape (batch_size, sequence_length)
        labels : torch.LongTensor
            Labels of shape (batch_size, sequence_length)

        Returns
        -------
        logits : torch.FloatTensor
            Output logits of shape (batch_size, sequence_length, vocab_size)
        """
        # cache: added (past_key_values, use_cache) to support fast decode with KV-cache.
        #        - past_key_values: list of (past_key, past_value) per layer, or None
        #        - when use_cache=True, we also return present_key_values for the next step

        # Get the batch size and sequence length from the input token IDs
        batch_size, seq_len = input_token_ids.shape
        
        # Get the token embeddings for the input token IDs
        #print("Input token IDs shape:", input_token_ids.shape)
        #print("Token embeddings shape:", self.cfg["vocab_size"], self.cfg["emb_dim"])
        #print("Input token IDs min:", input_token_ids.min().item(), "max:", input_token_ids.max().item())

        tok_embeds = self.token_embeddings(input_token_ids)
        
        # Get the positional embeddings for the input token IDs
        pos_embeds = self.positional_embeddings(
            torch.arange(seq_len, device=input_token_ids.device)
        )
        
        # Add the token embeddings and positional embeddings
        x = tok_embeds if self.cfg.get("use_rope", True) else (tok_embeds + pos_embeds)
        
        # Apply dropout to the embeddings
        x = self.drop_embeddings(x)
        
        # cache: normalize past_key_values length and structure
        if past_key_values is None:
            past_key_values = [(None, None)] * len(self.transformers_block)

        # Pass the embeddings through the transformer blocks
        # cache: we need to loop blocks to pass per-layer caches in/out (can't call nn.Sequential directly).
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformers_block):
            pk, pv = past_key_values[i]
            x, pk_new, pv_new = block(x, past_key=pk, past_value=pv, use_cache=use_cache)
            if use_cache:
                present_key_values.append((pk_new, pv_new))
        
        # Apply layer normalization to the output of the transformer blocks
        x = self.final_norm(x)
        
        # Get the logits for the output of the final layer normalization
        logits = self.out_head(x)
        
        loss = None
        # Return the logits
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(logits.view(-1, self.cfg["vocab_size"]), labels.view(-1))
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # cache: return present_key_values only when use_cache=True (inference-time)
        if use_cache:
            return {"logits": logits, "loss": loss, "present_key_values": present_key_values}
        else:
            return {"logits": logits, "loss": loss}
