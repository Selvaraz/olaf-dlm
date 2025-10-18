import torch
import torch.nn as nn
from .OpalTransformer import OpalTransformerBlock
from .OpalLayerNormalization import OpalLayerNormalization
# LoRA Domain Adaptation: Import LoRA utilities for model injection
from ..attention.lora_utils import LoRAConfig, inject_lora_into_model, freeze_base_model_weights

class OpalGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # ✅ MPS INIT FIX: Set proper default dtype and memory management
        if torch.backends.mps.is_available():
            torch.set_default_dtype(torch.float32)  # MPS works best with float32
            print("🍎 MPS DETECTED: Setting float32 as default dtype")

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
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        # tie embeddings
        self.out_head.weight = self.token_embeddings.weight

        self.cfg = cfg

        # LoRA Domain Adaptation: Handle LoRA injection if enabled
        self.lora_config = LoRAConfig.from_dict(cfg)  # LoRA Domain Adaptation: Extract LoRA config
        self.lora_injected_modules = []  # LoRA Domain Adaptation: Track LoRA-injected modules
        
        if self.lora_config.use_lora:
            print(f"🎯 LoRA Domain Adaptation: Injecting LoRA adapters into model...")
            # LoRA Domain Adaptation: Inject LoRA adapters into attention layers
            self, self.lora_injected_modules = inject_lora_into_model(
                model=self,
                lora_config=self.lora_config,
                verbose=True
            )
            
            # LoRA Domain Adaptation: Freeze base model weights, keep only LoRA trainable
            frozen_params = freeze_base_model_weights(self, verbose=True)
            print(f"🎯 LoRA Domain Adaptation: Frozen {frozen_params:,} base parameters")

        # ✅ CRITICAL DEBUG: Print actual model dimensions to detect vocab mismatches
        print(f"🔍 MODEL INITIALIZATION DEBUG:")
        print(f"   Config vocab_size: {cfg['vocab_size']}")
        print(f"   Token embedding vocab size: {self.token_embeddings.num_embeddings}")
        print(f"   Output head vocab size: {self.out_head.out_features}")
        print(f"   Embedding dim: {cfg['emb_dim']}")
        
        # LoRA Domain Adaptation: Print LoRA status
        if self.lora_config.use_lora:
            print(f"   LoRA enabled: rank={self.lora_config.rank}, alpha={self.lora_config.alpha}")
            print(f"   LoRA modules: {len(self.lora_injected_modules)} injected")
        else:
            print(f"   LoRA disabled: using full model training")
        
        # Check for mismatches
        if self.token_embeddings.num_embeddings != cfg["vocab_size"]:
            print(f"🚨 CRITICAL MISMATCH: Token embedding size != config vocab_size")
        if self.out_head.out_features != cfg["vocab_size"]:
            print(f"🚨 CRITICAL MISMATCH: Output head size != config vocab_size")

        # ✅ MPS MEMORY OPTIMIZATION: Initialize tracking variables
        self._mps_step_counter = 0
        self._mps_memory_cleanup_frequency = 50

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

    def _mps_safe_preprocessing(self, input_token_ids, labels=None):
        """MPS-specific preprocessing to ensure tensor compatibility."""
        device = input_token_ids.device
        
        if device.type == 'mps':
            # ✅ MPS FIX: Ensure proper tensor types
            input_token_ids = input_token_ids.long()
            if labels is not None:
                labels = labels.long()
            
            # ✅ MPS FIX: Periodic memory management
            self._mps_step_counter += 1
            if self._mps_step_counter % self._mps_memory_cleanup_frequency == 0:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                torch.mps.synchronize()
        
        return input_token_ids, labels

    def forward(self, input_token_ids, labels=None, past_key_values=None, use_cache=False):
        """
        MPS-optimized forward pass with comprehensive error handling.
        """
        # ✅ MPS PREPROCESSING: Handle MPS-specific requirements
        input_token_ids, labels = self._mps_safe_preprocessing(input_token_ids, labels)
        
        batch_size, seq_len = input_token_ids.shape
        
        # ✅ CRITICAL FIX: Add bounds checking to prevent CUDA index out of bounds
        vocab_size = self.cfg["vocab_size"]
        
        # Check for any out-of-bounds token IDs
        min_token = input_token_ids.min().item()
        max_token = input_token_ids.max().item()
        
        if min_token < 0 or max_token >= vocab_size:
            print(f"🚨 CUDA BOUNDS ERROR DETECTED IN MODEL FORWARD!")
            print(f"   Token range: [{min_token}, {max_token}]")
            print(f"   Vocab size: {vocab_size}")
            print(f"   Input shape: {input_token_ids.shape}")
            print(f"   Out-of-bounds tokens: {(input_token_ids >= vocab_size).sum().item()}")
            print(f"   Negative tokens: {(input_token_ids < 0).sum().item()}")
            
            # Emergency fix: Clamp tokens to valid range
            print(f"   🛡️  EMERGENCY FIX: Clamping tokens to [0, {vocab_size-1}]")
            input_token_ids = torch.clamp(input_token_ids, 0, vocab_size - 1)

        tok_embeds = self.token_embeddings(input_token_ids)
        
        # ✅ BOUNDS CHECK: Ensure sequence length doesn't exceed context_length
        context_length = self.cfg["context_length"]
        if seq_len > context_length:
            print(f"🚨 CRITICAL: Sequence length {seq_len} > context_length {context_length}")
            print(f"   🛡️  EMERGENCY TRUNCATE: Limiting to {context_length}")
            input_token_ids = input_token_ids[:, :context_length]
            seq_len = context_length
            tok_embeds = self.token_embeddings(input_token_ids)
        
        if seq_len > self.positional_embeddings.num_embeddings:
            max_pos_len = self.positional_embeddings.num_embeddings
            input_token_ids = input_token_ids[:, :max_pos_len]
            seq_len = max_pos_len
            tok_embeds = self.token_embeddings(input_token_ids)
        
        pos_embeds = self.positional_embeddings(
            torch.arange(seq_len, device=input_token_ids.device)
        )
        
        x = tok_embeds if self.cfg.get("use_rope", True) else (tok_embeds + pos_embeds)
        x = self.drop_embeddings(x)
        
        # ✅ MPS FIX: Ensure proper dtype consistency
        if input_token_ids.device.type == 'mps':
            x = x.float()  # Ensure float32 for MPS stability
        
        if past_key_values is None:
            past_key_values = [(None, None)] * len(self.transformers_block)

        present_key_values = [] if use_cache else None
        
        try:
            for i, block in enumerate(self.transformers_block):
                pk, pv = past_key_values[i]
                
                # ✅ MPS FIX: Synchronize before each attention block
                if input_token_ids.device.type == 'mps':
                    torch.mps.synchronize()
                
                x, pk_new, pv_new = block(x, past_key=pk, past_value=pv, use_cache=use_cache)
                
                if use_cache:
                    present_key_values.append((pk_new, pv_new))
                    
        except Exception as e:
            print(f"🚨 CRITICAL: Error in transformer block {i+1}")
            print(f"   Error: {e}")
            print(f"   Input shape: {x.shape}")
            print(f"   Block index: {i}")
            print(f"   Device: {x.device}")
            if input_token_ids.device.type == 'mps':
                print(f"   🍎 MPS Error - trying memory cleanup...")
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            raise
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        loss = None
        if labels is not None:
            # ✅ CRITICAL: Check labels bounds before loss calculation to prevent CUDA errors
            vocab_size = self.cfg["vocab_size"]
            labels_flat = labels.view(-1)
            valid_labels = labels_flat[labels_flat != -100]
            
            if len(valid_labels) > 0:
                labels_min, labels_max = valid_labels.min().item(), valid_labels.max().item()
                
                if labels_max >= vocab_size or labels_min < 0:
                    print(f"🚨 CRITICAL: Labels out of bounds in loss calculation!")
                    print(f"   Labels range: [{labels_min}, {labels_max}]")
                    print(f"   Vocab size: {vocab_size}")
                    print(f"   Out-of-bounds count: {(valid_labels >= vocab_size).sum().item()}")
                    print(f"   Negative count: {(valid_labels < 0).sum().item()}")
                    print(f"   🛡️  EMERGENCY CLAMP: Fixing labels before loss calculation")
                    
                    # Emergency clamp labels while preserving -100
                    labels_clamped = labels.clone()
                    valid_mask = labels_clamped != -100
                    labels_clamped[valid_mask] = torch.clamp(labels_clamped[valid_mask], 0, vocab_size - 1)
                    labels = labels_clamped
            
            # ✅ MPS FIX: Ensure loss computation uses proper dtypes
            if input_token_ids.device.type == 'mps':
                logits = logits.float()
                labels = labels.long()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # ✅ MPS FINAL CLEANUP: Post-forward synchronization
        if input_token_ids.device.type == 'mps':
            torch.mps.synchronize()
            
            # Periodic deep cleanup for long training runs
            if self._mps_step_counter % (self._mps_memory_cleanup_frequency * 10) == 0:
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                print(f"🍎 MPS: Deep memory cleanup at step {self._mps_step_counter}")
        
        if use_cache:
            return {"logits": logits, "loss": loss, "present_key_values": present_key_values}
        else:
            return {"logits": logits, "loss": loss}

    def get_mps_memory_stats(self):
        """Get MPS memory statistics for debugging."""
        if hasattr(torch.mps, 'current_allocated_memory'):
            allocated = torch.mps.current_allocated_memory()
            return f"MPS Memory: {allocated / 1024**2:.1f} MB"
        return "MPS Memory: Stats unavailable"
    
    def is_lora_enabled(self) -> bool:
        """LoRA Domain Adaptation: Check if LoRA is enabled for this model."""
        return self.lora_config.use_lora and len(self.lora_injected_modules) > 0
    
    def get_lora_parameters(self):
        """
        LoRA Domain Adaptation: Get all LoRA parameters for optimizer creation.
        
        Returns:
            List[torch.nn.Parameter]: List of LoRA parameters that require gradients
        """
        if not self.is_lora_enabled():
            print("❌ LoRA Domain Adaptation: LoRA not enabled, no parameters to return")
            return []
        
        # LoRA Domain Adaptation: Directly collect LoRA parameters from model
        lora_params = []
        for name, param in self.named_parameters():
            if ('lora_A' in name or 'lora_B' in name) and param.requires_grad:
                lora_params.append(param)
        
        if not lora_params:
            print("❌ LoRA Domain Adaptation: No LoRA parameters found in model!")
            print("❌ This indicates LoRA injection failed or parameters were not properly initialized")
            
            # Debug: Check what parameters we have
            print("🔍 Available parameters:")
            for name, param in self.named_parameters():
                if 'lora' in name.lower() or 'Wq' in name or 'Wk' in name or 'Wv' in name:
                    print(f"   {name}: requires_grad={param.requires_grad}, shape={param.shape}")
        
        return lora_params
    
    def merge_lora_weights(self, verbose: bool = True):
        """LoRA Domain Adaptation: Merge LoRA adapter weights into base model."""
        if not self.is_lora_enabled():
            if verbose:
                print("LoRA Domain Adaptation: No LoRA adapters to merge")
            return self
            
        from ..attention.lora_utils import merge_lora_weights
        return merge_lora_weights(self, verbose=verbose)
    
    def unload_lora_weights(self, verbose: bool = True):
        """LoRA Domain Adaptation: Unload LoRA weights from base model."""
        if not self.is_lora_enabled():
            if verbose:
                print("LoRA Domain Adaptation: No LoRA adapters to unload")
            return self
            
        from ..attention.lora_utils import unload_lora_weights
        return unload_lora_weights(self, verbose=verbose)
    
    def get_lora_info(self):
        """LoRA Domain Adaptation: Get detailed information about LoRA adapters."""
        from ..attention.lora_utils import get_model_lora_info
        return get_model_lora_info(self)


