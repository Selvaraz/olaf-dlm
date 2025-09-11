import torch
from torch.utils.data import Dataset
import json
from typing import List, Dict
from ..config.opal_config import TRAINING_CONFIG, OPAL_MODEL_CONFIG

class OpalFinetuneDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer):
        """
        Args:
            data: List of dicts with keys {"prompt", "response"}
            tokenizer: SentencePieceProcessor instance
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = OPAL_MODEL_CONFIG["context_length"]
        self.device = TRAINING_CONFIG["device"]

        self.samples = self._prepare_data()
        
        # Check for potential tokenization issues
        self._validate_tokenization()

    def _validate_tokenization(self):
        """Check for common tokenization issues that could cause repetitive generation."""
        if not self.samples:
            return
            
        unk_id = self.tokenizer.unk_id() if hasattr(self.tokenizer, 'unk_id') else -1
        total_tokens = 0
        unk_tokens = 0
        
        for sample in self.samples[:min(100, len(self.samples))]:  # Check first 100 samples
            input_ids = sample["input_ids"]
            total_tokens += len(input_ids)
            if unk_id >= 0:
                unk_tokens += (input_ids == unk_id).sum().item()
        
        if total_tokens > 0:
            unk_percentage = (unk_tokens / total_tokens) * 100
            print(f"   → Unknown token rate: {unk_percentage:.2f}% ({unk_tokens}/{total_tokens})")
            
            if unk_percentage > 5.0:
                print(f"   ⚠️ WARNING: High unknown token rate! This could cause repetitive generation.")
                print(f"      Consider using a tokenizer trained on similar data.")
            elif unk_percentage > 10.0:
                print(f"   🚨 CRITICAL: Very high unknown token rate! This will likely cause poor generation quality.")

    def _json_to_natural_format(self, json_obj):
        """
        Convert JSON response to a more natural language format that's better for fine-tuning.
        This reduces repetitive JSON syntax and makes responses more human-readable.
        """
        if isinstance(json_obj, dict):
            if "action" in json_obj and json_obj["action"] == "orbit.troubleshoot":
                return self._format_troubleshoot_response(json_obj)
            else:
                # Generic JSON to natural language conversion
                return self._generic_json_to_text(json_obj)
        else:
            return str(json_obj)
    
    def _format_troubleshoot_response(self, response):
        """Format orbit.troubleshoot responses in a natural way."""
        output = []
        
        # Start with action
        output.append(f"Action: {response.get('action', 'unknown')}")
        
        # Process execution steps
        execution = response.get("execution", {})
        steps = execution.get("steps", [])
        
        if steps:
            output.append("Execution Steps:")
            
            for i, step in enumerate(steps, 1):
                step_type = step.get("step_type", "unknown")
                descriptions = step.get("description", [])
                commands = step.get("commands", [])
                
                # Format step header
                if step_type == "step_explain":
                    output.append(f"Step {i} - Explanation:")
                elif step_type == "step_conf":
                    output.append(f"Step {i} - Configuration:")
                elif step_type == "step_exec":
                    output.append(f"Step {i} - Execution:")
                else:
                    output.append(f"Step {i} - {step_type}:")
                
                # Add descriptions
                if descriptions:
                    for desc in descriptions:
                        output.append(f"  - {desc}")
                
                # Add commands
                if commands:
                    output.append("  Commands:")
                    for cmd in commands:
                        output.append(f"    {cmd}")
        
        result = "\n".join(output)
        
        # Limit length to prevent overly long responses
        if len(result) > 600:
            result = result[:600] + "\n  [Response truncated for training efficiency]"
        
        return result
    
    def _generic_json_to_text(self, obj, indent=0):
        """Convert generic JSON to more natural text format."""
        lines = []
        prefix = "  " * indent
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)) and value:
                    lines.append(f"{prefix}{key}:")
                    lines.append(self._generic_json_to_text(value, indent + 1))
                else:
                    lines.append(f"{prefix}{key}: {value}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    lines.append(f"{prefix}Item {i+1}:")
                    lines.append(self._generic_json_to_text(item, indent + 1))
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{obj}")
        
        return "\n".join(lines)

    def _prepare_data(self):
        samples = []
        for item in self.data:
            if "prompt" not in item or "response" not in item:
                raise ValueError(f"Invalid JSONL sample format: {item}")

            prompt = item["prompt"].strip()
            # Convert entire response object to a more natural format
            if isinstance(item["response"], str):
                response_text = item["response"].strip()
            else:
                # Convert JSON to natural language format instead of raw JSON
                response_text = self._json_to_natural_format(item["response"])

            # Validate minimum length
            if len(prompt) < 5:
                print(f"⚠ Skipping too short prompt: {prompt[:50]}...")
                continue

            # Since <BOS> and <EOS> are not in tokenizer vocabulary, use the built-in special tokens
            bos_token = self.tokenizer.bos_id() if hasattr(self.tokenizer, 'bos_id') and self.tokenizer.bos_id() >= 0 else None
            eos_token = self.tokenizer.eos_id() if hasattr(self.tokenizer, 'eos_id') and self.tokenizer.eos_id() >= 0 else None
            
            # Add clear separators to help the model distinguish prompt from response
            # Use tokens that are likely in the vocabulary
            prompt_marker = "PROMPT:"
            response_marker = "RESPONSE:"
            
            # Build the full sequence with clear structure
            full_text = f"{prompt_marker} {prompt} {response_marker} {response_text}"
            
            # Encode the full text
            input_ids = self.tokenizer.encode(full_text, out_type=int)
            
            # Add BOS token at the beginning if available
            if bos_token is not None:
                input_ids = [bos_token] + input_ids
            
            # Add EOS token at the end if available
            if eos_token is not None:
                input_ids = input_ids + [eos_token]
            
            # CRITICAL: Add bounds checking to prevent CUDA index out of bounds error
            vocab_size = self.tokenizer.get_piece_size()
            max_token_id = max(input_ids) if input_ids else -1
            min_token_id = min(input_ids) if input_ids else -1
            
            # FIX: Actually clamp the out-of-bounds tokens instead of skipping
            if max_token_id >= vocab_size or min_token_id < 0:
                print(f"🚨 FIXING out-of-bounds tokens in sample:")
                print(f"   Token range: [{min_token_id}, {max_token_id}], vocab_size: {vocab_size}")
                print(f"   Sample text: {full_text[:100]}...")
                
                # Count issues
                out_of_bounds_count = len([t for t in input_ids if t >= vocab_size])
                negative_count = len([t for t in input_ids if t < 0])
                print(f"   Out-of-bounds tokens: {out_of_bounds_count}, Negative tokens: {negative_count}")
                
                # FIX: Clamp all tokens to valid range
                unk_id = self.tokenizer.unk_id() if hasattr(self.tokenizer, 'unk_id') and self.tokenizer.unk_id() >= 0 else 3
                input_ids = [
                    max(0, min(t, vocab_size - 1)) if 0 <= t < vocab_size 
                    else unk_id 
                    for t in input_ids
                ]
                print(f"   → Fixed: clamped to [0, {vocab_size-1}], using unk_id={unk_id} for invalid tokens")
                
                # Verify fix
                new_max = max(input_ids) if input_ids else -1
                new_min = min(input_ids) if input_ids else -1
                print(f"   → After fix: token range [{new_min}, {new_max}] ✅")
            
            # Encode the prompt part with marker to determine masking boundary more accurately
            prompt_with_marker = f"{prompt_marker} {prompt} {response_marker}"
            prompt_ids = self.tokenizer.encode(prompt_with_marker, out_type=int)
            if bos_token is not None:
                prompt_ids = [bos_token] + prompt_ids
                
            # FIX: Also bounds-check prompt_ids
            if prompt_ids:
                max_prompt_token = max(prompt_ids)
                min_prompt_token = min(prompt_ids)
                if max_prompt_token >= vocab_size or min_prompt_token < 0:
                    print(f"🚨 FIXING out-of-bounds tokens in prompt_ids:")
                    print(f"   Prompt token range: [{min_prompt_token}, {max_prompt_token}]")
                    unk_id = self.tokenizer.unk_id() if hasattr(self.tokenizer, 'unk_id') and self.tokenizer.unk_id() >= 0 else 3
                    prompt_ids = [
                        max(0, min(t, vocab_size - 1)) if 0 <= t < vocab_size 
                        else unk_id 
                        for t in prompt_ids
                    ]
                    print(f"   → Prompt IDs fixed ✅")
                
            # Truncate if too long FIRST - before creating labels
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                
            prompt_len = min(len(prompt_ids), len(input_ids))

            # CRITICAL FIX: Create labels array with SAME length as input_ids
            # The bug was: labels = [-100] * prompt_len + input_ids[prompt_len:]
            # This creates labels longer than input_ids when input_ids is truncated
            labels = []
            for i, token_id in enumerate(input_ids):
                if i < prompt_len:
                    labels.append(-100)  # Mask prompt tokens
                else:
                    labels.append(token_id)  # Keep response tokens
            
            # VERIFY: labels and input_ids must have EXACTLY the same length
            assert len(labels) == len(input_ids), f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}"

            # CRITICAL: Validate label token IDs to prevent CUDA errors
            valid_label_tokens = [l for l in labels if l != -100]
            if valid_label_tokens:
                max_label_token = max(valid_label_tokens)
                if max_label_token >= vocab_size:
                    print(f"🚨 CRITICAL ERROR: Label token ID {max_label_token} >= vocab_size {vocab_size}")
                    print(f"   → Skipping this sample to prevent training crash")
                    continue

            # Debug: Print first few samples to verify format
            if len(samples) < 3:  # Only for first few samples
                print(f"   → Sample {len(samples) + 1} debug:")
                print(f"     Prompt: {prompt[:100]}...")
                print(f"     Response length: {len(response_text)} chars")
                print(f"     Full text: {full_text[:150]}...")
                print(f"     Input IDs length: {len(input_ids)}, Prompt boundary: {prompt_len}")
                
                # Check if response_text is properly structured JSON
                if isinstance(item["response"], dict):
                    print(f"     JSON keys: {list(item['response'].keys())}")

            samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            })

        print(f"📊 Fine-tune dataset prepared: {len(samples)} valid samples")
        if len(samples) > 0:
            avg_input_len = sum(len(s["input_ids"]) for s in samples) / len(samples)
            avg_label_len = sum((s["labels"] != -100).sum().item() for s in samples) / len(samples)
            print(f"   → Avg input length: {avg_input_len:.1f}, Avg response length: {avg_label_len:.1f}")
            
            # Debug: Check if BOS/EOS tokens are properly used
            sample_input = samples[0]["input_ids"]
            sample_labels = samples[0]["labels"]
            bos_token = self.tokenizer.bos_id() if hasattr(self.tokenizer, 'bos_id') and self.tokenizer.bos_id() >= 0 else None
            eos_token = self.tokenizer.eos_id() if hasattr(self.tokenizer, 'eos_id') and self.tokenizer.eos_id() >= 0 else None
            
            print(f"   → BOS token ID: {bos_token}, EOS token ID: {eos_token}")
            print(f"   → Sample input first 5 tokens: {sample_input[:5].tolist()}")
            print(f"   → Sample input last 5 tokens: {sample_input[-5:].tolist()}")
            print(f"   → Sample labels (non-masked): {(sample_labels != -100).sum().item()}/{len(sample_labels)}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]["input_ids"], self.samples[idx]["labels"]


"""
=== EXAMPLE: JSONL TO TRAINING FORMAT CONVERSION ===

Given a single JSONL entry like this:
{
    "prompt": "How do I configure OSPF on a Cisco router?",
    "response": {
        "action": "orbit.troubleshoot",
        "execution": {
            "steps": [
                {
                    "step_type": "step_conf",
                    "description": ["Enable OSPF routing protocol"],
                    "commands": ["router ospf 1"]
                },
                {
                    "step_type": "step_conf", 
                    "description": ["Configure network statement"],
                    "commands": ["network 192.168.1.0 0.0.0.255 area 0"]
                }
            ]
        }
    }
}

This dataset converts it to the following training format:

1. NATURAL LANGUAGE CONVERSION:
   The JSON response gets converted to human-readable format:
   
   "Action: orbit.troubleshoot
   Execution Steps:
   Step 1 - Configuration:
     - Enable OSPF routing protocol
     Commands:
       router ospf 1
   Step 2 - Configuration:
     - Configure network statement
     Commands:
       network 192.168.1.0 0.0.0.255 area 0"

2. TRAINING SEQUENCE STRUCTURE:
   The final training sequence becomes:
   
   Input Text: "PROMPT: How do I configure OSPF on a Cisco router? RESPONSE: Action: orbit.troubleshoot..."
   
   Tokenized as:
   - BOS token (if available)
   - PROMPT: How do I configure OSPF on a Cisco router? RESPONSE: [MASKED - labels = -100]
   - Action: orbit.troubleshoot... [NOT MASKED - labels = actual token IDs]
   - EOS token (if available)

3. MASKING STRATEGY:
   - Everything up to and including "RESPONSE:" marker is MASKED (labels = -100)
   - Only the actual response content is used for loss calculation
   - This teaches the model to generate responses, not prompts

4. TOKEN STRUCTURE:
   input_ids = [BOS, tok1, tok2, ..., tokN, EOS]  # Full sequence
   labels    = [-100, -100, -100, ..., tokX, tokY, tokZ, EOS]  # Only response tokens
   
   Where:
   - Prompt tokens (tok1...tokN-X) are masked with -100
   - Response tokens (tokX...tokZ) have actual token IDs for training
   - Model learns to predict the response given the prompt context

This format ensures:
✅ Clean separation between prompt and response
✅ Model only trains on generating responses, not repeating prompts  
✅ Natural language format reduces JSON syntax repetition
✅ Clear structure helps model understand instruction-following pattern
✅ Proper masking prevents the model from learning to repeat inputs
"""
