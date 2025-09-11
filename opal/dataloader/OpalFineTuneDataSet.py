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
            # Handle responses with steps at top level (like your sample)
            if "steps" in json_obj:
                return self._format_steps_response(json_obj)
            # Handle orbit.troubleshoot responses
            elif "action" in json_obj and json_obj["action"] == "orbit.troubleshoot":
                return self._format_troubleshoot_response(json_obj)
            else:
                # Generic JSON to natural language conversion
                return self._generic_json_to_text(json_obj)
        else:
            return str(json_obj)
    
    def _format_steps_response(self, response):
        """Format responses with steps at top level (like your sample)."""
        output = []
        
        # Add action if present
        if "action" in response:
            output.append(f"Action: {response.get('action')}")
        
        # Process steps directly from top level
        steps = response.get("steps", [])
        
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
            # Always convert response to natural language format for better fine-tuning
            if isinstance(item["response"], str):
                try:
                    # Try to parse as JSON first in case it's a JSON string
                    json_response = json.loads(item["response"])
                    response_text = self._json_to_natural_format(json_response)
                except (json.JSONDecodeError, ValueError):
                    # If not valid JSON, use as-is
                    response_text = item["response"].strip()
            else:
                # Convert JSON object to natural language format
                response_text = self._json_to_natural_format(item["response"])

            # Validate minimum length
            if len(prompt) < 5:
                print(f"⚠ Skipping too short prompt: {prompt[:50]}...")
                continue

            # Since <BOS> and <EOS> are not in tokenizer vocabulary, use the built-in special tokens
            bos_token = self.tokenizer.bos_id() if hasattr(self.tokenizer, 'bos_id') and self.tokenizer.bos_id() >= 0 else None
            eos_token = self.tokenizer.eos_id() if hasattr(self.tokenizer, 'eos_id') and self.tokenizer.eos_id() >= 0 else None
            
            # Use a more standard chat format that's better for fine-tuning
            # This follows the format used by many successful chat models
            system_prompt = "You are Olaf, a helpful network troubleshooting assistant. Provide clear, step-by-step solutions for network configuration and debugging tasks."
            
            # Build the full sequence with proper chat formatting
            # This format is more likely to be recognized by the tokenizer
            full_text = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant: {response_text}"
            
            # Encode the full text
            input_ids = self.tokenizer.encode(full_text, out_type=int)
            
            # Add BOS token at the beginning if available
            if bos_token is not None:
                input_ids = [bos_token] + input_ids
            
            # Add EOS token at the end if available
            if eos_token is not None:
                input_ids = input_ids + [eos_token]
            
            # Encode the prompt part to determine masking boundary more accurately
            # We want to mask everything up to and including "Assistant: "
            prompt_with_system = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant: "
            prompt_ids = self.tokenizer.encode(prompt_with_system, out_type=int)
            if bos_token is not None:
                prompt_ids = [bos_token] + prompt_ids
                
            # Truncate if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                
            prompt_len = min(len(prompt_ids), len(input_ids))

            # Apply the masks for the prompt tokens, so our DLM does not 
            # Learn about the prompt :-D
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            
            # Ensure labels are also truncated to match input_ids length
            if len(labels) > self.max_length:
                labels = labels[:self.max_length]

            # Debug: Print first few samples to verify format
            if len(samples) < 3:  # Only for first few samples
                print(f"   → Sample {len(samples) + 1} debug:")
                print(f"     Prompt: {prompt[:100]}...")
                print(f"     Response length: {len(response_text)} chars")
                print(f"     Full text: {full_text[:200]}...")
                print(f"     Input IDs length: {len(input_ids)}, Prompt boundary: {prompt_len}")
                
                # Check if response_text is properly structured JSON
                if isinstance(item["response"], dict):
                    print(f"     JSON keys: {list(item['response'].keys())}")
                
                # Show the actual masking boundary
                decoded_prompt_part = self.tokenizer.decode(input_ids[:prompt_len])
                print(f"     Masked portion: {decoded_prompt_part[:100]}...")

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
