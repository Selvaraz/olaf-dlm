#!/usr/bin/env python3
"""
OPAL Fine-tuned Model Tester - Unified Interactive Testing Script

Test your fine-tuned OPAL model with customizable generation parameters.
Supports various temperature, top-k, top-p settings for exploring model behavior.

Features:
- Interactive mode with commands
- Single prompt testing
- Built-in generation presets
- Custom parameter support
- Guided usage examples
- Automatic model/tokenizer detection

Usage:
    python3 finetune_model_tester.py                                    # Interactive mode
    python3 finetune_model_tester.py --prompt "Your prompt"             # Single prompt
    python3 finetune_model_tester.py --preset creative                  # With preset
    python3 finetune_model_tester.py --params "temp=0.8,top_k=50"      # Custom params
    python3 finetune_model_tester.py --guide                           # Guided examples
    python3 finetune_model_tester.py --validate                        # Validate setup
"""

import sys
import os
import argparse
import torch
import json
import subprocess
from pathlib import Path

# Add OPAL modules to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def load_model_and_tokenizer(checkpoint_path, tokenizer_path, device="auto"):
    """
    Load the fine-tuned model and tokenizer
    
    Args:
        checkpoint_path (str): Path to the model checkpoint (.pt file)
        tokenizer_path (str): Path to the tokenizer model (.model file)
        device (str): Device to use ('auto', 'cpu', 'cuda', 'mps')
    
    Returns:
        tuple: (model, tokenizer, config, device)
    """
    try:
        import sentencepiece as smp
        from opal.transformer.OpalGPTModel import OpalGPT
        from opal.config.opal_config import OPAL_MODEL_CONFIG
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        print(f"🔧 Loading model on device: {device}")
        
        # Load tokenizer
        print(f"📄 Loading tokenizer from: {tokenizer_path}")
        tokenizer = smp.SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        
        # Validate tokenizer
        test_encode = tokenizer.encode("hello", out_type=int)
        if not test_encode:
            raise ValueError("Tokenizer failed to encode test string")
        
        vocab_size = tokenizer.get_piece_size()
        print(f"✅ Tokenizer loaded - Vocab size: {vocab_size}")
        
        # Load checkpoint
        print(f"🧠 Loading model checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract configuration and model state
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"✅ Using config from checkpoint")
            else:
                config = OPAL_MODEL_CONFIG
                print(f"⚠️ No config in checkpoint, using default fine-tune config")
            
            model_state = checkpoint.get('model_state_dict', checkpoint)
        else:
            config = OPAL_MODEL_CONFIG
            model_state = checkpoint
            print(f"⚠️ Legacy checkpoint format, using default config")
        
        # Ensure vocab size matches
        if config.get('vocab_size') != vocab_size:
            print(f"⚠️ Config vocab_size ({config.get('vocab_size')}) != tokenizer vocab_size ({vocab_size})")
            print(f"🔧 Updating config vocab_size to match tokenizer")
            config['vocab_size'] = vocab_size
        
        # Create and load model
        model = OpalGPT(config)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"   📊 Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"   📐 Context length: {config.get('context_length', 'unknown')}")
        print(f"   🧮 Embedding dim: {config.get('emb_dim', 'unknown')}")
        
        return model, tokenizer, config, device
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None, None

def find_latest_finetune_checkpoint():
    """Find the latest fine-tune checkpoint"""
    checkpoint_dir = Path("checkpoints")
    
    # Check for symlink first
    latest_link = checkpoint_dir / "finetune-latest.pt"
    if latest_link.exists():
        return str(latest_link.resolve())
    
    # Search in _finetune_ directory
    finetune_dir = checkpoint_dir / "_finetune_"
    if finetune_dir.exists():
        pt_files = list(finetune_dir.rglob("*.pt"))
        if pt_files:
            return str(max(pt_files, key=lambda p: p.stat().st_mtime))
    
    return None

def find_tokenizer():
    """Find the tokenizer model"""
    # Common locations
    locations = [
        "checkpoints/pretrained/opal_tokenizer.model",
        "checkpoints/opal_tokenizer.model", 
        "opal_tokenizer.model"
    ]
    
    for loc in locations:
        if Path(loc).exists():
            return loc
    
    return None

    return logits

def format_alternatives_table(token_alternatives, show_table=True):
    """
    Format the token alternatives into a nice table
    
    Args:
        token_alternatives (list): List of alternatives for each token
        show_table (bool): Whether to show the detailed table
    
    Returns:
        str: Formatted table string
    """
    if not show_table or not token_alternatives:
        return ""
    
    # Prepare table data
    table_rows = []
    for i, alternatives in enumerate(token_alternatives):
        if not alternatives:
            continue
            
        # Find the accepted token (marked as True in the tuple)
        accepted_token = None
        accepted_prob = 0.0
        alternative_tokens = []
        
        for token_text, prob, is_selected in alternatives:
            if is_selected:
                accepted_token = token_text
                accepted_prob = prob
            else:
                alternative_tokens.append((token_text, prob))
        
        # Get top 5 alternatives (excluding the accepted one)
        alternative_tokens = sorted(alternative_tokens, key=lambda x: x[1], reverse=True)[:5]
        
        # Format alternatives string
        alt_str = ", ".join([f"{token}[{prob:.3f}]" for token, prob in alternative_tokens])
        if not alt_str:
            alt_str = "No alternatives"
        
        table_rows.append({
            'position': i + 1,
            'accepted': f"{accepted_token}[{accepted_prob:.3f}]" if accepted_token else "N/A",
            'alternatives': alt_str
        })
    
    if not table_rows:
        return "\n🔍 No token alternatives data available\n"
    
    # Calculate column widths
    pos_width = max(8, max(len(str(row['position'])) for row in table_rows))
    accepted_width = max(15, max(len(row['accepted']) for row in table_rows))
    alt_width = max(30, max(len(row['alternatives']) for row in table_rows))
    
    # Ensure minimum readable widths
    alt_width = min(alt_width, 80)  # Cap at 80 chars for readability
    
    # Create table
    table_lines = []
    
    # Header
    header = f"{'Position':<{pos_width}} | {'Accepted Token':<{accepted_width}} | {'Alternative Tokens & Probabilities':<{alt_width}}"
    separator = "-" * len(header)
    
    table_lines.append("")
    table_lines.append("🔍 TOKEN ALTERNATIVES TABLE")
    table_lines.append("=" * 50)
    table_lines.append(header)
    table_lines.append(separator)
    
    # Data rows
    for row in table_rows:
        # Truncate alternatives if too long
        alternatives = row['alternatives']
        if len(alternatives) > alt_width:
            alternatives = alternatives[:alt_width-3] + "..."
        
        line = f"{row['position']:<{pos_width}} | {row['accepted']:<{accepted_width}} | {alternatives:<{alt_width}}"
        table_lines.append(line)
    
    table_lines.append(separator)
    table_lines.append(f"Total tokens analyzed: {len(table_rows)}")
    table_lines.append("")
    
    return "\n".join(table_lines)

def nucleus_sampling(logits, top_k=None, top_p=None, temperature=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits
    
    Args:
        logits (torch.Tensor): The logits from the model
        top_k (int): Keep only top k tokens with highest probability
        top_p (float): Keep the smallest possible set whose cumulative prob > top_p
        temperature (float): Temperature for sampling (1.0 = no temperature)
        filter_value (float): Value to use for filtered logits
    
    Returns:
        torch.Tensor: Filtered logits
    """
    logits = logits / temperature
    
    # Top-k filtering
    if top_k is not None and top_k > 0:
        # Get the top-k values and indices
        top_k = min(top_k, logits.size(-1))  # Safety check
        values, indices = torch.topk(logits, top_k)
        
        # Create a mask for tokens to keep
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(0, indices, True)
        
        # Set filtered tokens to filter_value
        logits = logits.masked_fill(~mask, filter_value)
    
    # Top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift to ensure we keep at least the first token
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        
        # Create indices to remove in original order
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits

def generate_with_params(model, tokenizer, prompt, config, device, **gen_params):
    """
    Generate text with specified parameters
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt (str): Input prompt
        config (dict): Model configuration
        device (str): Device to use
        **gen_params: Generation parameters
    
    Returns:
        tuple: (full_text, generated_text, metadata)
    """
    # Default generation parameters
    max_new_tokens = gen_params.get('max_new_tokens', 512)
    temperature = gen_params.get('temperature', 1.0)
    top_k = gen_params.get('top_k', None)
    top_p = gen_params.get('top_p', None)
    repetition_penalty = gen_params.get('repetition_penalty', 1.0)
    do_sample = gen_params.get('do_sample', True)
    show_probabilities = gen_params.get('show_probabilities', True)
    show_alternatives = gen_params.get('show_alternatives', True)
    
    try:
        # Format prompt for fine-tuned model (add conversation markers)
        formatted_prompt = f"{prompt}"
        
        # Encode prompt
        input_ids = tokenizer.encode(formatted_prompt, out_type=int)
        current_ids = input_ids.copy()
        context_length = config.get('context_length', 512)
        
        # Track generation metadata
        metadata = {
            'prompt_length': len(input_ids),
            'generated_length': 0,
            'stopped_reason': 'max_tokens',
            'parameters_used': gen_params,
            'formatted_prompt': formatted_prompt,
            'token_probabilities': [],  # Store token probabilities for display
            'token_alternatives': []  # Store top-k alternatives for each token
        }
        
        print(f"🚀 Generating with: temp={temperature}, top_k={top_k}, top_p={top_p}")
        print(f"   Max tokens: {max_new_tokens}, Context: {context_length}")
        print(f"   📝 Formatted prompt: '{formatted_prompt}'")
        print(f"   📏 Prompt tokens: {len(input_ids)}")
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare input (keep within context window)
            if len(current_ids) >= context_length:
                # Truncate from the beginning, keeping recent context
                truncate_len = len(current_ids) - context_length + 1
                current_ids = current_ids[truncate_len:]
            
            input_tensor = torch.tensor([current_ids], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Forward pass
                output = model(input_tensor)
                logits = output['logits'] if isinstance(output, dict) else output
                next_token_logits = logits[0, -1, :].float()
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    # Simple repetition penalty: reduce prob of recently used tokens
                    recent_tokens = set(current_ids[-50:])  # Last 50 tokens
                    for token_id in recent_tokens:
                        if token_id < len(next_token_logits):
                            if next_token_logits[token_id] > 0:
                                next_token_logits[token_id] /= repetition_penalty
                            else:
                                next_token_logits[token_id] *= repetition_penalty
                
                # Apply sampling filters
                if do_sample:
                    filtered_logits = nucleus_sampling(
                        next_token_logits.clone(), 
                        top_k=top_k, 
                        top_p=top_p, 
                        temperature=temperature
                    )
                    probs = torch.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    # Get the probability of the selected token
                    token_prob = probs[next_token].item()
                else:
                    # Greedy decoding
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.argmax(next_token_logits).item()
                    # Get the probability of the selected token
                    token_prob = probs[next_token].item()
                
                # Get top-6 alternatives (including the selected token)
                top_probs, top_indices = torch.topk(probs, min(6, probs.size(-1)))
                alternatives = []
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token_text = tokenizer.decode([idx.item()])
                    alternatives.append((token_text, prob.item(), idx.item() == next_token))
                
                current_ids.append(next_token)
                metadata['generated_length'] += 1
                
                # Store token and its probability for display
                token_text = tokenizer.decode([next_token])
                metadata['token_probabilities'].append((token_text, token_prob))
                metadata['token_alternatives'].append(alternatives)
                
                # Check stopping conditions
                if next_token == tokenizer.eos_id():
                    metadata['stopped_reason'] = 'eos_token'
                    break
                elif next_token == tokenizer.pad_id():
                    metadata['stopped_reason'] = 'pad_token'
                    break
        
        # Decode results
        full_text = tokenizer.decode(current_ids)
        generated_text = tokenizer.decode(current_ids[len(input_ids):])
        
        # Create text with probabilities if requested
        if show_probabilities and metadata.get('token_probabilities'):
            generated_with_probs = ""
            for token_text, prob in metadata['token_probabilities']:
                generated_with_probs += f"{token_text}[{prob:.3f}]"
            metadata['generated_with_probabilities'] = generated_with_probs
        
        # Create alternatives table if requested
        if show_alternatives and metadata.get('token_alternatives'):
            metadata['alternatives_table'] = format_alternatives_table(
                metadata['token_alternatives'], show_table=True
            )
        
        return full_text, generated_text, metadata
        
    except Exception as e:
        return prompt, f"Error during generation: {e}", {'error': str(e)}

def get_generation_presets():
    """Get available generation presets"""
    presets = {
        'conservative': {
            'temperature': 0.7,
            'top_k': 30,
            'top_p': 0.8,
            'repetition_penalty': 1.1,
            'description': 'Safe, coherent output with low randomness'
        },
        'balanced': {
            'temperature': 1.0,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.05,
            'description': 'Balanced creativity and coherence'
        },
        'creative': {
            'temperature': 1.2,
            'top_k': 100,
            'top_p': 0.95,
            'repetition_penalty': 1.02,
            'description': 'High creativity, more diverse outputs'
        },
        'focused': {
            'temperature': 0.3,
            'top_k': 10,
            'top_p': 0.7,
            'repetition_penalty': 1.2,
            'description': 'Very focused, deterministic output'
        },
        'greedy': {
            'temperature': 1.0,
            'do_sample': False,
            'description': 'Greedy decoding (most likely tokens)'
        }
    }
    return presets

def print_generation_presets():
    """Print available generation presets"""
    presets = get_generation_presets()
    
    print("\n🎯 Available Presets:")
    print("=" * 50)
    for name, params in presets.items():
        desc = params.get('description', '')
        param_items = {k: v for k, v in params.items() if k != 'description'}
        param_str = ', '.join([f"{k}={v}" for k, v in param_items.items()])
        print(f"  {name:<12} - {desc}")
        print(f"  {'':<12}   Parameters: {param_str}")
    print()
    
    return presets

def parse_generation_params(param_string):
    """
    Parse generation parameters from string
    Format: temp=0.8,top_k=50,top_p=0.9,rep_penalty=1.1,max_tokens=256
    """
    params = {}
    if not param_string:
        return params
    
    try:
        for part in param_string.split(','):
            if '=' not in part:
                continue
            key, value = part.strip().split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Map parameter names
            param_map = {
                'temp': 'temperature',
                'temperature': 'temperature',
                'top_k': 'top_k',
                'top_p': 'top_p',
                'rep_penalty': 'repetition_penalty',
                'repetition_penalty': 'repetition_penalty',
                'max_tokens': 'max_new_tokens',
                'max_new_tokens': 'max_new_tokens',
                'sample': 'do_sample',
                'show_probs': 'show_probabilities',
                'probabilities': 'show_probabilities',
                'show_alts': 'show_alternatives',
                'alternatives': 'show_alternatives',
                'show_alternatives': 'show_alternatives'
            }
            
            if key in param_map:
                param_key = param_map[key]
                
                # Convert values to appropriate types
                if param_key in ['top_k', 'max_new_tokens']:
                    params[param_key] = int(value)
                elif param_key in ['temperature', 'top_p', 'repetition_penalty']:
                    params[param_key] = float(value)
                elif param_key == 'do_sample':
                    params[param_key] = value.lower() in ('true', '1', 'yes')
                elif param_key == 'show_probabilities':
                    params[param_key] = value.lower() in ('true', '1', 'yes')
                elif param_key == 'show_alternatives':
                    params[param_key] = value.lower() in ('true', '1', 'yes')
                else:
                    params[param_key] = value
                    
    except Exception as e:
        print(f"⚠️ Error parsing parameters: {e}")
    
    return params

def validate_setup():
    """Validate that the model and tokenizer can be loaded"""
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        import sentencepiece as smp
        print(f"✅ SentencePiece available")
        
        from opal.transformer.OpalGPTModel import OpalGPT
        from opal.config.opal_config import OPAL_MODEL_CONFIG
        print(f"✅ OPAL modules available")
        
        # Check for model files
        checkpoint_dir = Path("checkpoints")
        
        # Look for tokenizer
        tokenizer_path = find_tokenizer()
        if tokenizer_path:
            print(f"✅ Tokenizer found: {tokenizer_path}")
            
            # Test tokenizer loading
            tokenizer = smp.SentencePieceProcessor()
            tokenizer.load(str(tokenizer_path))
            
            test_text = "Hello world"
            encoded = tokenizer.encode(test_text, out_type=int)
            decoded = tokenizer.decode(encoded)
            
            print(f"✅ Tokenizer test: '{test_text}' -> {encoded} -> '{decoded}'")
            print(f"   Vocab size: {tokenizer.get_piece_size()}")
        else:
            print(f"❌ Tokenizer not found in expected locations")
        
        # Look for model checkpoint
        checkpoint_to_test = find_latest_finetune_checkpoint()
        if not checkpoint_to_test:
            pretrained_checkpoint = checkpoint_dir / "pretrained" / "opal_gpt_checkpoint_20250908_141411.pt"
            if pretrained_checkpoint.exists():
                checkpoint_to_test = str(pretrained_checkpoint)
                print(f"✅ Pretrained checkpoint found: {checkpoint_to_test}")
            else:
                print(f"❌ No checkpoint found")
                return False
        else:
            print(f"✅ Fine-tune checkpoint found: {checkpoint_to_test}")
        
        if checkpoint_to_test and tokenizer_path:
            print(f"\n🧠 Testing model loading...")
            
            # Load checkpoint
            device = "cpu"  # Use CPU for validation
            checkpoint = torch.load(str(checkpoint_to_test), map_location=device)
            
            if isinstance(checkpoint, dict):
                config = checkpoint.get('config', OPAL_MODEL_CONFIG)
                model_state = checkpoint.get('model_state_dict', checkpoint)
                print(f"✅ Checkpoint structure valid")
                print(f"   Config vocab_size: {config.get('vocab_size')}")
                print(f"   Config emb_dim: {config.get('emb_dim')}")
                print(f"   Config n_layers: {config.get('n_layers')}")
            else:
                config = OPAL_MODEL_CONFIG
                model_state = checkpoint
                print(f"⚠️ Legacy checkpoint format")
            
            # Create model
            model = OpalGPT(config)
            model.load_state_dict(model_state)
            model.eval()
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {param_count:,} (~{param_count/1e6:.1f}M)")
            
            # Quick inference test
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)  # Dummy input
            with torch.no_grad():
                output = model(input_ids)
                logits = output['logits'] if isinstance(output, dict) else output
                print(f"✅ Model forward pass successful")
                print(f"   Output shape: {logits.shape}")
                print(f"   Expected: [1, 3, {config.get('vocab_size')}]")
        
        print(f"\n🎉 All validations passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Make sure you have installed requirements:")
        print(f"   pip install torch sentencepiece")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def interactive_mode(model, tokenizer, config, device):
    """
    Interactive mode for testing the model
    """
    print("\n🎮 INTERACTIVE FINE-TUNED MODEL TESTER")
    print("=" * 50)
    print("Commands:")
    print("  <prompt>           - Generate with current settings")
    print("  preset <name>      - Use generation preset")
    print("  params <params>    - Set custom parameters")
    print("  show               - Show current parameters")
    print("  presets            - List available presets")
    print("  probs on/off       - Toggle probability display")
    print("  alts on/off        - Toggle alternatives table display")
    print("  help               - Show this help")
    print("  quit/exit          - Exit")
    print("=" * 50)
    
    # Default parameters
    current_params = {
        'max_new_tokens': 512,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'repetition_penalty': 1.05,
        'do_sample': True,
        'show_probabilities': True,
        'show_alternatives': True
    }
    
    presets = print_generation_presets()
    
    while True:
        try:
            user_input = input("\n🎯 Input: ").strip()
            
            if not user_input or user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'help':
                print("\nParameter format: temp=0.8,top_k=50,top_p=0.9,rep_penalty=1.1,max_tokens=256")
                print("Available parameters: temp, top_k, top_p, rep_penalty, max_tokens, sample, show_probs, show_alts")
                continue
            elif user_input.lower() == 'presets':
                print_generation_presets()
                continue
            elif user_input.lower() == 'show':
                print(f"\n📊 Current parameters:")
                for key, value in current_params.items():
                    print(f"   {key}: {value}")
                continue
            elif user_input.lower().startswith('probs '):
                toggle = user_input[6:].strip().lower()
                if toggle in ['on', 'true', '1', 'yes']:
                    current_params['show_probabilities'] = True
                    print("✅ Probability display enabled")
                elif toggle in ['off', 'false', '0', 'no']:
                    current_params['show_probabilities'] = False
                    print("✅ Probability display disabled")
                else:
                    print("❌ Use 'probs on' or 'probs off'")
                continue
            elif user_input.lower().startswith('alts '):
                toggle = user_input[5:].strip().lower()
                if toggle in ['on', 'true', '1', 'yes']:
                    current_params['show_alternatives'] = True
                    print("✅ Alternatives table display enabled")
                elif toggle in ['off', 'false', '0', 'no']:
                    current_params['show_alternatives'] = False
                    print("✅ Alternatives table display disabled")
                else:
                    print("❌ Use 'alts on' or 'alts off'")
                continue
            elif user_input.lower().startswith('preset '):
                preset_name = user_input[7:].strip()
                if preset_name in presets:
                    preset_params = presets[preset_name].copy()
                    preset_params.pop('description', None)
                    current_params.update(preset_params)
                    print(f"✅ Applied preset '{preset_name}'")
                    print(f"📊 New parameters: {current_params}")
                else:
                    print(f"❌ Unknown preset '{preset_name}'. Use 'presets' to see available options.")
                continue
            elif user_input.lower().startswith('params '):
                param_string = user_input[7:].strip()
                new_params = parse_generation_params(param_string)
                if new_params:
                    current_params.update(new_params)
                    print(f"✅ Updated parameters: {new_params}")
                    print(f"📊 Current parameters: {current_params}")
                else:
                    print("❌ No valid parameters found. Format: temp=0.8,top_k=50,top_p=0.9")
                continue
            
            # Treat as generation prompt
            prompt = user_input
            print(f"\n🚀 Generating for prompt: '{prompt}'")
            print("-" * 50)
            
            # Generate text
            full_text, generated_text, metadata = generate_with_params(
                model, tokenizer, prompt, config, device, **current_params
            )
            
            # Display results
            if 'error' not in metadata:
                print(f"📝 Generated text ({metadata['generated_length']} tokens):")
                print(f"'{generated_text.strip()}'")
                
                # Show text with probabilities if available
                if metadata.get('generated_with_probabilities'):
                    print(f"\n🎯 Generated text with probabilities:")
                    print(f"'{metadata['generated_with_probabilities']}'")
                
                print(f"\n📄 Full conversation:")
                print(f"'{full_text.strip()}'")
                print(f"\n📊 Generation info:")
                print(f"   Stopped: {metadata['stopped_reason']}")
                print(f"   Prompt tokens: {metadata['prompt_length']}")
                print(f"   Generated tokens: {metadata['generated_length']}")
                
                # Show alternatives table if available
                if metadata.get('alternatives_table'):
                    print(metadata['alternatives_table'])
            else:
                print(f"❌ Generation failed: {metadata['error']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")

def guided_examples_mode():
    """Guided examples mode from the original run_finetune_tester.py"""
    print("🎮 OPAL Fine-tuned Model Tester - Guided Examples")
    print("=" * 60)
    
    # Find paths automatically
    model_path = find_latest_finetune_checkpoint()
    tokenizer_path = find_tokenizer()
    
    print(f"🔍 Auto-detected paths:")
    print(f"   Model: {model_path or 'Not found'}")
    print(f"   Tokenizer: {tokenizer_path or 'Not found'}")
    print()
    
    if not model_path or not tokenizer_path:
        print("❌ Required files not found. Please ensure you have:")
        print("   - A fine-tuned model checkpoint")
        print("   - A tokenizer model file")
        return
    
    print("Available guided modes:")
    print("  1. single        - Test single prompt")
    print("  2. presets       - Test with different presets")
    print("  3. custom        - Test with custom parameters")
    print("  4. help          - Show detailed help")
    print()
    
    try:
        choice = input("Choose mode (1-4): ").strip()
        modes = {
            '1': 'single',
            '2': 'presets',
            '3': 'custom',
            '4': 'help'
        }
        mode = modes.get(choice, 'single')
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return
    
    # Load model for guided examples
    model, tokenizer, config, device = load_model_and_tokenizer(
        model_path, tokenizer_path, "auto"
    )
    
    if model is None:
        print("❌ Failed to load model and tokenizer")
        return
    
    if mode == 'single':
        prompt = input("Enter prompt: ").strip()
        if prompt:
            print(f"\n🚀 Testing single prompt with balanced settings...")
            gen_params = {
                'max_new_tokens': 512,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.9,
                'repetition_penalty': 1.05,
                'do_sample': True
            }
            
            full_text, generated_text, metadata = generate_with_params(
                model, tokenizer, prompt, config, device, **gen_params
            )
            
            if 'error' not in metadata:
                print(f"\n📝 Generated ({metadata['generated_length']} tokens):")
                print(f"{generated_text.strip()}")
                
                # Show text with probabilities if available
                if metadata.get('generated_with_probabilities'):
                    print(f"\n🎯 Generated text with probabilities:")
                    print(f"{metadata['generated_with_probabilities']}")
                
                # Show alternatives table if available
                if metadata.get('alternatives_table'):
                    print(metadata['alternatives_table'])
                
                print(f"\n📊 Stopped: {metadata['stopped_reason']}")
            else:
                print(f"❌ Error: {metadata['error']}")
        
    elif mode == 'presets':
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            prompt = "Help me understand the basics of networking"
        
        presets = get_generation_presets()
        preset_names = ['conservative', 'balanced', 'creative', 'focused']
        
        for preset_name in preset_names:
            print(f"\n🎯 Testing with preset: {preset_name}")
            print("-" * 30)
            
            preset_params = presets[preset_name].copy()
            preset_params.pop('description', None)
            preset_params['max_new_tokens'] = 512
            
            full_text, generated_text, metadata = generate_with_params(
                model, tokenizer, prompt, config, device, **preset_params
            )
            
            if 'error' not in metadata:
                print(f"📝 Generated ({metadata['generated_length']} tokens):")
                print(f"{generated_text.strip()}")
                
                # Show text with probabilities if available
                if metadata.get('generated_with_probabilities'):
                    print(f"\n🎯 Generated text with probabilities:")
                    print(f"{metadata['generated_with_probabilities']}")
                
                # Show alternatives table if available
                if metadata.get('alternatives_table'):
                    print(metadata['alternatives_table'])
                
                print(f"📊 Stopped: {metadata['stopped_reason']}")
            else:
                print(f"❌ Error: {metadata['error']}")
            
            if preset_name != preset_names[-1]:
                input("\nPress Enter to continue to next preset...")
        
    elif mode == 'custom':
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            prompt = "Explain how OSPF routing protocol works"
        
        params = input("Enter parameters (e.g., temp=0.8,top_k=30,max_tokens=256): ").strip()
        
        print(f"\n🎯 Testing with custom parameters...")
        
        gen_params = {
            'max_new_tokens': 512,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.05,
            'do_sample': True
        }
        
        if params:
            custom_params = parse_generation_params(params)
            gen_params.update(custom_params)
            print(f"✅ Custom parameters: {custom_params}")
        
        full_text, generated_text, metadata = generate_with_params(
            model, tokenizer, prompt, config, device, **gen_params
        )
        
        if 'error' not in metadata:
            print(f"\n📝 Generated ({metadata['generated_length']} tokens):")
            print(f"{generated_text.strip()}")
            
            # Show text with probabilities if available
            if metadata.get('generated_with_probabilities'):
                print(f"\n🎯 Generated text with probabilities:")
                print(f"{metadata['generated_with_probabilities']}")
            
            # Show alternatives table if available
            if metadata.get('alternatives_table'):
                print(metadata['alternatives_table'])
            
            print(f"\n📊 Stopped: {metadata['stopped_reason']}")
        else:
            print(f"❌ Error: {metadata['error']}")
        
    elif mode == 'help':
        print("\n📖 DETAILED USAGE GUIDE")
        print("=" * 50)
        print()
        print("GENERATION PRESETS:")
        print("  - conservative: Safe, coherent (temp=0.7, top_k=30)")
        print("  - balanced: Good mix (temp=1.0, top_k=50)")
        print("  - creative: High diversity (temp=1.2, top_k=100)")
        print("  - focused: Very deterministic (temp=0.3, top_k=10)")
        print("  - greedy: Most likely tokens (no sampling)")
        print()
        print("CUSTOM PARAMETERS:")
        print("  Format: temp=0.8,top_k=50,top_p=0.9,rep_penalty=1.1,max_tokens=256,show_probs=true,show_alts=true")
        print("  - temp/temperature: Randomness (0.1-2.0)")
        print("  - top_k: Keep top K tokens (1-vocab_size)")
        print("  - top_p: Nucleus sampling threshold (0.1-1.0)")
        print("  - rep_penalty: Repetition penalty (1.0-2.0)")
        print("  - max_tokens: Maximum tokens to generate")
        print("  - sample: Enable sampling (true/false)")
        print("  - show_probs: Show token probabilities (true/false)")
        print("  - show_alts: Show alternative tokens table (true/false)")

def main():
    parser = argparse.ArgumentParser(description="OPAL Fine-tuned Model Tester - Unified Script")
    parser.add_argument("--model-path", "-m", type=str, 
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--tokenizer-path", "-t", type=str,
                       help="Path to tokenizer model (.model file)")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to use for inference")
    parser.add_argument("--prompt", "-p", type=str,
                       help="Single prompt to test (non-interactive mode)")
    parser.add_argument("--preset", type=str,
                       help="Generation preset to use")
    parser.add_argument("--params", type=str,
                       help="Custom generation parameters (format: temp=0.8,top_k=50)")
    parser.add_argument("--guide", "-g", action="store_true",
                       help="Run guided examples mode")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Validate setup and exit")
    
    args = parser.parse_args()
    
    # Handle validation mode
    if args.validate:
        print("🔍 OPAL Fine-tuned Model Setup Validation")
        print("=" * 50)
        success = validate_setup()
        if success:
            print(f"\n✅ Setup validation successful!")
            print(f"🚀 You can now use the tester:")
            print(f"   python3 finetune_model_tester.py")
            return 0
        else:
            print(f"\n❌ Setup validation failed!")
            return 1
    
    # Handle guided examples mode
    if args.guide:
        guided_examples_mode()
        return 0
    
    # Auto-detect paths if not provided
    if not args.model_path:
        args.model_path = find_latest_finetune_checkpoint()
        if args.model_path:
            print(f"🔍 Using auto-detected model: {args.model_path}")
    
    if not args.tokenizer_path:
        args.tokenizer_path = find_tokenizer()
        if args.tokenizer_path:
            print(f"🔍 Using auto-detected tokenizer: {args.tokenizer_path}")
    
    # Check required files
    if not args.model_path or not Path(args.model_path).exists():
        print("❌ Model checkpoint not found!")
        print("   Specify with --model-path or ensure checkpoint exists in expected location")
        print("   Use --validate to check setup")
        return 1
    
    if not args.tokenizer_path or not Path(args.tokenizer_path).exists():
        print("❌ Tokenizer model not found!")
        print("   Specify with --tokenizer-path or ensure tokenizer exists in expected location")
        print("   Use --validate to check setup")
        return 1
    
    # Load model and tokenizer
    model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, args.device
    )
    
    if model is None:
        print("❌ Failed to load model and tokenizer")
        return 1
    
    # Handle single prompt mode
    if args.prompt:
        # Parse generation parameters
        gen_params = {
            'max_new_tokens': 512,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.05,
            'do_sample': True,
            'show_probabilities': True,
            'show_alternatives': True
        }
        
        # Apply preset if specified
        if args.preset:
            presets = get_generation_presets()
            if args.preset in presets:
                preset_params = presets[args.preset].copy()
                preset_params.pop('description', None)
                gen_params.update(preset_params)
                print(f"✅ Using preset: {args.preset}")
            else:
                print(f"⚠️ Unknown preset '{args.preset}', using defaults")
        
        # Apply custom parameters if specified
        if args.params:
            custom_params = parse_generation_params(args.params)
            gen_params.update(custom_params)
            print(f"✅ Custom parameters: {custom_params}")
        
        # Generate
        print(f"\n🚀 Generating for: '{args.prompt}'")
        full_text, generated_text, metadata = generate_with_params(
            model, tokenizer, args.prompt, config, device, **gen_params
        )
        
        # Display results
        if 'error' not in metadata:
            print(f"\n📝 Generated ({metadata['generated_length']} tokens):")
            print(f"{generated_text.strip()}")
            
            # Show text with probabilities if available
            if metadata.get('generated_with_probabilities'):
                print(f"\n🎯 Generated text with probabilities:")
                print(f"{metadata['generated_with_probabilities']}")
            
            # Show alternatives table if available
            if metadata.get('alternatives_table'):
                print(metadata['alternatives_table'])
            
            print(f"\n📊 Stopped: {metadata['stopped_reason']}")
        else:
            print(f"❌ Error: {metadata['error']}")
    else:
        # Interactive mode (default)
        interactive_mode(model, tokenizer, config, device)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
