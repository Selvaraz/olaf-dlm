import torch
from torch.utils.data import DataLoader

def create_mps_safe_dataloader(dataset, batch_size=4, **kwargs):
    """Create MPS-optimized DataLoader for Apple Silicon."""
    
    mps_config = {
        'num_workers': 0,  # ✅ MPS FIX: Must use 0 workers on Apple Silicon
        'pin_memory': False,  # ✅ MPS FIX: Disable pin_memory for MPS
        'persistent_workers': False,
        'prefetch_factor': None,  # ✅ MPS FIX: Must be None when num_workers=0
        'drop_last': True,  # Consistent batch sizes for MPS
    }
    
    # Override with user kwargs but keep MPS-safe defaults
    final_config = {**mps_config, **kwargs}
    
    print("🍎 MPS DataLoader Configuration:")
    for key, value in final_config.items():
        print(f"   {key}: {value}")
    
    return DataLoader(dataset, batch_size=batch_size, **final_config)

def mps_safe_optimizer_step(optimizer, model, gradient_accumulation_steps=1, current_step=0, max_grad_norm=1.0):
    """MPS-safe optimizer step with proper synchronization.
    
    Note: This function assumes backward() has already been called on the loss.
    It only handles gradient clipping, optimizer step, and MPS synchronization.
    """
    
    # ✅ MPS FIX: Synchronize after backward pass (already done by caller)
    if next(model.parameters()).device.type == 'mps':
        if hasattr(torch.mps, 'synchronize'):
            try:
                torch.mps.synchronize()
            except Exception as e:
                print(f"⚠️ MPS synchronization warning: {e}")
    
    if (current_step + 1) % gradient_accumulation_steps == 0:
        # ✅ MPS FIX: Gradient clipping before optimizer step
        if next(model.parameters()).device.type == 'mps':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        
        # ✅ MPS FIX: Synchronize after optimizer step
        if next(model.parameters()).device.type == 'mps':
            if hasattr(torch.mps, 'synchronize'):
                try:
                    torch.mps.synchronize()
                except Exception as e:
                    print(f"⚠️ MPS synchronization warning: {e}")
        
        return True  # Step was taken
    
    return False  # No step taken

def setup_mps_environment():
    """Setup optimal MPS environment for training."""
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available, falling back to CPU")
        return torch.device('cpu')
    
    print("🍎 Setting up MPS environment for fine-tuning...")
    
    # Set memory fraction (be very conservative - try different values)
    if hasattr(torch.mps, 'set_per_process_memory_fraction'):
        try:
            # Try progressively smaller fractions (extremely aggressive for memory pressure)
            for fraction in [0.1, 0.08, 0.06]:  # Even smaller for memory pressure
                try:
                    torch.mps.set_per_process_memory_fraction(fraction)
                    print(f"   ✅ Memory fraction set to {int(fraction*100)}% (extreme conservation for memory pressure)")
                    break
                except Exception as inner_e:
                    print(f"   ⚠️ Fraction {fraction} failed: {inner_e}")
                    continue
            else:
                print("   ⚠️ Could not set any memory fraction, using defaults")
        except Exception as e:
            print(f"   ⚠️ Memory fraction API not working: {e}")
    
    # Set default dtype
    torch.set_default_dtype(torch.float32)
    print("   ✅ Default dtype set to float32")
    
    # Clear MPS cache if available (with error handling)
    try:
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            print("   ✅ MPS cache cleared")
    except Exception as e:
        print(f"   ⚠️ Could not clear MPS cache: {e}")
    
    # Test basic MPS functionality with smaller tensors
    try:
        test_tensor = torch.randn(10, 10, device='mps')  # Very small test tensor
        test_result = torch.mm(test_tensor, test_tensor.t())
        print("   ✅ MPS basic operations test passed")
        
        # Clean up test tensors immediately
        del test_tensor, test_result
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    except Exception as e:
        print(f"   ❌ MPS test failed: {e}")
        print("   🔄 Falling back to CPU for stability")
        return torch.device('cpu')
    
    return torch.device('mps')

def aggressive_mps_cleanup():
    """Perform aggressive MPS memory cleanup for fine-tuning."""
    if torch.backends.mps.is_available():
        try:
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            print("🍎 Aggressive MPS memory cleanup completed")
        except Exception as e:
            print(f"⚠️ MPS cleanup warning: {e}")

def get_mps_memory_info():
    """Get current MPS memory usage information."""
    if torch.backends.mps.is_available():
        try:
            # Get memory info (if available)
            allocated = torch.mps.current_allocated_memory() / (1024**3)  # GB
            cached = torch.mps.driver_allocated_memory() / (1024**3)      # GB
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': allocated + cached
            }
        except Exception:
            return {'allocated_gb': 0, 'cached_gb': 0, 'total_gb': 0}
    return {'allocated_gb': 0, 'cached_gb': 0, 'total_gb': 0}

def check_mps_memory_safety(threshold_gb=2.5):  # Even more conservative for consistent training
    """Check if MPS memory usage is within safe limits."""
    memory_info = get_mps_memory_info()
    total_memory = memory_info['total_gb']
    
    if total_memory > threshold_gb:
        print(f"⚠️ MPS Memory Warning: {total_memory:.2f}GB > {threshold_gb}GB threshold")
        extreme_mps_cleanup()  # Use extreme cleanup
        return False
    
    return True

def ultra_conservative_mps_cleanup():
    """Ultra-conservative MPS cleanup for memory-constrained training."""
    if torch.backends.mps.is_available():
        try:
            # Multiple cache clears with synchronization
            for _ in range(5):  # Increased from 3 to 5
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            
            # Force Python garbage collection
            import gc
            gc.collect()
            gc.collect()  # Call twice for thoroughness
            
            #print("🍎 Ultra-conservative MPS cleanup completed")
        except Exception as e:
            print(f"⚠️ Ultra-conservative cleanup warning: {e}")

def extreme_mps_cleanup():
    """Extreme MPS cleanup for memory crisis situations."""
    if torch.backends.mps.is_available():
        try:
            # Clear all possible caches
            for _ in range(10):  # Extreme cleanup
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                if hasattr(torch.mps, 'synchronize'):
                    torch.mps.synchronize()
            
            # Force multiple garbage collections
            import gc
            for _ in range(5):
                gc.collect()
            
            print("🍎 EXTREME MPS cleanup completed")
        except Exception as e:
            print(f"⚠️ Extreme cleanup warning: {e}")

def mps_memory_safe_backward(loss, cleanup_before=True, cleanup_after=True):
    """Memory-safe backward pass for MPS with aggressive cleanup."""
    if cleanup_before:
        extreme_mps_cleanup()
    
    try:
        loss.backward()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("🚨 MPS OOM during backward pass - attempting recovery")
            extreme_mps_cleanup()
            # Try to recover by clearing everything and retrying
            raise e  # Re-raise for handling by caller
        else:
            raise e
    
    if cleanup_after:
        extreme_mps_cleanup()