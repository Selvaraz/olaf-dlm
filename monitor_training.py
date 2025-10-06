#!/usr/bin/env python3
"""
Training Monitor for Large Corpus Training
Monitor memory usage, training progress, and system health during large corpus training.
"""

import psutil
import torch
import time
import os
from datetime import datetime

def get_memory_info():
    """Get current memory usage information."""
    # System memory
    memory = psutil.virtual_memory()
    system_ram_used = memory.used / (1024**3)  # GB
    system_ram_total = memory.total / (1024**3)  # GB
    system_ram_percent = memory.percent
    
    # GPU memory (if available)
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
        gpu_info = {
            "gpu_used": gpu_memory_used,
            "gpu_total": gpu_memory_total,
            "gpu_percent": gpu_memory_percent
        }
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory query, use process memory as approximation
        process = psutil.Process(os.getpid())
        mps_memory = process.memory_info().rss / (1024**3)  # GB
        gpu_info = {
            "mps_memory": mps_memory
        }
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_ram_used": system_ram_used,
        "system_ram_total": system_ram_total,
        "system_ram_percent": system_ram_percent,
        **gpu_info
    }

def monitor_training(interval=30):
    """Monitor training progress with specified interval (seconds)."""
    print("🔍 Training Monitor Started")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        while True:
            info = get_memory_info()
            elapsed = time.time() - start_time
            
            print(f"\n📊 {info['timestamp']} (Elapsed: {elapsed/3600:.1f}h)")
            print(f"🖥️  System RAM: {info['system_ram_used']:.1f}GB / {info['system_ram_total']:.1f}GB ({info['system_ram_percent']:.1f}%)")
            
            if 'gpu_used' in info:
                print(f"🎮 GPU Memory: {info['gpu_used']:.1f}GB / {info['gpu_total']:.1f}GB ({info['gpu_percent']:.1f}%)")
            elif 'mps_memory' in info:
                print(f"🍎 MPS Memory: {info['mps_memory']:.1f}GB")
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            print(f"⚡ CPU Usage: {cpu_percent:.1f}%")
            
            # Check for potential issues
            if info['system_ram_percent'] > 90:
                print("⚠️  WARNING: System RAM usage very high!")
            if 'gpu_percent' in info and info['gpu_percent'] > 95:
                print("⚠️  WARNING: GPU memory usage very high!")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        print(f"Total monitoring time: {(time.time() - start_time)/3600:.1f} hours")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--interval", type=int, default=30, help="Monitoring interval in seconds")
    args = parser.parse_args()
    
    monitor_training(args.interval)