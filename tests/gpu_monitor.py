#!/usr/bin/env python3
"""
GPU Monitor for TEMPEST Training
=================================
Real-time monitoring of GPU utilization during training.

Run in a separate terminal while training:
    python gpu_monitor.py

Press Ctrl+C to stop.
"""

import sys
import time
import subprocess
from datetime import datetime

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not found. Install with: pip install tensorflow")
    sys.exit(1)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def get_gpu_info():
    """Get GPU information using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', 
             '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            return None
            
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'utilization': float(parts[2]),
                        'memory_used': float(parts[3]),
                        'memory_total': float(parts[4]),
                        'temperature': float(parts[5]),
                        'power': float(parts[6])
                    })
        return gpus
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_tensorflow_gpu_info():
    """Get GPU information from TensorFlow."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        return gpus
    except:
        return []


def draw_bar(percentage, width=30):
    """Draw a progress bar."""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return bar


def format_memory(mb):
    """Format memory in MB to readable format."""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb:.0f}MB"


def print_gpu_status(gpus):
    """Print formatted GPU status."""
    print("\n" + "="*80)
    print(f"  GPU Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    if not gpus:
        print("⚠ No GPU information available")
        print("  - Check if nvidia-smi is installed")
        print("  - Verify NVIDIA drivers are working")
        return
    
    for gpu in gpus:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print("-" * 80)
        
        # Utilization
        util_bar = draw_bar(gpu['utilization'])
        print(f"  Utilization: {util_bar} {gpu['utilization']:5.1f}%")
        
        # Memory
        mem_percent = (gpu['memory_used'] / gpu['memory_total']) * 100
        mem_bar = draw_bar(mem_percent)
        mem_used = format_memory(gpu['memory_used'])
        mem_total = format_memory(gpu['memory_total'])
        print(f"  Memory:      {mem_bar} {mem_percent:5.1f}% ({mem_used}/{mem_total})")
        
        # Temperature
        temp_color = ""
        if gpu['temperature'] > 80:
            temp_color = "\033[91m"  # Red
        elif gpu['temperature'] > 70:
            temp_color = "\033[93m"  # Yellow
        else:
            temp_color = "\033[92m"  # Green
        print(f"  Temperature: {temp_color}{gpu['temperature']:5.1f}°C\033[0m")
        
        # Power
        print(f"  Power:       {gpu['power']:5.1f}W")
        print()


def print_tensorflow_info():
    """Print TensorFlow GPU configuration."""
    tf_gpus = get_tensorflow_gpu_info()
    
    if tf_gpus:
        print("TensorFlow GPU Configuration:")
        for i, gpu in enumerate(tf_gpus):
            print(f"  {i}: {gpu.name}")
    else:
        print("⚠ TensorFlow: No GPUs detected")
    print()


def print_instructions():
    """Print usage instructions."""
    print("\nMonitoring GPUs...")
    print("  - Updates every 2 seconds")
    print("  - Press Ctrl+C to stop")
    print("  - Run alongside training in a separate terminal")
    print()


def main():
    """Main monitoring loop."""
    print("\n" + "="*80)
    print("  TEMPEST GPU Monitor")
    print("="*80)
    
    # Initial TensorFlow check
    print_tensorflow_info()
    print_instructions()
    
    # Check if nvidia-smi is available
    gpus = get_gpu_info()
    if gpus is None:
        print("✗ nvidia-smi not found or not working")
        print("  Cannot monitor GPU status without NVIDIA drivers")
        return 1
    
    try:
        while True:
            clear_screen()
            gpus = get_gpu_info()
            print_gpu_status(gpus)
            
            # Usage tips
            print("Tips:")
            print("  - Utilization should be >80% during training for good GPU usage")
            print("  - If utilization is low, increase batch size")
            print("  - If memory is full, decrease batch size")
            print("  - Temperature should stay below 85°C")
            print("\nPress Ctrl+C to exit")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        return 0
    except Exception as e:
        print(f"\n\nError: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
