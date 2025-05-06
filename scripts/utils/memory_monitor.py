"""
Memory monitoring utilities to track and log memory usage during model training.

This module provides functions to:
1. Monitor CPU and RAM usage during model training
2. Monitor GPU memory if available
3. Create memory usage summaries and plots
4. Optimize batch sizes based on memory constraints
5. Provide TensorFlow callbacks for easy integration
"""

import os
import time
import json
import threading
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable, Optional, Union, Tuple, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs/memory")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Global variables for memory tracking
_memory_data = {
    "timestamps": [],
    "cpu_percent": [],
    "ram_used_gb": [],
    "ram_percent": []
}

# Try to import TensorFlow for GPU monitoring
try:
    import tensorflow as tf
    _memory_data["gpu_memory_used_gb"] = []
    _memory_data["gpu_memory_percent"] = []
    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False
    
try:
    import GPUtil
    _HAS_GPUTIL = True
except ImportError:
    _HAS_GPUTIL = False

# Configure TensorFlow memory growth if available
def configure_memory_growth():
    """Configure TensorFlow to allow memory growth on GPU to prevent OOM errors."""
    if _HAS_TENSORFLOW:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled on {len(gpus)} GPUs")
            except RuntimeError as e:
                print(f"Memory growth configuration error: {e}")
        else:
            print("No GPUs found. Using CPU only.")
    else:
        print("TensorFlow not available. Memory growth configuration skipped.")

# Function to get current memory usage
def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics for CPU and GPU if available."""
    # Get CPU and RAM info
    memory_info = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    ram_used_gb = memory_info.used / (1024**3)  # Convert to GB
    ram_percent = memory_info.percent
    
    result = {
        "cpu_percent": cpu_percent,
        "ram_used_gb": ram_used_gb,
        "ram_percent": ram_percent
    }
    
    # Get GPU info if available
    if _HAS_TENSORFLOW:
        try:
            # TensorFlow GPU memory monitoring
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_memory_info = tf.config.experimental.get_memory_info(gpus[0])
                total_memory = gpu_memory_info['current'] / (1024**3)  # Convert to GB
                result["gpu_memory_used_gb"] = total_memory
                # We don't have total memory from tf.config, so skip percentage
        except:
            pass
    
    if _HAS_GPUTIL:
        try:
            # GPUtil GPU memory monitoring (more detailed)
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                result["gpu_memory_used_gb"] = gpu.memoryUsed / 1024  # Convert to GB
                result["gpu_memory_percent"] = gpu.memoryUtil * 100
        except:
            pass
            
    return result

def log_memory_usage(label: str = ""):
    """Log current memory usage with optional label."""
    memory_info = get_memory_usage()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Memory usage {label}:")
    print(f"  CPU: {memory_info['cpu_percent']:.1f}%")
    print(f"  RAM: {memory_info['ram_used_gb']:.2f} GB ({memory_info['ram_percent']:.1f}%)")
    
    if 'gpu_memory_used_gb' in memory_info:
        print(f"  GPU: {memory_info['gpu_memory_used_gb']:.2f} GB", end="")
        if 'gpu_memory_percent' in memory_info:
            print(f" ({memory_info['gpu_memory_percent']:.1f}%)")
        else:
            print()
    
    return memory_info

class MemoryMonitor:
    """Class to monitor memory usage during model training."""
    
    def __init__(self, 
                 log_dir: Union[str, Path] = "logs/memory", 
                 interval: float = 1.0,
                 plot: bool = True,
                 save_csv: bool = True,
                 verbose: bool = False):
        """
        Initialize memory monitor.
        
        Args:
            log_dir: Directory to save logs and plots
            interval: Sampling interval in seconds
            plot: Whether to generate plots when stopped
            save_csv: Whether to save data as CSV when stopped
            verbose: Whether to print memory info during monitoring
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.plot = plot
        self.save_csv = save_csv
        self.verbose = verbose
        
        # Initialize memory data storage
        self.data = {
            "timestamps": [],
            "cpu_percent": [],
            "ram_used_gb": [],
            "ram_percent": []
        }
        
        if _HAS_TENSORFLOW or _HAS_GPUTIL:
            self.data["gpu_memory_used_gb"] = []
            self.data["gpu_memory_percent"] = []
        
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _monitor_loop(self):
        """Internal monitoring loop that collects memory data."""
        while self.running:
            try:
                # Get current time
                current_time = datetime.now()
                elapsed = (current_time - self.start_time).total_seconds()
                
                # Get memory usage
                memory_info = get_memory_usage()
                
                # Store data
                self.data["timestamps"].append(elapsed)
                self.data["cpu_percent"].append(memory_info["cpu_percent"])
                self.data["ram_used_gb"].append(memory_info["ram_used_gb"])
                self.data["ram_percent"].append(memory_info["ram_percent"])
                
                if "gpu_memory_used_gb" in memory_info:
                    self.data["gpu_memory_used_gb"].append(memory_info["gpu_memory_used_gb"])
                    if "gpu_memory_percent" in memory_info:
                        self.data["gpu_memory_percent"].append(memory_info["gpu_memory_percent"])
                
                # Print info if verbose
                if self.verbose:
                    print(f"[{elapsed:.1f}s] CPU: {memory_info['cpu_percent']:.1f}%, "
                          f"RAM: {memory_info['ram_used_gb']:.2f} GB ({memory_info['ram_percent']:.1f}%)", 
                          end="")
                    
                    if "gpu_memory_used_gb" in memory_info:
                        print(f", GPU: {memory_info['gpu_memory_used_gb']:.2f} GB", end="")
                        if "gpu_memory_percent" in memory_info:
                            print(f" ({memory_info['gpu_memory_percent']:.1f}%)")
                        else:
                            print()
                    else:
                        print()
                
                # Sleep for interval
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                time.sleep(self.interval)
    
    def start(self):
        """Start memory monitoring in a background thread."""
        if self.running:
            print("Memory monitoring is already running.")
            return
        
        self.running = True
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"Memory monitoring started. Sampling interval: {self.interval}s")
    
    def stop(self):
        """Stop memory monitoring and generate reports if configured."""
        if not self.running:
            print("Memory monitoring is not running.")
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Save data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        if self.save_csv and len(self.data["timestamps"]) > 0:
            df = pd.DataFrame(self.data)
            csv_path = self.log_dir / f"memory_log_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Memory log saved to {csv_path}")
        
        # Generate summary report
        summary = self._generate_summary()
        summary_path = self.log_dir / f"memory_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Memory summary saved to {summary_path}")
        
        # Generate plots
        if self.plot and len(self.data["timestamps"]) > 0:
            self._generate_plots(timestamp)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from collected memory data."""
        if not self.data["timestamps"]:
            return {"error": "No data collected"}
        
        summary = {
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": self.data["timestamps"][-1],
            "samples": len(self.data["timestamps"]),
            "sampling_interval": self.interval,
            "cpu_percent": {
                "mean": np.mean(self.data["cpu_percent"]),
                "max": np.max(self.data["cpu_percent"]),
                "min": np.min(self.data["cpu_percent"]),
                "std": np.std(self.data["cpu_percent"])
            },
            "ram_used_gb": {
                "mean": np.mean(self.data["ram_used_gb"]),
                "max": np.max(self.data["ram_used_gb"]),
                "min": np.min(self.data["ram_used_gb"]),
                "std": np.std(self.data["ram_used_gb"])
            },
            "ram_percent": {
                "mean": np.mean(self.data["ram_percent"]),
                "max": np.max(self.data["ram_percent"]),
                "min": np.min(self.data["ram_percent"]),
                "std": np.std(self.data["ram_percent"])
            }
        }
        
        # Add GPU stats if available
        if "gpu_memory_used_gb" in self.data and self.data["gpu_memory_used_gb"]:
            summary["gpu_memory_used_gb"] = {
                "mean": np.mean(self.data["gpu_memory_used_gb"]),
                "max": np.max(self.data["gpu_memory_used_gb"]),
                "min": np.min(self.data["gpu_memory_used_gb"]),
                "std": np.std(self.data["gpu_memory_used_gb"])
            }
            
            if "gpu_memory_percent" in self.data and self.data["gpu_memory_percent"]:
                summary["gpu_memory_percent"] = {
                    "mean": np.mean(self.data["gpu_memory_percent"]),
                    "max": np.max(self.data["gpu_memory_percent"]),
                    "min": np.min(self.data["gpu_memory_percent"]),
                    "std": np.std(self.data["gpu_memory_percent"])
                }
        
        return summary
    
    def _generate_plots(self, timestamp: str):
        """Generate plots of memory usage over time."""
        try:
            # Set up the figure with subplots
            has_gpu = "gpu_memory_used_gb" in self.data and len(self.data["gpu_memory_used_gb"]) > 0
            
            n_plots = 3 if has_gpu else 2
            fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), sharex=True)
            
            # Plot CPU usage
            axes[0].plot(self.data["timestamps"], self.data["cpu_percent"], 'b-')
            axes[0].set_title('CPU Usage')
            axes[0].set_ylabel('Percentage (%)')
            axes[0].grid(True)
            
            # Plot RAM usage
            ax1 = axes[1]
            ax1.plot(self.data["timestamps"], self.data["ram_used_gb"], 'g-')
            ax1.set_title('RAM Usage')
            ax1.set_ylabel('Used (GB)')
            ax1.grid(True)
            
            ax2 = ax1.twinx()
            ax2.plot(self.data["timestamps"], self.data["ram_percent"], 'r--')
            ax2.set_ylabel('Usage (%)')
            
            # Plot GPU usage if available
            if has_gpu:
                ax3 = axes[2]
                ax3.plot(self.data["timestamps"], self.data["gpu_memory_used_gb"], 'm-')
                ax3.set_title('GPU Memory Usage')
                ax3.set_ylabel('Used (GB)')
                ax3.grid(True)
                
                if "gpu_memory_percent" in self.data and len(self.data["gpu_memory_percent"]) > 0:
                    ax4 = ax3.twinx()
                    ax4.plot(self.data["timestamps"], self.data["gpu_memory_percent"], 'c--')
                    ax4.set_ylabel('Usage (%)')
            
            # Common x-axis label
            axes[-1].set_xlabel('Time (seconds)')
            
            # Add title and adjust layout
            fig.suptitle('Memory Usage Monitoring')
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save figure
            plot_path = self.log_dir / f"memory_plot_{timestamp}.png"
            fig.savefig(plot_path)
            print(f"Memory plot saved to {plot_path}")
            
            # Close figure to prevent memory leaks
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating memory plots: {e}")

# Function to get a memory monitor instance
def get_memory_monitor(**kwargs) -> MemoryMonitor:
    """Get a memory monitor instance with the specified configuration."""
    monitor = MemoryMonitor(**kwargs)
    return monitor

# Function to create a TensorFlow callback for memory monitoring
def memory_callback(name: str = "", log_interval: int = 10) -> Any:
    """
    Create a TensorFlow callback that logs memory usage during training.
    
    Args:
        name: Name prefix for the logs
        log_interval: How often to log memory usage (in epochs)
        
    Returns:
        TensorFlow callback object
    """
    if not _HAS_TENSORFLOW:
        print("TensorFlow not available. Memory callback not created.")
        return None
    
    class MemoryMonitorCallback(tf.keras.callbacks.Callback):
        def __init__(self, name: str, log_interval: int):
            super().__init__()
            self.name = name
            self.log_interval = log_interval
            
        def on_train_begin(self, logs=None):
            log_memory_usage(f"{self.name} - training start")
            
        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.log_interval == 0:
                log_memory_usage(f"{self.name} - epoch {epoch + 1}")
                
        def on_train_end(self, logs=None):
            log_memory_usage(f"{self.name} - training end")
    
    return MemoryMonitorCallback(name, log_interval)

# Function to optimize batch size based on memory constraints
def optimize_batch_size(
    model_fn: Callable[[int], Any],
    X: np.ndarray,
    y: np.ndarray,
    min_batch: int = 16,
    max_batch: int = 256,
    target_memory_usage: float = 0.8,
    verbose: bool = True
) -> int:
    """
    Find optimal batch size that fits within memory constraints.
    
    Args:
        model_fn: Function that creates model given batch size
        X: Input data
        y: Target data
        min_batch: Minimum batch size to try
        max_batch: Maximum batch size to try
        target_memory_usage: Target memory usage (0.0-1.0)
        verbose: Whether to print detailed information
        
    Returns:
        Optimal batch size
    """
    if verbose:
        print("Optimizing batch size...")
    
    # Start with maximum batch size and try to fit in memory
    batch_size = max_batch
    
    # Binary search approach
    low = min_batch
    high = max_batch
    optimal_batch_size = min_batch  # Default to minimum
    
    while low <= high:
        batch_size = (low + high) // 2
        if verbose:
            print(f"Trying batch size: {batch_size}")
        
        # Get current memory
        initial_memory = get_memory_usage()
        
        try:
            # Create model
            model = model_fn(batch_size)
            
            # Try to fit model with small number of steps
            X_sample = X[:min(1000, X.shape[0])]
            y_sample = y[:min(1000, y.shape[0])]
            
            # Compile and run a single batch
            model.fit(X_sample, y_sample, epochs=1, batch_size=batch_size, verbose=0)
            
            # Get memory after fitting
            post_memory = get_memory_usage()
            
            # Calculate memory increase
            ram_usage = post_memory["ram_percent"] / 100.0
            
            if verbose:
                print(f"  Memory usage with batch size {batch_size}: {ram_usage:.2%}")
            
            # Check if we're within limits
            if ram_usage < target_memory_usage:
                optimal_batch_size = batch_size
                low = batch_size + 1  # Try larger batch size
            else:
                high = batch_size - 1  # Try smaller batch size
                
            # Clean up
            del model
            import gc
            gc.collect()
            
            if _HAS_TENSORFLOW:
                tf.keras.backend.clear_session()
                
        except (tf.errors.ResourceExhaustedError, ValueError, Exception) as e:
            # Memory error or other exception
            if verbose:
                print(f"  Error with batch size {batch_size}: {str(e)}")
            high = batch_size - 1  # Try smaller batch size
    
    if verbose:
        print(f"Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size 