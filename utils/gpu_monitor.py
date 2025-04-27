"""GPU monitoring utilities for deep learning models."""

import logging
import threading
import time
from typing import Optional, Dict, List, Any
import platform
import psutil
import tensorflow as tf
import os

logger = logging.getLogger(__name__)

# Mac doesn't need GPU monitoring
HAS_GPU = False

class GPUMonitor:
    """A simplified GPU monitor that works on Mac M1/M2."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.keep_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.gpu_logs: List[Dict] = []
        self.enabled = False  # Mac GPU monitoring not implemented
        
    def start(self) -> None:
        """Start monitoring."""
        if not self.enabled:
            logger.info("GPU monitoring not enabled on Mac")
            return
            
        if self.monitor_thread is not None:
            logger.warning("Monitor already running")
            return
            
        self.keep_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self) -> None:
        """Stop monitoring."""
        self.keep_running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join()
            self.monitor_thread = None
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU stats."""
        return {
            'timestamp': time.time(),
            'gpu_utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'memory_free': 0,
            'temperature': 0
        }
        
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.keep_running:
            try:
                stats = self.get_stats()
                self.gpu_logs.append(stats)
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {e}")
            time.sleep(self.interval)
            
    def log_stats(self) -> None:
        """Log current GPU stats."""
        if not self.enabled:
            return
            
        try:
            stats = self.get_stats()
            logger.info(
                f"GPU Memory: {stats['memory_used']:.0f}MB / {stats['memory_total']:.0f}MB "
                f"({stats['memory_used']/stats['memory_total']*100:.1f}%) "
                f"Utilization: {stats['gpu_utilization']:.1f}%"
            )
        except Exception as e:
            logger.error(f"Error logging GPU stats: {e}")
            
    def get_gpu_stats(self) -> Dict:
        """Get current GPU statistics.
        
        Returns:
            Dictionary containing GPU statistics
        """
        if not self.enabled:
            return {
                'available': False,
                'error': 'GPU monitoring not enabled'
            }
        
        try:
            # Get TensorFlow memory growth settings
            tf_memory_growth = tf.config.experimental.get_memory_growth(
                tf.config.list_physical_devices('GPU')[0]
            ) if tf.config.list_physical_devices('GPU') else None
            
            return {
                'available': True,
                'name': 'Apple M1/M2 GPU',
                'memory_total': 0,
                'memory_used': 0,
                'memory_free': 0,
                'memory_used_percent': 0,
                'gpu_load': 0,
                'gpu_temperature': 0,
                'tf_memory_growth': tf_memory_growth
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e)
            }

    def get_memory_usage_summary(self) -> Dict:
        """Get summary of GPU memory usage during monitoring."""
        return {'error': 'GPU monitoring not available on Mac'}

    def get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization percentage."""
        return None
            
    def get_memory_usage(self) -> Optional[dict]:
        """Get current GPU memory usage in MB."""
        return None

def get_gpu_monitor(interval: float = 1.0) -> GPUMonitor:
    """Factory function to get a GPU monitor instance."""
    return GPUMonitor(interval)

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = get_gpu_monitor()
    monitor.start()
    time.sleep(2)
    monitor.stop() 