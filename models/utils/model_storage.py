"""
Utilities for managing model storage, caching, and checkpoints.
"""
import os
import time
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

def save_model_atomic(data: Dict[str, Any], filepath: Path) -> bool:
    """
    Atomically save model data to avoid corruption during writes.
    
    Args:
        data: Dictionary containing model data and metadata
        filepath: Target file path
    
    Returns:
        bool: True if save successful, False otherwise
    """
    temp_file = filepath.with_suffix('.tmp.pkl')
    try:
        # Check write permissions before attempting to save
        if not os.access(filepath.parent, os.W_OK):
            logger.error(f"No write permission for directory: {filepath.parent}")
            return False
            
        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to temporary file
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f)
            
        # Atomic rename
        os.replace(temp_file, filepath)
        return True
        
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {str(e)}")
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception as e2:
                logger.warning(f"Could not remove temporary file {temp_file}: {str(e2)}")
        return False

def cleanup_old_checkpoints(checkpoint_dir: Path, pattern: str = "*.h5", keep_last: int = 5) -> None:
    """
    Remove old model checkpoints, keeping only the N most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: File pattern to match checkpoints
        keep_last: Number of recent checkpoints to keep
    """
    try:
        checkpoints = sorted(
            Path(checkpoint_dir).glob(pattern),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep the N most recent checkpoints
        for checkpoint in checkpoints[keep_last:]:
            try:
                checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")
                
    except Exception as e:
        logger.error(f"Error during checkpoint cleanup: {e}")

def manage_model_cache(
    cache_dir: Path,
    max_size_mb: float = 1000,
    max_age_days: float = 7
) -> None:
    """
    Manage model cache by removing old/large files.
    
    Args:
        cache_dir: Cache directory path
        max_size_mb: Maximum cache size in MB
        max_age_days: Maximum age of cache files in days
    """
    try:
        if not cache_dir.exists():
            return
            
        cache_files = list(cache_dir.glob("*.pkl"))
        current_time = time.time()
        
        # Remove expired files first
        for cache_file in cache_files[:]:
            file_age_days = (current_time - cache_file.stat().st_mtime) / (24 * 3600)
            if file_age_days > max_age_days:
                try:
                    cache_file.unlink()
                    cache_files.remove(cache_file)
                    logger.info(f"Removed expired cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove expired cache file {cache_file}: {e}")
        
        # Check total size and remove oldest files if needed
        total_size = sum(f.stat().st_size for f in cache_files) / (1024 * 1024)
        if total_size > max_size_mb:
            sorted_files = sorted(cache_files, key=lambda x: x.stat().st_mtime)
            
            for cache_file in sorted_files:
                if total_size <= max_size_mb:
                    break
                    
                try:
                    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                    cache_file.unlink()
                    total_size -= file_size_mb
                    logger.info(f"Removed cache file due to size limit: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
                    
    except Exception as e:
        logger.error(f"Error during cache management: {e}")

def get_storage_info(directory: Path) -> Dict[str, Any]:
    """
    Get storage information for a directory.
    
    Args:
        directory: Directory to check
        
    Returns:
        Dict containing storage information
    """
    try:
        total_size = 0
        file_count = 0
        
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
                    file_count += 1
                    
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "file_count": file_count,
            "exists": directory.exists()
        }
    except Exception as e:
        logger.error(f"Error getting storage info for {directory}: {e}")
        return {
            "total_size_mb": 0,
            "file_count": 0,
            "exists": False,
            "error": str(e)
        } 