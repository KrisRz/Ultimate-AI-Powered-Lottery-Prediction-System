"""
Tests for model storage utilities
"""
import os
import time
import shutil
import pytest
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from models.utils.model_storage import (
    save_model_atomic,
    cleanup_old_checkpoints,
    manage_model_cache,
    get_storage_info
)

@pytest.fixture
def test_model():
    model = Sequential()
    model.add(Dense(units=1, input_shape=(1,)))
    model.compile(optimizer='adam', loss='mse')
    return model

@pytest.fixture
def test_dirs(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    cache_dir = tmp_path / "cache"
    checkpoint_dir.mkdir()
    cache_dir.mkdir()
    return checkpoint_dir, cache_dir

def test_save_model_atomic(test_model, tmp_path):
    model_data = {
        'model': test_model,
        'timestamp': time.time(),
        'metadata': {'test': True}
    }
    
    # Test successful save
    save_path = tmp_path / "test_model.pkl"
    assert save_model_atomic(model_data, save_path) is True
    assert save_path.exists()
    
    # Test failed save to read-only directory
    read_only_dir = tmp_path / "read_only"
    read_only_dir.mkdir()
    os.chmod(read_only_dir, 0o444)  # Read-only for all
    
    fail_path = read_only_dir / "should_fail.pkl"
    assert save_model_atomic(model_data, fail_path) is False
    
    # Check file doesn't exist (handle permission error)
    try:
        exists = fail_path.exists()
    except PermissionError:
        exists = False
    assert not exists
    
    # Cleanup
    os.chmod(read_only_dir, 0o777)  # Restore permissions for cleanup
    shutil.rmtree(read_only_dir)

def test_cleanup_old_checkpoints(test_dirs):
    checkpoint_dir = test_dirs[0]
    
    # Create test checkpoint files
    for i in range(10):
        with open(checkpoint_dir / f"model_checkpoint_{i}.h5", 'w') as f:
            f.write("test")
        time.sleep(0.1)  # Ensure different timestamps
    
    cleanup_old_checkpoints(checkpoint_dir, pattern="model_checkpoint_*.h5", keep_last=5)
    remaining = list(checkpoint_dir.glob("model_checkpoint_*.h5"))
    assert len(remaining) == 5

def test_manage_model_cache(test_dirs):
    cache_dir = test_dirs[1]
    
    # Create test cache files
    for i in range(5):
        with open(cache_dir / f"model_{i}.pkl", 'wb') as f:
            f.write(b'0' * 1024 * 1024)  # 1MB file
        time.sleep(0.1)
    
    # Test size-based cleanup
    manage_model_cache(cache_dir, max_size_mb=2.5)
    remaining = list(cache_dir.glob("*.pkl"))
    assert len(remaining) <= 3
    
    # Test age-based cleanup
    old_time = time.time() - 8 * 24 * 3600  # 8 days old
    for file in remaining:
        os.utime(file, (old_time, old_time))
    
    manage_model_cache(cache_dir, max_age_days=7)
    assert len(list(cache_dir.glob("*.pkl"))) == 0

def test_get_storage_info(test_dirs):
    cache_dir = test_dirs[1]
    
    # Create test files
    for i in range(3):
        with open(cache_dir / f"model_{i}.pkl", 'wb') as f:
            f.write(b'0' * 1024 * 1024)  # 1MB file
    
    info = get_storage_info(cache_dir)
    assert info['exists'] is True
    assert info['file_count'] == 3
    assert abs(info['total_size_mb'] - 3.0) < 0.1  # About 3MB
    
    # Test non-existent directory
    info = get_storage_info(cache_dir / "nonexistent")
    assert info['exists'] is False
    assert info['file_count'] == 0
    assert info['total_size_mb'] == 0 