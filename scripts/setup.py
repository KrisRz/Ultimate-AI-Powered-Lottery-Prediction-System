"""Setup script for installing dependencies and checking GPU availability."""

import subprocess
import sys
import logging
import os
from typing import List, Tuple
import pkg_resources
import tensorflow as tf

logger = logging.getLogger(__name__)

def check_gpu() -> Tuple[bool, str]:
    """Check GPU availability and return status."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = []
            for gpu in gpus:
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    gpu_info.append(f"- {gpu_details.get('device_name', 'Unknown GPU')}")
                except:
                    gpu_info.append(f"- {gpu.name}")
            return True, f"Found {len(gpus)} GPU(s):\n" + "\n".join(gpu_info)
        return False, "No GPU devices found"
    except Exception as e:
        return False, f"Error checking GPU: {str(e)}"

def get_required_packages() -> List[str]:
    """Get list of required packages from requirements.txt."""
    try:
        with open('requirements.txt', 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        logger.error(f"Error reading requirements.txt: {e}")
        return []

def check_installed_packages(required_packages: List[str]) -> Tuple[List[str], List[str]]:
    """Check which packages are installed and which are missing."""
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            pkg_resources.require(package)
            installed.append(package)
        except:
            missing.append(package)
            
    return installed, missing

def install_packages(packages: List[str]) -> bool:
    """Install missing packages using pip."""
    if not packages:
        return True
        
    try:
        # First try installing all packages at once
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning("Bulk installation failed, trying individual installation")
            
            # Try installing packages one by one
            failed_packages = []
            for package in packages:
                cmd = [sys.executable, "-m", "pip", "install", package]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    failed_packages.append(package)
                    logger.error(f"Failed to install {package}: {result.stderr}")
            
            if failed_packages:
                logger.error(f"Failed to install packages: {', '.join(failed_packages)}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Error installing packages: {e}")
        return False

def setup_environment():
    """Set up the environment by installing dependencies and checking GPU."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting environment setup...")
    
    # Check GPU availability
    has_gpu, gpu_message = check_gpu()
    logger.info(f"GPU Status: {gpu_message}")
    
    if has_gpu:
        logger.info("Configuring GPU memory growth...")
        try:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU memory growth configured successfully")
        except Exception as e:
            logger.warning(f"Could not configure GPU memory growth: {e}")
    
    # Get required packages
    required_packages = get_required_packages()
    if not required_packages:
        logger.error("Could not read requirements.txt")
        return False
    
    # Check installed packages
    installed, missing = check_installed_packages(required_packages)
    
    logger.info(f"Found {len(installed)} installed packages")
    if missing:
        logger.info(f"Missing {len(missing)} packages: {', '.join(missing)}")
        
        # Install missing packages
        logger.info("Installing missing packages...")
        if install_packages(missing):
            logger.info("All missing packages installed successfully")
        else:
            logger.error("Failed to install some packages")
            return False
    else:
        logger.info("All required packages are already installed")
    
    # Final GPU check after installations
    if has_gpu:
        try:
            # Try to import and initialize CUDA
            import tensorflow as tf
            if tf.test.is_built_with_cuda():
                logger.info("TensorFlow is built with CUDA")
                logger.info(f"TensorFlow GPU devices: {tf.config.list_physical_devices('GPU')}")
            else:
                logger.warning("TensorFlow is not built with CUDA")
        except Exception as e:
            logger.warning(f"Error checking CUDA status: {e}")
    
    logger.info("Environment setup completed")
    return True

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1) 