"""Feature visualization utilities for lottery prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from pathlib import Path
import time
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scripts.utils import setup_logging, LOG_DIR

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def visualize_features(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> None:
    """Visualize features using various methods."""
    try:
        logger.info(f"Creating visualizations for {len(feature_names)} features")
        
        # Create output directory
        output_dir = Path("outputs/visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save visualization
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(output_dir / f"feature_visualization_{timestamp}.png")
        logger.info(f"Visualization saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error visualizing features: {str(e)}")
        logger.debug(traceback.format_exc()) 