"""
Progress tracking utilities for model training.
"""
import logging
from typing import Dict, Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

def create_progress_bar(total: int, desc: str = "", position: int = 0, leave: bool = True) -> tqdm:
    """
    Create a progress bar for tracking operations.
    
    Args:
        total: Total number of steps
        desc: Description of the progress bar
        position: Position of the progress bar (for multiple bars)
        leave: Whether to leave the progress bar after completion
        
    Returns:
        tqdm progress bar instance
    """
    return tqdm(
        total=total,
        desc=desc,
        position=position,
        leave=leave,
        ncols=80,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

def create_model_progress_tracker(model_list: list, desc: str = "Training models") -> Dict[str, Any]:
    """
    Create a progress tracker for multiple models.
    
    Args:
        model_list: List of model names to track
        desc: Description of the overall progress
        
    Returns:
        Dictionary containing progress bars
    """
    try:
        # Create main progress bar
        main_bar = create_progress_bar(
            total=len(model_list),
            desc=desc,
            position=0
        )
        
        # Initialize tracker
        tracker = {
            'main': main_bar,
            'current_model': None,
            'status': {}
        }
        
        # Initialize status for each model
        for model in model_list:
            tracker['status'][model] = 'pending'
            
        return tracker
        
    except Exception as e:
        logger.error(f"Error creating progress tracker: {str(e)}")
        return {'main': None, 'current_model': None, 'status': {}}

def update_model_progress(tracker: Dict[str, Any], model_name: str, desc: Optional[str] = None) -> None:
    """
    Update progress for a specific model.
    
    Args:
        tracker: Progress tracker dictionary
        model_name: Name of the model to update
        desc: Optional new description for the progress bar
    """
    try:
        if not tracker or 'main' not in tracker:
            return
            
        # Update main progress bar description if provided
        if desc and hasattr(tracker['main'], 'set_description'):
            tracker['main'].set_description(desc)
            
        # Update model status
        if model_name in tracker.get('status', {}):
            prev_status = tracker['status'][model_name]
            tracker['status'][model_name] = 'complete'
            
            # Only update progress if this is a new completion
            if prev_status != 'complete':
                tracker['main'].update(1)
                
        # Update current model
        tracker['current_model'] = model_name
        
    except Exception as e:
        logger.error(f"Error updating progress for {model_name}: {str(e)}") 