#!/usr/bin/env python3
"""
Simple script to run lottery predictions using the configured model ensemble.
"""

from main import main
import logging

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run prediction with default settings (10 predictions)
    results = main(retrain=True)  # Set to False to use existing trained models
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        exit(1) 