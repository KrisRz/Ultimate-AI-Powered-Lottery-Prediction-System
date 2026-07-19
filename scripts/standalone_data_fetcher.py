#!/usr/bin/env python3
"""
Standalone lottery data fetcher - downloads fresh data without complex dependencies
"""

from datetime import datetime
import logging
from pathlib import Path
import sys

import pandas as pd
import requests

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = Path("data")
DOWNLOADED_FILE = DATA_DIR / "lotto-draw-history.csv"
EXISTING_FILE = DATA_DIR / "lottery_data_1995_2025.csv"
MERGED_FILE = DATA_DIR / "merged_lottery_data.csv"

def download_fresh_data():
    """Download fresh lottery data from UK National Lottery website"""
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(exist_ok=True, parents=True)
        
        logger.info("Downloading fresh lottery data from UK National Lottery...")
        
        # UK National Lottery CSV URL
        url = "https://www.national-lottery.co.uk/results/lotto/draw-history/csv"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save the downloaded data
        with open(DOWNLOADED_FILE, 'wb') as f:
            f.write(response.content)
            
        logger.info(f"Successfully downloaded fresh data to {DOWNLOADED_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        return False

def merge_data_files():
    """Merge downloaded data with existing historical data"""
    try:
        if not DOWNLOADED_FILE.exists():
            logger.error(f"Downloaded file not found: {DOWNLOADED_FILE}")
            return False
            
        # Load new data
        logger.info("Loading downloaded data...")
        new_data = pd.read_csv(DOWNLOADED_FILE)
        logger.info(f"Loaded {len(new_data)} new records")
        
        # Load existing data if available
        if EXISTING_FILE.exists():
            logger.info("Loading existing historical data...")
            existing_data = pd.read_csv(EXISTING_FILE)
            logger.info(f"Loaded {len(existing_data)} existing records")
            
            # Combine dataframes
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            
            # Remove duplicates based on Draw Date if that column exists
            if 'Draw Date' in combined_data.columns:
                combined_data['Draw Date'] = pd.to_datetime(combined_data['Draw Date'])
                combined_data = combined_data.drop_duplicates(subset=['Draw Date'], keep='last')
                combined_data = combined_data.sort_values('Draw Date')
                logger.info(f"After removing duplicates: {len(combined_data)} records")
            else:
                # Just remove exact duplicates
                combined_data = combined_data.drop_duplicates()
                logger.info(f"After removing duplicates: {len(combined_data)} records")
                
        else:
            logger.info("No existing data found, using only downloaded data")
            combined_data = new_data
        
        # Save merged data
        combined_data.to_csv(MERGED_FILE, index=False)
        logger.info(f"Saved merged data to {MERGED_FILE} with {len(combined_data)} total records")
        
        # Show date range if available
        if 'Draw Date' in combined_data.columns:
            min_date = combined_data['Draw Date'].min()
            max_date = combined_data['Draw Date'].max()
            logger.info(f"Data covers: {min_date} to {max_date}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error merging data: {str(e)}")
        return False

def main():
    """Main function to download and merge lottery data"""
    print("="*60)
    print("LOTTERY DATA FETCHER")
    print("="*60)
    
    # Download fresh data
    print("\n1. Downloading fresh lottery data...")
    download_success = download_fresh_data()
    
    if not download_success:
        print("❌ Download failed!")
        return False
    
    print("✅ Download successful!")
    
    # Merge data
    print("\n2. Merging with existing data...")
    merge_success = merge_data_files()
    
    if not merge_success:
        print("❌ Merge failed!")
        return False
        
    print("✅ Data merge successful!")
    
    print("\n" + "="*60)
    print("DATA UPDATE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 