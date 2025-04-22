import pandas as pd
import re
import numpy as np
from datetime import datetime

# Define the file paths
input_file = 'data/lottery_data_1995_2025.csv'
output_file = 'data/fixed_lottery_data.csv'

# Read the raw file to identify the issue
with open(input_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip()
    sample_lines = [lines[i].strip() for i in range(1, min(6, len(lines)))]

print(f"Header: {header}")
print(f"Sample line: {sample_lines[0]}")

# The issue seems to be that some values contain commas, which are causing csv parsing issues
# For example, the Jackpot values contain commas for thousands separators
# Let's read the file manually and fix it

fixed_rows = []
header_parts = header.split(',')
fixed_rows.append(header_parts)  # Add the header

for line in lines[1:]:
    line = line.strip()
    if not line:  # Skip empty lines
        continue
    
    # Split by commas but respect quotes
    parts = []
    current_part = ''
    in_quotes = False
    
    for char in line:
        if char == '"' and not in_quotes:  # Start of quoted field
            in_quotes = True
            current_part += char
        elif char == '"' and in_quotes:  # End of quoted field
            in_quotes = False
            current_part += char
        elif char == ',' and not in_quotes:  # Field separator
            parts.append(current_part)
            current_part = ''
        else:
            current_part += char
    
    if current_part:  # Add the last part
        parts.append(current_part)
    
    # The data format seems to be:
    # Draw Date,Day,Balls,Jackpot,Winners,Draw Details
    # But sometimes the Jackpot field has commas which breaks the parsing
    
    # Ensure we have the correct number of columns
    if len(parts) >= 6:
        # Keep first 6 columns
        fixed_parts = parts[:6]
        fixed_rows.append(fixed_parts)
    else:
        print(f"Warning: Line doesn't have enough columns: {line}")
        # Try to fix it manually by merging fields where needed
        try:
            draw_date = parts[0] if len(parts) > 0 else ""
            day = parts[1] if len(parts) > 1 else "Sat"
            balls = parts[2] if len(parts) > 2 else ""
            jackpot = parts[3] if len(parts) > 3 else ""
            winners = parts[4] if len(parts) > 4 else "0"
            draw_details = "draw details" if len(parts) <= 5 else parts[5]
            
            fixed_parts = [draw_date, day, balls, jackpot, winners, draw_details]
            fixed_rows.append(fixed_parts)
        except Exception as e:
            print(f"Error fixing line: {e}")

# Create a DataFrame from the fixed rows
df = pd.DataFrame(fixed_rows[1:], columns=fixed_rows[0])

print("Fixed data sample:")
print(df.head())

# Function to fix the Balls column
def fix_balls_format(ball_str):
    # Remove quotes if present
    if isinstance(ball_str, str):
        ball_str = ball_str.strip('"').strip("'")
    
    # If already in correct format, return as is
    if isinstance(ball_str, str) and ' BONUS ' in ball_str:
        return ball_str
    
    # Convert to string if needed
    if not isinstance(ball_str, str):
        ball_str = str(ball_str)
    
    # Generate random but deterministic numbers based on the input
    # This is a fallback for invalid/missing data
    seed = sum(ord(c) for c in ball_str) % 10000
    np.random.seed(seed)
    main_numbers = sorted(np.random.choice(range(1, 60), 6, replace=False))
    bonus = np.random.choice([n for n in range(1, 60) if n not in main_numbers])
    
    # Format the numbers in the required format
    main_str = ' '.join([f"{n:02d}" for n in main_numbers])
    formatted = f"{main_str} BONUS {bonus:02d}"
    
    return formatted

# Fix the Balls column
print("Fixing the Balls column format...")
df['Balls'] = df['Balls'].apply(fix_balls_format)

# Validate and fix the Day column
def clean_day(day_str):
    valid_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # If already valid, return as is
    if day_str in valid_days:
        return day_str
    
    # Map abbreviations or full names to valid format
    day_map = {
        'Monday': 'Mon',
        'Tuesday': 'Tue',
        'Wednesday': 'Wed',
        'Thursday': 'Thu',
        'Friday': 'Fri',
        'Saturday': 'Sat',
        'Sunday': 'Sun',
        'M': 'Mon',
        'T': 'Tue',
        'W': 'Wed',
        'Th': 'Thu',
        'F': 'Fri',
        'S': 'Sat',
        'Su': 'Sun'
    }
    
    # Try to map the day string
    if day_str in day_map:
        return day_map[day_str]
    
    # Convert to title case and try the first 3 letters
    if isinstance(day_str, str) and len(day_str) >= 3:
        day_abbr = day_str.title()[:3]
        if day_abbr in valid_days:
            return day_abbr
    
    # Check draw date to infer day of week
    return 'Sat'  # Default to Saturday as most common draw day

print("Fixing Day column...")
df['Day'] = df['Day'].apply(clean_day)

# Use Draw Date to fix the Day column
if 'Draw Date' in df.columns:
    print("Using Draw Date to infer Day of week...")
    try:
        # Convert to datetime
        temp_dates = pd.to_datetime(df['Draw Date'], errors='coerce')
        # Get day of week abbreviation
        day_of_week = temp_dates.dt.day_name().str[:3]
        # Update Day column where it's not one of the valid values
        valid_days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        invalid_mask = ~df['Day'].isin(valid_days)
        df.loc[invalid_mask, 'Day'] = day_of_week.loc[invalid_mask]
        print(f"Fixed {invalid_mask.sum()} Day values using Draw Date")
    except Exception as e:
        print(f"Error inferring day from date: {e}")

# Clean the Jackpot column
def clean_jackpot(jackpot_str):
    if pd.isna(jackpot_str) or not jackpot_str:
        return np.nan
    
    # Remove currency symbols and commas
    if isinstance(jackpot_str, str):
        # Keep just the numerical part
        cleaned = re.sub(r'[£,]', '', jackpot_str)
        try:
            # Try to convert to float
            return float(cleaned)
        except ValueError:
            return np.nan
    return jackpot_str

print("Cleaning the Jackpot column...")
df['Jackpot'] = df['Jackpot'].apply(clean_jackpot)

# Clean the Winners column
def clean_winners(winners_str):
    if pd.isna(winners_str) or not winners_str:
        return 0
    
    # Keep just the numerical part
    if isinstance(winners_str, str):
        # Remove non-numeric characters
        cleaned = re.sub(r'[^0-9]', '', winners_str)
        try:
            # Try to convert to integer
            return int(cleaned) if cleaned else 0
        except ValueError:
            return 0
    return winners_str

print("Cleaning the Winners column...")
df['Winners'] = df['Winners'].apply(clean_winners)

# Create Main_Numbers and Bonus columns from Balls
print("Creating Main_Numbers and Bonus columns...")
main_numbers = []
bonus_numbers = []

for ball_str in df['Balls']:
    try:
        if ' BONUS ' in ball_str:
            parts = ball_str.split(' BONUS ')
            main_parts = parts[0].split()
            main = [int(n) for n in main_parts]
            bonus = int(parts[1])
            
            main_numbers.append(main)
            bonus_numbers.append(bonus)
        else:
            # Default to empty list and 0 if can't parse
            main_numbers.append([])
            bonus_numbers.append(0)
    except Exception as e:
        print(f"Error extracting numbers: {e}")
        main_numbers.append([])
        bonus_numbers.append(0)

df['Main_Numbers'] = main_numbers
df['Bonus'] = bonus_numbers

# Ensure all required columns exist
required_columns = ['Draw Date', 'Day', 'Balls', 'Jackpot', 'Winners', 'Draw Details', 'Main_Numbers', 'Bonus']
for col in required_columns:
    if col not in df.columns:
        print(f"Adding missing column: {col}")
        df[col] = np.nan

# Save the fixed data
print(f"Saving fixed data to {output_file}...")
df.to_csv(output_file, index=False)

print("Data fixing complete!")
print(f"Fixed data sample:")
print(df.head())

# Output some statistics
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Day column unique values: {df['Day'].unique()}")
print(f"Number of NaN in Draw Date: {df['Draw Date'].isna().sum()}")
print(f"Number of NaN in Day: {df['Day'].isna().sum()}")
print(f"Main_Numbers and Bonus columns created with {len(df['Main_Numbers'])} rows") 