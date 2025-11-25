#!/usr/bin/env python3
"""
Script to extract PhysicsStep and RenderStep mean_ns values from CSV files
and generate a summary.csv file.
"""

import csv
import os
from pathlib import Path


def extract_mean_ns(csv_file_path):
    """
    Extract mean_ns values for PhysicsStep and RenderStep from a CSV file.
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        Tuple of (physicsstep_mean_ns, renderstep_mean_ns) or (None, None) if not found
    """
    physicsstep_mean = None
    renderstep_mean = None
    
    try:
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['name'] == 'PhysicsStep':
                    physicsstep_mean = row['mean_ns']
                elif row['name'] == 'RenderStep':
                    renderstep_mean = row['mean_ns']
                    
                # Early exit if we found both
                if physicsstep_mean and renderstep_mean:
                    break
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        
    return physicsstep_mean, renderstep_mean


def main():
    # Target directory containing CSV files
    eval_dir = Path(__file__).parent / 'eval'
    
    # Find all CSV files in the eval directory
    csv_files = sorted([f for f in os.listdir(eval_dir) 
                       if f.endswith('.csv') and f != 'summary.csv'])
    
    # Prepare output data
    results = []
    
    for csv_file in csv_files:
        csv_path = eval_dir / csv_file
        physicsstep_mean, renderstep_mean = extract_mean_ns(csv_path)
        
        results.append({
            'csv_file_name': csv_file,
            'physicsstep_mean_ns': physicsstep_mean if physicsstep_mean else 'N/A',
            'renderstep_mean_ns': renderstep_mean if renderstep_mean else 'N/A'
        })
    
    # Write summary.csv to eval directory
    output_path = eval_dir / 'summary.csv'
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['csv_file_name', 'physicsstep_mean_ns', 'renderstep_mean_ns'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Summary written to {output_path}")
    print(f"Processed {len(results)} CSV files")


if __name__ == '__main__':
    main()
