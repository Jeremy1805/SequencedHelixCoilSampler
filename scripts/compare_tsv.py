#!/usr/bin/env python3
"""
Compare two TSV files with numerical tolerance for floating-point values.
"""

import sys
import csv
import math

def is_number(s):
    """Check if string represents a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def compare_values(val1, val2, tolerance=1e-7):
    """Compare two values with numerical tolerance."""
    # If both are strings and not numbers, compare exactly
    if not is_number(val1) and not is_number(val2):
        return val1 == val2
    
    # If one is number and other isn't, they're different
    if is_number(val1) != is_number(val2):
        return False
    
    # Both are numbers - compare with tolerance
    try:
        num1 = float(val1)
        num2 = float(val2)
        
        # Handle special cases
        if math.isnan(num1) and math.isnan(num2):
            return True
        if math.isinf(num1) and math.isinf(num2):
            return num1 == num2  # Same sign infinity
        if math.isnan(num1) or math.isnan(num2) or math.isinf(num1) or math.isinf(num2):
            return False
        
        # For very small numbers, use absolute tolerance
        if abs(num1) < tolerance and abs(num2) < tolerance:
            return abs(num1 - num2) < tolerance
        
        # For larger numbers, use relative tolerance
        max_val = max(abs(num1), abs(num2))
        return abs(num1 - num2) < tolerance * max_val
        
    except ValueError:
        return False

def compare_tsv_files(file1_path, file2_path, tolerance=1e-7):
    """Compare two TSV files with numerical tolerance."""
    
    try:
        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            reader1 = csv.reader(f1, delimiter='\t')
            reader2 = csv.reader(f2, delimiter='\t')
            
            differences = []
            row_num = 0
            
            for row1, row2 in zip(reader1, reader2):
                row_num += 1
                
                # Check if rows have same number of columns
                if len(row1) != len(row2):
                    differences.append(f"Row {row_num}: Different number of columns ({len(row1)} vs {len(row2)})")
                    continue
                
                # Compare each column
                for col_num, (val1, val2) in enumerate(zip(row1, row2)):
                    if not compare_values(val1, val2, tolerance):
                        differences.append(f"Row {row_num}, Col {col_num + 1}: '{val1}' vs '{val2}'")
            
            # Check if one file has more rows
            remaining1 = list(reader1)
            remaining2 = list(reader2)
            
            if remaining1:
                differences.append(f"File 1 has {len(remaining1)} extra rows starting at row {row_num + 1}")
            if remaining2:
                differences.append(f"File 2 has {len(remaining2)} extra rows starting at row {row_num + 1}")
            
            return differences
            
    except FileNotFoundError as e:
        return [f"File not found: {e}"]
    except Exception as e:
        return [f"Error reading files: {e}"]

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_tsv.py <file1.tsv> <file2.tsv>")
        print("Compares two TSV files with numerical tolerance of 1e-7")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    print(f"Comparing {file1} with {file2}...")
    
    differences = compare_tsv_files(file1, file2, tolerance=1e-7)
    
    if not differences:
        print("✅ Files are identical within tolerance (1e-7)")
        sys.exit(0)
    else:
        print("❌ Files differ:")
        for diff in differences[:20]:  # Limit output to first 20 differences
            print(f"  {diff}")
        
        if len(differences) > 20:
            print(f"  ... and {len(differences) - 20} more differences")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
