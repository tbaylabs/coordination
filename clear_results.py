#!/usr/bin/env python3
import json
import os
from pathlib import Path
import argparse

def clear_results_array(file_path):
    """
    Read a JSON file, clear the 'results' array while preserving all other data,
    and write back to the same file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'results' not in data:
            print(f"Warning: No 'results' array found in {file_path}")
            return False
        
        # Store the original length for reporting
        original_length = len(data['results'])
        
        # Clear the results array
        data['results'] = []
        
        # Write the modified data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"Cleared {original_length} results from {file_path}")
        return True
    
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {file_path}: {str(e)}")
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clear results arrays from JSON files in a directory')
    parser.add_argument('directory', help='Directory containing the JSON files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()
    
    # Convert to Path object and resolve to absolute path
    directory = Path(args.directory).resolve()
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return 1
    
    # Find all JSON files in the directory
    json_files = list(directory.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {directory}")
        return 1
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    if args.dry_run:
        print("Dry run - no changes will be made")
    
    success_count = 0
    for file_path in json_files:
        if args.dry_run:
            print(f"Would process: {file_path}")
        else:
            if clear_results_array(file_path):
                success_count += 1
    
    if args.dry_run:
        print(f"\nDry run complete. {len(json_files)} files would be processed")
    else:
        print(f"\nProcessing complete. Successfully cleared results in {success_count} of {len(json_files)} files")
    
    return 0

if __name__ == '__main__':
    exit(main())