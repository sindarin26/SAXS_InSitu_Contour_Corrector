# The extracted data dictionary will have the following structure:
# {
#     "filename1": {
#         "File Path": "path/to/file",
#         "Index": 1,
#         "Exposure Time": 0.1,
#         "2": 0.0,
#         "3": 0.0,
#         "Current": 0.0,
#         "KeV": 0.0,
#         "6": 0.0,
#         "7": 0.0,
#         "8": 0.0,
#         "Temperature": 0.0,
#         "9": 0.0,
#         "Time": "YYYY-MM-DD HH:MM:SS",
#         "Elapsed Time": 0,
#         "q": [0.219288, 0.274110, ...],
#         "Intensity": [0.012137, -0.001686, ...]
#     },
#     "filename2": { ... },
#     ...
#     "Error Log": [
#         "Error message 1",
#         "Error message 2",
#         ...
#     ]
# }


import json
import argparse
from pathlib import Path
import numpy as np
from spec_log_extractor import parse_log_file

def parse_dat_file(dat_path):
    dat_path = Path(dat_path).resolve()
    if not dat_path.exists():
        return None, f"[parse_dat_file] Data file not found: {dat_path}"
    
    q_values, intensity_values = [], []
    
    with dat_path.open('r', encoding='utf-8') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                q_values.append(float(parts[0]))
                intensity_values.append(float(parts[1]) if parts[1].lower() != 'nan' else np.nan)
            except ValueError as e:
                return None, f"[parse_dat_file] Value parsing error in file {dat_path}: {e}"
    
    return {"q": q_values, "Intensity": intensity_values}, None

def process_dat_files(dat_dir, log_path, output_dir):
    log_path = Path(log_path).resolve()
    extracted_data = parse_log_file(log_path)
    error_log = extracted_data.get("Error Log", [])
    
    dat_dir = Path(dat_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dat_files = list(dat_dir.glob("*.dat"))
    
    for dat_file in dat_files:
        filename = dat_file.stem  # Remove extension
        if filename not in extracted_data:
            error_log.append(f"[process_dat_files] {dat_file} does not match any log entry. Skipping.")
            continue
        
        parsed_data, error = parse_dat_file(dat_file)
        if error:
            error_log.append(error)
            continue
        
        extracted_data[filename]["q"] = parsed_data["q"]
        extracted_data[filename]["Intensity"] = parsed_data["Intensity"]
    
    extracted_data["Error Log"] = error_log
    
    output_file = output_dir / f"{log_path.stem}_extracted_dat.json"
    with output_file.open('w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    
    print(f"Processing complete. Data saved to {output_file}")

def main(dat_dir, log_path, output_dir):
    process_dat_files(dat_dir, log_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from .dat files and match with log data.")
    parser.add_argument("input", nargs="?", default='/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/Averaged/1st/', help="Directory containing .dat files")
    parser.add_argument("log", nargs="?", default="./ready/log241219-HS", help="Path to the log file")
    parser.add_argument("output", nargs="?", default="./ready/test_output", help="Directory to save the extracted JSON file")
    args = parser.parse_args()
    
    main(args.input, args.log, args.output)
