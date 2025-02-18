#asset/dat_extractor.py
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
#         "Series": "series_name",
#         "Series Index": 1
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

def process_dat_files(dat_dir, extracted_data):
    dat_dir = Path(dat_dir).resolve()
    error_log = extracted_data.get("Error Log", [])
    dat_files = list(dat_dir.glob("*.dat"))
    processed_series = set()
    
    for dat_file in dat_files:
        filename = dat_file.stem
        if filename not in extracted_data:
            error_log.append(f"[process_dat_files] {dat_file} does not match any log entry. Skipping.")
            continue
        
        parsed_data, error = parse_dat_file(dat_file)
        if error:
            error_log.append(error)
            continue
        
        extracted_data[filename]["q"] = parsed_data["q"]
        extracted_data[filename]["Intensity"] = parsed_data["Intensity"]
        
        if "Series" in extracted_data[filename]:
            processed_series.add(extracted_data[filename]["Series"])
    
    for filename, data in extracted_data.items():
        if "Series" in data and data["Series"] in processed_series:
            if "q" not in data or "Intensity" not in data:
                error_log.append(f"[process_dat_files] {filename} is missing q/Intensity data. Dat file missing for series {data['Series']}.")
    
    extracted_data["Error Log"] = error_log
    return extracted_data

def main(dat_dir, log_path, output_dir):

    from spec_log_extractor import parse_log_file

    extracted_data = parse_log_file(log_path)
    extracted_data = process_dat_files(dat_dir, extracted_data)
    
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_stem = Path(log_path).resolve().stem
    output_file = output_dir / f"{log_stem}_extracted_with_dat.json"
    
    with output_file.open('w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    
    total_items = len(extracted_data) - 1
    non_series_items = sum(1 for data in extracted_data.values() if isinstance(data, dict) and "Series" not in data)
    series_items = total_items - non_series_items
    series_types = {data["Series"] for data in extracted_data.values() if isinstance(data, dict) and "Series" in data}
    
    print(f"Total extracted items: {total_items}")
    print(f"Non-series items: {non_series_items}")
    print(f"Series items: {series_items}")
    print("Extracted series types:")
    for series in series_types:
        print(f" - {series}")
    
    print(f"Extraction complete. Data saved to {output_file}")
    
    if extracted_data["Error Log"]:
        print("Error Log:")
        for error in extracted_data["Error Log"]:
            print(error)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from .dat files and match with log data.")
    parser.add_argument("input", nargs="?", default='/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/Averaged', help="Directory containing .dat files")
    parser.add_argument("log", nargs="?", default="./ready/log241219-HS", help="Path to the log file")
    parser.add_argument("output", nargs="?", default="./ready/test_output", help="Directory to save the extracted JSON file")
    args = parser.parse_args()
    
    main(args.input, args.log, args.output)