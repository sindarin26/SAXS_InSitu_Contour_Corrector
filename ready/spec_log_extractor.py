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
#         "Elapsed Time": 0
#     },
#     "filename2": { ... },
#     ...
#     "Error Log": [
#         "Error message 1",
#         "Error message 2",
#         ...
#     ]
# }

import re
import json
import datetime
import argparse
from pathlib import Path

def parse_log_file(log_path):
    extracted_data = {}
    error_log = []
    
    log_path = Path(log_path).resolve()
    if not log_path.exists():
        error_log.append(f"[parse_log_file] Log file not found: {log_path}")
        extracted_data["Error Log"] = error_log
        return extracted_data
    
    with log_path.open('r', encoding='utf-8') as file:
        lines = file.readlines()
    
    z_lines = [line.strip() for line in lines if line.startswith("#Z")]
    
    if not z_lines:
        error_log.append("[parse_log_file] No #Z lines found in the log file.")
        extracted_data["Error Log"] = error_log
        return extracted_data
    
    time_reference = None
    
    for line in z_lines:
        parts = line.split()
        
        if len(parts) < 14:
            error_log.append(f"[parse_log_file] Malformed line: {line}")
            continue
        
        file_path = Path(parts[1]).resolve()
        filename = file_path.stem
        
        match = re.search(r'_(\d{4})$', filename)
        if not match:
            error_log.append(f"[parse_log_file] No valid index found in filename: {filename}")
            continue
        
        index = int(match.group(1))
        
        try:
            exposure_time = round(float(parts[2]), 1)
            value_2 = float(parts[3])
            value_3 = float(parts[4])
            current = round(float(parts[5]), 2)
            kev = round(float(parts[6]), 3)
            value_6 = float(parts[7])
            value_7 = round(float(parts[8]), 1)
            value_8 = round(float(parts[9]), 1)
            temperature = float(parts[10])
            value_9 = float(parts[11])
            time_str = ' '.join(parts[12:])
            time_obj = datetime.datetime.strptime(time_str, "%a %b %d %H:%M:%S %Y")
        except ValueError as e:
            error_log.append(f"[parse_log_file] Value parsing error in line: {line}. Error: {e}")
            continue
        
        if time_reference is None:
            time_reference = time_obj
        elapsed_time = int((time_obj - time_reference).total_seconds())
        
        extracted_data[filename] = {
            "File Path": str(file_path),
            "Index": index,
            "Exposure Time": exposure_time,
            "2": value_2,
            "3": value_3,
            "Current": current,
            "KeV": kev,
            "6": value_6,
            "7": value_7,
            "8": value_8,
            "Temperature": temperature,
            "9": value_9,
            "Time": time_obj.strftime("%Y-%m-%d %H:%M:%S"),
            "Elapsed Time": elapsed_time
        }
    
    indices = sorted([data["Index"] for data in extracted_data.values()])
    if indices and indices[0] != 1:
        error_log.append("[parse_log_file] Missing index 1 in extracted data.")
    
    extracted_data["Error Log"] = error_log
    return extracted_data

def main(log_path, output_path):
    log_path = Path(log_path).resolve()
    output_path = Path(output_path).resolve()
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    extracted_data = parse_log_file(log_path)
    
    base_filename = log_path.stem
    output_file = output_path / f"{base_filename}_extracted.json"
    
    with output_file.open('w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, indent=4)
    
    print(f"Extraction complete. Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract specific data from log files.")
    parser.add_argument("input", nargs="?", default="./ready/log241219-HS", help="Path to the log file")
    parser.add_argument("output", nargs="?", default="./ready/test_output", help="Directory to save the extracted JSON file")
    args = parser.parse_args()
    
    main(args.input, args.output)
