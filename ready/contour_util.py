import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files

def select_series(extracted_data):
    """Extract series with q and Intensity, prompt selection if multiple."""
    series_list = {data["Series"] for data in extracted_data.values() if isinstance(data, dict) and "Series" in data and "q" in data and "Intensity" in data}
    
    if not series_list:
        raise ValueError("No valid series found with q and Intensity data.")
    
    if len(series_list) == 1:
        return list(series_list)[0]
    
    return prompt_series_selection(series_list)

def prompt_series_selection(series_list):
    """Prompt user to select a series from available options."""
    print("Available Series:")
    series_list = sorted(series_list)
    for i, series in enumerate(series_list, 1):
        print(f"{i}. {series}")
    
    while True:
        try:
            selection = int(input("Select a series by number: "))
            if 1 <= selection <= len(series_list):
                return series_list[selection - 1]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def extract_contour_data(selected_series, extracted_data):
    """Extract q, time, and Intensity values for the selected series."""
    series_entries = [
        data for data in extracted_data.values()
        if isinstance(data, dict) and data.get("Series") == selected_series and isinstance(data.get("q"), list) and isinstance(data.get("Intensity"), list)
    ]
    
    if not series_entries:
        raise ValueError("No valid data found for the selected series.")
    
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])  # Sort by time
    
    times = [entry["Series Elapsed Time"] for entry in series_entries]
    temperatures = [entry["Temperature"] for entry in series_entries]
    q_values = [entry["q"] for entry in series_entries]
    intensity_values = [entry["Intensity"] for entry in series_entries]
    
    return {
        "Series": selected_series,
        "Time-temp": (times, temperatures),
        "Data": [{"Time": times[i], "q": q_values[i], "Intensity": intensity_values[i]} for i in range(len(times))]
    }

def plot_contour(contour_data):
    """Generate contour plot with interpolated data, using percentile scaling and colormap."""
    times, temperatures = contour_data["Time-temp"]
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])
    
    # Generate a regular grid for q values with max resolution from original x-axis data
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)  # Use unique times without interpolation
    
    # Interpolate intensity values onto the regular grid
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    
    # Apply percentile-based scaling for contrast adjustment
    if np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return
    
    lower_bound = np.nanpercentile(grid_intensity, 0.1)
    upper_bound = np.nanpercentile(grid_intensity, 98)
    
    plt.figure(figsize=(12, 8), dpi=300)
    cp = plt.contourf(grid_q, grid_time, grid_intensity, levels=100, cmap='inferno', vmin=lower_bound, vmax=upper_bound)
    
    # Colorbar setup
    c = plt.colorbar(cp, orientation='vertical')
    c.set_label("log10(Intensity)", fontsize=14, fontweight='bold', fontname='Times New Roman')
    
    # Labels
    plt.xlabel("2theta (Cu K-alpha)", fontsize=14, fontweight='bold', fontname='Times New Roman')
    plt.ylabel("Elapsed Time", fontsize=14, fontweight='bold', fontname='Times New Roman')
    
    # Axis settings
    plt.xticks(fontsize=12, fontweight='bold', fontname='Times New Roman')
    plt.yticks(fontsize=12, fontweight='bold', fontname='Times New Roman')
    
    plt.title(f"Contour Plot for {contour_data['Series']}", fontsize=16, fontweight='bold', fontname='Times New Roman')
    plt.show()

def main(dat_dir, log_path, output_dir):
    log_data = parse_log_file(log_path)
    extracted_data, output_file = process_dat_files(dat_dir, log_data, output_dir)
    
    selected_series = select_series(extracted_data)
    
    # Check for missing data before contour plotting
    for entry in extracted_data.values():
        if isinstance(entry, dict) and "q" in entry and "Intensity" in entry:
            if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
                print(f"Warning: Missing data detected in {entry['File Path']}")
    
    contour_data = extract_contour_data(selected_series, extracted_data)
    
    for entry in contour_data["Data"]:
        if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
            print(f"Warning: Missing data detected in time frame {entry['Time']}")
    
    plot_contour(contour_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from .dat files and match with log data.")
    parser.add_argument("input", nargs="?", default='/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/Averaged', help="Directory containing .dat files")
    parser.add_argument("log", nargs="?", default="./ready/log241219-HS", help="Path to the log file")
    parser.add_argument("output", nargs="?", default="./ready/test_output", help="Directory to save the extracted JSON file")
    args = parser.parse_args()
    
    main(args.input, args.log, args.output)
