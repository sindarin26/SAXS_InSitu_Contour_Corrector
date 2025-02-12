import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit  # 가우시안 피팅용
import pandas as pd  # CSV (엑셀) 출력용
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files

# ============================================================
# 기존 함수들: 시리즈 선택, 데이터 추출, 플롯 등
# ============================================================
def select_series(extracted_data):
    """Extract series with q and Intensity, prompt selection if multiple."""
    series_list = {
        data["Series"]
        for data in extracted_data.values()
        if isinstance(data, dict) and "Series" in data and "q" in data and "Intensity" in data
    }
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
        if isinstance(data, dict)
        and data.get("Series") == selected_series
        and isinstance(data.get("q"), list)
        and isinstance(data.get("Intensity"), list)
    ]
    if not series_entries:
        raise ValueError("No valid data found for the selected series.")
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])
    times = [entry["Series Elapsed Time"] for entry in series_entries]
    temperatures = [entry["Temperature"] for entry in series_entries]
    q_values = [entry["q"] for entry in series_entries]
    intensity_values = [entry["Intensity"] for entry in series_entries]
    return {
        "Series": selected_series,
        "Time-temp": (times, temperatures),
        "Data": [
            {"Time": times[i], "q": q_values[i], "Intensity": intensity_values[i]}
            for i in range(len(times))
        ]
    }

def plot_contour(contour_data, temp=False, legend=True, graph_option=None):
    """
    Generate contour plot with interpolated data, optionally with a Temperature vs Time subplot.
    """
    default_graph_option = {
        "figure_size": (12, 8),
        "figure_dpi": 300,
        "figure_title_enable": False,
        "figure_title_text": "",
        "font_label": "Times New Roman",
        "font_tick": "Times New Roman",
        "font_title": "Times New Roman",
        "axes_label_size": 14,
        "tick_label_size": 12,
        "title_size": 16,
        "contour_xlabel_enable": True,
        "contour_xlabel_text": "2theta (Cu K-alpha)",
        "contour_ylabel_enable": True,
        "contour_ylabel_text": "Elapsed Time",
        "contour_title_enable": True,
        "contour_title_text": "Contour Plot",
        "contour_xlim": None,
        "contour_grid": False,
        "temp_xlabel_enable": True,
        "temp_xlabel_text": "Temperature",
        "temp_ylabel_enable": True,
        "temp_ylabel_text": "Elapsed Time",
        "temp_title_enable": False,
        "temp_title_text": "Temperature Plot",
        "temp_xlim": None,
        "temp_grid": True,
        "global_ylim": None,
        "wspace": 0.00,
        "width_ratios": [7, 2],
        "contour_levels": 100,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "colorbar_label": "log10(Intensity)",
        "cbar_location": "left",
        "cbar_pad": 0.15
    }
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}
    times, temperatures = contour_data["Time-temp"]
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return
    lower_bound = np.nanpercentile(grid_intensity, final_opt["contour_lower_percentile"])
    upper_bound = np.nanpercentile(grid_intensity, final_opt["contour_upper_percentile"])
    if temp:
        fig, (ax_contour, ax_temp) = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=final_opt["figure_size"],
            dpi=final_opt["figure_dpi"],
            gridspec_kw={"width_ratios": final_opt["width_ratios"]}
        )
        plt.subplots_adjust(wspace=final_opt["wspace"])
        cp = ax_contour.contourf(
            grid_q, grid_time, grid_intensity,
            levels=final_opt["contour_levels"],
            cmap=final_opt["contour_cmap"],
            vmin=lower_bound, vmax=upper_bound
        )
        if final_opt["contour_xlim"] is not None:
            ax_contour.set_xlim(final_opt["contour_xlim"])
        if final_opt["global_ylim"] is not None:
            ax_contour.set_ylim(final_opt["global_ylim"])
        if final_opt["contour_xlabel_enable"]:
            ax_contour.set_xlabel(final_opt["contour_xlabel_text"],
                                  fontsize=final_opt["axes_label_size"],
                                  fontweight='bold',
                                  fontname=final_opt["font_label"])
        if final_opt["contour_ylabel_enable"]:
            ax_contour.set_ylabel(final_opt["contour_ylabel_text"],
                                  fontsize=final_opt["axes_label_size"],
                                  fontweight='bold',
                                  fontname=final_opt["font_label"])
        if final_opt["contour_title_enable"]:
            ax_contour.set_title(final_opt["contour_title_text"],
                                 fontsize=final_opt["title_size"],
                                 fontweight='bold',
                                 fontname=final_opt["font_title"])
        ax_contour.tick_params(axis='x', labelsize=final_opt["tick_label_size"])
        ax_contour.tick_params(axis='y', labelsize=final_opt["tick_label_size"])
        for lbl in ax_contour.get_xticklabels() + ax_contour.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])
        if final_opt["contour_grid"]:
            ax_contour.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if legend:
            c = fig.colorbar(cp, ax=ax_contour,
                             orientation='vertical',
                             location=final_opt["cbar_location"],
                             pad=final_opt["cbar_pad"])
            c.set_label(final_opt["colorbar_label"],
                        fontsize=final_opt["axes_label_size"],
                        fontweight='bold',
                        fontname=final_opt["font_label"])
            c.ax.tick_params(labelsize=final_opt["tick_label_size"])
            for clbl in c.ax.get_yticklabels():
                clbl.set_fontname(final_opt["font_tick"])
        ax_temp.plot(temperatures, times, 'r-')
        ax_temp.invert_xaxis()
        if final_opt["temp_xlim"] is not None:
            ax_temp.set_xlim(final_opt["temp_xlim"])
        if final_opt["global_ylim"] is not None:
            ax_temp.set_ylim(final_opt["global_ylim"])
        if final_opt["temp_xlabel_enable"]:
            ax_temp.set_xlabel(final_opt["temp_xlabel_text"],
                               fontsize=final_opt["axes_label_size"],
                               fontweight='bold',
                               fontname=final_opt["font_label"])
        else:
            ax_temp.set_xlabel("")
        if final_opt["temp_ylabel_enable"]:
            ax_temp.yaxis.set_label_position("right")
            ax_temp.yaxis.tick_right()
            ax_temp.set_ylabel(final_opt["temp_ylabel_text"],
                               fontsize=final_opt["axes_label_size"],
                               fontweight='bold',
                               fontname=final_opt["font_label"])
        else:
            ax_temp.yaxis.set_label_position("right")
            ax_temp.yaxis.tick_right()
            ax_temp.set_ylabel("")
        if final_opt["temp_title_enable"]:
            ax_temp.set_title(final_opt["temp_title_text"],
                              fontsize=final_opt["title_size"],
                              fontweight='bold',
                              fontname=final_opt["font_title"])
        ax_temp.tick_params(axis='x', labelsize=final_opt["tick_label_size"], direction='in')
        ax_temp.tick_params(axis='y', labelsize=final_opt["tick_label_size"], direction='in')
        for lbl in ax_temp.get_xticklabels() + ax_temp.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])
        if final_opt["temp_grid"]:
            ax_temp.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if final_opt["figure_title_enable"]:
            fig.suptitle(final_opt["figure_title_text"],
                         fontsize=final_opt["title_size"],
                         fontweight='bold',
                         fontname=final_opt["font_title"])
    else:
        fig, ax = plt.subplots(figsize=final_opt["figure_size"],
                               dpi=final_opt["figure_dpi"])
        cp = ax.contourf(grid_q, grid_time, grid_intensity,
                         levels=final_opt["contour_levels"],
                         cmap=final_opt["contour_cmap"],
                         vmin=lower_bound,
                         vmax=upper_bound)
        if final_opt["contour_xlim"] is not None:
            ax.set_xlim(final_opt["contour_xlim"])
        if final_opt["global_ylim"] is not None:
            ax.set_ylim(final_opt["global_ylim"])
        if final_opt["contour_xlabel_enable"]:
            ax.set_xlabel(final_opt["contour_xlabel_text"],
                          fontsize=final_opt["axes_label_size"],
                          fontweight='bold',
                          fontname=final_opt["font_label"])
        if final_opt["contour_ylabel_enable"]:
            ax.set_ylabel(final_opt["contour_ylabel_text"],
                          fontsize=final_opt["axes_label_size"],
                          fontweight='bold',
                          fontname=final_opt["font_label"])
        if final_opt["contour_title_enable"]:
            ax.set_title(final_opt["contour_title_text"],
                         fontsize=final_opt["title_size"],
                         fontweight='bold',
                         fontname=final_opt["font_title"])
        ax.tick_params(axis='x', labelsize=final_opt["tick_label_size"])
        ax.tick_params(axis='y', labelsize=final_opt["tick_label_size"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])
        if final_opt["contour_grid"]:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        if legend:
            c = fig.colorbar(cp, ax=ax,
                             orientation='vertical',
                             location=final_opt["cbar_location"],
                             pad=final_opt["cbar_pad"])
            c.set_label(final_opt["colorbar_label"],
                        fontsize=final_opt["axes_label_size"],
                        fontweight='bold',
                        fontname=final_opt["font_label"])
            c.ax.tick_params(labelsize=final_opt["tick_label_size"])
            for clbl in c.ax.get_yticklabels():
                clbl.set_fontname(final_opt["font_tick"])
        if final_opt["figure_title_enable"]:
            fig.suptitle(final_opt["figure_title_text"],
                         fontsize=final_opt["title_size"],
                         fontweight='bold',
                         fontname=final_opt["font_title"])
    plt.show()

# ============================================================
# Temperature Correction Functions
# ============================================================
def select_temperature_ranges(extracted_data, selected_series, debug=False):
    """
    사용자 인터랙션을 통해 normal_range, adjust_range (시간 구간)을 지정.
    """
    series_entries = [
        data for data in extracted_data.values()
        if isinstance(data, dict) and data.get("Series") == selected_series
    ]
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])
    x_vals = [entry["Series Elapsed Time"] for entry in series_entries]
    y_vals = [entry["Temperature"] for entry in series_entries]
    plt.figure()
    plt.plot(x_vals, y_vals, 'ko-', label='Original Temperature')
    plt.xlabel("Series Elapsed Time")
    plt.ylabel("Temperature")
    plt.title("Select Normal Range by Clicking Two Points (Start, End)")
    plt.legend()
    plt.grid(True)
    pts = plt.ginput(2, timeout=-1)
    normal_range = (min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0]))
    plt.title("Select Adjust Range by Clicking Two Points (Start, End)")
    pts = plt.ginput(2, timeout=-1)
    adjust_range = (min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0]))
    if debug:
        print(f"[Debug] normal_range = {normal_range}")
        print(f"[Debug] adjust_range = {adjust_range}")
    plt.close()
    return normal_range, adjust_range

def temp_correction(extracted_data, selected_series, normal_range, adjust_range, debug=False):
    """
    normal_range에서 선형 피팅 후, adjust_range 내에서 온도 교정을 수행.
    """
    series_entries = [
        data for data in extracted_data.values()
        if isinstance(data, dict) and data.get("Series") == selected_series
    ]
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])
    times = np.array([entry["Series Elapsed Time"] for entry in series_entries])
    temps = np.array([entry["Temperature"] for entry in series_entries])
    normal_mask = (times >= normal_range[0]) & (times <= normal_range[1])
    normal_times = times[normal_mask]
    normal_temps = temps[normal_mask]
    if len(normal_times) < 2:
        print("Warning: Normal range has insufficient points for fitting.")
        return extracted_data
    a, b = np.polyfit(normal_times, normal_temps, 1)
    adjust_mask = (times >= adjust_range[0]) & (times <= adjust_range[1])
    adjust_indices = np.where(adjust_mask)[0]
    diffs = np.diff(temps[adjust_mask])
    zero_diff_indices = np.where(diffs == 0)[0]
    zero_diff_global = adjust_indices[zero_diff_indices+1]
    fitted_temps = a * times + b
    for i, entry in enumerate(series_entries):
        if "Temperature before adjust" not in entry:
            entry["Temperature before adjust"] = entry["Temperature"]
        if i in zero_diff_global:
            entry["Adjusted Temperature"] = fitted_temps[i]
            entry["Temperature"] = fitted_temps[i]
        else:
            entry["Adjusted Temperature"] = entry["Temperature"]
    if debug:
        plt.figure()
        plt.plot(times, temps, 'ko-', label='Original Temperature')
        plt.plot(times, fitted_temps, 'r--', label='Fitted (Normal Range)')
        corrected_temps = [e["Temperature"] for e in series_entries]
        plt.plot(times, corrected_temps, 'bx-', label='Corrected Temperature')
        plt.xlabel("Series Elapsed Time")
        plt.ylabel("Temperature")
        plt.title("Temperature Correction Debug View")
        plt.legend()
        plt.grid(True)
        plt.show()
    return extracted_data

# ============================================================
# Additional functions for contour_data and tracked_peaks processing
# ============================================================
def select_q_range(contour_data, index=None):
    """
    contour_data의 n번째 데이터에서 q 범위를 선택 (ginput 사용).
    """
    if index is None:
        index = 0
    try:
        data_entry = contour_data["Data"][index]
    except IndexError:
        raise ValueError(f"Index {index} is out of range for contour_data.")
    q_vals = data_entry["q"]
    intensity_vals = data_entry["Intensity"]
    plt.figure()
    plt.plot(q_vals, intensity_vals, 'ko-', label='Intensity')
    plt.xlabel("q")
    plt.ylabel("Intensity")
    plt.title("Select q Range by Clicking Two Points (Start, End)")
    plt.legend()
    plt.grid(True)
    pts = plt.ginput(2, timeout=-1)
    plt.close()
    if len(pts) < 2:
        raise RuntimeError("Not enough points were selected.")
    q_min = min(pts[0][0], pts[1][0])
    q_max = max(pts[0][0], pts[1][0])
    return (q_min, q_max)

def gaussian(x, a, mu, sigma, offset):
    """Gaussian 함수: a * exp(-((x-mu)^2)/(2*sigma^2)) + offset"""
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

def find_peak(contour_data, Index_number=0, input_range=None, peak_info=None):
    """
    주어진 contour_data의 Index_number 데이터에서, input_range 내의 q, Intensity 데이터를
    대상으로 가우시안 피팅을 수행하여 peak의 q 위치와 intensity를 구하고,
    input_range의 중심을 기준으로 출력 range를 반환.
    """
    if input_range is None:
        raise ValueError("input_range must be provided.")
    try:
        data_entry = contour_data["Data"][Index_number]
    except IndexError:
        raise ValueError(f"Index {Index_number} is out of range for contour_data.")
    q_arr = np.array(data_entry["q"])
    intensity_arr = np.array(data_entry["Intensity"])
    mask = (q_arr >= input_range[0]) & (q_arr <= input_range[1])
    if not np.any(mask):
        print("No data points found within the specified input_range.")
        return None
    q_subset = q_arr[mask]
    intensity_subset = intensity_arr[mask]
    a_guess = np.max(intensity_subset) - np.min(intensity_subset)
    mu_guess = q_subset[np.argmax(intensity_subset)]
    sigma_guess = (input_range[1] - input_range[0]) / 4.0
    offset_guess = np.min(intensity_subset)
    p0 = [a_guess, mu_guess, sigma_guess, offset_guess]
    try:
        popt, _ = curve_fit(gaussian, q_subset, intensity_subset, p0=p0)
    except Exception as e:
        print("Gaussian fit failed:", e)
        return None
    peak_q = popt[1]
    peak_intensity = gaussian(peak_q, *popt)
    center = (input_range[0] + input_range[1]) / 2.0
    shift = peak_q - center
    output_range = (input_range[0] + shift, input_range[1] + shift)
    if Index_number > 0 and peak_info is not None:
        if peak_info.get("q_threshold") is None:
            prev_peak_q = peak_info.get("peak_q")
            if prev_peak_q is None:
                print("Previous peak_q is missing in peak_info.")
            else:
                peak_info["q_threshold"] = 1 if prev_peak_q >= 3 else 0.1
        q_threshold = peak_info.get("q_threshold", 0.1)
        if peak_info.get("intensity_threshold") is None:
            peak_info["intensity_threshold"] = 0.5
        intensity_threshold = peak_info.get("intensity_threshold", 0.5)
        prev_peak_q = peak_info.get("peak_q")
        prev_peak_intensity = peak_info.get("peak_intensity")
        if prev_peak_q is not None and abs(peak_q - prev_peak_q) > q_threshold:
            print("Peak q difference exceeds threshold.")
            return None
        if prev_peak_intensity is not None and abs(peak_intensity - prev_peak_intensity) > intensity_threshold * prev_peak_intensity:
            print("Peak intensity difference exceeds threshold.")
            return None
    return peak_q, peak_intensity, output_range

def adjust_q_range_on_failure(contour_data, peak_data, failure_index):
    """
    피크 추적 실패 시, 지금까지의 peak 데이터를 표시하고 수동으로 q 범위를 재설정.
    """
    print(f"Peak tracking failure at index {failure_index}.")
    print("Displaying contour plot with tracked peaks so far for reference.")
    plot_contour_with_peaks(contour_data, peak_data)
    print("Please manually adjust the q range for the current index using the plot.")
    new_q_range = select_q_range(contour_data, index=failure_index)
    return new_q_range

def track_peaks(contour_data, input_range, tracking_option=None):
    """
    contour_data의 각 데이터에 대해 연속적으로 피크를 추적.
    만약 피크 추적에 실패하면 수동 보정 후 재시도하며, 온도 정보도 함께 저장.
    """
    times_list = []
    peak_q_list = []
    peak_intensity_list = []
    temp_list = []  # 온도 저장용
    prev_peak_info = None
    n = len(contour_data["Data"])
    for i in range(n):
        time_val = contour_data["Data"][i]["Time"]
        temp_val = contour_data["Time-temp"][1][i]
        if i == 0:
            result = find_peak(contour_data, Index_number=0, input_range=input_range, peak_info=None)
            if result is not None:
                peak_q, peak_intensity, _ = result
                prev_peak_info = {"peak_q": peak_q, "peak_intensity": peak_intensity}
            else:
                raise RuntimeError("Peak tracking failed at index 0.")
        else:
            result = find_peak(contour_data, Index_number=i, input_range=input_range, peak_info=prev_peak_info)
            if result is None:
                partial_peak_data = {
                    "Series": contour_data["Series"],
                    "Time-temp": contour_data["Time-temp"],
                    "Data": [
                        {"Time": contour_data["Data"][j]["Time"],
                         "peak_q": peak_q_list[j],
                         "peak_Intensity": peak_intensity_list[j],
                         "Temperature": temp_list[j]}
                        for j in range(i)
                    ]
                }
                new_input_range = adjust_q_range_on_failure(contour_data, partial_peak_data, failure_index=i)
                result = find_peak(contour_data, Index_number=i, input_range=new_input_range, peak_info=None)
                if result is None:
                    raise RuntimeError(f"Peak tracking failed at index {i} even after manual adjustment.")
                else:
                    input_range = new_input_range
                    peak_q, peak_intensity, _ = result
                    prev_peak_info = {"peak_q": peak_q, "peak_intensity": peak_intensity}
            else:
                peak_q, peak_intensity, _ = result
                prev_peak_info = {
                    "peak_q": peak_q,
                    "peak_intensity": peak_intensity,
                    "q_threshold": prev_peak_info.get("q_threshold"),
                    "intensity_threshold": prev_peak_info.get("intensity_threshold")
                }
        times_list.append(time_val)
        peak_q_list.append(peak_q)
        peak_intensity_list.append(peak_intensity)
        temp_list.append(temp_val)
    new_contour = {
        "Series": contour_data["Series"],
        "Time-temp": contour_data["Time-temp"],
        "Data": [
            {"Time": t, "Temperature": temp, "peak_q": pq, "peak_Intensity": pi}
            for t, temp, pq, pi in zip(times_list, temp_list, peak_q_list, peak_intensity_list)
        ]
    }
    return new_contour

def plot_contour_with_peaks(contour_data, peak_data, graph_option=None):
    """
    contour_data를 기반으로 칸투어 플롯을 그리고, 그 위에 peak_data의 피크 위치들을 빨간 원으로 표시.
    """
    default_graph_option = {
        "figure_size": (12, 8),
        "figure_dpi": 300,
        "contour_levels": 100,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "contour_xlabel_text": "2theta (Cu K-alpha)",
        "contour_ylabel_text": "Elapsed Time",
        "global_ylim": None,
        "contour_xlim": None,
    }
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}
    times, _ = contour_data["Time-temp"]
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return
    lower_bound = np.nanpercentile(grid_intensity, final_opt["contour_lower_percentile"])
    upper_bound = np.nanpercentile(grid_intensity, final_opt["contour_upper_percentile"])
    fig, ax = plt.subplots(figsize=final_opt["figure_size"], dpi=final_opt["figure_dpi"])
    cp = ax.contourf(grid_q, grid_time, grid_intensity,
                     levels=final_opt["contour_levels"],
                     cmap=final_opt["contour_cmap"],
                     vmin=lower_bound,
                     vmax=upper_bound)
    if final_opt["contour_xlim"] is not None:
        ax.set_xlim(final_opt["contour_xlim"])
    if final_opt["global_ylim"] is not None:
        ax.set_ylim(final_opt["global_ylim"])
    ax.set_xlabel(final_opt["contour_xlabel_text"], fontsize=14, fontweight='bold')
    ax.set_ylabel(final_opt["contour_ylabel_text"], fontsize=14, fontweight='bold')
    ax.set_title("Contour Plot with Peak Positions", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    added_label = False
    for entry in peak_data["Data"]:
        peak_q = entry.get("peak_q")
        time_val = entry.get("Time")
        if peak_q is not None and not np.isnan(peak_q):
            if not added_label:
                ax.scatter(peak_q, time_val, color="red", edgecolors="black", s=100, label="Peak Position", zorder=10)
                added_label = True
            else:
                ax.scatter(peak_q, time_val, color="red", edgecolors="black", s=100, zorder=10)
    ax.legend()
    plt.show()

# ============================================================
# [New Functionality] - (C) SDD recalculation based on peak fitting and tracked_peaks update
# ============================================================
def calculate_sdd_for_tracked_peaks(tracked_peaks, fit_params, fit_index_range, original_sdd, beam_center_x, beam_center_y, pixel_size):
    """
    For each tracked peak, calculate the predicted 2theta using the linear fit:
         twotheta_predicted = a * Temperature + b,
    and calculate new SDD for indices greater than fit_index_range[1].
    """
    a, b = fit_params
    new_data = []
    fit_end = fit_index_range[1]
    
    for i, entry in enumerate(tracked_peaks["Data"]):
        T = entry["Temperature"]
        measured_twotheta = entry.get("peak_q", 0)  # 현재는 2theta 값이 "peak_q"에 저장되어 있음
        predicted_twotheta = a * T + b
        
        if (i > fit_end) and (measured_twotheta is not None) and (measured_twotheta != 0):
            # 측정된 2theta로부터 실제 거리 계산
            twotheta_rad = np.radians(predicted_twotheta) / 2  # 2theta/2 = theta
            r_pixels = original_sdd * np.tan(twotheta_rad) / pixel_size  # 예상되는 픽셀 반지름
            
            # 측정된 2theta로부터 실제 SDD 계산
            measured_twotheta_rad = np.radians(measured_twotheta) / 2
            new_sdd = r_pixels * pixel_size / np.tan(measured_twotheta_rad)
        else:
            new_sdd = original_sdd
            
        new_entry = entry.copy()
        new_entry["sdd"] = new_sdd
        new_entry["predicted_twotheta"] = predicted_twotheta
        new_data.append(new_entry)
    
    tracked_peaks["Data"] = new_data
    return tracked_peaks


# ============================================================
# [New Functionality] - (D) Recalculate q values using the theoretical formula
# ============================================================
def recalc_q_values(contour_data, tracked_peaks, original_sdd, beam_center_x, beam_center_y, pixel_size, image_size_x=1920, image_size_y=1920):
    """
    각 스캔에서 2theta 값을 재계산
    1. 원본 2theta 백업
    2. 픽셀 위치로부터 새로운 SDD를 사용하여 2theta 재계산
    """
    for i, entry in enumerate(contour_data["Data"]):
        original_twotheta = np.array(entry["q"])  # 현재는 2theta 값이 "q"에 저장되어 있음
        entry["twotheta_raw"] = original_twotheta.tolist()
        
        new_sdd = tracked_peaks["Data"][i].get("sdd")
        if new_sdd is None:
            new_sdd = original_sdd
            
        if abs(new_sdd - original_sdd) > 1e-6:
            # 각 픽셀 위치에 대한 2D 그리드 생성
            x = np.arange(image_size_x)
            y = np.arange(image_size_y)
            X, Y = np.meshgrid(x, y)
            
            # 빔 센터로부터의 반지름 계산 (픽셀 단위)
            R = np.sqrt((X - beam_center_x)**2 + (Y - beam_center_y)**2)
            
            # 원본 2theta에 해당하는 R 값 찾기
            original_theta_rad = np.radians(original_twotheta) / 2
            r_pixels = original_sdd * np.tan(original_theta_rad) / pixel_size
            
            # 새로운 SDD로 2theta 재계산
            theta_new = np.arctan(r_pixels * pixel_size / new_sdd)
            twotheta_new = np.degrees(theta_new * 2)
            
            entry["q"] = twotheta_new.tolist()  # 여전히 "q"로 저장하지만 실제로는 2theta 값
        else:
            entry["q"] = original_twotheta.tolist()
            
    return contour_data


# ============================================================
# [New Functionality] - Export tracked_peaks to CSV
# ============================================================
def export_tracked_peaks_to_excel(tracked_peaks, filename="tracked_peaks.csv"):
    """
    Export Time, Temperature, sdd, peak_q (2theta), predicted_twotheta, and peak_intensity from tracked_peaks to a CSV file.
    """
    rows = []
    for entry in tracked_peaks["Data"]:
        row = {
            "Time": entry.get("Time"),
            "Temperature": entry.get("Temperature"),
            "sdd": entry.get("sdd"),
            "peak_2theta": entry.get("peak_q"),  # 실제로는 2theta 값
            "predicted_2theta": entry.get("predicted_twotheta"),
            "peak_Intensity": entry.get("peak_Intensity")  # intensity 값도 추가
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Tracked peaks exported to {filename}")


# ============================================================
# [New Functionality] - Temperature-Peak Fitting Functions
# ============================================================
def select_index_range_for_temp_fitting(tracked_peaks):
    """
    Plot index vs Temperature from tracked_peaks and allow the user to select a linear fitting range.
    Returns a tuple (start_index, end_index).
    """
    temps = tracked_peaks["Time-temp"][1]
    indices = np.arange(len(temps))
    plt.figure()
    plt.plot(indices, temps, 'bo-', label="Temperature")
    plt.xlabel("Index")
    plt.ylabel("Temperature")
    plt.title("Select Index Range for Temperature Fitting (Click two points)")
    plt.legend()
    plt.grid(True)
    pts = plt.ginput(2, timeout=-1)
    plt.close()
    if len(pts) < 2:
        raise RuntimeError("Not enough points were selected for index range.")
    start_index = int(round(min(pts[0][0], pts[1][0])))
    end_index = int(round(max(pts[0][0], pts[1][0])))
    return (start_index, end_index)

def select_peak_fit_range(tracked_peaks, index_range):
    """
    Within the given index range, plot Temperature vs tracked peak_q and allow the user to select a temperature range.
    Returns a tuple (temp_min, temp_max).
    """
    start_idx, end_idx = index_range
    sub_temps = np.array(tracked_peaks["Time-temp"][1])[start_idx:end_idx+1]
    sub_peak_q = np.array([entry["peak_q"] for entry in tracked_peaks["Data"][start_idx:end_idx+1]])
    plt.figure()
    plt.plot(sub_temps, sub_peak_q, 'ro-', label="Tracked Peak q")
    plt.xlabel("Temperature")
    plt.ylabel("Peak q")
    plt.title("Select Temperature Range for Peak Fitting (Click two points)")
    plt.legend()
    plt.grid(True)
    pts = plt.ginput(2, timeout=-1)
    plt.close()
    if len(pts) < 2:
        raise RuntimeError("Not enough points were selected for peak fit range.")
    temp_min = min(pts[0][0], pts[1][0])
    temp_max = max(pts[0][0], pts[1][0])
    return (temp_min, temp_max)

def fit_peak_vs_temp(tracked_peaks, fit_temp_range):
    """
    Perform linear fitting of tracked peak_q vs Temperature for data within fit_temp_range.
    Returns (a, b) where q = a * Temperature + b.
    """
    temp_array = np.array(tracked_peaks["Time-temp"][1])
    peak_q_array = np.array([entry["peak_q"] for entry in tracked_peaks["Data"]])
    mask = (temp_array >= fit_temp_range[0]) & (temp_array <= fit_temp_range[1])
    if np.sum(mask) < 2:
        raise RuntimeError("Insufficient data points in the selected peak fit range.")
    fit_temps = temp_array[mask]
    fit_peak_q = peak_q_array[mask]
    a, b = np.polyfit(fit_temps, fit_peak_q, 1)
    return (a, b)

# ============================================================
# MAIN
# ============================================================
def main(dat_dir, log_path, output_dir, original_sdd, beam_center_x, beam_center_y, pixel_size, wavelength=1.5406):
    log_data = parse_log_file(log_path)
    extracted_data = process_dat_files(dat_dir, log_data)
    selected_series = select_series(extracted_data)
    normal_range, adjust_range = select_temperature_ranges(extracted_data, selected_series, debug=True)
    extracted_data = temp_correction(extracted_data, selected_series, normal_range, adjust_range, debug=True)
    for entry in extracted_data.values():
        if isinstance(entry, dict) and "q" in entry and "Intensity" in entry:
            if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
                print(f"Warning: Missing data detected in {entry.get('File Path', 'Unknown File')}")
    contour_data = extract_contour_data(selected_series, extracted_data)
    for entry in contour_data["Data"]:
        if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
            print(f"Warning: Missing data detected in time frame {entry['Time']}")
    plot_contour(contour_data, temp=True, legend=True)
    q_range = select_q_range(contour_data)
    print("Selected q range:", q_range)
    peak_result = find_peak(contour_data, Index_number=0, input_range=q_range)
    if peak_result is not None:
        peak_q, peak_intensity, output_range = peak_result
        print(f"First dataset peak: q = {peak_q}, intensity = {peak_intensity}, adjusted range = {output_range}")
    else:
        print("Peak not found in first dataset.")
    tracked_peaks = track_peaks(contour_data, input_range=q_range)
    plot_contour_with_peaks(contour_data, tracked_peaks)
    index_range = select_index_range_for_temp_fitting(tracked_peaks)
    print("Selected index range for temperature fitting:", index_range)
    peak_fit_temp_range = select_peak_fit_range(tracked_peaks, index_range)
    print("Selected temperature range for peak fitting:", peak_fit_temp_range)
    fit_params = fit_peak_vs_temp(tracked_peaks, peak_fit_temp_range)
    print("Fitted parameters (a, b):", fit_params)
    tracked_peaks = calculate_sdd_for_tracked_peaks(tracked_peaks, fit_params, index_range, 
                                                  original_sdd, beam_center_x, beam_center_y, pixel_size)
    export_tracked_peaks_to_excel(tracked_peaks, filename="tracked_peaks.csv")
    corrected_contour_data = recalc_q_values(contour_data, tracked_peaks, original_sdd, 
                                           beam_center_x, beam_center_y, pixel_size)
    plot_contour(corrected_contour_data, temp=True, legend=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from .dat files and match with log data.")
    parser.add_argument("input", nargs="?", 
                        default='/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/Averaged', 
                        help="Directory containing .dat files")
    parser.add_argument("log", nargs="?", 
                        default="./ready/log241219-HS", 
                        help="Path to the log file")
    parser.add_argument("output", nargs="?", 
                        default="./ready/test_output", 
                        help="Directory to save the extracted JSON file")
    parser.add_argument("--orig_sdd", type=float, default=227.7524,
                        help="Original SDD in mm")
    parser.add_argument("--beam_center_x", type=float, default=954.1370,
                        help="Beam center X position in pixels (Python coords)")
    parser.add_argument("--beam_center_y", type=float, default=632.0930,
                        help="Beam center Y position in pixels (Python coords)")
    parser.add_argument("--pixel_size", type=float, default=0.0886,
                        help="Pixel size in mm")
    parser.add_argument("--wavelength", type=float, default=1.5406,
                        help="X-ray wavelength in Angstroms")
    args = parser.parse_args()
    
    main(args.input, args.log, args.output, args.orig_sdd, 
         args.beam_center_x, args.beam_center_y, args.pixel_size, args.wavelength)
