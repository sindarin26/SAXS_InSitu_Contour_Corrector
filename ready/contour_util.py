import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit  # 가우시안 피팅용
import pandas as pd  # CSV (엑셀) 출력용
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files
import time

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
        "contour_xlim": [30, 60],
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
    Matplotlib을 사용한 대화형 q 범위 선택 함수입니다.
    
    이 함수는 다음과 같은 대화형 기능을 제공합니다:
    - 그래프 영역에서의 휠: 마우스 포인트를 중심으로 양방향 확대/축소
    - 축 주변에서의 휠: 해당 축만 확대/축소
    - 왼쪽 클릭: 범위 지점 선택
    - 오른쪽 클릭: 선택 초기화
    - Enter: 선택 확정
    
    Parameters
    ----------
    contour_data : dict
        데이터 딕셔너리
    index : int, optional
        분석할 데이터의 인덱스 (기본값: 0)
        
    Returns
    -------
    tuple
        선택된 q 범위 (q_min, q_max)
    """
    if index is None:
        index = 0
    try:
        data_entry = contour_data["Data"][index]
    except IndexError:
        raise ValueError(f"Index {index} is out of range for contour_data.")
    
    q_vals = np.array(data_entry["q"])
    intensity_vals = np.array(data_entry["Intensity"])
    
    # 그래프 설정
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.2, left=0.12)
    
    # 기본 플롯
    line, = ax.plot(q_vals, intensity_vals, 'ko-', label='Intensity', zorder=1)
    ax.set_xlabel("q")
    ax.set_ylabel("Intensity")
    ax.set_title("Select q Range\nUse mouse wheel on axes or plot area to zoom")
    ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.legend()
    
    # 상태 추적용 변수들
    selected_points = []
    selection_plots = []
    last_wheel_time = [0]  # 리스트로 만들어 참조로 전달
    wheel_center = [None, None]  # [x, y] 좌표
    WHEEL_TIMEOUT = 0.1  # 100ms timeout

    # 상태 메시지 표시용 텍스트
    status_text = ax.text(0.5, -0.2, "", 
                         transform=ax.transAxes,
                         horizontalalignment='center',
                         bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
    
    def update_status(message):
        """상태 메시지 업데이트"""
        status_text.set_text(message)
        plt.draw()
    
    def get_axis_from_position(event):
        """이벤트 위치에 따른 축 판별"""
        bbox = ax.get_position()
        margin = 0.05
        
        # 그래프 영역 밖에서의 위치 판별
        if event.inaxes != ax:
            fig_coord_x = event.x / fig.get_figwidth()
            fig_coord_y = event.y / fig.get_figheight()
            
            # x축 영역 (아래 또는 위)
            if bbox.x0 <= fig_coord_x <= bbox.x1:
                if abs(fig_coord_y - bbox.y0) <= margin:
                    return 'x'
            
            # y축 영역 (왼쪽 또는 오른쪽)
            if bbox.y0 <= fig_coord_y <= bbox.y1:
                if abs(fig_coord_x - bbox.x0) <= margin:
                    return 'y'
            
            return None
        
        return 'both'
    
    def on_scroll(event):
        """휠 스크롤 이벤트 처리"""
        current_time = time.time()
        axis_type = get_axis_from_position(event)
        
        if axis_type is None:
            return
            
        # 휠 타이밍 체크 및 중심점 업데이트
        if current_time - last_wheel_time[0] > WHEEL_TIMEOUT:
            wheel_center[0] = event.xdata
            wheel_center[1] = event.ydata
        last_wheel_time[0] = current_time
        
        # 줌 비율 설정
        base_scale = 1.1
        scale_factor = 1.0 / base_scale if event.button == 'up' else base_scale
        
        # 현재 축 범위
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # 중심점 설정
        center_x = wheel_center[0] if wheel_center[0] is not None else event.xdata
        center_y = wheel_center[1] if wheel_center[1] is not None else event.ydata
        
        # 축별 줌 적용
        if axis_type in ['both', 'x'] and center_x is not None:
            ax.set_xlim([
                center_x - (center_x - xmin) * scale_factor,
                center_x + (xmax - center_x) * scale_factor
            ])
                
        if axis_type in ['both', 'y'] and center_y is not None:
            ax.set_ylim([
                center_y - (center_y - ymin) * scale_factor,
                center_y + (ymax - center_y) * scale_factor
            ])
        
        plt.draw()
    
    def on_click(event):
        """마우스 클릭 이벤트 처리"""
        if event.inaxes != ax:
            return
        
        if event.button == 1:  # 왼쪽 클릭
            if len(selected_points) >= 2:
                update_status("Already selected 2 points. Right click to reset.")
                return
            
            selected_points.append(event.xdata)
            line = ax.axvline(x=event.xdata, color='r', linestyle='--', alpha=0.7)
            selection_plots.append(line)
            
            if len(selected_points) == 2:
                update_status("Press Enter to confirm, right click to reset")
            else:
                update_status(f"Selected point at q = {event.xdata:.3f}. Select one more point.")
            plt.draw()
            
        elif event.button == 3:  # 오른쪽 클릭
            selected_points.clear()
            for line in selection_plots:
                line.remove()
            selection_plots.clear()
            update_status("Selection reset. Select two points.")
            plt.draw()
    
    def on_key(event):
        """키보드 이벤트 처리"""
        if event.key == 'enter':
            if len(selected_points) == 2:
                plt.close()
            else:
                update_status("Please select two points before confirming.")
    
    # 이벤트 핸들러 연결
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    update_status("Select two points using left click. Use mouse wheel to zoom.")
    plt.show()
    
    # 선택 검증
    if len(selected_points) != 2:
        raise RuntimeError("Selection was not completed properly.")
    
    q_min = min(selected_points)
    q_max = max(selected_points)
    
    # 선택 범위 검증
    if q_max - q_min < 0.1:
        raise ValueError("Selected range is too narrow (< 0.1). Please select a wider range.")
    
    if q_min < min(q_vals) or q_max > max(q_vals):
        raise ValueError("Selected range is outside the data range.")
    
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
def interactive_peak_adjustment(contour_data, tracked_peaks, original_sdd, experiment_energy, converted_energy=8.042):
    """
    대화형으로 피크를 재조정하는 함수입니다. contour plot에서 피크를 클릭하면
    해당 시점의 데이터에 대해 q 범위를 다시 선택하고 피크를 재계산합니다.
    
    마우스 조작:
    - 휠: 줌 인/아웃
    - 왼쪽 클릭: 피크 선택
    - 오른쪽 클릭: 선택 취소
    - Enter: 선택 확정
    """
    times, temperatures = contour_data["Time-temp"]
    
    # 컨투어 플롯 데이터 준비
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])
    
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    
    # 그래프 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)  # 상태 메시지를 위한 여백
    
    # 컨투어 플롯
    cp = ax.contourf(grid_q, grid_time, grid_intensity, levels=100, cmap='inferno')
    fig.colorbar(cp, ax=ax, label='log10(Intensity)')
    
    # 피크 포인트 플롯
    peak_points = []
    for entry in tracked_peaks["Data"]:
        point = ax.scatter(entry["peak_q"], entry["Time"], 
                         color="red", edgecolors="black", s=100, zorder=10)
        peak_points.append(point)
    
    ax.set_xlabel("q")
    ax.set_ylabel("Time")
    ax.set_title("Click on a peak to readjust - Use mouse wheel to zoom")
    
    # 상태 메시지 표시용 텍스트 박스
    status_text = ax.text(0.5, -0.15, "", 
                         transform=ax.transAxes,
                         horizontalalignment='center',
                         bbox=dict(facecolor='white', alpha=0.8))
    
    def update_status(message):
        status_text.set_text(message)
        plt.draw()
    
    def find_nearest_peak(click_x, click_y):
        min_dist = float('inf')
        nearest_idx = None
        
        for i, entry in enumerate(tracked_peaks["Data"]):
            peak_x = entry["peak_q"]
            peak_y = entry["Time"]
            
            # 정규화된 거리 계산 (x와 y의 스케일이 다르므로)
            x_range = max(q_all) - min(q_all)
            y_range = max(times) - min(times)
            dx = (click_x - peak_x) / x_range
            dy = (click_y - peak_y) / y_range
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # 너무 멀리 있는 점은 무시 (임계값: 0.1)
        return nearest_idx if min_dist < 0.1 else None
    
    def on_click(event):
        if event.inaxes != ax:
            return
            
        if event.button == 1:  # 왼쪽 클릭
            idx = find_nearest_peak(event.xdata, event.ydata)
            if idx is not None:
                update_status(f"Selected peak at index {idx}. Adjusting q range...")
                plt.draw()
                
                # 현재 창 숨기기
                fig.canvas.draw_idle()
                plt.close()
                
                # q 범위 재선택
                try:
                    new_q_range = select_q_range(contour_data, index=idx)
                    result = find_peak(contour_data, Index_number=idx, 
                                     input_range=new_q_range, peak_info=None)
                    
                    if result is not None:
                        peak_q, peak_intensity, _ = result
                        tracked_peaks["Data"][idx]["peak_q"] = peak_q
                        tracked_peaks["Data"][idx]["peak_Intensity"] = peak_intensity
                        
                        # SDD 재계산이 필요한 경우
                        if tracked_peaks["Data"][idx].get("corrected_SDD") is not None:
                            new_sdd = calculate_corrected_sdd(
                                peak_q, original_sdd,
                                tracked_peaks["Data"][idx]["corrected_peak_q"],
                                experiment_energy, converted_energy
                            )
                            tracked_peaks["Data"][idx]["corrected_SDD"] = new_sdd
                        
                        print(f"Peak updated: q = {peak_q:.4f}, intensity = {peak_intensity:.4f}")
                        
                        # 업데이트된 결과로 다시 플롯
                        interactive_peak_adjustment(contour_data, tracked_peaks, 
                                                 original_sdd, experiment_energy, 
                                                 converted_energy)
                    else:
                        print("Failed to find peak in the selected range")
                        interactive_peak_adjustment(contour_data, tracked_peaks, 
                                                 original_sdd, experiment_energy, 
                                                 converted_energy)
                except Exception as e:
                    print(f"Error during peak adjustment: {e}")
                    interactive_peak_adjustment(contour_data, tracked_peaks, 
                                             original_sdd, experiment_energy, 
                                             converted_energy)
            else:
                update_status("No peak found near click position")
    
    def on_scroll(event):
        if event.inaxes != ax:
            return
        
        # 현재 축 범위
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # 줌 비율
        if event.button == 'up':
            scale_factor = 0.9
        else:
            scale_factor = 1.1
        
        # 마우스 위치 기준으로 줌
        center_x = event.xdata
        center_y = event.ydata
        
        new_width = (xmax - xmin) * scale_factor
        new_height = (ymax - ymin) * scale_factor
        
        ax.set_xlim([center_x - new_width/2, center_x + new_width/2])
        ax.set_ylim([center_y - new_height/2, center_y + new_height/2])
        plt.draw()
    
    # 이벤트 핸들러 연결
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    update_status("Click on a peak to readjust its position. Use mouse wheel to zoom.")
    plt.show()
    
    return tracked_peaks
# ============================================================
# [New Functionality] - (C) SDD recalculation based on peak fitting and tracked_peaks update
# ============================================================
def q_to_2theta(q, energy_keV):
    """
    q (Å⁻¹)를 주어진 에너지(keV)에서의 2θ (°)로 변환하는 함수.
    
    Parameters
    ----------
    q : float or array
        q 값 (Å⁻¹)
    energy_keV : float
        X-ray 에너지 (keV)
    
    Returns
    -------
    float or array
        변환된 2θ 값 (°)
    """
    wavelength = 12.398 / energy_keV  # λ = 12.398 / E
    theta_rad = np.arcsin((q * wavelength) / (4 * np.pi))
    theta_deg = np.degrees(2 * theta_rad)
    return theta_deg

def theta_to_q(theta_2, energy_keV):
    """
    2θ (°)를 주어진 에너지(keV)에서의 q (Å⁻¹)로 변환하는 함수.
    
    Parameters
    ----------
    theta_2 : float or array
        2θ 값 (°)
    energy_keV : float
        X-ray 에너지 (keV)
    
    Returns
    -------
    float or array
        변환된 q 값 (Å⁻¹)
    """
    wavelength = 12.398 / energy_keV
    theta_rad = np.radians(theta_2 / 2)
    q_value = (4 * np.pi / wavelength) * np.sin(theta_rad)
    return q_value

def max_SDD_calculation(theta_2_max, pixel_size, beam_center_x, beam_center_y, image_size):
    """
    이미지의 네 모서리를 기준으로, 최대 방사 거리(R)를 구한 후 이를 이용하여
    최대 SDD (sample-to-detector distance)를 계산하는 함수.
    
    Parameters
    ----------
    theta_2_max : float
        최대 2θ 값 (°)
    pixel_size : float
        픽셀 크기 (mm)
    beam_center_x : float
        빔 센터 x 좌표 (MATLAB 좌표 기준)
    beam_center_y : float
        빔 센터 y 좌표 (MATLAB 좌표 기준)
    image_size : tuple
        이미지 크기 (width, height)
    
    Returns
    -------
    tuple
        (최대 SDD (mm), 최대 R (mm))
    """
    theta_max_rad = np.radians(theta_2_max)
    corners = [(1, 1), (1, image_size[1]), (image_size[0], 1), (image_size[0], image_size[1])]
    
    R_pixel_max = max(np.sqrt((corner_x - beam_center_x) ** 2 + 
                               (corner_y - beam_center_y) ** 2) 
                      for corner_x, corner_y in corners)
    corrected_R_pixel_max = round(R_pixel_max, 0) - 1
    R_mm_max = corrected_R_pixel_max * pixel_size
    
    SDD_max = R_mm_max / np.tan(theta_max_rad)
    return SDD_max, R_mm_max

def calculate_corrected_sdd(original_2theta, original_sdd, corrected_2theta, exp_energy, converted_energy=8.042):
    """
    [단일 포인트용] 원래 데이터(원래 2θ 값 또는 q 값)를 기반으로,
    SDD 보정 후의 새로운 SDD 값을 계산한다.
    
    데이터 체계:
      - 만약 converted_energy가 None이면, 원본 데이터는 q 값으로 저장됨.
      - converted_energy가 주어지면, 원본 데이터는 변환 전 CuKα 2θ 값으로 저장됨.
    
    절차
    -------
    1) 입력 데이터(원래 2θ 또는 q)를, 만약 converted_energy가 주어졌다면 CuKα 기준의 2θ → q로 변환.
    2) 해당 q 값을 exp_energy 기준의 2θ로 변환.
    3) 기존 SDD를 이용하여, 각 포인트의 detector 상 방사거리 R = original_sdd * tan(2θ_exp)를 계산.
    4) 같은 R에서, 보정 SDD를 적용하면 새로운 2θ (exp_energy 기준)는 arctan(R/newSDD)로 구해짐.
    5) 이를 다시 q (exp_energy 기준)로 변환.
       - 만약 converted_energy가 주어지면 최종 결과를 다시 CuKα 2θ (즉, theta_to_q와 q_to_2theta를 거쳐)
         변환하여 출력한다.
    
    Parameters
    ----------
    original_2theta : float
        원래 데이터 값; converted_energy가 None이면 q 값, 아니면 CuKα 기준 2θ (°)
    original_sdd : float
        원래 SDD (mm)
    corrected_2theta : float
        보정하고자 하는 2θ 값; converted_energy가 None이면 q 값, 아니면 CuKα 기준 2θ (°)
    exp_energy : float
        실험 X-ray 에너지 (keV) – R 계산 시 사용
    converted_energy : float, optional
        입력 데이터가 변환된 CuKα 에너지 기준 값 (keV). 
        None이면 입력/출력은 q 값로 처리함.
    
    Returns
    -------
    corrected_sdd : float
        보정 후의 SDD (mm), 계산은 exp_energy 기준 2θ를 사용.
    """
    if converted_energy is None:
        # 입력 데이터가 q 값으로 저장됨
        q_original = original_2theta
        q_corrected = corrected_2theta
    else:
        # 입력 데이터가 CuKα 기준의 2θ 값으로 저장됨 → q로 변환
        q_original = theta_to_q(original_2theta, converted_energy)
        q_corrected = theta_to_q(corrected_2theta, converted_energy)
    
    # exp_energy 기준 2θ 계산
    exp_2theta = q_to_2theta(q_original, exp_energy)
    
    # 원래 SDD에서의 detector 반경 R 계산
    original_r = original_sdd * np.tan(np.radians(exp_2theta))
    
    # 보정된 2θ (exp_energy 기준) 계산: R = newSDD * tan(2θ_new)
    corrected_exp_2theta = q_to_2theta(q_corrected, exp_energy)
    
    # 보정 후 SDD = R / tan(2θ_new)
    corrected_sdd = original_r / np.tan(np.radians(corrected_exp_2theta))
    
    return corrected_sdd

def recalc_q_list(
    q_list,        
    original_sdd,  
    corrected_sdd, 
    energy_keV,     
    converted_energy=8.042  
):
    """
    [배열 단위] 원래 데이터(저장형태에 따라 q 값 또는 CuKα 기준 2θ 값)를,
    기존 SDD(original_sdd)에서 측정된 것으로부터 보정 SDD(corrected_sdd)를 적용할 경우
    최종적으로 exp_energy 기준으로 얻어야 하는 데이터(출력 형식은 입력과 동일)를 재계산한다.
    
    데이터 체계
    -----------
    - 만약 converted_energy가 None이면, 입력 q_list는 q 값이며 최종 결과도 q 값이다.
    - converted_energy가 주어지면, 입력 q_list는 CuKα 기준의 2θ 값(°)으로 저장되어 있으며,
      최종 결과도 동일한 방식(즉, 2θ 값)으로 출력한다.
    
    계산 절차
    -----------
    1) (converted_energy가 주어졌다면) 입력 2θ(CuKα) → q (CuKα)로 변환.
    2) 위 q 값을 exp_energy 기준의 2θ로 변환.
    3) 원래 SDD를 사용하여, 각 포인트의 반경 R = original_sdd * tan( exp_energy 기준 2θ ) 계산.
    4) 같은 R에 대해, 보정 SDD를 적용하면 새로운 2θ (exp_energy 기준)는 arctan(R / corrected_sdd)로 구해짐.
    5) 이 새로운 2θ를 exp_energy 기준의 q로 변환.
    6) 만약 converted_energy가 주어졌다면, 최종 결과를 다시 CuKα 기준의 2θ 값으로 변환.
    
    Parameters
    ----------
    q_list : array-like
        원래 데이터. 
        - converted_energy가 None이면 q 값 (Å⁻¹)
        - converted_energy가 주어지면 CuKα 기준의 2θ 값 (°)
    original_sdd : float
        원래 SDD (mm)
    corrected_sdd : float
        보정된 SDD (mm)
    energy_keV : float
        exp_energy (q 또는 2θ 변환 시 사용되는 에너지, keV)
    converted_energy : float, optional
        입력 데이터가 CuKα 기준일 경우의 에너지 (keV). 
        None이면 변환 없이 q 값으로 처리.
    
    Returns
    -------
    corrected_q_list : ndarray
        - converted_energy가 None이면: 보정 후 q 값 (Å⁻¹)
        - converted_energy가 주어지면: 보정 후 CuKα 기준의 2θ 값 (°)
    """
    # 만약 converted_energy가 주어졌다면, 입력 데이터는 2θ 값(°)이므로 q로 변환
    if converted_energy is None:
        # 입력이 q 값으로 저장됨 → 별도 변환 없이 사용
        q_conv = q_list
    else:
        q_conv = theta_to_q(q_list, converted_energy)
    
    # 1) exp_energy 기준의 2θ 값으로 변환
    exp_2theta = q_to_2theta(q_conv, energy_keV)
    
    # 2) 원래 SDD에서의 반경 R 계산
    R = original_sdd * np.tan(np.radians(exp_2theta))
    
    # 3) 보정 SDD 적용 시의 새로운 2θ (exp_energy 기준)
    corrected_2theta = np.degrees(np.arctan(R / corrected_sdd))
    
    # 4) 새로운 2θ를 exp_energy 기준의 q로 변환
    q_temp = theta_to_q(corrected_2theta, energy_keV)
    
    # 5) 만약 converted_energy가 주어졌다면, 최종 결과를 CuKα 기준의 2θ 값으로 변환
    if converted_energy is None:
        corrected_q_list = q_temp
    else:
        corrected_q_list = q_to_2theta(q_temp, converted_energy)
    
    return corrected_q_list

def add_corrected_peak_q(tracked_peaks, fit_params, peak_fit_temp_range, original_SDD, experiment_energy, converted_energy):
    """
    fit_params를 사용하여 corrected_peak_q 값을 계산하고 tracked_peaks에 추가

    Parameters:
        tracked_peaks (dict): tracked_peaks 데이터
        fit_params (tuple): (a, b) - 피팅 파라미터
        peak_fit_temp_range (tuple): (temp_min, temp_max) - 피팅에 사용된 온도 범위

    Returns:
        dict: corrected_peak_q가 SDD, original_SDD가 추가된 tracked_peaks
    """
    a, b = fit_params
    temp_max = peak_fit_temp_range[1]
    
    for entry in tracked_peaks["Data"]:
        T = entry["Temperature"]
        entry["original_SDD"] = original_SDD
        if T > temp_max:  # 피팅 온도 범위 이후의 데이터에 대해서만
            corrected_q = a * T + b  # 피팅 결과로 계산된 q 값
            entry["corrected_peak_q"] = corrected_q
            entry["corrected_SDD"] = round(calculate_corrected_sdd(entry["peak_q"], original_SDD, corrected_q, experiment_energy, converted_energy), 4)
        else:
            entry["corrected_peak_q"] = None
            entry["corrected_SDD"] = None
            
    return tracked_peaks


def calculate_sdd_for_tracked_peaks_refactored(contour_data, tracked_peaks, original_sdd, experiment_energy, converted_energy=8.042):
    """
    각 타임프레임의 contour_data["Data"] 항목에 대해,
      1. 원래의 q 리스트를 "q_raw"라는 키에 백업하고,
      2. 해당 타임프레임에 대응하는 tracked_peaks["Data"] 항목에 corrected_SDD와 corrected_peak_q가 존재하면,
         re-calculation을 통해 새 SDD에 맞는 q 리스트를 계산하여 "q" 값을 덮어쓴다.
    
    데이터 체계
    -----------
    - contour_data["Data"]의 각 항목은 원래 q 리스트(입력 형식에 따라 q 또는 CuKα 기준의 2θ 값)를 포함한다.
    - tracked_peaks["Data"]의 각 항목은 같은 타임프레임에 해당하며, 
      corrected_SDD와 corrected_peak_q가 있으면 해당 타임프레임의 q 리스트를 새 SDD에 맞게 재계산할 대상이다.
    - converted_energy가 None이면 입력/출력 모두 q 값으로 처리되고,
      converted_energy가 주어지면 입력 데이터는 CuKα 기준의 2θ 값(°)으로 저장되어 있으며, 최종 출력도 2θ 값 형식으로 처리된다.
    
    계산 절차
    -----------
    1) 각 항목에 대해 원래 q 리스트를 "q_raw"에 백업한다.
    2) 만약 해당 타임프레임의 tracked_peaks 데이터에 corrected_SDD와 corrected_peak_q가 존재하면,
       recalc_q_list 함수를 사용하여 q_raw를 원래 SDD(original_sdd)에서 보정 SDD(corrected_SDD)를 적용한 
       새 q 리스트로 재계산한다.
    3) 재계산된 q 리스트로 해당 항목의 "q" 값을 덮어쓴다.
    
    Parameters
    ----------
    contour_data : dict
        contour 데이터가 저장된 dict. "Data" 키 아래 각 항목은 최소한 "q" (및 "Intensity", "Time" 등)를 포함.
    tracked_peaks : dict
        피크 추적 결과 데이터가 저장된 dict. "Data" 키 아래 각 항목은 피크 관련 정보와 함께 corrected_SDD,
        corrected_peak_q 등이 포함될 수 있음.
    original_sdd : float
        원래 사용된 SDD (mm)
    experiment_energy : float
        q/2θ 변환 시 사용할 실험 X-ray 에너지 (keV)
    converted_energy : float, optional
        입력 데이터가 CuKα 기준으로 변환된 에너지 (keV). None이면 입력/출력은 q 값으로 처리.
    
    Returns
    -------
    dict
        새 SDD 보정이 적용된 contour_data (각 항목의 "q" 값이 재계산됨).
    """
    # contour_data의 각 항목에 대해 원래 q 리스트 백업 (q_raw)하기
    for entry in contour_data["Data"]:
        if "q_raw" not in entry:
            # np.array()로 복사하면 안전하게 백업 가능 (리스트일 경우에도)
            entry["q_raw"] = np.array(entry["q"])
    
    # contour_data와 tracked_peaks의 항목이 같은 순서라고 가정하고 index-wise로 처리
    for i, entry in enumerate(contour_data["Data"]):
        # 해당 타임프레임의 tracked_peaks 데이터 가져오기
        try:
            peak_entry = tracked_peaks["Data"][i]
        except IndexError:
            # 만약 tracked_peaks 데이터가 부족하면 건너뛰기
            continue
        
        # corrected_SDD와 corrected_peak_q가 존재하는 경우에만 q 리스트 재계산
        if peak_entry.get("corrected_SDD") is not None and peak_entry.get("corrected_peak_q") is not None:
            new_sdd = peak_entry["corrected_SDD"]
            # q_list 재계산: 입력은 백업된 원래 q_raw
            new_q = recalc_q_list(
                q_list=entry["q_raw"],
                original_sdd=original_sdd,
                corrected_sdd=new_sdd,
                energy_keV=experiment_energy,
                converted_energy=converted_energy
            )
            # 재계산된 q 리스트로 덮어쓰기
            entry["q"] = new_q
        # 만약 해당 타임프레임에 보정 정보가 없다면 원래 q (또는 q_raw) 그대로 유지
    
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

def export_tracked_peaks_to_excel_with_correction(tracked_peaks, filename="tracked_peaks_with_correction.csv"):
    """
    Export tracked peaks data including corrected_peak_q to a CSV file.
    
    Parameters:
        tracked_peaks (dict): tracked_peaks 데이터
        filename (str): 저장할 파일 이름
    """
    rows = []
    for entry in tracked_peaks["Data"]:
        row = {
            "Time": entry.get("Time"),
            "Temperature": entry.get("Temperature"),
            "peak_q": entry.get("peak_q"),
            "peak_Intensity": entry.get("peak_Intensity"),
            "corrected_peak_q": entry.get("corrected_peak_q"),
            "corrected_SDD": entry.get("corrected_SDD"),
            "original_SDD": entry.get("original_SDD")
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
def main(dat_dir, log_path, original_sdd, image_size, beam_center, pixel_size, experiment_energy, converted_energy, q_format):
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


    while True:
        tracked_peaks = interactive_peak_adjustment(contour_data, tracked_peaks, 
                                             original_sdd, experiment_energy, 
                                             converted_energy)
        proceed = input("Do you want to proceed with the adjusted peaks? (y/n): ").strip().lower()
        if proceed == 'y':
            break
        elif proceed == 'n':
            print("Readjusting peaks...")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


    index_range = select_index_range_for_temp_fitting(tracked_peaks)
    print("Selected index range for temperature fitting:", index_range)

    peak_fit_temp_range = select_peak_fit_range(tracked_peaks, index_range)
    print("Selected temperature range for peak fitting:", peak_fit_temp_range)

    fit_params = fit_peak_vs_temp(tracked_peaks, peak_fit_temp_range)
    print("Fitted parameters (a, b):", fit_params)

    # 새로운 함수 호출
    tracked_peaks = add_corrected_peak_q(tracked_peaks, fit_params, index_range, original_sdd, experiment_energy, converted_energy)
    
    # 결과 확인을 위한 export
    export_tracked_peaks_to_excel_with_correction(tracked_peaks, "tracked_peaks_with_correction.csv")


    corrected_contour_data = calculate_sdd_for_tracked_peaks_refactored(contour_data, tracked_peaks, original_sdd, experiment_energy, converted_energy)
    

    #export_tracked_peaks_to_excel(tracked_peaks, filename="tracked_peaks.csv")
    
    
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
    parser.add_argument("--beam_center_x", type=float, default=955.1370,
                        help="Beam center X position in pixels (Matlab coords, which is 1+based)")
    parser.add_argument("--beam_center_y", type=float, default=633.0930,
                        help="Beam center Y position in pixels (Matlab coords, which is 1+based)")
    parser.add_argument("--pixel_size", type=float, default=0.0886,
                        help="Pixel size in mm")
    parser.add_argument("--experiment_energy", type=float, default=19.78,
                        help="X-ray energy in keV")
    parser.add_argument("--converted_energy", type=float, default=8.042,
                        help="Converted X-ray energy in keV, Default: 8.042 keV; Cu K-alpha")
    parser.add_argument("--image_size_x", type=int, default=1920,
                        help="Image size in X direction")
    parser.add_argument("--image_size_y", type=int, default=1920,
                        help="Image size in Y direction")
    parser.add_argument("--q_format", type=str, default='CuKalpha',
                        help="q_format, Default: 'CuKalpha', Options: 'q', 'CuKalpha'")
    args = parser.parse_args()

    image_size = (args.image_size_x, args.image_size_y)
    beam_center = (args.beam_center_x, args.beam_center_y)

    main(args.input, args.log, args.orig_sdd, image_size, beam_center, args.pixel_size, args.experiment_energy, args.converted_energy, args.q_format)