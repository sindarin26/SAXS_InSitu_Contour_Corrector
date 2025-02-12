import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import curve_fit  # 가우시안 피팅을 위한 함수
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files

# 기존 함수들 ---------------------------------------------------------

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
    
    # 정렬 키가 "Series Elapsed Time"인 경우 (혹은 "Series Index"로 할 수도 있음)
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])  
    
    times = [entry["Series Elapsed Time"] for entry in series_entries]
    temperatures = [entry["Temperature"] for entry in series_entries]
    q_values = [entry["q"] for entry in series_entries]
    intensity_values = [entry["Intensity"] for entry in series_entries]
    
    return {
        "Series": selected_series,
        "Time-temp": (times, temperatures),
        "Data": [
            {
                "Time": times[i], 
                "q": q_values[i], 
                "Intensity": intensity_values[i]
            } 
            for i in range(len(times))
        ]
    }

def plot_contour(contour_data, temp=False, legend=True, graph_option=None):
    """
    Generate contour plot with interpolated data, optionally with a Temperature vs Time subplot.

    Parameters
    ----------
    contour_data : dict
        {
          "Series": str,
          "Time-temp": ([times], [temperatures]),
          "Data": [
              {"Time": t1, "q": [...], "Intensity": [...]},
              {"Time": t2, "q": [...], "Intensity": [...]},
              ...
          ]
        }
    temp : bool
        False -> 칸투어만 그림
        True  -> 오른쪽에 Temperature vs Time 플롯 추가
    legend : bool
        True  -> 컬러바(범례) 표시
        False -> 컬러바 표시 안 함
    graph_option : dict
        그래프의 폰트/범위/레이블 등 사용자 지정 옵션 딕셔너리.
        None 또는 빈 딕셔너리 이면 기본 옵션을 사용.
    """

    # ========== (1) 기본 옵션 정의 ==========
    default_graph_option = {
        # 1) Figure 전체 설정
        "figure_size": (12, 8),
        "figure_dpi": 300,
        "figure_title_enable": False,    # 전체 Figure에 suptitle 달지 여부
        "figure_title_text": "",

        # 2) 폰트 및 크기 설정
        "font_label": "Times New Roman",      # x, y축 레이블/컬러바 레이블 폰트
        "font_tick": "Times New Roman",       # 틱 라벨 폰트
        "font_title": "Times New Roman",      # 그래프 타이틀 폰트
        "axes_label_size": 14,                # 축 라벨 폰트 크기
        "tick_label_size": 12,                # 틱 라벨 폰트 크기
        "title_size": 16,                     # 타이틀 폰트 크기

        # 3) Contour subplot 설정
        "contour_xlabel_enable": True,
        "contour_xlabel_text": "2theta (Cu K-alpha)",
        "contour_ylabel_enable": True,
        "contour_ylabel_text": "Elapsed Time",
        "contour_title_enable": True,
        "contour_title_text": "Contour Plot",
        "contour_xlim": None,    # (xmin, xmax) or None
        "contour_grid": False,   # Grid 표시 여부

        # 4) Temp subplot 설정 (temp=True 일 때만)
        "temp_xlabel_enable": True,
        "temp_xlabel_text": "Temperature",
        "temp_ylabel_enable": True,
        "temp_ylabel_text": "Elapsed Time",
        "temp_title_enable": False,
        "temp_title_text": "Temperature Plot",
        "temp_xlim": None,       # (xmin, xmax) or None
        "temp_grid": True,       # Temp 그래프 Grid 표시 여부

        # 5) Y축 범위(공유)
        "global_ylim": None,     # (ymin, ymax) or None

        # 6) 서브플롯 간격
        "wspace": 0.00,
        "width_ratios": [7, 2],  # 칸투어 : 온도 그래프 너비 비율

        # 7) 칸투어 옵션
        "contour_levels": 100,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,

        # 8) Colorbar 옵션
        "colorbar_label": "log10(Intensity)",
        "cbar_location": "left",   # 'left' or 'right' (Matplotlib 3.3+)
        "cbar_pad": 0.15
    }

    # (1b) 사용자 옵션 merge
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}

    # ========== (2) contour_data 에서 기본 데이터 추출 ==========
    times, temperatures = contour_data["Time-temp"]

    # 모든 q, intensity, time 1D array화
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])

    # q축 / time축 공통 그리드
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)

    # 보간
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return

    # 퍼센타일로 명암 대비 설정
    lower_bound = np.nanpercentile(grid_intensity, final_opt["contour_lower_percentile"])
    upper_bound = np.nanpercentile(grid_intensity, final_opt["contour_upper_percentile"])

    # ========== (3) Figure 생성 ==========
    if temp:
        # temp=True → 2개 subplot (왼쪽: 칸투어, 오른쪽: temp), sharey=True
        fig, (ax_contour, ax_temp) = plt.subplots(
            ncols=2,
            sharey=True,
            figsize=final_opt["figure_size"],
            dpi=final_opt["figure_dpi"],
            gridspec_kw={"width_ratios": final_opt["width_ratios"]}
        )
        plt.subplots_adjust(wspace=final_opt["wspace"])

        # -------- (A) 칸투어 subplot --------
        cp = ax_contour.contourf(
            grid_q,
            grid_time,
            grid_intensity,
            levels=final_opt["contour_levels"],
            cmap=final_opt["contour_cmap"],
            vmin=lower_bound,
            vmax=upper_bound
        )

        # x, y축 범위
        if final_opt["contour_xlim"] is not None:
            ax_contour.set_xlim(final_opt["contour_xlim"])
        if final_opt["global_ylim"] is not None:
            ax_contour.set_ylim(final_opt["global_ylim"])

        # X 라벨
        if final_opt["contour_xlabel_enable"]:
            ax_contour.set_xlabel(
                final_opt["contour_xlabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
        # Y 라벨
        if final_opt["contour_ylabel_enable"]:
            ax_contour.set_ylabel(
                final_opt["contour_ylabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )

        # 타이틀
        if final_opt["contour_title_enable"]:
            ax_contour.set_title(
                final_opt["contour_title_text"],
                fontsize=final_opt["title_size"],
                fontweight='bold',
                fontname=final_opt["font_title"]
            )

        # 틱 라벨 폰트
        ax_contour.tick_params(axis='x', labelsize=final_opt["tick_label_size"])
        ax_contour.tick_params(axis='y', labelsize=final_opt["tick_label_size"])
        for lbl in ax_contour.get_xticklabels() + ax_contour.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])

        # Grid 여부
        if final_opt["contour_grid"]:
            ax_contour.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # 컬러바
        if legend:
            c = fig.colorbar(
                cp, ax=ax_contour,
                orientation='vertical',
                location=final_opt["cbar_location"],
                pad=final_opt["cbar_pad"]
            )
            c.set_label(
                final_opt["colorbar_label"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
            c.ax.tick_params(labelsize=final_opt["tick_label_size"])
            for clbl in c.ax.get_yticklabels():
                clbl.set_fontname(final_opt["font_tick"])

        # -------- (B) Temp subplot --------
        ax_temp.plot(temperatures, times, 'r-')
        ax_temp.invert_xaxis()  # 왼쪽=고온, 오른쪽=저온

        # x, y축 범위
        if final_opt["temp_xlim"] is not None:
            ax_temp.set_xlim(final_opt["temp_xlim"])
        # y축은 sharey=True → contour와 동일
        if final_opt["global_ylim"] is not None:
            ax_temp.set_ylim(final_opt["global_ylim"])

        # Temp X 라벨
        if final_opt["temp_xlabel_enable"]:
            ax_temp.set_xlabel(
                final_opt["temp_xlabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
        else:
            # 숨기기
            ax_temp.set_xlabel("")

        # Temp Y 라벨
        if final_opt["temp_ylabel_enable"]:
            ax_temp.yaxis.set_label_position("right")
            ax_temp.yaxis.tick_right()
            ax_temp.set_ylabel(
                final_opt["temp_ylabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
        else:
            # 숨길 경우
            ax_temp.yaxis.set_label_position("right")
            ax_temp.yaxis.tick_right()
            ax_temp.set_ylabel("")

        # Temp 타이틀
        if final_opt["temp_title_enable"]:
            ax_temp.set_title(
                final_opt["temp_title_text"],
                fontsize=final_opt["title_size"],
                fontweight='bold',
                fontname=final_opt["font_title"]
            )

        # 틱 라벨 폰트
        ax_temp.tick_params(axis='x', labelsize=final_opt["tick_label_size"], direction='in')
        ax_temp.tick_params(axis='y', labelsize=final_opt["tick_label_size"], direction='in')
        for lbl in ax_temp.get_xticklabels() + ax_temp.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])

        # Grid
        if final_opt["temp_grid"]:
            ax_temp.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # Figure 전체 타이틀
        if final_opt["figure_title_enable"]:
            fig.suptitle(
                final_opt["figure_title_text"],
                fontsize=final_opt["title_size"],
                fontweight='bold',
                fontname=final_opt["font_title"]
            )

    else:
        # temp=False → 칸투어 플롯만
        fig, ax = plt.subplots(
            figsize=final_opt["figure_size"],
            dpi=final_opt["figure_dpi"]
        )
        cp = ax.contourf(
            grid_q,
            grid_time,
            grid_intensity,
            levels=final_opt["contour_levels"],
            cmap=final_opt["contour_cmap"],
            vmin=lower_bound,
            vmax=upper_bound
        )

        # x, y축 범위
        if final_opt["contour_xlim"] is not None:
            ax.set_xlim(final_opt["contour_xlim"])
        if final_opt["global_ylim"] is not None:
            ax.set_ylim(final_opt["global_ylim"])

        # X 라벨
        if final_opt["contour_xlabel_enable"]:
            ax.set_xlabel(
                final_opt["contour_xlabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
        # Y 라벨
        if final_opt["contour_ylabel_enable"]:
            ax.set_ylabel(
                final_opt["contour_ylabel_text"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )

        # 타이틀
        if final_opt["contour_title_enable"]:
            ax.set_title(
                final_opt["contour_title_text"],
                fontsize=final_opt["title_size"],
                fontweight='bold',
                fontname=final_opt["font_title"]
            )

        # 틱 라벨 폰트
        ax.tick_params(axis='x', labelsize=final_opt["tick_label_size"])
        ax.tick_params(axis='y', labelsize=final_opt["tick_label_size"])
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname(final_opt["font_tick"])

        # Grid
        if final_opt["contour_grid"]:
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # 컬러바
        if legend:
            c = fig.colorbar(
                cp, ax=ax,
                orientation='vertical',
                location=final_opt["cbar_location"],
                pad=final_opt["cbar_pad"]
            )
            c.set_label(
                final_opt["colorbar_label"],
                fontsize=final_opt["axes_label_size"],
                fontweight='bold',
                fontname=final_opt["font_label"]
            )
            c.ax.tick_params(labelsize=final_opt["tick_label_size"])
            for clbl in c.ax.get_yticklabels():
                clbl.set_fontname(final_opt["font_tick"])

        # Figure 전체 타이틀
        if final_opt["figure_title_enable"]:
            fig.suptitle(
                final_opt["figure_title_text"],
                fontsize=final_opt["title_size"],
                fontweight='bold',
                fontname=final_opt["font_title"]
            )

    plt.show()

# ------------------------------------------------------------------------
# FUNCTIONS FOR TEMPERATURE CORRECTION
# ------------------------------------------------------------------------

def select_temperature_ranges(extracted_data, selected_series, debug=False):
    """
    사용자 인터랙션을 통해 normal_range, adjust_range를 지정.
    normal_range, adjust_range는 (xmin, xmax) 형태의 시간 구간으로 반환.
    """
    # series 데이터만 추출
    series_entries = [
        data for data in extracted_data.values()
        if isinstance(data, dict)
        and data.get("Series") == selected_series
    ]
    # (Series Elapsed Time, Temperature, 인덱스) 정렬
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

    # ginput 등으로 두 점을 찍어 normal_range 구간을 잡는다.
    # ※ jupyter 환경에서 동작 안 될 수 있음. console/matplotlib-backend에서 시도.
    pts = plt.ginput(2, timeout=-1)  # 2번 클릭
    normal_range = (min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0]))
    
    plt.title("Select Adjust Range by Clicking Two Points (Start, End)")
    pts = plt.ginput(2, timeout=-1)  # 또 2번 클릭
    adjust_range = (min(pts[0][0], pts[1][0]), max(pts[0][0], pts[1][0]))
    
    if debug:
        print(f"[Debug] normal_range = {normal_range}")
        print(f"[Debug] adjust_range = {adjust_range}")
    
    plt.close()  # 그래프 창 닫기
    return normal_range, adjust_range


def temp_correction(extracted_data, selected_series, normal_range, adjust_range, debug=False):
    """
    normal_range에서 선형 피팅 후, adjust_range 구간 내에서
    '온도 변화가 0인 구간'을 찾아 해당 구간의 Temperature를 피팅값으로 교정한다.
    """
    # 시리즈 데이터만 추출 및 정렬
    series_entries = [
        data for data in extracted_data.values()
        if isinstance(data, dict)
        and data.get("Series") == selected_series
    ]
    series_entries.sort(key=lambda x: x["Series Elapsed Time"])
    
    times = np.array([entry["Series Elapsed Time"] for entry in series_entries])
    temps = np.array([entry["Temperature"] for entry in series_entries])
    
    # 1) normal_range에 해당하는 부분 추출
    normal_mask = (times >= normal_range[0]) & (times <= normal_range[1])
    normal_times = times[normal_mask]
    normal_temps = temps[normal_mask]
    
    if len(normal_times) < 2:
        print("Warning: Normal range has insufficient points for fitting.")
        return extracted_data
    
    # 선형 피팅 (y = a*x + b)
    # np.polyfit( x, y, 차수 ) => [a, b]
    a, b = np.polyfit(normal_times, normal_temps, 1)
    
    # 2) adjust_range 추출
    adjust_mask = (times >= adjust_range[0]) & (times <= adjust_range[1])
    adjust_indices = np.where(adjust_mask)[0]  # adjust_range 내에 해당하는 인덱스
    
    # 3) adjust_range 내에서 온도 변화량이 0인 구간 찾기
    #  diff(temp) = 0 인 부분만 찾아서 교정
    diffs = np.diff(temps[adjust_mask])
    zero_diff_indices = np.where(diffs == 0)[0]  # local index
    
    # 실제 global index로 변환 (adjust_indices[i], adjust_indices[i+1] ...)
    # zero_diff_indices는 [0,1,2,...] 형태
    # zero_diff_indices+1 은 다음 포인트
    zero_diff_global = adjust_indices[zero_diff_indices+1]
    
    # 4) 보정값 계산: fitted_temp = a * times[i] + b
    fitted_temps = a * times + b
    
    # 5) "Temperature before adjust" 백업 키를 만들고,
    #    zero_diff_global에 해당하는 포인트만 보정
    for i, entry in enumerate(series_entries):
        if "Temperature before adjust" not in entry:
            entry["Temperature before adjust"] = entry["Temperature"]
        
        # adjust_range 안에 있고, 그리고 diff=0 구간으로 찍힌다면 보정 적용
        if i in zero_diff_global:
            entry["Adjusted Temperature"] = fitted_temps[i]
            # 최종적으로 "Temperature" 자체도 덮어씌운다.
            entry["Temperature"] = fitted_temps[i]
        else:
            # adjust_range 에 있지만 diff=0이 아닌 구간은 그대로 두거나
            # 필요에 따라 전부 덮어씌울 수도 있음(사용자 요구사항에 맞추어 수정)
            entry["Adjusted Temperature"] = entry["Temperature"]  # 실제론 기존값 그대로
        
    # debug=True라면, 원본 vs 피팅 vs 보정 결과를 그래프에 표시
    if debug:
        plt.figure()
        plt.plot(times, temps, 'ko-', label='Original Temperature')
        plt.plot(times, fitted_temps, 'r--', label='Fitted (Normal Range)')
        
        # 보정 후 데이터
        corrected_temps = [e["Temperature"] for e in series_entries]
        plt.plot(times, corrected_temps, 'bx-', label='Corrected Temperature')
        
        plt.xlabel("Series Elapsed Time")
        plt.ylabel("Temperature")
        plt.title("Temperature Correction Debug View")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 다시 extracted_data에 업데이트
    # series_entries의 수정 내용이 extracted_data 원본 객체들에 반영됨(참조 공유).
    return extracted_data

# ------------------------------------------------------------------------
# 추가 메서드들 (contour_data를 입력으로 작동)
# ------------------------------------------------------------------------

def select_q_range(contour_data, index=None):
    """
    contour_data의 n번째 데이터에서 q 범위를 선택하도록 사용자에게 입력받는다.
    플롯은 x축에 q, y축에 Intensity를 사용한다.
    
    Parameters
    ----------
    contour_data : dict
        extract_contour_data()로 생성된 데이터
    index : int, optional
        선택할 데이터의 인덱스 (기본값 0)
    
    Returns
    -------
    q_range : tuple
        (q_min, q_max)
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
    input_range의 중심을 피크 위치로 이동시킨 output_range를 반환한다.
    
    Parameters
    ----------
    contour_data : dict
        extract_contour_data()로 생성된 데이터
    Index_number : int, optional
        분석할 데이터 인덱스 (기본값 0)
    input_range : tuple
        (q_min, q_max)의 범위 (반드시 제공)
    peak_info : dict, optional
        이전 피크 정보 딕셔너리 (Index가 0보다 큰 경우에 사용)
        예: {"peak_q": <double>, "peak_intensity": <double>, "q_threshold": None, "intensity_threshold": None}
    
    Returns
    -------
    (peak_q, peak_intensity, output_range) 또는 조건에 맞지 않으면 None
    """
    if input_range is None:
        raise ValueError("input_range must be provided.")
    
    try:
        data_entry = contour_data["Data"][Index_number]
    except IndexError:
        raise ValueError(f"Index {Index_number} is out of range for contour_data.")
    
    q_arr = np.array(data_entry["q"])
    intensity_arr = np.array(data_entry["Intensity"])
    
    # input_range 내 데이터 선택
    mask = (q_arr >= input_range[0]) & (q_arr <= input_range[1])
    if not np.any(mask):
        print("No data points found within the specified input_range.")
        return None
    q_subset = q_arr[mask]
    intensity_subset = intensity_arr[mask]
    
    # 초기 추정값: amplitude, peak 위치, sigma, offset
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
    
    # input_range의 중심을 기준으로 이동 (예: 1~5 범위에서 피크가 4이면, 중심 3 -> shift=+1 → 출력 range 2~6)
    center = (input_range[0] + input_range[1]) / 2.0
    shift = peak_q - center
    output_range = (input_range[0] + shift, input_range[1] + shift)
    
    # Index가 0보다 큰 경우 peak_info를 활용하여 임계값 비교
    if Index_number > 0 and peak_info is not None:
        # q_threshold 결정
        if peak_info.get("q_threshold") is None:
            prev_peak_q = peak_info.get("peak_q")
            if prev_peak_q is None:
                print("Previous peak_q is missing in peak_info.")
            else:
                peak_info["q_threshold"] = 1 if prev_peak_q >= 3 else 0.1
        q_threshold = peak_info.get("q_threshold", 0.1)
        
        # intensity_threshold (기본 +-50% -> 0.5)
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
    피크 추적이 failure된 인덱스에서, 지금까지의 peak 데이터와 함께
    칸투어 플롯을 표시하고, 사용자가 수동으로 q 범위를 재설정하도록 한다.
    
    Parameters
    ----------
    contour_data : dict
        extract_contour_data()로 생성된 원본 데이터
    peak_data : dict
        지금까지 추적된 peak 데이터 (부분 결과)
    failure_index : int
        피크 추적 실패가 발생한 인덱스
    
    Returns
    -------
    new_q_range : tuple
        사용자가 선택한 새로운 (q_min, q_max) 범위
    """
    print(f"Peak tracking failure at index {failure_index}.")
    print("Displaying contour plot with tracked peaks so far for reference.")
    plot_contour_with_peaks(contour_data, peak_data)
    print("Please manually adjust the q range for the current index using the plot.")
    new_q_range = select_q_range(contour_data, index=failure_index)
    return new_q_range


def track_peaks(contour_data, input_range, tracking_option=None):
    """
    contour_data의 각 데이터에 대해 연속적으로 피크를 추적하는 함수.
    만약 피크 추적에 실패하면, 수동 보정을 위한 창을 띄워 사용자가
    새로운 q 범위를 선택하게 하고, threshold 비교 없이 재시도한다.
    
    Parameters
    ----------
    contour_data : dict
        extract_contour_data()로 생성된 데이터
    input_range : tuple
        초기 (q_min, q_max) 범위
    tracking_option : dict, optional
        추가 옵션 (예: q_threshold, intensity_threshold). 현재 예제에서는 사용하지 않음.
    
    Returns
    -------
    dict
        {
            "Series": <series_name>,
            "Time-temp": (times, temperatures),
            "Data": [
                {"Time": t1, "peak_q": <value>, "peak_Intensity": <value>},
                {"Time": t2, "peak_q": <value>, "peak_Intensity": <value>},
                ...
            ]
        }
    
    Raises
    ------
    RuntimeError
        - 첫 번째 인덱스(0)에서 피크 찾기 실패 시
        - 수동 보정 후에도 피크 찾기가 실패 시
    """
    times_list = []
    peak_q_list = []
    peak_intensity_list = []

    prev_peak_info = None
    n = len(contour_data["Data"])
    
    for i in range(n):
        time_val = contour_data["Data"][i]["Time"]
        
        if i == 0:
            # 첫 번째 인덱스 → peak_info 없음
            result = find_peak(contour_data, Index_number=0, input_range=input_range, peak_info=None)
            if result is not None:
                peak_q, peak_intensity, _ = result
                # 첫 피크 정보 저장 (threshold 비교 위한 info)
                prev_peak_info = {"peak_q": peak_q, "peak_intensity": peak_intensity}
            else:
                # 첫 번째 인덱스 자체가 실패면 더 진행 불가
                raise RuntimeError("Peak tracking failed at index 0.")
        else:
            # 두 번째 인덱스부터
            result = find_peak(contour_data, Index_number=i, input_range=input_range, peak_info=prev_peak_info)
            
            if result is None:
                # 지금까지의 결과를 partial_peak_data로 구성 (시각화용)
                partial_peak_data = {
                    "Series": contour_data["Series"],
                    "Time-temp": contour_data["Time-temp"],
                    "Data": [
                        {"Time": contour_data["Data"][j]["Time"], 
                         "peak_q": peak_q_list[j], 
                         "peak_Intensity": peak_intensity_list[j]}
                        for j in range(i)
                    ]
                }
                
                # 실패 시, 수동으로 q 범위 재설정을 요청
                new_input_range = adjust_q_range_on_failure(
                    contour_data=contour_data,
                    peak_data=partial_peak_data,
                    failure_index=i
                )
                
                # "직전 피크와의 threshold 비교"를 건너뛰기 위해 peak_info=None으로 재시도
                result = find_peak(
                    contour_data, 
                    Index_number=i, 
                    input_range=new_input_range, 
                    peak_info=None   # 여기서 threshold 비교 무시
                )
                
                if result is None:
                    raise RuntimeError(f"Peak tracking failed at index {i} even after manual adjustment.")
                else:
                    # 성공하면 새 범위를 이후에도 사용
                    input_range = new_input_range
                    # 이번에 찾은 피크를 새 기준으로 저장
                    peak_q, peak_intensity, _ = result
                    prev_peak_info = {"peak_q": peak_q, "peak_intensity": peak_intensity}
            else:
                # 정상적으로 찾았으면 그대로 업데이트
                peak_q, peak_intensity, _ = result
                # threshold 체크에 성공했으므로, prev_peak_info 갱신
                prev_peak_info = {
                    "peak_q": peak_q, 
                    "peak_intensity": peak_intensity,
                    "q_threshold": prev_peak_info.get("q_threshold"),
                    "intensity_threshold": prev_peak_info.get("intensity_threshold")
                }

        # i번째 결과 저장
        times_list.append(time_val)
        peak_q_list.append(peak_q)
        peak_intensity_list.append(peak_intensity)
    
    # 모든 인덱스에 대해 피크 추적 완료
    new_contour = {
        "Series": contour_data["Series"],
        "Time-temp": contour_data["Time-temp"],
        "Data": [
            {"Time": t, "peak_q": pq, "peak_Intensity": pi}
            for t, pq, pi in zip(times_list, peak_q_list, peak_intensity_list)
        ]
    }
    return new_contour

def plot_contour_with_peaks(contour_data, peak_data, graph_option=None):
    """
    contour_data를 기반으로 칸투어 플롯을 그리고, 그 위에 peak_data에 기록된
    피크 위치들을 빨간 원(marker)으로 오버레이하여 표시한다.
    
    Parameters
    ----------
    contour_data : dict
        extract_contour_data()로 생성된 데이터
    peak_data : dict
        track_peaks()의 결과 데이터 (각 시간별 peak_q와 peak_Intensity 포함)
    graph_option : dict, optional
        사용자 지정 그래프 옵션 (없으면 기본값 사용)
    """
    # 기본 옵션 (plot_contour와 유사한 옵션 사용)
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
    cp = ax.contourf(
        grid_q,
        grid_time,
        grid_intensity,
        levels=final_opt["contour_levels"],
        cmap=final_opt["contour_cmap"],
        vmin=lower_bound,
        vmax=upper_bound
    )
    
    if final_opt["contour_xlim"] is not None:
        ax.set_xlim(final_opt["contour_xlim"])
    if final_opt["global_ylim"] is not None:
        ax.set_ylim(final_opt["global_ylim"])
    
    ax.set_xlabel(final_opt["contour_xlabel_text"], fontsize=14, fontweight='bold')
    ax.set_ylabel(final_opt["contour_ylabel_text"], fontsize=14, fontweight='bold')
    ax.set_title("Contour Plot with Peak Positions", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # peak_data의 각 데이터에 대해, peak_q와 해당 시간을 빨간 원으로 표시
    added_label = False
    for entry in peak_data["Data"]:
        peak_q = entry.get("peak_q")
        time_val = entry.get("Time")
        # 유효한 값인 경우만 표시
        if peak_q is not None and not np.isnan(peak_q):
            # marker 크기를 충분히 크게 설정 (예: s=100)
            if not added_label:
                ax.scatter(peak_q, time_val, color="red", edgecolors="black", s=100, label="Peak Position", zorder=10)
                added_label = True
            else:
                ax.scatter(peak_q, time_val, color="red", edgecolors="black", s=100, zorder=10)
    
    ax.legend()
    plt.show()

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main(dat_dir, log_path, output_dir):
    log_data = parse_log_file(log_path)
    extracted_data = process_dat_files(dat_dir, log_data)
    
    # 시리즈 선택
    selected_series = select_series(extracted_data)
    
    # --- (Option) 온도 보정 구간 선택 단계 ---
    # 사용자에게 normal_range, adjust_range를 선택하게 함.
    normal_range, adjust_range = select_temperature_ranges(extracted_data, selected_series, debug=True)
    
    # 실제로 온도 보정 수행
    extracted_data = temp_correction(extracted_data, selected_series, normal_range, adjust_range, debug=True)
    
    # Check for missing data before contour plotting
    for entry in extracted_data.values():
        if isinstance(entry, dict) and "q" in entry and "Intensity" in entry:
            if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
                print(f"Warning: Missing data detected in {entry.get('File Path', 'Unknown File')}")
    
    # 실제 contour_data 생성
    contour_data = extract_contour_data(selected_series, extracted_data)
    
    for entry in contour_data["Data"]:
        if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
            print(f"Warning: Missing data detected in time frame {entry['Time']}")
    
    # 최종 플롯 (온도 subplot 포함)
    plot_contour(contour_data, temp=True, legend=True)
    
    # ===== 추가 메서드 호출 예시 =====
    # 1. contour_data의 첫번째 데이터에서 q 범위 선택
    q_range = select_q_range(contour_data)  # index 기본값 0
    print("Selected q range:", q_range)
    
    # 2. 첫번째 데이터에서 피크 찾기 (Index_number=0일 땐 peak_info 사용하지 않음)
    peak_result = find_peak(contour_data, Index_number=0, input_range=q_range)
    if peak_result is not None:
        peak_q, peak_intensity, output_range = peak_result
        print(f"First dataset peak: q = {peak_q}, intensity = {peak_intensity}, adjusted range = {output_range}")
    else:
        print("Peak not found in first dataset.")
    
    # 3. 모든 데이터에 대해 피크 추적 (연속 추적)
    tracked_peaks = track_peaks(contour_data, input_range=q_range)
    #print("Tracked peaks:", tracked_peaks)
    
    # 4. contour 플롯에 피크 위치 오버레이하여 확인 (빨간 원 표시)
    plot_contour_with_peaks(contour_data, tracked_peaks)

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
    args = parser.parse_args()
    
    main(args.input, args.log, args.output)