import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files

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
    # 데이터에 따라 조정하세요.
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
# MAIN
# ------------------------------------------------------------------------

def main(dat_dir, log_path, output_dir):
    log_data = parse_log_file(log_path)
    extracted_data, output_file = process_dat_files(dat_dir, log_data, output_dir)
    
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
                print(f"Warning: Missing data detected in {entry['File Path']}")
    
    # 실제 contour_data 생성
    contour_data = extract_contour_data(selected_series, extracted_data)
    
    for entry in contour_data["Data"]:
        if len(entry["q"]) == 0 or len(entry["Intensity"]) == 0:
            print(f"Warning: Missing data detected in time frame {entry['Time']}")
    
    # 최종 플롯
    plot_contour(contour_data, temp=True, legend=True)

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