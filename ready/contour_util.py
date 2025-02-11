import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import griddata
from spec_log_extractor import parse_log_file
from dat_extractor import process_dat_files
from collections import OrderedDict

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

def plot_contour(contour_data, temp=False, legend=True):
    """
    Generate contour plot with interpolated data.
    
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
        False -> 기존처럼 칸투어만 그림
        True -> 오른쪽에 Temperature vs Time 플롯을 추가
    legend : bool
        True -> 컬러바(범례) 표시, False -> 컬러바 표시 안 함
    """
    times, temperatures = contour_data["Time-temp"]

    # ---------- 1) 칸투어 그리기 위한 데이터 전처리 ----------
    # 모든 q, intensity, time 데이터 펼치기
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])

    # q축의 공통 그리드 & time축의 공통 그리드
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)

    # 보간
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata(
        (q_all, time_all), 
        intensity_all, 
        (grid_q, grid_time), 
        method='nearest'
    )

    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return
    
    # 퍼센타일 기반 명암 대비 조정
    lower_bound = np.nanpercentile(grid_intensity, 0.1)
    upper_bound = np.nanpercentile(grid_intensity, 98)

    # ---------- 2) 서브플롯 레이아웃 결정 ----------
    if temp:
        # temp=True 이면, 좌측(칸투어), 우측(온도-시간) 2개 서브플롯
        fig, (ax_contour, ax_temp) = plt.subplots(
            ncols=2, 
            sharey=True,            # y축(시간) 동일
            figsize=(12, 8), 
            dpi=300, 
            gridspec_kw={"width_ratios": [7, 2]}  # 왼쪽은 넓게, 오른쪽은 좁게
        )
        plt.subplots_adjust(wspace=0.00)  # 두 플롯 사이 간격 최소화

        # ---- 2.1) 왼쪽(칸투어) ----
        cp = ax_contour.contourf(
            grid_q, 
            grid_time, 
            grid_intensity, 
            levels=100, 
            cmap='inferno', 
            vmin=lower_bound, 
            vmax=upper_bound
        )
        ax_contour.set_xlabel("2theta (Cu K-alpha)", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax_contour.set_ylabel("Elapsed Time", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax_contour.set_title(f"Contour Plot for {contour_data['Series']}",
                             fontsize=16, fontweight='bold', fontname='Times New Roman')
        ax_contour.tick_params(axis='x', labelsize=12)
        ax_contour.tick_params(axis='y', labelsize=12)

        # 컬러바(범례) 왼쪽 배치
        if legend:
            # Matplotlib >=3.3에서는 location='left' 사용 가능
            c = fig.colorbar(cp, ax=ax_contour, orientation='vertical', location='left', pad=0.15)
            c.set_label("log10(Intensity)", fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        # ---- 2.2) 오른쪽(Temperature vs Time) ----
        # time이 y축, temp가 x축
        ax_temp.plot(temperatures, times, 'r-')  # 온도를 x, 시간을 y
        # x축 범위: 왼쪽(높은온도) -> 오른쪽(낮은온도)이 되도록 반전
        ax_temp.invert_xaxis()

        # y축 라벨을 오른쪽으로
        ax_temp.yaxis.set_label_position("right")
        ax_temp.yaxis.tick_right()
        ax_temp.set_ylabel("Elapsed Time", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax_temp.set_xlabel("Temperature", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax_temp.tick_params(axis='x', labelsize=12)
        ax_temp.tick_params(axis='y', labelsize=12)

    else:
        # temp=False 이면, 기존처럼 1개 플롯(칸투어)만
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        cp = ax.contourf(
            grid_q, 
            grid_time, 
            grid_intensity, 
            levels=100, 
            cmap='inferno', 
            vmin=lower_bound, 
            vmax=upper_bound
        )
        
        if legend:
            c = fig.colorbar(cp, ax=ax, orientation='vertical', location='left', pad=0.15)
            c.set_label("log10(Intensity)", fontsize=14, fontweight='bold', fontname='Times New Roman')
        
        ax.set_xlabel("2theta (Cu K-alpha)", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_ylabel("Elapsed Time", fontsize=14, fontweight='bold', fontname='Times New Roman')
        ax.set_title(f"Contour Plot for {contour_data['Series']}",
                     fontsize=16, fontweight='bold', fontname='Times New Roman')
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    plt.show()

# ------------------------------------------------------------------------
# NEW FUNCTIONS FOR TEMPERATURE CORRECTION
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
    #    예) i, i-1 의 temp 차이가 0 => 연속적으로 0이 2번 이상인 구간
    #    실은 "변화량이 0인 지점"이 여러개 있을 수 있으므로,
    #    간단히는 adjust_range 내부 전체를 "피팅값"으로 강제 덮어씌우는 방법도 가능.
    #
    #    문제에서 '승온중에 "10 11 11 13" 이렇게 중간이 0증가 구간' => "11->11" 이 문제
    #    여기서는 조금 단순화하여, "변화량 = 0인 지점"을 모두 피팅값으로 교정한다고 가정.
    
    # (여기서는 예시로, adjust_range 범위 전체에 대해서
    #  diff(temp) = 0 인 부분만 찾아서 교정한다고 예시 코드를 작성합니다.)
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
    # 아래 코드는 예시이며, 실제로는 --debug 플래그나 특정 사용자 선택 시에만 수행하도록 구성할 수도 있음.
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