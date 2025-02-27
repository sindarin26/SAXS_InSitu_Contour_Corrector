import numpy as np
from scipy.optimize import curve_fit
from asset.contour_storage import FITTING_THRESHOLD
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from asset.contour_util import interpolate_contour_data
    

def gaussian(x, a, mu, sigma, offset):
    """
    Gaussian function: a * exp(-((x-mu)^2)/(2*sigma^2)) + offset
    
    Parameters
    ----------
    x : array-like
        Independent variable
    a : float
        Amplitude
    mu : float
        Center
    sigma : float
        Standard deviation
    offset : float
        Vertical offset
    """
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

def lorentzian(x, a, x0, gamma, offset):
    """
    Lorentzian function: a * gamma^2 / ((x-x0)^2 + gamma^2) + offset
    
    Parameters
    ----------
    x : array-like
        Independent variable
    a : float
        Amplitude
    x0 : float
        Center
    gamma : float
        Half-width at half-maximum (HWHM)
    offset : float
        Vertical offset
    """
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + offset

def voigt(x, a, mu, sigma, gamma, offset):
    """
    Voigt function: Convolution of Gaussian and Lorentzian
    Approximated using the Faddeeva function (real part)
    
    Parameters
    ----------
    x : array-like
        Independent variable
    a : float
        Amplitude
    mu : float
        Center
    sigma : float
        Gaussian sigma
    gamma : float
        Lorentzian gamma (HWHM)
    offset : float
        Vertical offset
    """
    from scipy.special import voigt_profile
    z = (x - mu + 1j*gamma) / (sigma * np.sqrt(2))
    return a * voigt_profile(x - mu, sigma, gamma) + offset

def find_peak(contour_data, Index_number=0, input_range=None, peak_info=None, 
             fitting_function="gaussian", threshold_config=None):
    """
    Find peak in the given data range using specified fitting function and thresholds.
    
    Returns
    -------
    tuple or str
        If successful: (peak_q, peak_intensity, output_range, fwhm)
        If failed: Error message string explaining the failure reason
    """
    if threshold_config is None:
        threshold_config = FITTING_THRESHOLD

    if input_range is None:
        return "No input range provided"
        
    try:
        data_entry = contour_data["Data"][Index_number]
    except IndexError:
        return f"Index {Index_number} is out of range for contour_data"
    
    q_arr = np.array(data_entry["q"])
    intensity_arr = np.array(data_entry["Intensity"])
    mask = (q_arr >= input_range[0]) & (q_arr <= input_range[1])
    
    if not np.any(mask):
        return "No data points found within the specified input range"
        
    q_subset = q_arr[mask]
    intensity_subset = intensity_arr[mask]
    
    # Select fitting function and set initial parameters
    if fitting_function.lower() == "lorentzian":
        fit_func = lorentzian
        # Initial guesses for Lorentzian
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        x0_guess = q_subset[np.argmax(intensity_subset)]
        gamma_guess = (input_range[1] - input_range[0]) / 4.0
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, x0_guess, gamma_guess, offset_guess]
        
        # 감마에 대한 하한(lower bound)를 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, -np.inf]  # a, x0, gamma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)
        
    elif fitting_function.lower() == "voigt":
        fit_func = voigt
        # Initial guesses for Voigt
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 8.0
        gamma_guess = sigma_guess  # Start with equal Gaussian and Lorentzian contributions
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, gamma_guess, offset_guess]
        
        # 시그마와 감마에 대한 하한을 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, 0.0, -np.inf]  # a, mu, sigma, gamma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)
        
    else:  # default to Gaussian
        fit_func = gaussian
        # Initial guesses for Gaussian
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 4.0
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, offset_guess]
        
        # 시그마에 대한 하한을 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, -np.inf]  # a, mu, sigma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)
    
    try:
        # bounds 파라미터 추가
        popt, _ = curve_fit(fit_func, q_subset, intensity_subset, p0=p0, bounds=bounds)
    except Exception as e:
        return f"Fitting failed with {fitting_function}: {str(e)}"
    
    # Get peak position and intensity
    if fitting_function.lower() == "lorentzian":
        peak_q = popt[1]  # x0 parameter
    elif fitting_function.lower() == "voigt":
        peak_q = popt[1]  # mu parameter
    else:  # Gaussian
        peak_q = popt[1]  # mu parameter
    
    peak_intensity = fit_func(peak_q, *popt)
    
    # Calculate FWHM
    fwhm = calculate_fwhm(q_subset, intensity_subset, peak_q, fitting_function, popt)
    
    # Threshold checks
    if Index_number > 0 and peak_info is not None:
        error_msg = check_peak_thresholds(
            peak_q, peak_intensity, fwhm,
            {'peak_q': peak_info.get('peak_q'),
             'peak_intensity': peak_info.get('peak_intensity'),
             'fwhm': peak_info.get('fwhm')},
            threshold_config
        )
        if error_msg:
            return error_msg
    
    # Calculate output range
    center = (input_range[0] + input_range[1]) / 2.0
    shift = peak_q - center
    output_range = (input_range[0] + shift, input_range[1] + shift)
    
    return peak_q, peak_intensity, output_range, fwhm


def calculate_fwhm(x, y, peak_pos, fitting_function, popt):
    """Calculate FWHM based on fitting function and parameters"""
    if fitting_function == "gaussian":
        a, mu, sigma, offset = popt
        fwhm = 2.355 * sigma  # 2.355 = 2*sqrt(2*ln(2))
    elif fitting_function == "lorentzian":
        a, x0, gamma, offset = popt
        fwhm = 2 * gamma  # FWHM = 2γ for Lorentzian
    elif fitting_function == "voigt":
        a, mu, sigma, gamma, offset = popt
        # Approximation for Voigt FWHM
        # From: https://en.wikipedia.org/wiki/Voigt_profile#The_width_of_the_Voigt_profile
        fG = 2.355 * sigma
        fL = 2 * gamma
        fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
    else:
        raise ValueError(f"Unknown fitting function: {fitting_function}")
    
    return fwhm

def check_peak_thresholds(peak_q, peak_intensity, fwhm, prev_info, threshold_config):
    """
    Check all thresholds and return None if passed, or error message if failed
    
    Parameters
    ----------
    peak_q : float
        Current peak q position
    peak_intensity : float
        Current peak intensity
    fwhm : float
        Current peak FWHM
    prev_info : dict
        Previous peak information including q, intensity, and FWHM
    threshold_config : dict
        Threshold configuration dictionary
        
    Returns
    -------
    str or None
        Error message if any threshold is exceeded, None if all checks pass
    """
    if not prev_info:
        return None
        
    prev_q = prev_info.get('peak_q')
    prev_intensity = prev_info.get('peak_intensity')
    prev_fwhm = prev_info.get('fwhm')
    
    # Basic q threshold check
    if threshold_config['use_basic_q_threshold'] and prev_q is not None:
        if abs(peak_q - prev_q) > threshold_config['q_threshold']:
            return f"Peak position changed by {abs(peak_q - prev_q):.3f}, exceeding threshold {threshold_config['q_threshold']:.3f}"
    
    # Intensity threshold check
    if threshold_config['use_intensity_threshold'] and prev_intensity is not None:
        intensity_change = abs(peak_intensity - prev_intensity) / prev_intensity
        if intensity_change > threshold_config['intensity_threshold']:
            return f"Peak intensity changed by {intensity_change*100:.1f}%, exceeding threshold {threshold_config['intensity_threshold']*100:.1f}%"
    
    # FWHM-based q threshold check
    if threshold_config['use_fwhm_q_threshold'] and prev_fwhm is not None:
        fwhm_q_threshold = threshold_config['fwhm_q_factor'] * prev_fwhm
        if abs(peak_q - prev_q) > fwhm_q_threshold:
            return f"Peak position changed by {abs(peak_q - prev_q):.3f}, exceeding FWHM-based threshold {fwhm_q_threshold:.3f}"
    
    # FWHM comparison check
    if threshold_config['use_fwhm_comparison'] and prev_fwhm is not None:
        fwhm_change = abs(fwhm - prev_fwhm) / prev_fwhm
        if fwhm_change > threshold_config['fwhm_change_threshold']:
            return f"Peak FWHM changed by {fwhm_change*100:.1f}%, exceeding threshold {threshold_config['fwhm_change_threshold']*100:.1f}%"
    
    return None

def find_peak_extraction(
    contour_data,
    Index_number=0,
    input_range=None,
    fitting_function="gaussian",
    threshold_config=None,
    flag_auto_tracking=False,
    flag_manual_adjust=False,
    flag_start=False,
    start_index=None,
    current_peak_name=None,
    peak_info=None
):
    """
    플래그를 고려하여 특정 프레임(Index_number)에서 피크를 찾는 함수.

    Parameters
    ----------
    contour_data : dict
        원본 컨투어 데이터 (Time, Temperature, q, Intensity 등).
        예: {"Data": [...], "Series": "...", "Time-temp": (...)}
        
    Index_number : int
        현재 처리 중인 프레임 인덱스

    input_range : tuple(float, float) or None
        사용자가 지정한 q 범위 (예: (5.0, 6.0))
        None이면 에러 메시지 반환

    fitting_function : str
        "gaussian", "lorentzian", "voigt" 중 하나

    threshold_config : dict or None
        임계값(threshold) 설정 딕셔너리.
        None이면 default(FITTING_THRESHOLD) 사용.

    flag_auto_tracking : bool
        True → 자동 추적 중 (threshold를 적용할지 결정하는데 사용)
    
    flag_manual_adjust : bool
        True → 수동 조정 중 (threshold 적용 X)
    
    flag_start : bool
        True → 새로운 피크 시작 모드.
               current_peak_name가 None이면 새 이름 생성.
        False → 기존 피크 연속 추적(또는 수정 모드).
    
    start_index : int or None
        새로운 피크를 시작할 때, 피크 이름에 사용될 "시작 프레임 인덱스".
        예: 0 (첫 프레임). flag_start=True일 때 사용.

    current_peak_name : str or None
        현재 추적 중인 피크 이름 (예: "peak_0_5.6789")
        - flag_start=True이고 None이면 새 이름 생성
        - 그렇지 않으면 그대로 사용

    peak_info : dict or None
        이전 프레임에서 찾은 피크 정보 (peak_q, peak_intensity, fwhm, peak_name)
        threshold 체크에 사용됨.

    Returns
    -------
    (peak_q, peak_intensity, output_range, fwhm, peak_name) or str
        성공 시: 튜플
            - peak_q : float
            - peak_intensity : float
            - output_range : (float, float)
            - fwhm : float
            - peak_name : str (피크 그룹 이름)
        실패 시: 문자열(에러 메시지)
    """

    # 1) Threshold 기본값 확인
    if threshold_config is None:
        from asset.contour_storage import FITTING_THRESHOLD
        threshold_config = FITTING_THRESHOLD

    # 디버깅 출력 추가
    print(f"\nDEBUG: find_peak_extraction called for index {Index_number}")
    print(f"DEBUG: flag_auto_tracking={flag_auto_tracking}, flag_manual_adjust={flag_manual_adjust}")
    print(f"DEBUG: threshold_config: {threshold_config}")
    print(f"DEBUG: Fitting function: {fitting_function}")


    # 2) input_range 검증
    if input_range is None:
        return "No input range provided"

    # 3) contour_data 범위 체크
    try:
        data_entry = contour_data["Data"][Index_number]
    except (KeyError, IndexError):
        return f"Index {Index_number} is out of range for contour_data"

    q_arr = np.array(data_entry["q"])
    intensity_arr = np.array(data_entry["Intensity"])

    # 4) 사용자가 지정한 q 범위로 마스킹
    q_min, q_max = input_range
    mask = (q_arr >= q_min) & (q_arr <= q_max)
    if not np.any(mask):
        return "No data points found within the specified input range"

    q_subset = q_arr[mask]
    intensity_subset = intensity_arr[mask]

    # 5) 피팅 함수 선택 및 초기 추정값 설정
    if fitting_function.lower() == "lorentzian":
        fit_func = lorentzian
        a_guess = float(np.max(intensity_subset) - np.min(intensity_subset))
        x0_guess = float(q_subset[np.argmax(intensity_subset)])
        gamma_guess = (q_max - q_min) / 4.0
        offset_guess = float(np.min(intensity_subset))
        p0 = [a_guess, x0_guess, gamma_guess, offset_guess]
        
        # 감마에 대한 하한(lower bound)를 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, -np.inf]  # a, x0, gamma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)

    elif fitting_function.lower() == "voigt":
        fit_func = voigt
        a_guess = float(np.max(intensity_subset) - np.min(intensity_subset))
        mu_guess = float(q_subset[np.argmax(intensity_subset)])
        sigma_guess = (q_max - q_min) / 8.0
        gamma_guess = sigma_guess
        offset_guess = float(np.min(intensity_subset))
        p0 = [a_guess, mu_guess, sigma_guess, gamma_guess, offset_guess]
        
        # 시그마와 감마에 대한 하한을 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, 0.0, -np.inf]  # a, mu, sigma, gamma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)

    else:  # default: Gaussian
        from asset.fitting_util import gaussian
        fit_func = gaussian
        a_guess = float(np.max(intensity_subset) - np.min(intensity_subset))
        mu_guess = float(q_subset[np.argmax(intensity_subset)])
        sigma_guess = (q_max - q_min) / 4.0
        offset_guess = float(np.min(intensity_subset))
        p0 = [a_guess, mu_guess, sigma_guess, offset_guess]
        
        # 시그마에 대한 하한을 0으로 설정
        lower_bounds = [0.0, -np.inf, 0.0, -np.inf]  # a, mu, sigma, offset
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        bounds = (lower_bounds, upper_bounds)


    # 6) 피팅 시도
    try:
        # bounds 파라미터 추가
        popt, _ = curve_fit(fit_func, q_subset, intensity_subset, p0=p0, bounds=bounds)
    except Exception as e:
        return f"Fitting failed with {fitting_function}: {str(e)}"


    # 7) 피크 위치 및 강도 계산
    if fitting_function.lower() == "lorentzian":
        peak_q = popt[1]  # x0
        fitting_params = {
                    "a": popt[0],
                    "x0": popt[1],
                    "gamma": popt[2],
                    "offset": popt[3]
                }
    elif fitting_function.lower() == "voigt":
        peak_q = popt[1]  # mu
        fitting_params = {
            "a": popt[0],
            "mu": popt[1],
            "sigma": popt[2],
            "gamma": popt[3],
            "offset": popt[4]
        }

    else:  # gaussian
        peak_q = popt[1]
        fitting_params = {
            "a": popt[0],
            "mu": popt[1],
            "sigma": popt[2],
            "offset": popt[3]
        }


    peak_intensity = fit_func(peak_q, *popt)

    # 8) FWHM 계산
    from asset.fitting_util import calculate_fwhm
    fwhm = calculate_fwhm(q_subset, intensity_subset, peak_q, fitting_function.lower(), popt)

    # 9) 피크 이름 결정
    if flag_start:
        # "새 피크 시작" 모드
        if current_peak_name is None:
            # start_index가 None이면 Index_number로 대체
            use_start_index = start_index if (start_index is not None) else Index_number
            peak_name = f"peak_{use_start_index}_{peak_q:.4f}"
        else:
            peak_name = current_peak_name
    else:
        # 기존 피크 추적 / 수정 모드
        if current_peak_name:
            peak_name = current_peak_name
        elif peak_info and "peak_name" in peak_info:
            peak_name = peak_info["peak_name"]
        else:
            # 아무 정보도 없다면 임시 이름
            peak_name = f"peak_{Index_number}_{peak_q:.4f}"

    # 10) Threshold 체크
    #     (자동 추적이면서 수동 조정이 아니고, 이전 peak_info가 있으면)
    if flag_auto_tracking and (not flag_manual_adjust) and peak_info is not None:
        from asset.fitting_util import check_peak_thresholds
        prev_data = {
            "peak_q": peak_info.get("peak_q"),
            "peak_intensity": peak_info.get("peak_intensity"),
            "fwhm": peak_info.get("fwhm")
        }
        error_msg = check_peak_thresholds(peak_q, peak_intensity, fwhm, prev_data, threshold_config)
        if error_msg:
            return error_msg

    # 11) output_range 계산 - 여기가 핵심!
    if flag_auto_tracking:
        # 자동 추적 모드: 피크 위치 기준으로 범위 재조정 (기존 방식)
        center = 0.5 * (q_min + q_max)
        shift = peak_q - center
        output_range = (q_min + shift, q_max + shift)
        print(f"DEBUG: Auto-tracking mode - shifting range to center on peak")
        print(f"DEBUG: Original range: {input_range}, shifted range: {output_range}")
    else:
        # 수동 모드: 입력 범위를 그대로 사용 (사용자 선택 범위 유지)
        output_range = input_range
        print(f"DEBUG: Manual mode - using input range as-is: {output_range}")


    return peak_q, peak_intensity, output_range, fwhm, peak_name, fitting_function, fitting_params


def run_automatic_tracking(
    contour_data: dict,
    tracked_peaks: dict,
    current_peak_info: dict,       # {"peak_q": ..., "peak_intensity": ..., "peak_name": ... } 등
    current_index: int,
    max_index: int,
    get_q_range_func,              # q_range를 얻는 함수 (예: self.q_correction_helper.get_q_range)
    fitting_model="gaussian",
    threshold_config=None,
    flag_start=True,
    start_index=None,
    flag_auto_tracking=True,
    flag_manual_adjust=False,
    on_error_callback=None,
    on_update_callback=None,
    on_finish_callback=None
):
    """
    자동 추적 로직.
    
    첫번째 호출 시 flag_start가 True이면 새로운 피크로 처리한 후,
    이후 호출부터는 자동으로 flag_start를 False로 처리하여 연속 추적하게 함.
    
    Returns:
        (bool, int, dict): (성공 여부, 마지막 처리한 current_index, 마지막 peak_info)
    """
    # "첫 번째 프레임은 이미 처리됨"이라 가정하고, 현재 인덱스를 증가
    current_index += 1

    success = True
    last_peak_info = current_peak_info

    while current_index <= max_index:
        # q_range 얻기
        q_range = get_q_range_func()
        if not q_range or len(q_range) != 2:
            msg = f"No valid q_range at frame {current_index}"
            success = False
            if on_error_callback:
                on_error_callback(msg)
            break

        # 첫 호출이면 flag_start 그대로 사용, 이후는 False
        current_call_flag_start = flag_start if (last_peak_info is None) else False

        result = find_peak_extraction(
            contour_data=contour_data,
            Index_number=current_index,
            input_range=q_range,
            fitting_function=fitting_model,
            threshold_config=threshold_config,
            flag_auto_tracking=flag_auto_tracking,
            flag_manual_adjust=flag_manual_adjust,
            flag_start=current_call_flag_start,
            start_index=start_index,
            current_peak_name=last_peak_info.get("peak_name") if last_peak_info else None,
            peak_info=last_peak_info
        )

        if isinstance(result, str):
            success = False
            if on_error_callback:
                on_error_callback(f"Peak not found at frame {current_index}: {result}")
            break

        peak_q, peak_intensity, output_range, fwhm, peak_name, fitting_function, fitting_params = result

        current_entry = contour_data["Data"][current_index]

        current_time = tracked_peaks['Time-temp'][0][current_index]
        current_temp = tracked_peaks['Time-temp'][1][current_index]

        print(f"DEBUG: Current time and temp = {current_time}, {current_temp}")


        new_result = {
            "frame_index": current_index,
            "Time": current_time,
            "Temperature": current_temp,
            "peak_q": peak_q,
            "peak_Intensity": peak_intensity,
            "fwhm": fwhm,
            "peak_name": peak_name,
            "output_range": output_range,
            "fitting_function": fitting_function,
            "fitting_params": fitting_params
        }
        tracked_peaks["Data"].append(new_result)

        last_peak_info = {
            "peak_q": peak_q,
            "peak_intensity": peak_intensity,
            "fwhm": fwhm,
            "peak_name": peak_name,
            "output_range": output_range,
            "fitting_function": fitting_function,
            "fitting_params": fitting_params
        }

        if on_update_callback:
            on_update_callback(
                frame_index=current_index,
                peak_q=peak_q,
                peak_intensity=peak_intensity,
                fwhm=fwhm,
                peak_name=peak_name
            )

        # 첫 호출 이후, ensure flag_start is False
        flag_start = False

        current_index += 1

    if success and on_finish_callback:
        on_finish_callback(
            message=f"Auto tracking completed up to frame {current_index-1}.",
            last_peak_info=last_peak_info
        )

    return success, current_index, last_peak_info

def plot_contour_extraction(
    contour_data,
    tracked_peaks,
    found_peak_list,
    flag_adjust_mode=False,
    graph_option=None,
    on_peak_selected_callback=None
):
    """
    컨투어 플롯을 그리고, found_peak_list에 포함된 peak_name별로
    다른 색상으로 피크 궤적(peak_q vs. Time)을 표시한다.
    
    캐싱된 보간 데이터를 사용하여 성능 개선.
    """
    # 1) 그래프 옵션 기본값
    default_graph_option = {
        "figure_size": (12, 8),
        "figure_dpi": 150,
        "contour_levels": 100,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "contour_xlim": None,
        "global_ylim": None,
    }
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}

    # 2) 캐싱된 보간 데이터 사용
    interpolated_data = interpolate_contour_data(contour_data)
    grid_q = interpolated_data["grid_q"]
    grid_time = interpolated_data["grid_time"]
    grid_intensity = interpolated_data["grid_intensity"]
    
    # 사용자 지정 percentile 또는 캐싱된 값 사용
    lower_percentile = final_opt.get("contour_lower_percentile", 0.1)
    upper_percentile = final_opt.get("contour_upper_percentile", 98)
    
    if lower_percentile == 0.1 and "lower_bound" in interpolated_data:
        lower_bound = interpolated_data["lower_bound"]
    else:
        lower_bound = np.nanpercentile(grid_intensity, lower_percentile)
        
    if upper_percentile == 98 and "upper_bound" in interpolated_data:
        upper_bound = interpolated_data["upper_bound"]
    else:
        upper_bound = np.nanpercentile(grid_intensity, upper_percentile)
    
    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN.")
        # 빈 플롯 반환
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        return canvas

    # 3) Matplotlib Figure, Axes 생성
    fig, ax = plt.subplots(
        figsize=final_opt["figure_size"],
        dpi=final_opt["figure_dpi"]
    )

    # 4) 등고선 플롯
    cp = ax.contourf(
        grid_q, grid_time, grid_intensity,
        levels=final_opt["contour_levels"],
        cmap=final_opt["contour_cmap"],
        vmin=lower_bound,
        vmax=upper_bound
    )

    # 5) 축 범위 설정
    if final_opt["contour_xlim"] is not None:
        ax.set_xlim(final_opt["contour_xlim"])
    if final_opt["global_ylim"] is not None:
        ax.set_ylim(final_opt["global_ylim"])

    # 6) found_peak_list에 포함된 peak_name별로 산점도+라인 그리기
    data_by_peak = {}
    for entry in tracked_peaks.get("Data", []):
        pname = entry.get("peak_name")
        if pname in found_peak_list:
            if pname not in data_by_peak:
                data_by_peak[pname] = {"q": [], "time": [], "indices": []}
            data_by_peak[pname]["q"].append(entry["peak_q"])
            data_by_peak[pname]["time"].append(entry["Time"])
            data_by_peak[pname]["indices"].append(entry["frame_index"])

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    scatter_dict = {}
    selected_peak_name = [None]
    selected_frame_index = [None]  # 선택된 프레임 인덱스 저장 변수 추가
    highlight_lines = [None, None]

    for i, (pname, data_dict) in enumerate(data_by_peak.items()):
        color = color_cycle[i % len(color_cycle)]
        q_vals = np.array(data_dict["q"])
        t_vals = np.array(data_dict["time"])
        
        sc = ax.scatter(
            q_vals, t_vals,
            facecolor='none',  # 내부 채움 투명
            edgecolor='none',  # 외곽선 투명
            s=40,
            picker=5 if flag_adjust_mode else False
        )
        ax.plot(q_vals, t_vals, color=color, alpha=0.7, linewidth=1)
        scatter_dict[pname] = sc  # 콜백에서 식별하기 위해 저장

    # 7) 클릭 이벤트 (flag_adjust_mode=True)
    def on_pick(event):
        # 기존 하이라이트 제거
        for hl in highlight_lines:
            if hl: hl.remove()
        highlight_lines[0], highlight_lines[1] = None, None

        artist = event.artist
        for pname, sc_ in scatter_dict.items():
            if artist == sc_:
                idx = event.ind[0]  # 선택된 데이터 포인트 인덱스
                frame_index = data_by_peak[pname]["indices"][idx]  # 해당 데이터 포인트의 프레임 인덱스
                q_val = data_by_peak[pname]["q"][idx]
                t_val = data_by_peak[pname]["time"][idx]
                
                highlight_lines[0] = ax.axvline(q_val, color='yellow', linestyle='--', linewidth=2)
                highlight_lines[1] = ax.axhline(t_val, color='yellow', linestyle='--', linewidth=2)
                
                # 선택된 피크 이름과 프레임 인덱스 저장
                selected_peak_name[0] = pname
                selected_frame_index[0] = frame_index
                
                # 디버그 출력
                print(f"Selected peak: {pname}, frame index: {frame_index}")
                
                # 콜백 함수 호출 (존재하는 경우)
                if on_peak_selected_callback:
                    on_peak_selected_callback(pname, frame_index)
                
                fig.canvas.draw_idle()
                break

    if flag_adjust_mode:
        fig.canvas.mpl_connect('pick_event', on_pick)

    # 8) FigureCanvas 생성 & 반환
    canvas = FigureCanvas(fig)

    # 선택된 피크 이름을 반환하는 메서드 추가
    def get_selected_peak_name():
        return selected_peak_name[0]
    canvas.get_selected_peak_name = get_selected_peak_name
    
    # 선택된 프레임 인덱스를 반환하는 메서드 추가
    def get_selected_frame_index():
        return selected_frame_index[0]
    canvas.get_selected_frame_index = get_selected_frame_index

    canvas.draw()
    return canvas