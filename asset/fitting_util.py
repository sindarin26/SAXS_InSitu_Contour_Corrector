import numpy as np
from scipy.optimize import curve_fit
from asset.contour_storage import FITTING_THRESHOLD
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


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
    elif fitting_function.lower() == "voigt":
        fit_func = voigt
        # Initial guesses for Voigt
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 8.0
        gamma_guess = sigma_guess  # Start with equal Gaussian and Lorentzian contributions
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, gamma_guess, offset_guess]
    else:  # default to Gaussian
        fit_func = gaussian
        # Initial guesses for Gaussian
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 4.0
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, offset_guess]
    
    try:
        popt, _ = curve_fit(fit_func, q_subset, intensity_subset, p0=p0)
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

def find_peak_extraction(contour_data, Index_number=0, input_range=None, 
                      peak_info=None, fitting_function="gaussian", 
                      threshold_config=None, start_flag=None):
    """
    특정 피크를 찾는 함수. 기존 find_peak의 확장 버전.
    
    새로운 기능:
    - start_flag가 정수면 해당 인덱스에서 시작하는 새 피크 검출
    - start_flag가 None이면 기존 피크 수정 모드
    - 검출된 피크는 peak_{start_flag}_{peak_q} 형태로 저장됨
    
    Parameters
    ----------
    start_flag : int or None
        피크 검출 시작 인덱스. None이면 수정 모드.
        
    Returns
    -------
    tuple or str
        성공 시: (peak_q, peak_intensity, output_range, fwhm, peak_name)
        실패 시: 에러 메시지 문자열
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
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        x0_guess = q_subset[np.argmax(intensity_subset)]
        gamma_guess = (input_range[1] - input_range[0]) / 4.0
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, x0_guess, gamma_guess, offset_guess]
    elif fitting_function.lower() == "voigt":
        fit_func = voigt
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 8.0
        gamma_guess = sigma_guess
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, gamma_guess, offset_guess]
    else:  # default to Gaussian
        fit_func = gaussian
        a_guess = np.max(intensity_subset) - np.min(intensity_subset)
        mu_guess = q_subset[np.argmax(intensity_subset)]
        sigma_guess = (input_range[1] - input_range[0]) / 4.0
        offset_guess = np.min(intensity_subset)
        p0 = [a_guess, mu_guess, sigma_guess, offset_guess]
    
    try:
        popt, _ = curve_fit(fit_func, q_subset, intensity_subset, p0=p0)
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
    
    # 피크 이름 생성
    if start_flag is not None:
        # 새 피크 시작 모드
        peak_name = f"peak_{start_flag}_{peak_q:.4f}"
    else:
        # 수정 모드: 기존 이름 유지 또는 새 이름 생성
        peak_name = peak_info.get("peak_name") if peak_info else f"peak_modified_{peak_q:.4f}"
    
    # Threshold checks (start_flag가 None일 때만)
    if start_flag is None and Index_number > 0 and peak_info is not None:
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
    
    return peak_q, peak_intensity, output_range, fwhm, peak_name

def plot_contour_extraction(contour_data, current_peaks=None, graph_option=None, GUI=False):
    """
    컨투어 플롯과 여러 피크를 표시하는 함수.
    
    Parameters
    ----------
    contour_data : dict
        컨투어 데이터
    current_peaks : dict, optional
        현재까지 찾은 피크들의 정보를 담은 딕셔너리
    graph_option : dict, optional
        그래프 옵션
    GUI : bool, optional
        GUI 모드 여부
        
    Returns
    -------
    matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg or None
        GUI 모드일 때는 캔버스 반환, 아니면 None
    """
    # 기본 컨투어 플롯 설정
    default_graph_option = {
        "figure_size": (12, 8),
        "figure_dpi": 300,
        "contour_levels": 100,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "contour_xlabel_text": "q",
        "contour_ylabel_text": "Time",
        "global_ylim": None,
        "contour_xlim": None,
    }
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}

    # 컨투어 데이터 준비
    times, _ = contour_data["Time-temp"]
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) 
                              for entry in contour_data["Data"]])

    # 그리드 데이터 준비
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, 
                            (grid_q, grid_time), method='nearest')

    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN")
        return None if GUI else plt.show()

    # 강도 범위 설정
    lower_bound = np.nanpercentile(grid_intensity, 
                                 final_opt["contour_lower_percentile"])
    upper_bound = np.nanpercentile(grid_intensity, 
                                 final_opt["contour_upper_percentile"])

    # 플롯 생성
    fig, ax = plt.subplots(figsize=final_opt["figure_size"],
                          dpi=final_opt["figure_dpi"])
    
    # 컨투어 플롯
    cp = ax.contourf(grid_q, grid_time, grid_intensity,
                     levels=final_opt["contour_levels"],
                     cmap=final_opt["contour_cmap"],
                     vmin=lower_bound, vmax=upper_bound)

    # 축 범위 설정
    if final_opt["contour_xlim"] is not None:
        ax.set_xlim(final_opt["contour_xlim"])
    if final_opt["global_ylim"] is not None:
        ax.set_ylim(final_opt["global_ylim"])

    # 라벨 설정
    ax.set_xlabel(final_opt["contour_xlabel_text"], fontsize=14, fontweight='bold')
    ax.set_ylabel(final_opt["contour_ylabel_text"], fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 피크 플롯
    if current_peaks is not None and current_peaks.get("Data"):
        # 단순히 피크들을 선으로 연결
        peak_q_vals = [entry["peak_q"] for entry in current_peaks["Data"]]
        peak_times = [entry["Time"] for entry in current_peaks["Data"]]
        ax.plot(peak_q_vals, peak_times, 'r-', linewidth=1.5, zorder=10)

    # 컬러바 추가
    fig.colorbar(cp, ax=ax, label='log10(Intensity)')
    
    if GUI:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        canvas = FigureCanvasQTAgg(fig)
        canvas.draw()
        return canvas
    else:
        plt.show()