import numpy as np
from scipy.optimize import curve_fit
from asset.contour_storage import FITTING_THRESHOLD

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
    if threshold_config['use_q_threshold'] and prev_q is not None:
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
