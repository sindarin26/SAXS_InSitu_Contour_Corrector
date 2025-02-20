import numpy as np
from scipy.optimize import curve_fit

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

def find_peak(contour_data, Index_number=0, input_range=None, peak_info=None, fitting_function="gaussian"):
    """
    주어진 contour_data의 Index_number 데이터에서, input_range 내의 q, Intensity 데이터를
    대상으로 피팅을 수행하여 peak의 q 위치와 intensity를 구하고,
    input_range의 중심을 기준으로 출력 range를 반환.
    
    Parameters
    ----------
    contour_data : dict
        Contour plot data
    Index_number : int
        Index of the data point to analyze
    input_range : tuple
        (min, max) range for fitting
    peak_info : dict
        Previous peak information for validation
    fitting_function : str
        Type of fitting function to use ("gaussian", "lorentzian", or "voigt")
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
        print(f"Fitting failed with {fitting_function}:", e)
        return None
    
    # Extract peak position based on fitting function
    if fitting_function.lower() == "lorentzian":
        peak_q = popt[1]  # x0 parameter
    elif fitting_function.lower() == "voigt":
        peak_q = popt[1]  # mu parameter
    else:  # Gaussian
        peak_q = popt[1]  # mu parameter
    
    peak_intensity = fit_func(peak_q, *popt)
    
    # Calculate output range
    center = (input_range[0] + input_range[1]) / 2.0
    shift = peak_q - center
    output_range = (input_range[0] + shift, input_range[1] + shift)
    
    # Validation against previous peak if available
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