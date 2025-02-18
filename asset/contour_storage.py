# Global storage for sharing data between different parts of the application
DATA = {
    'log_data': None,          # Data from spec log file
    'extracted_data': None,    # Data from dat file processing
    'contour_data': None,      # Data for contour plotting
    'tracked_peaks': None,     # Peak tracking data
}

# File/Path information
PATH_INFO = {
    'spec_log': None,         # Spec log file path
    'dat_dir': None,          # Dat files directory
    'output_dir': None,       # Output directory
}

# Experiment parameters
PARAMS = {
    'original_sdd': 227.7524,
    'beam_center_x': 955.1370,
    'beam_center_y': 633.0930,
    'pixel_size': 0.0886,
    'experiment_energy': 19.78,
    'converted_energy': 8.042,
    'image_size_x': 1920,
    'image_size_y': 1920,
    'q_format': 'CuKalpha'
}