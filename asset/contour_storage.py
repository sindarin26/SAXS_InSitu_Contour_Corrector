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

# Plot options
PLOT_OPTIONS = {
    'temp': True,
    'legend': True,
    'graph_option': {
        "figure_size": (12, 8),
        "figure_dpi": 300,
        "figure_title_enable": False,
        "figure_title_text": "",
        "font_label": "Times New Roman",
        "font_tick": "Times New Roman",
        "font_title": "Times New Roman",
        "axes_label_size": 10,
        "tick_label_size": 8,
        "title_size": 12,
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
        "contour_levels": 200,
        "contour_cmap": "inferno",
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "colorbar_label": "log10(Intensity)",
        "cbar_location": "left",
        "cbar_pad": 0.15,
    }
}

# Process status and selections
PROCESS_STATUS = {
    'selected_series': None,  # Currently selected series
}
