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
    'legend': False,
    'graph_option': {
        # Figure 설정
        "figure_size": (12, 8),
        "figure_dpi": 300,
        
        # 타이틀 관련 (True/False 바로 아래에 텍스트 배치)
        "figure_title_enable": False,
        "figure_title_text": "",
        "contour_title_enable": False,
        "contour_title_text": "Contour Plot",
        "temp_title_enable": False,
        "temp_title_text": "Temperature Plot",

        # 폰트 관련 설정
        "font_label": "Times New Roman",
        "font_tick": "Times New Roman",
        "font_title": "Times New Roman",
        
        # Label 크기 설정
        "axes_label_size": 10,
        "tick_label_size": 8,
        "title_size": 12,

        # X축 & Y축 관련 설정 (X 먼저, Y 다음)
        "contour_xlabel_enable": True,
        "contour_xlabel_text": "2theta (Cu K-alpha)",
        "contour_ylabel_enable": True,
        "contour_ylabel_text": "Elapsed Time",
        "temp_xlabel_enable": True,
        "temp_xlabel_text": "Temperature",
        "temp_ylabel_enable": True,
        "temp_ylabel_text": "Elapsed Time",

        # Grid 및 범위 설정
        "contour_grid": False,
        "contour_xlim": None,
        "global_ylim": None,
        "temp_xlim": None,
        "temp_grid": True,

        # 컬러맵 및 색상 관련 설정
        "contour_cmap": "inferno",
        "contour_levels": 200,
        "contour_lower_percentile": 0.1,
        "contour_upper_percentile": 98,
        "colorbar_label": "log10(Intensity)",
        "cbar_location": "left",
        "cbar_pad": 0.15,

        # 배치 및 비율
        "wspace": 0.00,
        "width_ratios": [6, 2],
    }
}

# Process status and selections
PROCESS_STATUS = {
    'selected_series': None,  # Currently selected series
}
