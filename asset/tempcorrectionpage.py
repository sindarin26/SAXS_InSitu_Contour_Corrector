#asset.tempcorrectionpage.py
import numpy as np
from PyQt5 import QtWidgets, QtCore
from asset.contour_storage import DATA, PATH_INFO, PARAMS
from asset.page_asset import LoadingDialog
from asset.contour_util_gui import TempCorrectionHelper, create_plot_widget

class TempCorrectionPage(QtCore.QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        
        # Store UI elements
        self.CB_series = self.main.ui.CB_series
        self.PB_reset_temp = self.main.ui.PB_reset_temp
        self.GV_adjust_temp = self.main.ui.GV_adjust_temp
        self.GV_raw_temp = self.main.ui.GV_raw_temp
        self.GV_corrected_temp = self.main.ui.GV_corrected_temp
        self.PB_reset_adjust_temp = self.main.ui.PB_reset_adjust_temp
        self.PB_apply_temp = self.main.ui.PB_apply_temp
        self.L_min = self.main.ui.L_min
        self.L_max = self.main.ui.L_max
        self.PB_back_0 = self.main.ui.PB_back_0
        self.PB_next_2 = self.main.ui.PB_next_2
        self.PB_apply_corrected_temp = self.main.ui.PB_apply_corrected_temp

        # Initialize variables
        self.current_series = None
        self.temp_data = None
        self.corrected_temp_data = None

        # Setup plots
        self.setup_plots()
        
        # Initialize helpers
        self.temp_correction_helper = TempCorrectionHelper(self.adjust_plot)
        
        # Connect signals
        self.connect_signals()
        self.initialize_series_combobox()

    def setup_plots(self):
        """Initialize all plot widgets with proper settings"""
        # Setup GV_adjust_temp
        self.adjust_plot = create_plot_widget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.adjust_plot)
        self.GV_adjust_temp.setLayout(layout)
        
        # Setup GV_raw_temp
        self.raw_plot = create_plot_widget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.raw_plot)
        self.GV_raw_temp.setLayout(layout)
        
        # Setup GV_corrected_temp
        self.corrected_plot = create_plot_widget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.corrected_plot)
        self.GV_corrected_temp.setLayout(layout)

    def connect_signals(self):
        """Connect all UI signals"""
        self.CB_series.currentIndexChanged.connect(self.on_series_changed)
        self.PB_reset_temp.clicked.connect(self.reset_all)
        self.PB_reset_adjust_temp.clicked.connect(self.reset_adjustment)
        self.PB_apply_temp.clicked.connect(self.apply_temp_range)
        self.PB_apply_corrected_temp.clicked.connect(self.apply_correction)
        self.PB_back_0.clicked.connect(lambda: self.main.SW_Main_page.setCurrentIndex(0))

    def initialize_series_combobox(self):
        """Initialize series combobox with available series"""
        self.CB_series.clear()
        self.CB_series.addItem("None")
        
        if DATA['log_data']:
            series_list = {
                data["Series"]
                for data in DATA['log_data'].values()
                if isinstance(data, dict) and "Series" in data
            }
            for series in sorted(series_list):
                self.CB_series.addItem(series)

    def update_line_positions(self):
        """수직선 이동 시 L_min, L_max 라벨 업데이트"""
        # 현재 모드에 따른 라인 참조
        if self.temp_correction_helper.selection_mode == "steady":
            lines = self.temp_correction_helper.steady_lines
        elif self.temp_correction_helper.selection_mode == "adjust":
            lines = self.temp_correction_helper.adjust_lines
        else:
            # 특별히 다른 모드가 없다면 무시
            return
        
        # 라인이 2개 있을 때만 min/max 계산
        if len(lines) == 2:
            x1 = lines[0].value()
            x2 = lines[1].value()
            self.L_min.setText(str(int(min(x1, x2))))
            self.L_max.setText(str(int(max(x1, x2))))
            
    def on_series_changed(self, index):
        """Handle series selection change"""
        if index == 0:  # None selected
            self.clear_all_plots()
            self.current_series = None
            return

        self.current_series = self.CB_series.currentText()
        self.load_series_data()
        self.update_all_plots()

    def load_series_data(self):
        """Load temperature data for selected series"""
        if not self.current_series or not DATA['log_data']:
            return

        series_data = [
            (data["Series Index"], data["Temperature"])
            for data in DATA['log_data'].values()
            if isinstance(data, dict) and 
               data.get("Series") == self.current_series
        ]
        
        # Sort by series index
        series_data.sort(key=lambda x: x[0])
        self.temp_data = np.array([temp for _, temp in series_data])
        self.corrected_temp_data = self.temp_data.copy()
        
        # Set data for helper
        self.temp_correction_helper.set_data(self.temp_data)

    def clear_all_plots(self):
        """Clear all plots and reset lines"""
        self.adjust_plot.clear()
        self.raw_plot.clear()
        self.corrected_plot.clear()
        self.L_min.setText("0")
        self.L_max.setText("0")
        self.temp_correction_helper.steady_lines = []
        self.temp_correction_helper.adjust_lines = []

    def update_all_plots(self):
        """Update all plot displays"""
        self.plot_raw_data()
        self.plot_corrected_data()
        self.temp_correction_helper.add_selection_lines()

    def plot_raw_data(self):
        """Plot original temperature data"""
        if self.temp_data is None:
            return
            
        self.raw_plot.clear()
        indices = np.arange(len(self.temp_data))
        self.raw_plot.plot(indices, self.temp_data, pen='b')

    def plot_corrected_data(self):
        """Plot corrected temperature data"""
        if self.corrected_temp_data is None:
            return
            
        self.corrected_plot.clear()
        indices = np.arange(len(self.corrected_temp_data))
        self.corrected_plot.plot(indices, self.corrected_temp_data, pen='r')

    def apply_temp_range(self):
        """Apply selected temperature range"""
        if self.temp_correction_helper.selection_mode == "steady":
            steady_range = self.temp_correction_helper.get_steady_range()
            if steady_range:
                self.temp_correction_helper.selection_mode = "adjust"
                self.temp_correction_helper.add_selection_lines()
        
        elif self.temp_correction_helper.selection_mode == "adjust":
            steady_range = self.temp_correction_helper.get_steady_range()
            adjust_range = self.temp_correction_helper.get_adjust_range()
            
            if steady_range and adjust_range:
                new_temp_data = self.temp_correction_helper.apply_temp_correction(
                    steady_range, adjust_range)
                    
                if new_temp_data is not None:
                    self.corrected_temp_data = new_temp_data
                    self.plot_corrected_data()
                    
                    # Reset for next adjustment if needed
                    self.temp_correction_helper.selection_mode = "steady"
                    self.temp_correction_helper.set_data(self.corrected_temp_data)
                    self.temp_correction_helper.add_selection_lines()
                else:
                    QtWidgets.QMessageBox.warning(
                        self.main,
                        "Error",
                        "Failed to apply temperature correction. Please check the selected ranges."
                    )

    def reset_adjustment(self):
        """Reset temperature adjustment"""
        if self.temp_data is not None:
            self.corrected_temp_data = self.temp_data.copy()
            self.plot_corrected_data()
            self.temp_correction_helper.set_data(self.corrected_temp_data)
            self.temp_correction_helper.selection_mode = "steady"
            self.temp_correction_helper.add_selection_lines()

    def reset_all(self):
        """Reset everything to initial state"""
        self.clear_all_plots()
        self.CB_series.setCurrentIndex(0)
        self.current_series = None
        self.temp_data = None
        self.corrected_temp_data = None
        self.temp_correction_helper.temp_data = None
        self.temp_correction_helper.selection_mode = "steady"

    def apply_correction(self):
        """Apply final temperature correction to the original data"""
        if not self.current_series or self.corrected_temp_data is None:
            return
            
        # Find all entries for current series
        series_entries = []
        for key, data in DATA['log_data'].items():
            if isinstance(data, dict) and data.get("Series") == self.current_series:
                series_entries.append((key, data))
                
        # Sort by series index
        series_entries.sort(key=lambda x: x[1]["Series Index"])
        
        # Update temperatures
        for i, (key, data) in enumerate(series_entries):
            if i < len(self.corrected_temp_data):
                DATA['log_data'][key]["Temperature"] = self.corrected_temp_data[i]
        
        # Disable back button after applying correction
        self.PB_back_0.setEnabled(False)
        
        # Show success message
        QtWidgets.QMessageBox.information(
            self.main,
            "Success",
            "Temperature correction has been applied successfully."
        )