#asset.tempcorrectionpage.py
import numpy as np
from PyQt5 import QtWidgets, QtCore
from asset.contour_storage import DATA, PATH_INFO, PARAMS
from asset.page_asset import LoadingDialog
from asset.contour_util_gui import TempCorrectionHelper, create_plot_widget
import pyqtgraph as pg

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
        self.PB_next_2.clicked.connect(lambda: self.main.SW_Main_page.setCurrentIndex(2))

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
        self.temp_correction_helper.steady_lines = []
        self.temp_correction_helper.adjust_lines = []

    def update_all_plots(self):
        """Update all plot displays"""
        self.plot_raw_data()
        self.plot_corrected_data()
        self.temp_correction_helper.add_selection_lines()

    def plot_raw_data(self):
        if self.temp_data is None:
            return

        self.raw_plot.clear()
        indices = np.arange(len(self.temp_data))
        # 기존: self.raw_plot.plot(indices, self.temp_data, pen='b')
        self.raw_plot.plot(
            indices, 
            self.temp_data,
            pen=pg.mkPen(color='k', width=1), # 검정색 실선
            symbol='o',                       # 동그라미 마커
            symbolSize=5,                     # 마커 크기
            symbolPen=pg.mkPen('k'),          # 마커 테두리 검정
            symbolBrush=pg.mkBrush('w')       # 마커 내부 흰색
        )

    def plot_corrected_data(self):
        """Plot corrected temperature data"""
        if self.corrected_temp_data is None:
            return
                
        self.corrected_plot.clear()
        indices = np.arange(len(self.corrected_temp_data))
        self.corrected_plot.plot(
            indices, 
            self.corrected_temp_data,
            pen=pg.mkPen(color='r', width=1),  # 빨간색 실선
            symbol='o',                        # 동그라미 마커
            symbolSize=5,                      # 마커 크기
            symbolPen=pg.mkPen('r'),           # 마커 테두리 빨강
            symbolBrush=pg.mkBrush('w')        # 마커 내부 흰색
        )

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
        """Reset only selection lines"""
        if self.temp_data is not None:
            self.temp_correction_helper.selection_mode = "steady"
            self.temp_correction_helper.add_selection_lines()

    def reset_all(self):
        """Reset only current series data"""
        if self.current_series is not None:
            # Get original temperature data for current series
            series_data = [
                (data["Series Index"], data["Temperature"])
                for data in DATA['log_data'].values()
                if isinstance(data, dict) and 
                data.get("Series") == self.current_series
            ]
            series_data.sort(key=lambda x: x[0])
            self.temp_data = np.array([temp for _, temp in series_data])
            self.corrected_temp_data = self.temp_data.copy()
            
            # Update plots
            self.plot_raw_data()
            self.plot_corrected_data()
            self.temp_correction_helper.set_data(self.corrected_temp_data)
            self.temp_correction_helper.selection_mode = "steady"
            self.temp_correction_helper.add_selection_lines()

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
        
        # Reload data to update plots
        self.load_series_data()
        self.plot_raw_data()
        self.plot_corrected_data()
        
        # Show success message
        QtWidgets.QMessageBox.information(
            self.main,
            "Success",
            "Temperature correction has been applied successfully."
        )