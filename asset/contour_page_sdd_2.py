# asset/contour_page_sdd_2.py

from PyQt5 import QtWidgets, QtCore
import numpy as np
from asset.contour_storage import DATA
from asset.contour_util_gui import IndexRangeSelectionHelper, PeakTempRangeHelper
from asset.contour_util import (fit_peak_vs_temp, add_corrected_peak_q, 
                              calculate_sdd_for_tracked_peaks_refactored)
from asset.contour_storage import PARAMS
import pyqtgraph as pg

class SDDFittingPage(QtCore.QObject):
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # Initialize variables
        self.index_range = None
        self.peak_fit_temp_range = None
        self.fit_params = None
        
        # Initialize helpers
        self.index_range_helper = None
        self.peak_temp_helper = None
        
        # Setup initial state
        self.setup_initial_state()
        self.connect_signals()

    def setup_initial_state(self):
        """Initialize UI state"""
        self.ui.PB_final_apply.setEnabled(False)
        self.L_current_status_3 = self.ui.L_current_status_3
        
    def connect_signals(self):
        """Connect button signals"""
        self.ui.PB_back_3.clicked.connect(self.on_back)
        self.ui.PB_final_apply.clicked.connect(self.on_final_apply)
        
    def initialize_fitting(self):
        """Start the fitting process"""
        self.setup_index_range_selection()
        self.L_current_status_3.setText("Select Index Range for Temperature Fitting")
        
    def setup_index_range_selection(self):
        """Setup the index range selection view"""
        if hasattr(self.ui, 'QGV_final'):
            if self.ui.QGV_final.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())
                
            layout = QtWidgets.QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            self.ui.QGV_final.setLayout(layout)
            
            # Initialize helper
            self.index_range_helper = IndexRangeSelectionHelper(plot_widget)
            
            # Get temperature data from tracked peaks
            temp_data = DATA['tracked_peaks']["Time-temp"][1]
            self.index_range_helper.set_data(temp_data)
            self.index_range_helper.add_selection_lines()
            
    def setup_peak_temp_range_selection(self):
        """Setup the peak vs temperature range selection view"""
        if hasattr(self.ui, 'QGV_final'):
            if self.ui.QGV_final.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())
                
            layout = QtWidgets.QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            self.ui.QGV_final.setLayout(layout)
            
            # Initialize helper
            self.peak_temp_helper = PeakTempRangeHelper(plot_widget)
            
            # Get data within selected index range
            tracked_peaks = DATA['tracked_peaks']
            start_idx, end_idx = self.index_range
            temp_data = np.array(tracked_peaks["Time-temp"][1])[start_idx:end_idx+1]
            peak_q_data = np.array([entry["peak_q"] for entry in tracked_peaks["Data"][start_idx:end_idx+1]])
            
            self.peak_temp_helper.set_data(temp_data, peak_q_data)
            self.peak_temp_helper.add_selection_lines()
            
    def on_back(self):
        """Handle back button click"""
        self.index_range = None
        self.peak_fit_temp_range = None
        self.fit_params = None
        self.ui.stackedWidget.setCurrentIndex(2)
        
    def proceed_to_peak_temp_selection(self):
        """Proceed to peak vs temperature selection after index range selection"""
        self.index_range = self.index_range_helper.get_index_range()
        if self.index_range is None:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Warning",
                "Please select a valid index range first."
            )
            return
            
        self.L_current_status_3.setText("Select Temperature Range for Peak Fitting")
        self.setup_peak_temp_range_selection()
        
    def on_final_apply(self):
        """Handle final apply button click"""
        if self.index_range is None:
            # First phase: Get index range and proceed to temp range selection
            self.proceed_to_peak_temp_selection()
            return
            
        if self.peak_fit_temp_range is None:
            # Second phase: Get temperature range and calculate fit
            self.peak_fit_temp_range = self.peak_temp_helper.get_temp_range()
            if self.peak_fit_temp_range is None:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    "Please select a valid temperature range first."
                )
                return
                
            # Calculate fit parameters
            self.fit_params = fit_peak_vs_temp(DATA['tracked_peaks'], self.peak_fit_temp_range)
            
            # Apply corrections
            tracked_peaks = add_corrected_peak_q(
                DATA['tracked_peaks'],
                self.fit_params,
                self.peak_fit_temp_range,
                PARAMS['original_sdd'],
                PARAMS['experiment_energy'],
                PARAMS['converted_energy']
            )
            
            # Update contour data with new SDD corrections
            corrected_contour_data = calculate_sdd_for_tracked_peaks_refactored(
                DATA['contour_data'],
                tracked_peaks,
                PARAMS['original_sdd'],
                PARAMS['experiment_energy'],
                PARAMS['converted_energy']
            )
            
            # Update stored data
            DATA['tracked_peaks'] = tracked_peaks
            DATA['contour_data'] = corrected_contour_data
            
            # Show success message and return to previous page
            QtWidgets.QMessageBox.information(
                self.main,
                "Success",
                "SDD correction has been applied successfully."
            )
            self.on_back()