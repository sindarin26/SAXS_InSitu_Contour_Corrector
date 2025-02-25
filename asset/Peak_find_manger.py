#assst/Peak_find_manger.py

from asset.contour_peak_ui import Ui_SDD_correction
from PyQt5 import QtWidgets
from asset.contour_storage import DATA, PARAMS
from asset.Peak_find_manager_0 import PeakSettingsPage

class SDDCorrectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_SDD_correction()
        self.ui.setupUi(self)
        
        # Initialize pages
        self.settings_page = PeakSettingsPage(self)

        # Connect signals
        self.ui.PB_next_1.clicked.connect(self.start_peak_tracking)
        
        # Set initial states
        self.ui.stackedWidget.setCurrentIndex(0)
        
    def start_peak_tracking(self):
        """Start peak tracking process when Next button is clicked"""

        self.init_data_peak_find()
        contour_data = DATA.get('contour_data')
        if not contour_data:
            QtWidgets.QMessageBox.warning(
                self, 
                "Warning", 
                "No contour data available. Please process data first."
            )
            return
        
        # Initialize peak tracking with current contour data
        self.peak_tracking_page.initialize_peak_tracking(contour_data)
        
    def start_sdd_fitting(self):
        """Start SDD fitting process"""
        # We don't need to check DATA['tracked_peaks'] here because
        # the button is only enabled when peak tracking is complete
        self.ui.stackedWidget.setCurrentIndex(3)
        self.fitting_page.initialize_fitting()
        
    def check_peak_tracking_complete(self):
        """Check if peak tracking is complete and enable/disable SDD correction button"""
        if DATA.get('tracked_peaks') is None:
            self.ui.PB_sdd_correction_start.setEnabled(False)
            return
            
        tracked_peaks = DATA['tracked_peaks']
        total_frames = len(DATA['contour_data']['Data'])
        found_peaks = len(tracked_peaks.get('Data', []))
        
        self.ui.PB_sdd_correction_start.setEnabled(found_peaks == total_frames)

    def init_data_peak_find(self):
        self.PEAK_EXTRACT_DATA = {
            'NOTE': '',
            'PEAK': [],
            'tracked_peaks': None
        }