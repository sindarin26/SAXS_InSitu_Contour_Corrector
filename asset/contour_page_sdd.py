#asset/contour_page_sdd.py

from asset.contour_sdd_ui import Ui_SDD_correction
from PyQt5 import QtWidgets
from asset.contour_storage import DATA, PARAMS
from asset.contour_page_sdd_0 import SDDSettingsPage
from asset.contour_page_sdd_1 import SDDPeakTrackingPage
from asset.contour_page_sdd_2 import SDDFittingPage

class SDDCorrectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_SDD_correction()
        self.ui.setupUi(self)
        
        # Initialize pages
        self.settings_page = SDDSettingsPage(self)
        self.peak_tracking_page = SDDPeakTrackingPage(self)
        self.fitting_page = SDDFittingPage(self)
        
        # Connect signals
        self.ui.PB_next_1.clicked.connect(self.start_peak_tracking)
        self.ui.PB_sdd_correction_start.clicked.connect(self.start_sdd_fitting)
        self.ui.PB_quit.clicked.connect(self.close)  # Connect PB_quit to close the dialog
        
        # Set initial states
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.PB_sdd_correction_start.setEnabled(False)
        
    def start_peak_tracking(self):
        """Start peak tracking process when Next button is clicked"""
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