#assst/contour_page_sdd.py

from asset.contour_sdd_ui import Ui_SDD_correction
from PyQt5 import QtWidgets
from asset.contour_storage import DATA, PARAMS
from asset.contour_page_sdd_0 import SDDSettingsPage
from asset.contour_page_sdd_1 import SDDPeakTrackingPage

class SDDCorrectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_SDD_correction()
        self.ui.setupUi(self)
        
        # Initialize pages
        self.settings_page = SDDSettingsPage(self)
        self.peak_tracking_page = SDDPeakTrackingPage(self)
        
        # Connect signals
        self.ui.PB_next_1.clicked.connect(self.start_peak_tracking)
        
        # Set initial page
        self.ui.stackedWidget.setCurrentIndex(0)
    
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