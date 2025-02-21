#asset/contour_page_peak.py
from PyQt5 import QtWidgets
from asset.contour_peak_ui import Ui_Peak_export_manager
from asset.contour_page_peak_0 import PeakSettingsPage
from asset.contour_page_peak_1 import PeakTrackingPage
from asset.contour_storage import DATA

class PeakExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Peak_export_manager()
        self.ui.setupUi(self)

        # Initialize data dictionary for this export session
        self.PEAK_EXTRACT_DATA = {
            'NOTE': '',
            'PEAK': [],
            'tracked_peaks': None
        }
        
        # Initialize pages
        self.settings_page = PeakSettingsPage(self)
        self.peak_tracking_page = PeakTrackingPage(self)
        
        # Set initial state
        self.ui.stackedWidget.setCurrentIndex(0)
        
        # Connect signals
        self.setup_connections()

    def setup_connections(self):
        """Setup signal connections"""
        # Connect quit button
        self.ui.PB_quit.clicked.connect(self.close)

    def closeEvent(self, event):
        """Handle window close event"""
        event.accept()