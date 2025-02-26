#asset/contour_page_peak.py
from PyQt5 import QtWidgets
from asset.contour_peak_ui import Ui_Peak_export_manager
from asset.contour_page_peak_0 import PeakSettingsPage
from asset.contour_page_peak_1 import PeakTrackingPage
from asset.contour_page_peak_3 import DataExportPage  # Import the new page
from asset.contour_page_peak_4 import QRangeIndexPage
from asset.contour_storage import DATA
import copy

class PeakExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Peak_export_manager()
        self.ui.setupUi(self)

        # Initialize data dictionary for this export session with new structure
        self.init_peak_data()
        
        # Initialize pages
        self.settings_page = PeakSettingsPage(self)
        self.peak_tracking_page = PeakTrackingPage(self)
        self.data_export_page = DataExportPage(self)  # Initialize the new page
        self.qrange_index_page = QRangeIndexPage(self)

        # Set initial state
        self.ui.stackedWidget.setCurrentIndex(0)
        
        # Connect signals
        self.setup_connections()

    def init_peak_data(self, contour_data=None):
        """
        Initialize peak data structure
        
        Parameters:
            contour_data (dict, optional): If provided, copy this data as the basis for tracking
        """
        if contour_data is None and DATA.get('contour_data'):
            contour_data = DATA.get('contour_data')
        
        # 원본 데이터 깊은 복사
        peak_contour_data = copy.deepcopy(contour_data) if contour_data else None
        
        # 캐시된 보간 데이터가 있으면 함께 복사 (깊은 복사 X - 불필요한 메모리 사용 방지)
        if peak_contour_data and 'interpolated_data' in contour_data:
            peak_contour_data['interpolated_data'] = contour_data['interpolated_data']
            print("DEBUG: Interpolated data cache copied to peak tracking data")
        
        self.PEAK_EXTRACT_DATA = {
            "NOTE": "",
            "PEAK": peak_contour_data,
            "tracked_peaks": {
                "Series": contour_data["Series"] if contour_data else "",
                "Time-temp": contour_data["Time-temp"] if contour_data else ([], []),
                "Data": []
            },
            "found_peak_list": []
        }

    def setup_connections(self):
        """Setup signal connections"""
        # Connect quit button to close the dialog
        self.ui.PB_quit.clicked.connect(self.close)
        
        # Connect the export page navigation button
        self.ui.PB_go_to_export_page.clicked.connect(self.go_to_export_page)

    def go_to_export_page(self):
        """Navigate to the data export page and initialize it"""
        self.ui.stackedWidget.setCurrentIndex(3)  # Set to index 3 (DataExportPage)
        self.data_export_page.initialize_page()   # Initialize the page

    def closeEvent(self, event):
        """Handle window close event"""
        event.accept()