from PyQt5 import QtWidgets, QtCore
from asset.contour_peak_ui import Ui_Peak_export_manager
from asset.contour_page_peak_0 import PeakSettingsPage
from asset.contour_page_peak_1 import PeakTrackingPage
from asset.contour_page_peak_3 import DataExportPage
from asset.contour_page_peak_4 import QRangeIndexPage
from asset.contour_storage import DATA
import copy

class PeakExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Peak_export_manager()
        self.ui.setupUi(self)
        
        self.setWindowFlags(
            self.windowFlags() | 
            QtCore.Qt.WindowMinMaxButtonsHint | 
            QtCore.Qt.WindowCloseButtonHint & 
            ~QtCore.Qt.WindowContextHelpButtonHint
        )

        # Initialize data dictionary for this export session with new structure
        self.init_peak_data()

        
        # Add a flag to track if page 1 is being visited for the first time
        self.page_1_initial_visit = True
        
        # Initialize pages
        self.settings_page = PeakSettingsPage(self)
        self.peak_tracking_page = PeakTrackingPage(self)
        self.data_export_page = DataExportPage(self)
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
        
        # Connect to stackedWidget's currentChanged signal to track page transitions
        self.ui.stackedWidget.currentChanged.connect(self.on_page_changed)

    def on_page_changed(self, index):
        """Track page transitions to manage the initial visit flag"""
        if index == 1:  # If moving to page 1
            # Reset the initial visit flag when returning to settings and then back to page 1
            if self.ui.stackedWidget.currentIndex() == 0:
                self.page_1_initial_visit = True
            else:
                self.page_1_initial_visit = False

    def go_to_export_page(self):
        """Navigate to the data export page and initialize it"""
        self.ui.stackedWidget.setCurrentIndex(3)  # Set to index 3 (DataExportPage)
        self.data_export_page.initialize_page()   # Initialize the page

    def closeEvent(self, event):
        """Handle window close event"""
        event.accept()