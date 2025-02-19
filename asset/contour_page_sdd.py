#assst/contour_page_sdd.py
from PyQt5 import QtWidgets
from asset.contour_sdd_ui import Ui_SDD_correction
from asset.contour_page_sdd_0 import SDDSettingsPage

class SDDCorrectionDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_SDD_correction()
        self.ui.setupUi(self)
        
        # Initialize pages
        self.settings_page = SDDSettingsPage(self)
        
        # Set initial page
        self.ui.stackedWidget.setCurrentIndex(0)