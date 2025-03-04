# Standard library imports
import sys
import os
import copy
import time
import traceback
import datetime
import locale
import json
import re
from pathlib import Path

# PyQt5 imports
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pyqtgraph as pg
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

# Application UI imports
from asset.contour_main_ui import Ui_Contour_plot
from asset.contour_sdd_ui import Ui_SDD_correction
from asset.contour_peak_ui import Ui_Peak_export_manager

# Application module imports
from asset.contour_storage import DATA, PATH_INFO, PARAMS, PLOT_OPTIONS, PROCESS_STATUS, FITTING_THRESHOLD
from asset.spec_log_extractor import parse_log_file
from asset.dat_extractor import process_dat_files
from asset.contour_util import (
    extract_contour_data, interpolate_contour_data, plot_contour,
    select_q_range, find_peak, track_peaks, plot_contour_with_peaks,
    q_to_2theta, theta_to_q, calculate_corrected_sdd, fit_peak_vs_temp
)
from asset.contour_util_gui import (
    QRangeCorrectionHelper, plot_contour_with_peaks_gui,
    TempCorrectionHelper, IndexRangeSelectionHelper, PeakTempRangeHelper,
    DraggableLine, create_plot_widget
)
from asset.fitting_util import (
    gaussian, lorentzian, voigt, find_peak_extraction,
    run_automatic_tracking, plot_contour_extraction,
    calculate_fwhm, check_peak_thresholds
)
from asset.page_asset import LoadingDialog, DragDropLineEdit, normalize_path

# Application page imports
from asset.contour_page_0 import BrowsePage
from asset.contour_page_1 import TempCorrectionPage
from asset.contour_page_2 import ContourPlotPage
from asset.contour_page_sdd import SDDCorrectionDialog
from asset.contour_page_sdd_0 import SDDSettingsPage
from asset.contour_page_sdd_1 import SDDPeakTrackingPage
from asset.contour_page_sdd_2 import SDDFittingPage
from asset.contour_settings_dialog import ContourSettingsDialog
from asset.contour_page_peak import PeakExportDialog
from asset.contour_page_peak_0 import PeakSettingsPage
from asset.contour_page_peak_1 import PeakTrackingPage
from asset.contour_page_peak_3 import DataExportPage
from asset.contour_page_peak_4 import QRangeIndexPage

class MainWindow(QMainWindow):
    def __init__(self, use_thread=True):
        super().__init__()
        # PyQtGraph 전역 설정
        pg.setConfigOptions(background='w', foreground='k')
        
        # UI 설정
        self.ui = Ui_Contour_plot()  # 이 부분도 변경됨
        self.ui.setupUi(self)
        self.SW_Main_page = self.ui.SW_main
        self.use_thread = use_thread

        # 페이지 초기화
        self.browse_page = BrowsePage(self)
        self.temp_correction_page = TempCorrectionPage(self)
        self.contour_plot_page = ContourPlotPage(self)

        # 페이지 변경 시그널 연결
        self.SW_Main_page.currentChanged.connect(self.on_page_changed)

        # Worker 초기화
        self.worker = None
        self.worker_thread = None
        
    def on_page_changed(self, index):
        """Handle page changes"""
        if index == 2:  # ContourPlotPage
            self.contour_plot_page.on_page_entered()



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Set application icon
    if os.path.exists("9A.ico"):
        app_icon = QtGui.QIcon("9A.ico")
        app.setWindowIcon(app_icon)

    # 스타일 설정
    app.setStyle("Fusion")
    
    # 메인 윈도우 생성 및 표시
    win = MainWindow(use_thread=True)
    win.show()
    
    sys.exit(app.exec_())