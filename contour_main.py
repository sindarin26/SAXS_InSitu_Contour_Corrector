#contour_main.py
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph as pg
from asset.contour_main_ui import Ui_MainWindow
from asset.browsepage import BrowsePage
from asset.tempcorrectionpage import TempCorrectionPage

class MainWindow(QMainWindow):
    def __init__(self, use_thread=True):
        super().__init__()
        # PyQtGraph 전역 설정
        pg.setConfigOptions(background='w', foreground='k')
        
        # UI 설정
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.SW_Main_page = self.ui.SW_main
        self.use_thread = use_thread

        # 페이지 초기화
        self.browse_page = BrowsePage(self)
        self.temp_correction_page = TempCorrectionPage(self)

        # 페이지 변경 시그널 연결
        self.SW_Main_page.currentChanged.connect(self.on_page_changed)

        # Worker 초기화
        self.worker = None
        self.worker_thread = None
        
    def on_page_changed(self, index):
        """Handle page changes"""
        if index == 1:  # TempCorrectionPage
            self.temp_correction_page.initialize_series_combobox()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle("Fusion")
    
    # 메인 윈도우 생성 및 표시
    win = MainWindow(use_thread=True)
    win.show()
    
    sys.exit(app.exec_())