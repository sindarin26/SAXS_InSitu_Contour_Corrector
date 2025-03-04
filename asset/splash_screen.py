from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import time

class SplashScreen(QtWidgets.QSplashScreen):
    def __init__(self, width=500, height=300):
        """Create a splash screen with a progress bar"""
        # Create a pixmap for the splash screen
        pixmap = QtGui.QPixmap(width, height)
        pixmap.fill(QtGui.QColor(240, 240, 240))  # Light background
        
        super().__init__(pixmap)
        
        # Add layout to splash screen
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Add title
        self.title_label = QtWidgets.QLabel("Contour Plot for 9A, PAL")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont("Segoe UI", 16)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color: #333333;")
        self.layout.addWidget(self.title_label)
        
        # Add spacer
        self.layout.addSpacing(50)
        
        # Add status label
        self.status_label = QtWidgets.QLabel("Opening...")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setFont(QtGui.QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #333333;")
        self.layout.addWidget(self.status_label)
        
        # Create progress bar with Fusion style
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        # Light theme Fusion-like style for QProgressBar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdbdbd;
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                color: #333333;
                height: 25px;
            }
            
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 3px;
            }
        """)
        self.layout.addWidget(self.progress_bar)
        
        # Add credits
        self.credits_label = QtWidgets.QLabel("hyungju@postech.ac.kr, yeongsik@postech.ac.kr, yhkim26@postech.ac.kr")
        self.credits_label.setAlignment(QtCore.Qt.AlignCenter)
        self.credits_label.setFont(QtGui.QFont("Segoe UI", 9))
        self.credits_label.setStyleSheet("color: #333333;")
        self.layout.addWidget(self.credits_label)
        
        # Add version
        self.version_label = QtWidgets.QLabel("v1.0 - February 2025")
        self.version_label.setAlignment(QtCore.Qt.AlignCenter)
        self.version_label.setFont(QtGui.QFont("Segoe UI", 9))
        self.version_label.setStyleSheet("color: #333333;")
        self.layout.addWidget(self.version_label)
        
        # Center the splash screen on the monitor
        desktop = QtWidgets.QApplication.desktop()
        screen_rect = desktop.screenGeometry()
        self.move(
            (screen_rect.width() - width) // 2,
            (screen_rect.height() - height) // 2
        )
    
    def set_progress(self, value, message=None):
        """Update progress bar and optionally the status message"""
        self.progress_bar.setValue(value)
        if message:
            self.status_label.setText(message)
        
        # Process events to update the UI
        QtWidgets.QApplication.processEvents()


# 메인 애플리케이션에서 스플래시 화면을 사용할 때:
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    
    # Fusion 스타일 적용
    app.setStyle("Fusion")
    
    # 라이트 팔레트 설정
    light_palette = QtGui.QPalette()
    light_palette.setColor(QtGui.QPalette.Window, QtGui.QColor(240, 240, 240))
    light_palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(51, 51, 51))
    light_palette.setColor(QtGui.QPalette.Base, QtGui.QColor(255, 255, 255))
    light_palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(245, 245, 245))
    light_palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
    light_palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(51, 51, 51))
    light_palette.setColor(QtGui.QPalette.Text, QtGui.QColor(51, 51, 51))
    light_palette.setColor(QtGui.QPalette.Button, QtGui.QColor(240, 240, 240))
    light_palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(51, 51, 51))
    light_palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    light_palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 120, 215))
    light_palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(light_palette)
    
    splash = SplashScreen()
    splash.show()
    
    # 로딩 시뮬레이션
    for i in range(101):
        splash.set_progress(i, f"Loading... {i}%")
        time.sleep(0.05)  # 실제 앱에서는 실제 로딩 작업으로 대체
    
    # 여기서 메인 윈도우를 시작
    main_window = QtWidgets.QMainWindow()
    main_window.setWindowTitle("Main Application")
    main_window.resize(800, 600)
    
    splash.finish(main_window)
    main_window.show()
    
    sys.exit(app.exec_())