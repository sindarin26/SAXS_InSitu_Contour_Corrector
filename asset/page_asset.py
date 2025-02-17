from PyQt5 import QtWidgets, QtCore, QtGui

class LoadingDialog(QtWidgets.QDialog):
    """공통으로 사용되는 로딩 다이얼로그"""
    def __init__(self, parent=None, message="Processing"):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setMinimumWidth(400)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.CustomizeWindowHint | QtCore.Qt.WindowTitleHint)
        
        font = QtGui.QFont("Segoe UI", 9)
        self.setFont(font)
        
        layout = QtWidgets.QVBoxLayout()
        
        self.label = QtWidgets.QLabel(message)
        self.label.setFont(font)
        layout.addWidget(self.label)
        
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFont(font)
        layout.addWidget(self.progress)
        
        self.setLayout(layout)

class DragDropLineEdit(QtWidgets.QLineEdit):
    """드래그 앤 드롭 기능이 있는 라인에디트"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.setText(path)
            self.returnPressed.emit()

def normalize_path(path):
    """경로 문자열 정규화"""
    path = path.replace("\\", "/")
    path = path.rstrip("/")
    return path