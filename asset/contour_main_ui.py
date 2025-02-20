from PyQt5.QtWidgets import QApplication

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QPalette, QColor

QApplication.setStyle("Fusion")

app = QApplication.instance()
if app:
    # 폰트 설정
    font = QFont("Segoe UI", 10)  # 윈도우에서 가독성이 좋은 폰트
    app.setFont(font)

    # 다크 모드 팔레트 설정
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))  # 다크 배경
    palette.setColor(QPalette.WindowText, QColor(255, 255, 255))  # 흰색 글자
    palette.setColor(QPalette.Base, QColor(35, 35, 35))  # 입력창 배경
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197))  # 강조 색상
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    # 기본 스타일시트 적용 (버튼, 입력창 등)
    app.setStyleSheet("""
        QWidget {
            background-color: #353535;
            color: #ffffff;
        }
        QPushButton {
            background-color: #2c3e50;
            border: 1px solid #1a252f;
            padding: 5px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #1a252f;
        }
        QLineEdit {
            background-color: #1e1e1e;
            border: 1px solid #5a5a5a;
            padding: 4px;
            color: white;
        }
        QLabel {
            font-size: 12px;
            font-weight: bold;
        }
    """)
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Python/SAXS_InSitu_Contour_Corrector/contour_main_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Contour_plot(object):
    def setupUi(self, Contour_plot):
        Contour_plot.setObjectName("Contour_plot")
        Contour_plot.resize(1533, 1051)
        self.centralwidget = QtWidgets.QWidget(Contour_plot)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.SW_main = QtWidgets.QStackedWidget(self.centralwidget)
        self.SW_main.setObjectName("SW_main")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_4 = QtWidgets.QFrame(self.page)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_4)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.frame_4)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.LE_spec_log_dir = QtWidgets.QLineEdit(self.frame)
        self.LE_spec_log_dir.setText("")
        self.LE_spec_log_dir.setObjectName("LE_spec_log_dir")
        self.horizontalLayout.addWidget(self.LE_spec_log_dir)
        self.PB_spec_log_browse = QtWidgets.QPushButton(self.frame)
        self.PB_spec_log_browse.setObjectName("PB_spec_log_browse")
        self.horizontalLayout.addWidget(self.PB_spec_log_browse)
        self.PB_spec_apply = QtWidgets.QPushButton(self.frame)
        self.PB_spec_apply.setObjectName("PB_spec_apply")
        self.horizontalLayout.addWidget(self.PB_spec_apply)
        self.verticalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(self.frame_4)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.LE_dat_dir = QtWidgets.QLineEdit(self.frame_2)
        self.LE_dat_dir.setText("")
        self.LE_dat_dir.setObjectName("LE_dat_dir")
        self.horizontalLayout_2.addWidget(self.LE_dat_dir)
        self.PB_dat_browse = QtWidgets.QPushButton(self.frame_2)
        self.PB_dat_browse.setObjectName("PB_dat_browse")
        self.horizontalLayout_2.addWidget(self.PB_dat_browse)
        self.PB_dat_apply = QtWidgets.QPushButton(self.frame_2)
        self.PB_dat_apply.setObjectName("PB_dat_apply")
        self.horizontalLayout_2.addWidget(self.PB_dat_apply)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame_4)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_3 = QtWidgets.QLabel(self.frame_3)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.LE_output_dir = QtWidgets.QLineEdit(self.frame_3)
        self.LE_output_dir.setObjectName("LE_output_dir")
        self.horizontalLayout_3.addWidget(self.LE_output_dir)
        self.PB_output_browse = QtWidgets.QPushButton(self.frame_3)
        self.PB_output_browse.setObjectName("PB_output_browse")
        self.horizontalLayout_3.addWidget(self.PB_output_browse)
        self.PB_output_apply = QtWidgets.QPushButton(self.frame_3)
        self.PB_output_apply.setObjectName("PB_output_apply")
        self.horizontalLayout_3.addWidget(self.PB_output_apply)
        self.verticalLayout.addWidget(self.frame_3)
        self.verticalLayout_3.addWidget(self.frame_4)
        self.frame_8 = QtWidgets.QFrame(self.page)
        self.frame_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_8.setObjectName("frame_8")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_8)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.frame_9 = QtWidgets.QFrame(self.frame_8)
        self.frame_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_9.setObjectName("frame_9")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_9)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.frame_9)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_2.addWidget(self.label_6)
        self.TW_Process_List = QtWidgets.QTableWidget(self.frame_9)
        self.TW_Process_List.setObjectName("TW_Process_List")
        self.TW_Process_List.setColumnCount(0)
        self.TW_Process_List.setRowCount(0)
        self.verticalLayout_2.addWidget(self.TW_Process_List)
        self.horizontalLayout_8.addWidget(self.frame_9)
        self.verticalLayout_3.addWidget(self.frame_8)
        self.frame_5 = QtWidgets.QFrame(self.page)
        self.frame_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.frame_6 = QtWidgets.QFrame(self.frame_5)
        self.frame_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.frame_6)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(self.frame_6)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_7.addWidget(self.label_5)
        self.label_11 = QtWidgets.QLabel(self.frame_6)
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setIndent(6)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_7.addWidget(self.label_11)
        self.horizontalLayout_6.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_5)
        self.frame_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_5.setContentsMargins(12, 12, -1, -1)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.PB_reset = QtWidgets.QPushButton(self.frame_7)
        self.PB_reset.setObjectName("PB_reset")
        self.horizontalLayout_5.addWidget(self.PB_reset)
        self.PB_next_1 = QtWidgets.QPushButton(self.frame_7)
        self.PB_next_1.setObjectName("PB_next_1")
        self.horizontalLayout_5.addWidget(self.PB_next_1)
        self.horizontalLayout_6.addWidget(self.frame_7)
        self.verticalLayout_3.addWidget(self.frame_5)
        self.SW_main.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.frame_13 = QtWidgets.QFrame(self.page_2)
        self.frame_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_13.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_13.setObjectName("frame_13")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.frame_13)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.CB_series = QtWidgets.QComboBox(self.frame_13)
        self.CB_series.setObjectName("CB_series")
        self.horizontalLayout_12.addWidget(self.CB_series)
        self.verticalLayout_6.addWidget(self.frame_13)
        self.frame_14 = QtWidgets.QFrame(self.page_2)
        self.frame_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_14.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_14.setObjectName("frame_14")
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout(self.frame_14)
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        self.frame_16 = QtWidgets.QFrame(self.frame_14)
        self.frame_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_16.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_16.setObjectName("frame_16")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_16)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.frame_22 = QtWidgets.QFrame(self.frame_16)
        self.frame_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_22.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_22.setObjectName("frame_22")
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout(self.frame_22)
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.frame_23 = QtWidgets.QFrame(self.frame_22)
        self.frame_23.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_23.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_23.setObjectName("frame_23")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.frame_23)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_13 = QtWidgets.QLabel(self.frame_23)
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_13.addWidget(self.label_13)
        self.horizontalLayout_20.addWidget(self.frame_23)
        self.frame_24 = QtWidgets.QFrame(self.frame_22)
        self.frame_24.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_24.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_24.setObjectName("frame_24")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.frame_24)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.PB_reset_temp = QtWidgets.QPushButton(self.frame_24)
        self.PB_reset_temp.setObjectName("PB_reset_temp")
        self.horizontalLayout_19.addWidget(self.PB_reset_temp)
        self.horizontalLayout_20.addWidget(self.frame_24)
        self.verticalLayout_5.addWidget(self.frame_22)
        self.GV_adjust_temp = QtWidgets.QGraphicsView(self.frame_16)
        self.GV_adjust_temp.setObjectName("GV_adjust_temp")
        self.verticalLayout_5.addWidget(self.GV_adjust_temp)
        self.frame_17 = QtWidgets.QFrame(self.frame_16)
        self.frame_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_17.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_17.setObjectName("frame_17")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.frame_17)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.frame_19 = QtWidgets.QFrame(self.frame_17)
        self.frame_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_19.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_19.setObjectName("frame_19")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.frame_19)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.PB_reset_adjust_temp = QtWidgets.QPushButton(self.frame_19)
        self.PB_reset_adjust_temp.setObjectName("PB_reset_adjust_temp")
        self.horizontalLayout_15.addWidget(self.PB_reset_adjust_temp)
        self.PB_apply_temp = QtWidgets.QPushButton(self.frame_19)
        self.PB_apply_temp.setObjectName("PB_apply_temp")
        self.horizontalLayout_15.addWidget(self.PB_apply_temp)
        self.horizontalLayout_18.addWidget(self.frame_19)
        self.verticalLayout_5.addWidget(self.frame_17)
        self.horizontalLayout_26.addWidget(self.frame_16)
        self.frame_15 = QtWidgets.QFrame(self.frame_14)
        self.frame_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_15.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_15.setObjectName("frame_15")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_15)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_14 = QtWidgets.QLabel(self.frame_15)
        self.label_14.setObjectName("label_14")
        self.verticalLayout_4.addWidget(self.label_14)
        self.GV_raw_temp = QtWidgets.QGraphicsView(self.frame_15)
        self.GV_raw_temp.setObjectName("GV_raw_temp")
        self.verticalLayout_4.addWidget(self.GV_raw_temp)
        self.label_15 = QtWidgets.QLabel(self.frame_15)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_4.addWidget(self.label_15)
        self.GV_corrected_temp = QtWidgets.QGraphicsView(self.frame_15)
        self.GV_corrected_temp.setObjectName("GV_corrected_temp")
        self.verticalLayout_4.addWidget(self.GV_corrected_temp)
        self.frame_26 = QtWidgets.QFrame(self.frame_15)
        self.frame_26.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_26.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_26.setObjectName("frame_26")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout(self.frame_26)
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.frame_30 = QtWidgets.QFrame(self.frame_26)
        self.frame_30.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_30.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_30.setObjectName("frame_30")
        self.horizontalLayout_25 = QtWidgets.QHBoxLayout(self.frame_30)
        self.horizontalLayout_25.setObjectName("horizontalLayout_25")
        self.PB_apply_corrected_temp = QtWidgets.QPushButton(self.frame_30)
        self.PB_apply_corrected_temp.setObjectName("PB_apply_corrected_temp")
        self.horizontalLayout_25.addWidget(self.PB_apply_corrected_temp)
        self.horizontalLayout_21.addWidget(self.frame_30)
        self.verticalLayout_4.addWidget(self.frame_26)
        self.horizontalLayout_26.addWidget(self.frame_15)
        self.verticalLayout_6.addWidget(self.frame_14)
        self.frame_10 = QtWidgets.QFrame(self.page_2)
        self.frame_10.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_10.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_10.setObjectName("frame_10")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.frame_10)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.frame_11 = QtWidgets.QFrame(self.frame_10)
        self.frame_11.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_11.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_11.setObjectName("frame_11")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.frame_11)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_7 = QtWidgets.QLabel(self.frame_11)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_10.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.frame_11)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_10.addWidget(self.label_8)
        self.label_12 = QtWidgets.QLabel(self.frame_11)
        self.label_12.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_12.setIndent(6)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_10.addWidget(self.label_12)
        self.horizontalLayout_9.addWidget(self.frame_11)
        self.frame_12 = QtWidgets.QFrame(self.frame_10)
        self.frame_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_12.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_12.setObjectName("frame_12")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame_12)
        self.horizontalLayout_11.setContentsMargins(12, 12, -1, -1)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.PB_back_0 = QtWidgets.QPushButton(self.frame_12)
        self.PB_back_0.setObjectName("PB_back_0")
        self.horizontalLayout_11.addWidget(self.PB_back_0)
        self.PB_next_2 = QtWidgets.QPushButton(self.frame_12)
        self.PB_next_2.setObjectName("PB_next_2")
        self.horizontalLayout_11.addWidget(self.PB_next_2)
        self.horizontalLayout_9.addWidget(self.frame_12)
        self.verticalLayout_6.addWidget(self.frame_10)
        self.SW_main.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.frame_25 = QtWidgets.QFrame(self.page_3)
        self.frame_25.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_25.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_25.setObjectName("frame_25")
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout(self.frame_25)
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.L_current_series = QtWidgets.QLabel(self.frame_25)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.L_current_series.setFont(font)
        self.L_current_series.setObjectName("L_current_series")
        self.horizontalLayout_22.addWidget(self.L_current_series)
        self.verticalLayout_11.addWidget(self.frame_25)
        self.line_3 = QtWidgets.QFrame(self.page_3)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.verticalLayout_11.addWidget(self.line_3)
        self.frame_27 = QtWidgets.QFrame(self.page_3)
        self.frame_27.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_27.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_27.setObjectName("frame_27")
        self.horizontalLayout_27 = QtWidgets.QHBoxLayout(self.frame_27)
        self.horizontalLayout_27.setObjectName("horizontalLayout_27")
        self.GV_contour = QtWidgets.QGraphicsView(self.frame_27)
        self.GV_contour.setObjectName("GV_contour")
        self.horizontalLayout_27.addWidget(self.GV_contour)
        self.frame_28 = QtWidgets.QFrame(self.frame_27)
        self.frame_28.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_28.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_28.setObjectName("frame_28")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.frame_28)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem)
        self.frame_31 = QtWidgets.QFrame(self.frame_28)
        self.frame_31.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_31.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_31.setObjectName("frame_31")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_31)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.PB_contour_setting = QtWidgets.QPushButton(self.frame_31)
        self.PB_contour_setting.setObjectName("PB_contour_setting")
        self.verticalLayout_7.addWidget(self.PB_contour_setting)
        self.verticalLayout_9.addWidget(self.frame_31)
        self.frame_29 = QtWidgets.QFrame(self.frame_28)
        self.frame_29.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_29.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_29.setObjectName("frame_29")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.frame_29)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.PB_export = QtWidgets.QPushButton(self.frame_29)
        self.PB_export.setObjectName("PB_export")
        self.verticalLayout_8.addWidget(self.PB_export)
        self.line_2 = QtWidgets.QFrame(self.frame_29)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_8.addWidget(self.line_2)
        self.label_18 = QtWidgets.QLabel(self.frame_29)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_8.addWidget(self.label_18)
        self.frame_32 = QtWidgets.QFrame(self.frame_29)
        self.frame_32.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_32.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_32.setObjectName("frame_32")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout(self.frame_32)
        self.horizontalLayout_23.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_23.setSpacing(0)
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.PB_export_temp_on = QtWidgets.QPushButton(self.frame_32)
        self.PB_export_temp_on.setCheckable(True)
        self.PB_export_temp_on.setAutoExclusive(True)
        self.PB_export_temp_on.setObjectName("PB_export_temp_on")
        self.horizontalLayout_23.addWidget(self.PB_export_temp_on)
        self.PB_export_temp_off = QtWidgets.QPushButton(self.frame_32)
        self.PB_export_temp_off.setCheckable(True)
        self.PB_export_temp_off.setChecked(True)
        self.PB_export_temp_off.setAutoExclusive(True)
        self.PB_export_temp_off.setObjectName("PB_export_temp_off")
        self.horizontalLayout_23.addWidget(self.PB_export_temp_off)
        self.verticalLayout_8.addWidget(self.frame_32)
        self.line = QtWidgets.QFrame(self.frame_29)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout_8.addWidget(self.line)
        self.label_19 = QtWidgets.QLabel(self.frame_29)
        self.label_19.setObjectName("label_19")
        self.verticalLayout_8.addWidget(self.label_19)
        self.frame_33 = QtWidgets.QFrame(self.frame_29)
        self.frame_33.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_33.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_33.setObjectName("frame_33")
        self.horizontalLayout_24 = QtWidgets.QHBoxLayout(self.frame_33)
        self.horizontalLayout_24.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_24.setSpacing(0)
        self.horizontalLayout_24.setObjectName("horizontalLayout_24")
        self.PB_export_data_on = QtWidgets.QPushButton(self.frame_33)
        self.PB_export_data_on.setCheckable(True)
        self.PB_export_data_on.setAutoExclusive(True)
        self.PB_export_data_on.setObjectName("PB_export_data_on")
        self.horizontalLayout_24.addWidget(self.PB_export_data_on)
        self.PB_export_data_off = QtWidgets.QPushButton(self.frame_33)
        self.PB_export_data_off.setCheckable(True)
        self.PB_export_data_off.setChecked(True)
        self.PB_export_data_off.setAutoExclusive(True)
        self.PB_export_data_off.setObjectName("PB_export_data_off")
        self.horizontalLayout_24.addWidget(self.PB_export_data_off)
        self.verticalLayout_8.addWidget(self.frame_33)
        self.verticalLayout_9.addWidget(self.frame_29)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_9.addItem(spacerItem1)
        self.horizontalLayout_27.addWidget(self.frame_28)
        self.verticalLayout_11.addWidget(self.frame_27)
        self.frame_34 = QtWidgets.QFrame(self.page_3)
        self.frame_34.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_34.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_34.setObjectName("frame_34")
        self.horizontalLayout_30 = QtWidgets.QHBoxLayout(self.frame_34)
        self.horizontalLayout_30.setObjectName("horizontalLayout_30")
        self.frame_36 = QtWidgets.QFrame(self.frame_34)
        self.frame_36.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_36.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_36.setObjectName("frame_36")
        self.horizontalLayout_29 = QtWidgets.QHBoxLayout(self.frame_36)
        self.horizontalLayout_29.setObjectName("horizontalLayout_29")
        self.label_21 = QtWidgets.QLabel(self.frame_36)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_29.addWidget(self.label_21)
        self.L_current_output_dir = QtWidgets.QLabel(self.frame_36)
        self.L_current_output_dir.setText("")
        self.L_current_output_dir.setObjectName("L_current_output_dir")
        self.horizontalLayout_29.addWidget(self.L_current_output_dir)
        self.PB_open_output_folder = QtWidgets.QPushButton(self.frame_36)
        self.PB_open_output_folder.setObjectName("PB_open_output_folder")
        self.horizontalLayout_29.addWidget(self.PB_open_output_folder)
        self.horizontalLayout_29.setStretch(0, 5)
        self.horizontalLayout_29.setStretch(1, 5)
        self.horizontalLayout_29.setStretch(2, 1)
        self.horizontalLayout_30.addWidget(self.frame_36)
        self.frame_35 = QtWidgets.QFrame(self.frame_34)
        self.frame_35.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_35.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_35.setObjectName("frame_35")
        self.horizontalLayout_28 = QtWidgets.QHBoxLayout(self.frame_35)
        self.horizontalLayout_28.setObjectName("horizontalLayout_28")
        self.label_20 = QtWidgets.QLabel(self.frame_35)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_28.addWidget(self.label_20)
        self.LE_output_dir_2 = QtWidgets.QLineEdit(self.frame_35)
        self.LE_output_dir_2.setObjectName("LE_output_dir_2")
        self.horizontalLayout_28.addWidget(self.LE_output_dir_2)
        self.PB_output_browse_2 = QtWidgets.QPushButton(self.frame_35)
        self.PB_output_browse_2.setObjectName("PB_output_browse_2")
        self.horizontalLayout_28.addWidget(self.PB_output_browse_2)
        self.PB_output_apply_2 = QtWidgets.QPushButton(self.frame_35)
        self.PB_output_apply_2.setObjectName("PB_output_apply_2")
        self.horizontalLayout_28.addWidget(self.PB_output_apply_2)
        self.horizontalLayout_30.addWidget(self.frame_35)
        self.horizontalLayout_30.setStretch(0, 1)
        self.horizontalLayout_30.setStretch(1, 1)
        self.verticalLayout_11.addWidget(self.frame_34)
        self.frame_18 = QtWidgets.QFrame(self.page_3)
        self.frame_18.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_18.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_18.setObjectName("frame_18")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.frame_18)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.frame_20 = QtWidgets.QFrame(self.frame_18)
        self.frame_20.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_20.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_20.setObjectName("frame_20")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_20)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_9 = QtWidgets.QLabel(self.frame_20)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_16.addWidget(self.label_9)
        self.label_10 = QtWidgets.QLabel(self.frame_20)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_16.addWidget(self.label_10)
        self.label_16 = QtWidgets.QLabel(self.frame_20)
        self.label_16.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_16.setIndent(6)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_16.addWidget(self.label_16)
        self.horizontalLayout_14.addWidget(self.frame_20)
        self.frame_21 = QtWidgets.QFrame(self.frame_18)
        self.frame_21.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_21.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_21.setObjectName("frame_21")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame_21)
        self.horizontalLayout_17.setContentsMargins(12, 12, -1, -1)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.PB_back_1 = QtWidgets.QPushButton(self.frame_21)
        self.PB_back_1.setObjectName("PB_back_1")
        self.horizontalLayout_17.addWidget(self.PB_back_1)
        self.PB_SDD_correction = QtWidgets.QPushButton(self.frame_21)
        self.PB_SDD_correction.setObjectName("PB_SDD_correction")
        self.horizontalLayout_17.addWidget(self.PB_SDD_correction)
        self.horizontalLayout_14.addWidget(self.frame_21)
        self.verticalLayout_11.addWidget(self.frame_18)
        self.SW_main.addWidget(self.page_3)
        self.verticalLayout_10.addWidget(self.SW_main)
        Contour_plot.setCentralWidget(self.centralwidget)

        self.retranslateUi(Contour_plot)
        self.SW_main.setCurrentIndex(0)
        self.PB_export_temp_on.pressed.connect(self.PB_export_temp_off.toggle) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Contour_plot)

    def retranslateUi(self, Contour_plot):
        _translate = QtCore.QCoreApplication.translate
        Contour_plot.setWindowTitle(_translate("Contour_plot", "Contour Plot for 9A, PAL"))
        self.label.setText(_translate("Contour_plot", "SPEC log file"))
        self.PB_spec_log_browse.setText(_translate("Contour_plot", "Browse"))
        self.PB_spec_apply.setText(_translate("Contour_plot", "Apply"))
        self.label_2.setText(_translate("Contour_plot", ".dat file folder"))
        self.PB_dat_browse.setText(_translate("Contour_plot", "Browse"))
        self.PB_dat_apply.setText(_translate("Contour_plot", "Apply"))
        self.label_3.setText(_translate("Contour_plot", "Output folder"))
        self.PB_output_browse.setText(_translate("Contour_plot", "Browse"))
        self.PB_output_apply.setText(_translate("Contour_plot", "Apply"))
        self.label_6.setText(_translate("Contour_plot", "Process File List"))
        self.label_4.setText(_translate("Contour_plot", "Last Update: FEB, 2025"))
        self.label_5.setText(_translate("Contour_plot", "hyungju@postech.ac.kr, yhkim26@postech.ac.kr"))
        self.label_11.setText(_translate("Contour_plot", "9A Beamline, PAL"))
        self.PB_reset.setText(_translate("Contour_plot", "Reset"))
        self.PB_next_1.setText(_translate("Contour_plot", "Next"))
        self.label_13.setText(_translate("Contour_plot", "Temperature Signal Out Correction"))
        self.PB_reset_temp.setText(_translate("Contour_plot", "Reset All"))
        self.PB_reset_adjust_temp.setText(_translate("Contour_plot", "Reset"))
        self.PB_apply_temp.setText(_translate("Contour_plot", "Apply"))
        self.label_14.setText(_translate("Contour_plot", "Raw"))
        self.label_15.setText(_translate("Contour_plot", "Corrected"))
        self.PB_apply_corrected_temp.setText(_translate("Contour_plot", "Apply Corrected Data"))
        self.label_7.setText(_translate("Contour_plot", "Last Update: FEB, 2025"))
        self.label_8.setText(_translate("Contour_plot", "hyungju@postech.ac.kr, yhkim26@postech.ac.kr"))
        self.label_12.setText(_translate("Contour_plot", "9A Beamline, PAL"))
        self.PB_back_0.setText(_translate("Contour_plot", "Back"))
        self.PB_next_2.setText(_translate("Contour_plot", "Next"))
        self.L_current_series.setText(_translate("Contour_plot", "Current Series"))
        self.PB_contour_setting.setText(_translate("Contour_plot", "Contour Setting"))
        self.PB_export.setText(_translate("Contour_plot", "Export"))
        self.label_18.setText(_translate("Contour_plot", "Export Index - Temp"))
        self.PB_export_temp_on.setText(_translate("Contour_plot", "O"))
        self.PB_export_temp_off.setText(_translate("Contour_plot", "X"))
        self.label_19.setText(_translate("Contour_plot", "Export q - intensity"))
        self.PB_export_data_on.setText(_translate("Contour_plot", "O"))
        self.PB_export_data_off.setText(_translate("Contour_plot", "X"))
        self.label_21.setText(_translate("Contour_plot", "Current Output Folder: "))
        self.PB_open_output_folder.setText(_translate("Contour_plot", "Open Output Folder"))
        self.label_20.setText(_translate("Contour_plot", "Output"))
        self.PB_output_browse_2.setText(_translate("Contour_plot", "Browse"))
        self.PB_output_apply_2.setText(_translate("Contour_plot", "Apply"))
        self.label_9.setText(_translate("Contour_plot", "Last Update: FEB, 2025"))
        self.label_10.setText(_translate("Contour_plot", "hyungju@postech.ac.kr, yhkim26@postech.ac.kr"))
        self.label_16.setText(_translate("Contour_plot", "9A Beamline, PAL"))
        self.PB_back_1.setText(_translate("Contour_plot", "Back"))
        self.PB_SDD_correction.setText(_translate("Contour_plot", "SDD Correction"))
