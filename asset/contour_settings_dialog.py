# asset/contour_settings_dialog.py
import re
from PyQt5 import QtWidgets, QtCore, QtGui
from asset.contour_storage import PLOT_OPTIONS

class ContourSettingsDialog(QtWidgets.QDialog):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contour Plot Settings")
        self.callback = callback  # 옵션이 변경될 때마다 호출할 콜백 (예: replot)
        
        # 메인 레이아웃에 스크롤 영역 추가 (옵션이 많을 경우 대비)
        main_layout = QtWidgets.QVBoxLayout(self)
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        container = QtWidgets.QWidget()
        scroll_area.setWidget(container)
        
        self.grid_layout = QtWidgets.QGridLayout(container)
        # 4열 배치: 각 셀에 옵션 위젯(레이블+입력 위젯)
        self.grid_layout.setSpacing(10)
        
        self.build_fields()

    def build_fields(self):
        """
        PLOT_OPTIONS의 top-level (graph_option 제외)와 graph_option 내부의 모든 옵션을
        (키, 값, parent_key) 튜플의 리스트로 모은 후 4열 그리드에 추가합니다.
        """
        entries = []
        # top-level 옵션 (graph_option 제외)
        for key, value in PLOT_OPTIONS.items():
            if key == 'graph_option':
                continue
            entries.append( (key, value, None) )
        # graph_option 내부 옵션
        for key, value in PLOT_OPTIONS['graph_option'].items():
            entries.append( (key, value, 'graph_option') )
        
        col_count = 4
        row = 0
        col = 0
        for key, value, parent_key in entries:
            field_widget = self.create_field_widget(key, value, parent_key)
            self.grid_layout.addWidget(field_widget, row, col)
            col += 1
            if col >= col_count:
                col = 0
                row += 1

    def create_field_widget(self, key, value, parent_key):
        """
        옵션의 키와 현재 값, 상위 키(parent_key: 없으면 top-level, 있으면 'graph_option')를 받아
        레이블과 알맞은 입력 위젯(QLineEdit, QComboBox 등)을 포함하는 QWidget을 생성합니다.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        label = QtWidgets.QLabel(key)
        layout.addWidget(label)
        
        # xlim, ylim은 별도의 두 입력창으로 분리 (하나라도 빈칸이면 auto)
        if key in ("contour_xlim", "global_ylim"):
            container = QtWidgets.QWidget()
            h_layout = QtWidgets.QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            # min 입력창
            le_min = QtWidgets.QLineEdit()
            le_min.setMaximumWidth(50)
            # max 입력창
            le_max = QtWidgets.QLineEdit()
            le_max.setMaximumWidth(50)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                min_val, max_val = value
                le_min.setText("" if min_val is None else str(min_val))
                le_max.setText("" if max_val is None else str(max_val))
            else:
                le_min.setText("")
                le_max.setText("")
            le_min.editingFinished.connect(
                lambda k=key, pk=parent_key, lmin=le_min, lmax=le_max: self.on_lim_changed(k, pk, lmin, lmax)
            )
            le_max.editingFinished.connect(
                lambda k=key, pk=parent_key, lmin=le_min, lmax=le_max: self.on_lim_changed(k, pk, lmin, lmax)
            )
            h_layout.addWidget(QtWidgets.QLabel("Min"))
            h_layout.addWidget(le_min)
            h_layout.addWidget(QtWidgets.QLabel("Max"))
            h_layout.addWidget(le_max)
            layout.addWidget(container)
            return widget

        # 폰트 옵션: key가 "font_"로 시작하면 콤보박스 (현재 시스템에 설치된 폰트 목록도 확인)
        elif key.startswith("font_"):
            input_widget = QtWidgets.QComboBox()
            # 기본적으로 Times New Roman, Segoe UI 제공 (추후 추가하기 쉽게)
            input_widget.addItems(["Times New Roman", "Segoe UI"])
            current_font = value if value in ["Times New Roman", "Segoe UI"] else "Times New Roman"
            input_widget.setCurrentText(current_font)
            input_widget.previous_font = current_font  # 이전 선택 폰트 저장
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: self.on_font_changed(k, pk, w)
            )
        # 불리언 값인 경우
        elif isinstance(value, bool):
            input_widget = QtWidgets.QComboBox()
            input_widget.addItems(["True", "False"])
            input_widget.setCurrentText(str(value))
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: self.on_combo_changed(k, pk, w, is_bool=True)
            )
        # 컬러맵 옵션: "contour_cmap"는 미리 정의된 인기 컬러맵 목록을 콤보박스로 표시
        elif key == "contour_cmap":
            input_widget = QtWidgets.QComboBox()
            colormaps = ["inferno", "viridis", "plasma", "magma", "jet"]
            input_widget.addItems(colormaps)
            current_val = value if value in colormaps else "inferno"
            input_widget.setCurrentText(current_val)
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: self.on_combo_changed(k, pk, w)
            )
        # tuple 또는 list (예: figure_size, width_ratios 등)인 경우
        elif isinstance(value, (tuple, list)):
            input_widget = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout(input_widget)
            hlayout.setContentsMargins(0, 0, 0, 0)
            self._edits = []  # 각 요소에 대한 QLineEdit들을 저장할 리스트
            for item in value:
                le = QtWidgets.QLineEdit()
                le.setMaximumWidth(50)
                le.setText(str(item))
                le.editingFinished.connect(
                    lambda k=key, pk=parent_key, edits=self._edits: self.on_list_changed(k, pk, edits)
                )
                hlayout.addWidget(le)
                self._edits.append(le)
        # 숫자(int, float)나 None (빈칸이면 None 적용)
        elif isinstance(value, (int, float)) or value is None:
            input_widget = QtWidgets.QLineEdit()
            if value is not None:
                input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: self.on_lineedit_changed(k, pk, w)
            )
        # 문자열 (일반 텍스트, _text 항목 등)
        elif isinstance(value, str):
            input_widget = QtWidgets.QLineEdit()
            input_widget.setText(value)
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: self.on_lineedit_changed(k, pk, w)
            )
        else:
            # 기본 처리: QLineEdit로 문자열 변환
            input_widget = QtWidgets.QLineEdit()
            input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: self.on_lineedit_changed(k, pk, w)
            )
            
        layout.addWidget(input_widget)
        return widget

    def on_combo_changed(self, key, parent_key, widget, is_bool=False):
        text = widget.currentText()
        new_value = True if is_bool and text == "True" else (False if is_bool else text)
        if parent_key:
            PLOT_OPTIONS[parent_key][key] = new_value
        else:
            PLOT_OPTIONS[key] = new_value
        if self.callback:
            self.callback()

    def on_lineedit_changed(self, key, parent_key, widget):
        text = widget.text().strip()
        # 빈 문자열이면 None으로 처리
        if text == "":
            new_value = None
        else:
            # 숫자 형식이면 int 또는 float로 변환 시도, 아니면 문자열 그대로
            if re.match(r'^-?\d+$', text):
                new_value = int(text)
            else:
                try:
                    new_value = float(text)
                except ValueError:
                    new_value = text
        if parent_key:
            PLOT_OPTIONS[parent_key][key] = new_value
        else:
            PLOT_OPTIONS[key] = new_value
        if self.callback:
            self.callback()

    def on_list_changed(self, key, parent_key, edits):
        new_list = []
        for le in edits:
            txt = le.text().strip()
            if txt == "":
                continue
            else:
                if re.match(r'^-?\d+$', txt):
                    new_list.append(int(txt))
                else:
                    try:
                        new_list.append(float(txt))
                    except ValueError:
                        new_list.append(txt)
        new_value = new_list if new_list else None
        if parent_key:
            PLOT_OPTIONS[parent_key][key] = new_value
        else:
            PLOT_OPTIONS[key] = new_value
        if self.callback:
            self.callback()

    def on_lim_changed(self, key, parent_key, le_min, le_max):
        txt_min = le_min.text().strip()
        txt_max = le_max.text().strip()
        # 하나라도 빈칸이면 None (자동)
        if txt_min == "" or txt_max == "":
            new_value = None
        else:
            try:
                new_value = (float(txt_min), float(txt_max))
            except ValueError:
                new_value = None
        if parent_key:
            PLOT_OPTIONS[parent_key][key] = new_value
        else:
            PLOT_OPTIONS[key] = new_value
        if self.callback:
            self.callback()

    def on_font_changed(self, key, parent_key, widget):
        selected_font = widget.currentText()
        available_fonts = QtGui.QFontDatabase().families()
        if selected_font not in available_fonts:
            widget.blockSignals(True)
            widget.setCurrentText(widget.previous_font)
            widget.blockSignals(False)
            QtWidgets.QMessageBox.warning(self, "Font Warning",
                                          f"'{selected_font}' is not available on this system.\nReverting to previous font.")
            return
        widget.previous_font = selected_font
        if parent_key:
            PLOT_OPTIONS[parent_key][key] = selected_font
        else:
            PLOT_OPTIONS[key] = selected_font
        if self.callback:
            self.callback()