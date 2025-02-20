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
        옵션을 그룹화하여 배치하고, 3열 레이아웃을 유지하면서 빈 공간을 삽입.
        """
        entries = []
        
        # PLOT_OPTIONS에서 순서에 맞춰 항목 가져오기
        options_order = [
            "figure_size", "figure_dpi",  None, # Figure 설정

            "figure_title_enable", "figure_title_text", None,  # Title 관련
            "contour_title_enable", "contour_title_text", None,
            "temp_title_enable", "temp_title_text", None,

            "font_label", "font_tick", "font_title",  # 폰트 설정

            "axes_label_size", "tick_label_size", "title_size",  # 라벨 크기 설정

            "contour_xlabel_enable", "contour_xlabel_text", None,  # X축 설정
            "contour_ylabel_enable", "contour_ylabel_text", None,  # Y축 설정
            "temp_xlabel_enable", "temp_xlabel_text", None,
            "temp_ylabel_enable", "temp_ylabel_text", None,

            "contour_grid", "contour_xlim", "global_ylim",  # Grid 및 범위 설정
            "temp_grid", "temp_xlim", None,

            "contour_cmap", "colorbar_label", "cbar_location",  # 컬러맵 설정
            "cbar_pad", "contour_levels", None,
            "contour_lower_percentile", "contour_upper_percentile", None,

            "wspace", "width_ratios", None  # 배치 관련
        ]

        for key in options_order:
            if key is None:
                entries.append((None, None, None))  # 빈칸 추가
            else:
                entries.append((key, PLOT_OPTIONS['graph_option'][key], 'graph_option'))

        col_count = 3  # 3열 유지
        row = 0
        col = 0
        for key, value, parent_key in entries:
            if key is None:
                # 빈칸을 넣을 때는 QWidget을 추가하여 유지
                empty_widget = QtWidgets.QWidget()
                self.grid_layout.addWidget(empty_widget, row, col)
            else:
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

        # 폰트 옵션: key가 "font_"로 시작하면 FontComboBox 사용
        elif key.startswith("font_"):
            input_widget = QtWidgets.QFontComboBox()
            # 일반적으로 사용되는 폰트만 표시
            input_widget.setFontFilters(QtWidgets.QFontComboBox.ScalableFonts)
            
            # writingSystem 설정으로 호환성 확보
            input_widget.setWritingSystem(QtGui.QFontDatabase.Latin)
            
            # 현재 폰트 설정
            try:
                current_font = QtGui.QFont(value)
                if current_font.exactMatch():
                    input_widget.setCurrentFont(current_font)
                else:
                    # 폰트가 없으면 기본값으로 Times New Roman 사용
                    input_widget.setCurrentFont(QtGui.QFont("Times New Roman"))
            except Exception:
                # 오류 발생 시 기본값 사용
                input_widget.setCurrentFont(QtGui.QFont("Times New Roman"))
            
            def font_changed():
                """폰트 변경 시 호출되는 콜백"""
                try:
                    font_family = input_widget.currentFont().family()
                    if parent_key:
                        PLOT_OPTIONS[parent_key][key] = font_family
                    else:
                        PLOT_OPTIONS[key] = font_family
                    if self.callback:
                        self.callback()
                except Exception as e:
                    print(f"Font change error: {str(e)}")
                    # 오류 발생 시 기본값으로 복구
                    input_widget.setCurrentFont(QtGui.QFont("Times New Roman"))
            
            input_widget.currentFontChanged.connect(lambda _: font_changed())


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

    # def on_font_changed(self, key, parent_key, widget):
    #     selected_font = widget.currentText()
    #     available_fonts = QtGui.QFontDatabase().families()
    #     if selected_font not in available_fonts:
    #         widget.blockSignals(True)
    #         widget.setCurrentText(widget.previous_font)
    #         widget.blockSignals(False)
    #         QtWidgets.QMessageBox.warning(self, "Font Warning",
    #                                       f"'{selected_font}' is not available on this system.\nReverting to previous font.")
    #         return
    #     widget.previous_font = selected_font
    #     if parent_key:
    #         PLOT_OPTIONS[parent_key][key] = selected_font
    #     else:
    #         PLOT_OPTIONS[key] = selected_font
    #     if self.callback:
    #         self.callback()