# asset/contour_settings_dialog.py
import re
from PyQt5 import QtWidgets, QtCore, QtGui
from asset.contour_storage import PLOT_OPTIONS
from asset.page_asset import LoadingDialog

class ContourSettingsDialog(QtWidgets.QDialog):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Contour Plot Settings")
        
        # 도움말(?) 버튼 제거
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowContextHelpButtonHint)
        
        self.callback = callback  # 옵션이 변경될 때마다 호출할 콜백 (예: replot)
        
        # 변경된 옵션을 임시로 저장할 딕셔너리
        self.temp_options = {}
        
        # 폰트 설정
        font = QtGui.QFont("Segoe UI", 12)
        self.setFont(font)
        
        # 메인 레이아웃에 스크롤 영역 추가 (옵션이 많을 경우 대비)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        container = QtWidgets.QWidget()
        scroll_area.setWidget(container)
        
        # 수직 레이아웃으로 변경하여 섹션별로 그룹박스 추가
        self.container_layout = QtWidgets.QVBoxLayout(container)
        self.container_layout.setSpacing(15)
        
        # 적용 버튼 추가
        self.apply_button = QtWidgets.QPushButton("Apply Changes")
        self.apply_button.clicked.connect(self.apply_all_changes)
        main_layout.addWidget(self.apply_button)
        
        self.build_fields()
        
        # 원본 옵션의 깊은 복사 생성
        import copy
        self.temp_options = copy.deepcopy(PLOT_OPTIONS['graph_option'])
        
        # 창 크기 설정
        self.resize(800, 1200)

    def build_fields(self):
        """
        옵션을 섹션별로 그룹화하여 배치
        """
        # 섹션 정의: (섹션 이름, 포함할 옵션들)
        sections = {
            "Contour Settings": [
                "figure_size", "figure_dpi", 
                "contour_cmap", 
                "contour_levels",
                "contour_lower_percentile", "contour_upper_percentile",
                "width_ratios"
            ],
            "Title": [
                "figure_title_enable", "figure_title_text",
                "contour_title_enable", "contour_title_text",
                "temp_title_enable", "temp_title_text",
                "title_size"
            ],
            "Font": [
                "font_label", "font_tick", "font_title",
                "axes_label_size", "tick_label_size"
            ],
            "X Axis": [
                "contour_xlabel_enable", "contour_xlabel_text",
                "temp_xlabel_enable", "temp_xlabel_text",
                "contour_xlim", "temp_xlim"
            ],
            "Y Axis": [
                "contour_ylabel_enable", "contour_ylabel_text",
                "temp_ylabel_enable", "temp_ylabel_text",
                "global_ylim"
            ],
            "Grid": [
                "contour_grid", "temp_grid"
            ],
            "Legend": [
                "colorbar_label", "cbar_location",
                "cbar_pad", "wspace"
            ]
        }
        
        # 각 섹션별로 그룹박스 생성 및 옵션 추가
        for section_name, options in sections.items():
            group_box = self.create_section_group(section_name, options)
            self.container_layout.addWidget(group_box)
    
    def create_section_group(self, title, options):
        """
        섹션 제목과 옵션 목록을 받아 그룹박스를 생성
        """
        group_box = QtWidgets.QGroupBox(title)
        grid_layout = QtWidgets.QGridLayout(group_box)
        grid_layout.setSpacing(10)
        
        # 그룹박스 제목 폰트 설정 - Bold에서 일반으로 변경
        title_font = QtGui.QFont("Segoe UI", 12)  # Bold 제거
        group_box.setFont(title_font)
        
        # 옵션 배치 (2열로 배치)
        row = 0
        col = 0
        col_count = 2  # 2열 레이아웃
        
        for key in options:
            if key in PLOT_OPTIONS['graph_option']:
                field_widget = self.create_field_widget(key, PLOT_OPTIONS['graph_option'][key], 'graph_option')
                grid_layout.addWidget(field_widget, row, col)
                
                col += 1
                if col >= col_count:
                    col = 0
                    row += 1
        
        return group_box

    def create_field_widget(self, key, value, parent_key):
        """
        옵션의 키와 현재 값, 상위 키(parent_key: 없으면 top-level, 있으면 'graph_option')를 받아
        레이블과 알맞은 입력 위젯(QLineEdit, QComboBox 등)을 포함하는 QWidget을 생성합니다.
        """
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 옵션명을 읽기 쉽게 표시
        display_name = key.replace("_", " ").title()
        label = QtWidgets.QLabel(display_name)
        label.setFont(QtGui.QFont("Segoe UI", 12))
        layout.addWidget(label)
        
        # xlim, ylim은 별도의 두 입력창으로 분리 (하나라도 빈칸이면 auto)
        if key in ("contour_xlim", "global_ylim", "temp_xlim"):
            container = QtWidgets.QWidget()
            h_layout = QtWidgets.QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            # min 입력창
            le_min = QtWidgets.QLineEdit()
            le_min.setMaximumWidth(50)
            le_min.setFont(QtGui.QFont("Segoe UI", 12))
            # max 입력창
            le_max = QtWidgets.QLineEdit()
            le_max.setMaximumWidth(50)
            le_max.setFont(QtGui.QFont("Segoe UI", 12))
            
            if isinstance(value, (tuple, list)) and len(value) == 2:
                min_val, max_val = value
                le_min.setText("" if min_val is None else str(min_val))
                le_max.setText("" if max_val is None else str(max_val))
            else:
                le_min.setText("")
                le_max.setText("")
            
            # 값을 임시 저장하고 Enter 키 누를 때만 처리
            le_min.editingFinished.connect(
                lambda k=key, pk=parent_key, lmin=le_min, lmax=le_max: 
                self.store_lim_change(k, pk, lmin, lmax)
            )
            le_max.editingFinished.connect(
                lambda k=key, pk=parent_key, lmin=le_min, lmax=le_max: 
                self.store_lim_change(k, pk, lmin, lmax)
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
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
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
                    # 폰트가 없으면 기본값으로 Segoe UI 사용
                    input_widget.setCurrentFont(QtGui.QFont("Segoe UI"))
            except Exception:
                # 오류 발생 시 기본값 사용
                input_widget.setCurrentFont(QtGui.QFont("Segoe UI"))
            
            # 폰트 변경 시 임시 저장
            input_widget.currentFontChanged.connect(
                lambda font, k=key, pk=parent_key: 
                self.store_font_change(k, pk, font.family())
            )

        # 불리언 값인 경우
        elif isinstance(value, bool):
            input_widget = QtWidgets.QComboBox()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.addItems(["True", "False"])
            input_widget.setCurrentText(str(value))
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: 
                self.store_combo_change(k, pk, w, is_bool=True)
            )
            
        # 컬러맵 옵션: "contour_cmap"는 미리 정의된 인기 컬러맵 목록을 콤보박스로 표시
        elif key == "contour_cmap":
            input_widget = QtWidgets.QComboBox()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            colormaps = ["inferno", "viridis", "plasma", "magma", "jet", "hot", "cool", "rainbow"]
            input_widget.addItems(colormaps)
            current_val = value if value in colormaps else "inferno"
            input_widget.setCurrentText(current_val)
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: 
                self.store_combo_change(k, pk, w)
            )
            
        # tuple 또는 list (예: figure_size, width_ratios 등)인 경우
        elif isinstance(value, (tuple, list)):
            input_widget = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout(input_widget)
            hlayout.setContentsMargins(0, 0, 0, 0)
            
            # 이전 방식에서 self._edits를 인스턴스 변수로 사용했으나, 지역 변수로 변경
            edits = []
            for item in value:
                le = QtWidgets.QLineEdit()
                le.setFont(QtGui.QFont("Segoe UI", 12))
                le.setMaximumWidth(50)
                le.setText(str(item))
                hlayout.addWidget(le)
                edits.append(le)
            
            # 모든 QLineEdit을 저장하고 editingFinished 신호에 연결
            for le in edits:
                le.editingFinished.connect(
                    lambda k=key, pk=parent_key, e=edits: 
                    self.store_list_change(k, pk, e)
                )
                
        # 숫자(int, float)나 None (빈칸이면 None 적용)
        elif isinstance(value, (int, float)) or value is None:
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            if value is not None:
                input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
        # 문자열 (일반 텍스트, _text 항목 등)
        elif isinstance(value, str):
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.setText(value)
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
        else:
            # 기본 처리: QLineEdit로 문자열 변환
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
        layout.addWidget(input_widget)
        return widget

    # 나머지 메서드는 그대로 유지
    def store_combo_change(self, key, parent_key, widget, is_bool=False):
        """콤보박스 값 변경을 임시 저장"""
        text = widget.currentText()
        new_value = True if is_bool and text == "True" else (False if is_bool else text)
        if parent_key:
            self.temp_options[key] = new_value
        else:
            self.temp_options[key] = new_value

    def store_lineedit_change(self, key, parent_key, widget):
        """라인에디트 값 변경을 임시 저장"""
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
            self.temp_options[key] = new_value
        else:
            self.temp_options[key] = new_value

    def store_list_change(self, key, parent_key, edits):
        """리스트나 튜플 값 변경을 임시 저장"""
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
            self.temp_options[key] = new_value
        else:
            self.temp_options[key] = new_value

    def store_lim_change(self, key, parent_key, le_min, le_max):
        """limit 값 변경을 임시 저장"""
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
            self.temp_options[key] = new_value
        else:
            self.temp_options[key] = new_value

    def store_font_change(self, key, parent_key, font_family):
        """폰트 변경을 임시 저장"""
        if parent_key:
            self.temp_options[key] = font_family
        else:
            self.temp_options[key] = font_family

    def apply_all_changes(self):
        """모든 변경사항을 적용하고 콜백 호출"""
        # 임시 저장된 변경사항을 PLOT_OPTIONS에 적용
        for key, value in self.temp_options.items():
            PLOT_OPTIONS['graph_option'][key] = value
            
        # 콜백 호출 (플롯 업데이트)
        if self.callback:
            self.callback()

    def keyPressEvent(self, event):
        """Enter 키를 누르면 변경사항 적용"""
        if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
            self.apply_all_changes()
        else:
            super().keyPressEvent(event)