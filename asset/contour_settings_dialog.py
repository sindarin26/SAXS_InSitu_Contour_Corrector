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

        self.field_widgets = {}
        
        # 폰트 설정
        font = QtGui.QFont("Segoe UI", 12)
        self.setFont(font)
        
        # 메인 레이아웃에 스크롤 영역 추가 (옵션이 많을 경우 대비)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # 이미지 크기 정보 섹션 추가 (스크롤 영역 외부에)
        self.image_size_group = self.create_image_size_info_section()
        main_layout.addWidget(self.image_size_group)
        
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
        
        # 원본 옵션의 깊은 복사 생성
        import copy
        self.temp_options = copy.deepcopy(PLOT_OPTIONS['graph_option'])
        
        self.build_fields()
        
        # 이미지 크기 정보 초기 업데이트
        self.update_image_size_info()
        
        # 창 크기 설정
        self.resize(800, 1200)

    def create_image_size_info_section(self):
        """이미지 크기 정보를 표시하는 섹션 생성"""
        group_box = QtWidgets.QGroupBox("Export Image Size")
        grid_layout = QtWidgets.QGridLayout(group_box)
        grid_layout.setSpacing(10)
        
        # 그룹박스 제목 폰트 설정
        title_font = QtGui.QFont("Segoe UI", 12)
        title_font.setBold(True)
        group_box.setFont(title_font)
        
        # 인치 단위 레이블
        self.size_inches_label = QtWidgets.QLabel("Size (inches): --")
        self.size_inches_label.setFont(QtGui.QFont("Segoe UI", 12))
        grid_layout.addWidget(self.size_inches_label, 0, 0)
        
        # 픽셀 단위 레이블
        self.size_pixels_label = QtWidgets.QLabel("Size (pixels): --")
        self.size_pixels_label.setFont(QtGui.QFont("Segoe UI", 12))
        grid_layout.addWidget(self.size_pixels_label, 0, 1)
        
        return group_box
        
    def update_image_size_info(self):
        """현재 설정값에 따른 이미지 크기 정보 업데이트"""
        # 현재 figure_size와 dpi 값 가져오기
        fig_size = self.temp_options.get("figure_size", (10, 8))
        dpi = self.temp_options.get("figure_dpi", 100)
        
        # 인치 단위 크기
        width_inches, height_inches = fig_size
        self.size_inches_label.setText(f"Size (inches): {width_inches:.2f} x {height_inches:.2f}")
        
        # 픽셀 단위 크기
        width_pixels = int(width_inches * dpi)
        height_pixels = int(height_inches * dpi)
        self.size_pixels_label.setText(f"Size (pixels): {width_pixels} x {height_pixels}")
        
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
                "width_ratios", "wspace"
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
                "contour_xlim", "temp_xlim",
                "contour_xticks_count", "contour_xticks_interval",
                "temp_xticks_count", "temp_xticks_interval"
            ],
            "Y Axis": [
                "contour_ylabel_enable", "contour_ylabel_text",
                "temp_ylabel_enable", "temp_ylabel_text",
                "global_ylim",
                "contour_yticks_count", "contour_yticks_interval"
            ],
            "Grid": [
                "contour_grid", "temp_grid"
            ],
            "Legend": [
                "colorbar_label", "cbar_location",
                "cbar_pad"
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

    def store_cbar_location_change(self, key, widget):
        """컬러바 위치 콤보박스 값 변경을 임시 저장하고 legend 옵션도 같이 설정"""
        text = widget.currentText()
        
        # 'None'이 선택되면 None으로 저장하고 legend는 False로 설정
        if text == "None":
            self.temp_options[key] = None
            PLOT_OPTIONS['legend'] = False
        else:
            # 그 외는 선택된 텍스트 그대로 저장하고 legend는 True로 설정
            self.temp_options[key] = text
            PLOT_OPTIONS['legend'] = True

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

        # 위젯 저장을 위한 딕셔너리
        field_info = {'type': None, 'widgets': []}

        if key == "cbar_location":
            input_widget = QtWidgets.QComboBox()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            options = ["None", "left"]
            input_widget.addItems(options)
            
            # 현재 값 설정
            if value is None or (value == "None") or not PLOT_OPTIONS.get('legend', False):
                input_widget.setCurrentText("None")
            else:
                current_index = input_widget.findText(value) if value else -1
                if current_index >= 0:
                    input_widget.setCurrentIndex(current_index)
                else:
                    input_widget.setCurrentText("left")
            
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, w=input_widget: 
                self.store_cbar_location_change(k, w)
            )
            
            field_info['type'] = 'combo'
            field_info['widgets'].append(input_widget)
            
        elif key in ("contour_xlim", "global_ylim", "temp_xlim"):
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
            
            field_info['type'] = 'range'
            field_info['widgets'] = [le_min, le_max]
            
            return widget

        # 나머지 위젯 유형은 유사한 방식으로 field_info 업데이트
        elif key.startswith("font_"):
            # 폰트 위젯
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
            
            field_info['type'] = 'font'
            field_info['widgets'].append(input_widget)
            
        elif isinstance(value, bool):
            # 불리언 콤보박스
            input_widget = QtWidgets.QComboBox()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.addItems(["True", "False"])
            input_widget.setCurrentText(str(value))
            input_widget.currentIndexChanged.connect(
                lambda idx, k=key, pk=parent_key, w=input_widget: 
                self.store_combo_change(k, pk, w, is_bool=True)
            )
            
            field_info['type'] = 'bool_combo'
            field_info['widgets'].append(input_widget)
            
        elif key == "contour_cmap":
            # 컬러맵 콤보박스
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
            
            field_info['type'] = 'combo'
            field_info['widgets'].append(input_widget)
            
        elif isinstance(value, (tuple, list)):
            # 리스트/튜플 입력
            input_widget = QtWidgets.QWidget()
            hlayout = QtWidgets.QHBoxLayout(input_widget)
            hlayout.setContentsMargins(0, 0, 0, 0)
            
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
                
            field_info['type'] = 'list'
            field_info['widgets'] = edits
            
        elif isinstance(value, (int, float)) or value is None:
            # 숫자 입력
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            if value is not None:
                input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
            field_info['type'] = 'number'
            field_info['widgets'].append(input_widget)
            
        elif isinstance(value, str):
            # 문자열 입력
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.setText(value)
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
            field_info['type'] = 'text'
            field_info['widgets'].append(input_widget)
            
        else:
            # 기본 입력
            input_widget = QtWidgets.QLineEdit()
            input_widget.setFont(QtGui.QFont("Segoe UI", 12))
            input_widget.setText(str(value))
            input_widget.editingFinished.connect(
                lambda k=key, pk=parent_key, w=input_widget: 
                self.store_lineedit_change(k, pk, w)
            )
            
            field_info['type'] = 'text'
            field_info['widgets'].append(input_widget)
        
        layout.addWidget(input_widget)

        self.field_widgets[key] = field_info

        return widget

    # 나머지 메서드는 그대로 유지
    def store_combo_change(self, key, parent_key, widget, is_bool=False):
        """콤보박스 값 변경을 임시 저장"""
        # 기존 코드 유지
        text = widget.currentText()
        new_value = True if is_bool and text == "True" else (False if is_bool else text)
        if parent_key:
            self.temp_options[key] = new_value
        else:
            self.temp_options[key] = new_value
            
        # 크기 관련 설정이 변경된 경우 이미지 크기 정보 업데이트
        if key in ["figure_size", "figure_dpi"]:
            self.update_image_size_info()

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
        """Apply all changes and call callback"""
        # Validate all changes
        invalid_fields = self.validate_changes()
        
        if invalid_fields:
            # Show error message for invalid values and revert those fields
            error_msg = "Invalid values were entered in the following fields:\n\n"
            for field, reason in invalid_fields:
                error_msg += f"- {field}: {reason}\n"
            error_msg += "\nThese fields will be reset to their original values."
            
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Input",
                error_msg
            )
            
            # Restore invalid fields to original values and update UI
            for field, _ in invalid_fields:
                original_value = PLOT_OPTIONS['graph_option'][field]
                self.temp_options[field] = original_value
                
                # Update UI widget (depends on field type)
                self.update_field_ui(field, original_value)
        
        # Apply validated values
        for key, value in self.temp_options.items():
            PLOT_OPTIONS['graph_option'][key] = value
        
        # 이미지 크기 정보 업데이트
        self.update_image_size_info()
            
        # Call callback (update plot)
        if self.callback:
            self.callback()
            
    def update_field_ui(self, field, value):
        """
        필드 위젯을 원래 값으로 업데이트
        
        Parameters:
        -----------
        field : str
            업데이트할 필드 이름
        value : any
            원래 값으로 설정할 값
        """
        # 필드 위젯이 저장되어 있는지 확인
        if field not in self.field_widgets:
            return
        
        field_info = self.field_widgets[field]
        field_type = field_info.get('type')
        widgets = field_info.get('widgets', [])
        
        if not widgets:
            return
        
        # 필드 유형에 따라 다른 업데이트 로직 적용
        if field_type == 'range':
            # 범위 필드 (min, max)
            if isinstance(value, (tuple, list)) and len(value) == 2 and len(widgets) >= 2:
                min_val, max_val = value
                widgets[0].setText("" if min_val is None else str(min_val))
                widgets[1].setText("" if max_val is None else str(max_val))
            else:
                # 값이 유효하지 않으면 빈 문자열로 설정
                widgets[0].setText("")
                widgets[1].setText("")
        
        elif field_type == 'list':
            # 리스트/튜플 필드
            if isinstance(value, (tuple, list)):
                for i, widget in enumerate(widgets):
                    if i < len(value):
                        widget.setText(str(value[i]))
                    else:
                        widget.setText("")
            else:
                # 값이 유효하지 않으면 모두 빈 문자열로 설정
                for widget in widgets:
                    widget.setText("")
        
        elif field_type == 'combo' or field_type == 'bool_combo':
            # 콤보박스 (일반 또는 불리언)
            combo = widgets[0]
            
            if field_type == 'bool_combo':
                # 불리언 콤보박스
                combo.setCurrentText(str(value))
            else:
                # 일반 콤보박스
                # 해당 항목이 있으면 선택, 없으면 첫 번째 항목 선택
                idx = combo.findText(str(value) if value is not None else "")
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentIndex(0)
        
        elif field_type == 'font':
            # 폰트 콤보박스
            font_combo = widgets[0]
            try:
                current_font = QtGui.QFont(value)
                if current_font.exactMatch():
                    font_combo.setCurrentFont(current_font)
                else:
                    # 폰트가 없으면 기본값으로 Segoe UI 사용
                    font_combo.setCurrentFont(QtGui.QFont("Segoe UI"))
            except Exception:
                # 오류 발생 시 기본값 사용
                font_combo.setCurrentFont(QtGui.QFont("Segoe UI"))
        
        elif field_type in ('number', 'text'):
            # 숫자 또는 텍스트 입력 필드
            line_edit = widgets[0]
            if value is None:
                line_edit.setText("")
            else:
                line_edit.setText(str(value))

    def validate_changes(self):
        """
        Validate input values and return invalid fields
        
        Returns:
            list: List of invalid fields in [(field_name, error_reason), ...] format
        """
        invalid_fields = []
        
        # Validate ranges
        range_fields = ['contour_xlim', 'global_ylim', 'temp_xlim']
        for field in range_fields:
            if field in self.temp_options and self.temp_options[field] is not None:
                value = self.temp_options[field]
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    invalid_fields.append((field, "Range must be in (min, max) format"))
                elif value[0] >= value[1]:
                    invalid_fields.append((field, f"Minimum value ({value[0]}) must be less than maximum value ({value[1]})"))
        
        # Validate numeric values
        numeric_fields = {
            'figure_dpi': (50, 1200, "DPI must be between 50-1200"),
            'axes_label_size': (1, 72, "Label size must be between 1-72"),
            'tick_label_size': (1, 72, "Tick label size must be between 1-72"),
            'title_size': (1, 72, "Title size must be between 1-72"),
            'contour_levels': (2, 1000, "Contour levels must be between 2-1000"),
            'contour_lower_percentile': (0, 50, "Lower percentile must be between 0-50"),
            'contour_upper_percentile': (50, 100, "Upper percentile must be between 50-100"),
            'cbar_pad': (0, 1, "Colorbar padding must be between 0-1"),
            'wspace': (0, 1, "Subplot spacing must be between 0-1"),
            'contour_xticks_count': (2, 50, "X-axis tick count must be between 2-50"),
            'contour_yticks_count': (2, 50, "Y-axis tick count must be between 2-50"),
            'temp_xticks_count': (2, 50, "Temperature plot X-axis tick count must be between 2-50"),
            'contour_xticks_interval': (0.01, 1000, "X-axis tick interval must be between 0.01-1000"),
            'contour_yticks_interval': (0.01, 1000, "Y-axis tick interval must be between 0.01-1000"),
            'temp_xticks_interval': (0.01, 1000, "Temperature plot X-axis tick interval must be between 0.01-1000"),
        }
        
        for field, (min_val, max_val, error_msg) in numeric_fields.items():
            if field in self.temp_options and self.temp_options[field] is not None:
                value = self.temp_options[field]
                if not isinstance(value, (int, float)):
                    invalid_fields.append((field, "Must be a numeric value"))
                elif value < min_val or value > max_val:
                    invalid_fields.append((field, error_msg))
        
        # Validate lists/tuples
        tuple_fields = {
            'figure_size': (2, "Size must be in (width, height) format"),
            'width_ratios': (2, "Width ratios must have at least 2 values")
        }
        
        for field, (min_len, error_msg) in tuple_fields.items():
            if field in self.temp_options and self.temp_options[field] is not None:
                value = self.temp_options[field]
                if not isinstance(value, (list, tuple)):
                    invalid_fields.append((field, "Must be a list format"))
                elif len(value) < min_len:
                    invalid_fields.append((field, error_msg))
        
        return invalid_fields
    
    def keyPressEvent(self, event):
        """Enter 키를 누르면 변경사항 적용"""
        if event.key() == QtCore.Qt.Key_Return or event.key() == QtCore.Qt.Key_Enter:
            self.apply_all_changes()
        else:
            super().keyPressEvent(event)