from PyQt5 import QtWidgets, QtCore, QtGui
from asset.contour_storage import PARAMS, FITTING_THRESHOLD

class SDDSettingsPage(QtCore.QObject):
    # 라벨 업데이트 매핑: 항상 PARAMS 기준으로 업데이트됨
    LABEL_MAPPINGS = {
        'L_sdd': ('original_sdd', '.4f', 'mm'),
        'L_pixel_size': ('pixel_size', '.4f', 'pixel'),
        'L_image_size_x': ('image_size_x', '', 'px'),
        'L_image_size_y': ('image_size_y', '', 'px'),
        'L_experiment_energy': ('experiment_energy', '.2f', 'keV'),
        'L_converted_energy': ('converted_energy', '.3f', 'keV'),
        'L_beam_center_x': ('beam_center_x', '.4f', 'px'),
        'L_beam_center_y': ('beam_center_y', '.4f', 'px')
    }


    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # Add fitting model to PARAMS if not present
        if 'fitting_model' not in PARAMS:
            PARAMS['fitting_model'] = 'gaussian'  # Default model

        # 초기 UI 설정
        self.setup_ui()
        self.connect_signals()
        self.update_labels()
        self.set_initial_q_format()
        self.setup_initial_fitting_model()
        
        # Threshold UI 초기화
        self.setup_threshold_ui()
        self.connect_threshold_signals()

    def setup_ui(self):
        """PARAMS의 현재 값으로 QLineEdit들을 초기화"""
        self.ui.LE_sdd.setText(str(PARAMS['original_sdd']))
        self.ui.LE_pixel_size.setText(str(PARAMS['pixel_size']))
        self.ui.LE_image_size_x.setText(str(PARAMS['image_size_x']))
        self.ui.LE_image_size_y.setText(str(PARAMS['image_size_y']))
        self.ui.LE_experiment_energy.setText(str(PARAMS['experiment_energy']))
        self.ui.LE_converted_energy.setText(str(PARAMS['converted_energy']))
        self.ui.LE_beam_center_x.setText(str(PARAMS['beam_center_x']))
        self.ui.LE_beam_center_y.setText(str(PARAMS['beam_center_y']))
    
    def setup_threshold_ui(self):
        """Initialize threshold UI elements"""
        # Threshold checkboxes and values
        self.threshold_mappings = {
            'q_threshold': {
                'checkbox': self.ui.CkB_q_threshold,
                'lineedit': self.ui.L_q_threshold,
                'label': self.ui.LE_q_threshold,
                'param_key': 'use_basic_q_threshold',
                'value_key': 'q_threshold'
            },
            'intensity_threshold': {
                'checkbox': self.ui.CkB_intensity_threshold,
                'lineedit': self.ui.LE_intensity_threshold,
                'label': self.ui.L_intensity_threshold,
                'param_key': 'use_intensity_threshold',
                'value_key': 'intensity_threshold'
            },
            'fwhm_q_threshold': {
                'checkbox': self.ui.CkB_fwhm_q_threshold,
                'lineedit': self.ui.LE_fwhm_q_threshold,
                'label': self.ui.L_fwhm_q_threshold,
                'param_key': 'use_fwhm_q_threshold',
                'value_key': 'fwhm_q_factor'
            },
            'fwhm_comparison': {
                'checkbox': self.ui.CkB_fwhm_comparison,
                'lineedit': self.ui.LE_fwhm_comparison,
                'label': self.ui.L_fwhm_comparison,
                'param_key': 'use_fwhm_comparison',
                'value_key': 'fwhm_change_threshold'
            }
        }

        # Initialize UI states from FITTING_THRESHOLD
        for mapping in self.threshold_mappings.values():
            checkbox = mapping['checkbox']
            lineedit = mapping['lineedit']
            label = mapping['label']
            param_key = mapping['param_key']
            value_key = mapping['value_key']

            # Set checkbox state
            checkbox.setChecked(FITTING_THRESHOLD[param_key])
            
            # Set initial value
            lineedit.setText(str(FITTING_THRESHOLD[value_key]))
            
            # Set initial state
            self.update_threshold_state(
                checkbox.isChecked(),
                lineedit,
                label,
                FITTING_THRESHOLD[value_key]
            )

    def connect_signals(self):
        """모든 시그널을 한 번씩만 연결"""
        param_mappings = {
            'LE_sdd': ('original_sdd', float),
            'LE_pixel_size': ('pixel_size', float),
            'LE_image_size_x': ('image_size_x', int),
            'LE_image_size_y': ('image_size_y', int),
            'LE_experiment_energy': ('experiment_energy', float),
            'LE_converted_energy': ('converted_energy', float),
            'LE_beam_center_x': ('beam_center_x', float),
            'LE_beam_center_y': ('beam_center_y', float)
        }
        for le_name, (param_name, convert_type) in param_mappings.items():
            line_edit = getattr(self.ui, le_name)
            line_edit.returnPressed.connect(
                self.create_update_callback(line_edit, param_name, convert_type)
            )

        # 콤보박스: q_format 변경 시 처리
        self.ui.CB_q_format.currentIndexChanged.connect(self.q_format_changed)
        self.ui.CB_fitting_model_sdd.currentIndexChanged.connect(self.fitting_model_changed)

    def connect_threshold_signals(self):
        """Connect signals for threshold UI elements"""
        for mapping in self.threshold_mappings.values():
            checkbox = mapping['checkbox']
            lineedit = mapping['lineedit']
            label = mapping['label']
            param_key = mapping['param_key']
            value_key = mapping['value_key']

            # Connect checkbox state change
            checkbox.stateChanged.connect(
                lambda state, le=lineedit, lb=label, vk=value_key:
                self.update_threshold_state(
                    bool(state),
                    le,
                    lb,
                    FITTING_THRESHOLD[vk]
                )
            )

            # Connect lineedit changes
            lineedit.editingFinished.connect(
                lambda le=lineedit, vk=value_key, lb=label:
                self.update_threshold_value(le, vk, lb)
            )

    def update_threshold_state(self, enabled, lineedit, label, current_value):
        """Update UI state for threshold controls"""
        lineedit.setEnabled(enabled)
        if enabled:
            label.setText(f"{current_value:.3f}")
        else:
            label.setText("Disabled")

    def update_threshold_value(self, lineedit, value_key, label):
        """Update threshold value in FITTING_THRESHOLD"""
        try:
            new_value = float(lineedit.text())
            if new_value < 0:
                raise ValueError("Value must be positive")
            FITTING_THRESHOLD[value_key] = new_value
            label.setText(f"{new_value:.3f}")
        except ValueError as e:
            # Restore previous value
            lineedit.setText(str(FITTING_THRESHOLD[value_key]))
            QtWidgets.QMessageBox.warning(
                self.main,
                "Invalid Input",
                f"Please enter a valid positive number: {str(e)}"
            )

    def create_update_callback(self, line_edit, param_name, convert_type):
        """각 QLineEdit에 대해 현재 값을 캡쳐한 콜백 함수 생성"""
        def callback():
            self.update_param(line_edit, param_name, convert_type)
        return callback

    def update_labels(self):
        """PARAMS의 현재 값으로 라벨(L_*)들을 업데이트"""
        for label_name, (param_name, format_spec, unit) in self.LABEL_MAPPINGS.items():
            label = getattr(self.ui, label_name)
            value = PARAMS[param_name]
            
            # converted_energy가 0인 경우 "None"으로 표시
            if param_name == 'converted_energy' and value == 0:
                label.setText("None")
                continue
            
            text = f"{value:{format_spec}}" if format_spec else str(value)
            if unit:
                text += f" {unit}"
            label.setText(text)

    def update_param(self, line_edit, param_name, convert_type):
        """
        사용자가 QLineEdit에 입력 후 엔터를 치면,
        그 값으로 PARAMS를 업데이트하고 라벨만 새로 갱신
        """
        text = line_edit.text().strip()
        try:
            value = convert_type(text)
            PARAMS[param_name] = value
            self.update_labels()
        except ValueError:
            line_edit.setText(str(PARAMS[param_name]))
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Please enter a valid {convert_type.__name__} for {param_name}"
            )

    def q_format_changed(self, index):
        """
        콤보박스 선택에 따라 q_format을 업데이트하고,
        그에 따라 converted_energy도 갱신
        """
        PARAMS['q_format'] = 'q' if index == 0 else 'CuKalpha'
        new_converted_energy = 0 if index == 0 else 8.042
        PARAMS['converted_energy'] = new_converted_energy
        self.ui.LE_converted_energy.setText(str(new_converted_energy))
        self.update_labels()

    def set_initial_q_format(self):
        """PARAMS의 q_format 값에 따라 콤보박스의 초기 인덱스를 설정"""
        if PARAMS.get('q_format', 'CuKalpha') == 'CuKalpha':
            self.ui.CB_q_format.setCurrentIndex(1)
        else:
            self.ui.CB_q_format.setCurrentIndex(0)

    def setup_initial_fitting_model(self):
        """Set initial fitting model based on PARAMS"""
        model_index = {
            'gaussian': 0,
            'lorentzian': 1,
            'voigt': 2
        }.get(PARAMS.get('fitting_model', 'gaussian'), 0)
        self.ui.CB_fitting_model_sdd.setCurrentIndex(model_index)


    def fitting_model_changed(self, index):
        """Update fitting model in PARAMS when combobox selection changes"""
        models = ['gaussian', 'lorentzian', 'voigt']
        PARAMS['fitting_model'] = models[index]