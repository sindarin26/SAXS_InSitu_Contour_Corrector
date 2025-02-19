from PyQt5 import QtWidgets, QtCore
from asset.contour_storage import PARAMS

class SDDSettingsPage(QtCore.QObject):
    # 라벨 업데이트 매핑: 항상 PARAMS 기준으로 업데이트됨
    LABEL_MAPPINGS = {
        'L_sdd': ('original_sdd', '.4f', 'mm'),
        'L_pixel_size': ('pixel_size', '.4f', 'mm'),
        'L_image_size_x': ('image_size_x', '', ''),
        'L_image_size_y': ('image_size_y', '', ''),
        'L_experiment_energy': ('experiment_energy', '.2f', 'keV'),
        'L_converted_energy': ('converted_energy', '.3f', 'keV'),
        'L_beam_center_x': ('beam_center_x', '.4f', 'pixel'),
        'L_beam_center_y': ('beam_center_y', '.4f', 'pixel')
    }

    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # 초기 UI 설정: PARAMS의 현재 값으로 채움
        self.setup_ui()
        self.connect_signals()
        self.update_labels()
        self.set_initial_q_format()

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

    def connect_signals(self):
        """모든 시그널을 한 번씩만 연결 (PB_reset은 제거)"""
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
            text = f"{value:{format_spec}}" if format_spec else str(value)
            if unit:
                text += f" {unit}"
            label.setText(text)

    def update_param(self, line_edit, param_name, convert_type):
        """
        사용자가 QLineEdit에 입력 후 엔터를 치면,
        그 값으로 PARAMS를 업데이트하고 라벨만 새로 갱신합니다.
        (QLineEdit의 값은 사용자가 입력한 그대로 유지됩니다.)
        """
        text = line_edit.text().strip()
        try:
            value = convert_type(text)
            print(f"[DEBUG] update_param: {param_name} old value={PARAMS[param_name]}, new value={value}")
            PARAMS[param_name] = value
            self.update_labels()
            print(f"[DEBUG] PARAMS after update: {PARAMS}")
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
        그에 따라 converted_energy도 갱신합니다.
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