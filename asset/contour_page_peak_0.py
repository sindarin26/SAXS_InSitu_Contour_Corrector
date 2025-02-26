#asset/contour_page_peak_0.py

from PyQt5 import QtWidgets, QtCore, QtGui
from asset.contour_storage import PARAMS, FITTING_THRESHOLD, DATA
import copy

class PeakSettingsPage(QtCore.QObject):
    """First page of peak export window for managing settings"""
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # Set fitting model if not present
        if 'fitting_model' not in PARAMS:
            PARAMS['fitting_model'] = 'gaussian'  # Default model

        # Setup UI
        self.setup_ui()
        self.connect_signals()
        self.setup_threshold_ui()
        self.connect_threshold_signals()

    def setup_ui(self):
        """Initialize UI elements"""
        # Initialize note field with current stored note (if any)
        self.ui.LE_note.setText(self.main.PEAK_EXTRACT_DATA.get('NOTE', ''))

        # Initialize fitting model combobox
        model_index = {
            'gaussian': 0,
            'lorentzian': 1,
            'voigt': 2
        }.get(PARAMS.get('fitting_model', 'gaussian'), 0)
        self.ui.CB_fitting_model_sdd.setCurrentIndex(model_index)

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
        """Connect signals for UI elements"""
        # Connect note application
        self.ui.PB_apply_note.clicked.connect(self.apply_note)
        self.ui.LE_note.returnPressed.connect(self.apply_note)

        # Connect fitting model changes
        self.ui.CB_fitting_model_sdd.currentIndexChanged.connect(self.fitting_model_changed)

        # Connect next button - disconnect from old behavior and connect to our new function
        try:
            self.ui.PB_next_1.clicked.disconnect()
        except TypeError:
            # No connections yet
            pass
            
        self.ui.PB_next_1.clicked.connect(self.on_start_tracking)

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
                lambda state, le=lineedit, lb=label, pk=param_key, vk=value_key:
                self.on_checkbox_state_changed(state, le, lb, pk, vk)
            )

            # Connect lineedit changes
            lineedit.editingFinished.connect(
                lambda le=lineedit, vk=value_key, lb=label, pk=param_key:
                self.on_value_changed(le, vk, lb, pk)
            )


    def on_checkbox_state_changed(self, state, lineedit, label, param_key, value_key):
        """Checkbox 상태가 변경되었을 때 호출되는 함수"""
        enabled = bool(state)
        
        # UI 상태 업데이트
        lineedit.setEnabled(enabled)
        
        # FITTING_THRESHOLD 업데이트 
        FITTING_THRESHOLD[param_key] = enabled
        
        # 라벨 업데이트
        if enabled:
            label.setText(f"{FITTING_THRESHOLD[value_key]:.3f}")
        else:
            label.setText("Disabled")
        
        print(f"DEBUG: Threshold {param_key} enabled={enabled}, value={FITTING_THRESHOLD[value_key]}")
        print(f"DEBUG: Current FITTING_THRESHOLD: {FITTING_THRESHOLD}")

    def on_value_changed(self, lineedit, value_key, label, param_key):
        """라인에디트 값이 변경되었을 때 호출되는 함수"""
        try:
            new_value = float(lineedit.text())
            if new_value < 0:
                raise ValueError("Value must be positive")
            
            # 값 업데이트
            FITTING_THRESHOLD[value_key] = new_value
            
            # 라벨 업데이트
            if FITTING_THRESHOLD[param_key]:  # 체크박스가 활성화된 경우에만
                label.setText(f"{new_value:.3f}")
            
            print(f"DEBUG: Threshold {value_key} updated to {new_value}")
            print(f"DEBUG: Current FITTING_THRESHOLD: {FITTING_THRESHOLD}")
        except ValueError as e:
            # 오류 발생 시 원래 값으로 복원
            lineedit.setText(str(FITTING_THRESHOLD[value_key]))
            QtWidgets.QMessageBox.warning(
                self.main,
                "Invalid Input",
                f"Please enter a valid positive number: {str(e)}"
            )


    def update_threshold_state(self, enabled, lineedit, label, current_value):
        """Update UI state for threshold controls and update FITTING_THRESHOLD"""
        lineedit.setEnabled(enabled)
        if enabled:
            label.setText(f"{current_value:.3f}")
        else:
            label.setText("Disabled")
            
        # 여기서 각 threshold_mappings의 항목을 확인해서
        # 현재 호출된 위젯과 연결된 param_key를 찾아 업데이트
        for mapping in self.threshold_mappings.values():
            if mapping['checkbox'] == lineedit or mapping['label'] == label:
                param_key = mapping['param_key']
                FITTING_THRESHOLD[param_key] = enabled
                break



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

    def apply_note(self):
        """Apply note text to PEAK_EXTRACT_DATA"""
        note_text = self.ui.LE_note.text().strip()
        self.main.PEAK_EXTRACT_DATA['NOTE'] = note_text

    def fitting_model_changed(self, index):
        """Update fitting model in PARAMS when combobox selection changes"""
        models = ['gaussian', 'lorentzian', 'voigt']
        PARAMS['fitting_model'] = models[index]

    def on_start_tracking(self):
        """
        Start tracking button click handler
        Initializes peak data structure and moves to peak tracking page
        """
        # Check if contour data exists
        if not DATA.get('contour_data'):
            QtWidgets.QMessageBox.warning(
                self.main,
                "Warning",
                "No contour data available. Please process data first."
            )
            return
            
        # 현재 FITTING_THRESHOLD 값 출력
        print("\n=== DEBUG: Moving from page 0 to page 1 ===")
        print(f"DEBUG: Current PARAMS['fitting_model']: {PARAMS['fitting_model']}")
        print(f"DEBUG: Current FITTING_THRESHOLD: {FITTING_THRESHOLD}")
        print("=== Threshold details ===")
        for key, value in FITTING_THRESHOLD.items():
            print(f"DEBUG:   {key}: {value}")
        print("============================\n")
        
        # Initialize peak data with contour data
        self.main.init_peak_data(DATA['contour_data'])
        
        # Apply current note if any
        note_text = self.ui.LE_note.text().strip()
        self.main.PEAK_EXTRACT_DATA['NOTE'] = note_text
        
        # Move to peak tracking page
        self.ui.stackedWidget.setCurrentIndex(1)
        
        # Initialize peak tracking page with our contour data
        if hasattr(self.main, 'peak_tracking_page'):
            self.main.peak_tracking_page.initialize_peak_tracking(self.main.PEAK_EXTRACT_DATA['PEAK'])
