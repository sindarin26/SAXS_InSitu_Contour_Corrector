#asset/contour_page_peak_0.py

from PyQt5 import QtWidgets, QtCore
from asset.contour_storage import PARAMS, FITTING_THRESHOLD

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

        # Connect next button
        self.ui.PB_next_1.clicked.connect(lambda: self.main.ui.stackedWidget.setCurrentIndex(1))

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

    def apply_note(self):
        """Apply note text to PEAK_EXTRACT_DATA"""
        note_text = self.ui.LE_note.text().strip()
        self.main.PEAK_EXTRACT_DATA['NOTE'] = note_text

    def fitting_model_changed(self, index):
        """Update fitting model in PARAMS when combobox selection changes"""
        models = ['gaussian', 'lorentzian', 'voigt']
        PARAMS['fitting_model'] = models[index]
