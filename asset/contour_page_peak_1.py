#asset/contour_page_peak_1.py
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from asset.contour_storage import PARAMS, FITTING_THRESHOLD
from asset.fitting_util import find_peak_extraction, run_automatic_tracking, plot_contour_extraction
from asset.contour_util_gui import QRangeCorrectionHelper
from asset.page_asset import LoadingDialog

class PeakTrackingPage(QtCore.QObject):
    """
    Page for tracking peaks in contour data. Includes q-range selection,
    automatic tracking, and peak adjustment capabilities.
    """
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui
        
        # State variables
        self.contour_data = None
        self.current_index = 0
        self.max_index = 0
        self.auto_tracking = False
        self.manual_adjust = False
        self.check_peak_range_mode = False  # 피크 범위 체크 모드 플래그 추가
        self.current_peak_name = None
        
        # UI elements from page 1 (index 1)
        self.QGV_qrange = self.ui.QGV_qrange
        self.L_current_status_1 = self.ui.L_current_status_1
        self.PB_back_0 = self.ui.PB_back_0
        self.PB_apply_qrange = self.ui.PB_apply_qrange
        
        # UI elements from page 2 (index 2)
        self.QGV_contour = self.ui.QGV_contour
        self.L_current_status_2 = self.ui.L_current_status_2
        self.PB_next = self.ui.PB_next
        self.PB_stop_auto_tracking = self.ui.PB_stop_auto_tracking
        self.PB_adjust_mode = self.ui.PB_adjust_mode
        self.PB_find_another_peak = self.ui.PB_find_another_peak
        self.PB_check_peak_range = self.ui.PB_check_peak_range  # 피크 범위 확인 버튼 추가
        
        # Set initial button text
        self.PB_adjust_mode.setText("Adjust Mode")
        self.PB_check_peak_range.setText("Check Peak Range")
        
        # Helper for q-range selection
        self.q_correction_helper = None
        
        # Canvas for contour visualization
        self.contour_canvas = None
        
        # Timer for peak selection
        self.selection_timer = None
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to handlers"""
        # Page 1 buttons
        self.PB_back_0.clicked.connect(self.on_back_to_settings)
        self.PB_apply_qrange.clicked.connect(self.on_apply_qrange)
        
        # Page 2 buttons
        self.PB_next.clicked.connect(self.on_next)
        self.PB_stop_auto_tracking.clicked.connect(self.on_stop_tracking)
        self.PB_adjust_mode.clicked.connect(self.on_enter_adjust_mode)
        self.PB_find_another_peak.clicked.connect(self.on_find_another_peak)
        self.PB_check_peak_range.clicked.connect(self.on_check_peak_range)  # 피크 범위 확인 버튼 핸들러 연결


    
    def initialize_peak_tracking(self, contour_data):
        """
        Initialize page with contour data and prepare for peak tracking
        
        Parameters:
            contour_data (dict): The contour data to analyze
        """
        self.contour_data = contour_data
        self.current_index = 0
        self.max_index = len(contour_data['Data']) - 1
        
        # Reset state
        self.auto_tracking = False
        self.manual_adjust = False
        self.current_peak_name = None
        
        # Setup q-range selection graph
        self.setup_q_range_graph()
        
        # Update status
        self.L_current_status_1.setText(
            f"Select q range for frame {self.current_index} / {self.max_index}"
        )
        
        # Set initial button states
        self.update_ui_state()
    
    def setup_q_range_graph(self, mode=None):
        """
        Setup QGV_qrange with current frame's data
        
        Parameters:
            mode (str): 'auto_tracking_error' 또는 'manual_adjust' - 어떤 모드에서 호출되었는지 표시
        """
        if not self.contour_data or self.current_index >= len(self.contour_data['Data']):
            return
        
        # Clear any existing content
        if self.QGV_qrange.layout():
            QtWidgets.QWidget().setLayout(self.QGV_qrange.layout())
        
        # Create plot widget
        plot_widget = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(plot_widget)
        self.QGV_qrange.setLayout(layout)
        
        # 피크 정보와 범위 정보 가져오기
        peak_q = None
        q_range = None
        peak_index = None
        
        if mode == "auto_tracking_error":
            # 오토트래킹 에러 케이스 - 직전 피크 참조
            if self.current_peak_name:
                for entry in reversed(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
                    if entry.get('peak_name') == self.current_peak_name:
                        peak_q = entry.get('peak_q')
                        q_range = entry.get('output_range')
                        peak_index = entry.get('frame_index')
                        if peak_q is not None and q_range is not None:
                            break
        elif mode == "manual_adjust":
            # manual adjust 케이스 - 현재 선택된 피크 사용
            if self.current_peak_name:
                for entry in self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
                    if entry.get('frame_index') == self.current_index and entry.get('peak_name') == self.current_peak_name:
                        peak_q = entry.get('peak_q')
                        q_range = entry.get('output_range')
                        peak_index = entry.get('frame_index')
                        break
        else:
            # 기본 케이스 - 이전과 동일하게 처리
            if self.current_peak_name:
                for entry in reversed(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
                    if entry.get('peak_name') == self.current_peak_name:
                        peak_q = entry.get('peak_q')
                        q_range = entry.get('output_range')
                        peak_index = entry.get('frame_index')
                        if peak_q is not None and q_range is not None:
                            break
        
        # Initialize helper with peak and range information
        current_entry = self.contour_data['Data'][self.current_index]
        self.q_correction_helper = QRangeCorrectionHelper(plot_widget)
        self.q_correction_helper.set_data(
            current_entry['q'], 
            current_entry['Intensity'],
            peak=peak_q,
            index=peak_index,
            current_index=self.current_index,
            q_range=q_range
        )
        
        self.q_correction_helper.add_selection_lines()
    
    def update_ui_state(self):
        """
        Update button states based on current tracking state
        
        There are several states:
        1. Selecting q-range (Page 1)
        2. Auto tracking in progress (Page 2)
        3. Tracking failed (Page 2)
        4. Waiting state after tracking completes/stops (Page 2)
        5. Adjustment mode (Page 2)
        6. Check peak range mode (Page 2)
        """
        # Page 1 state
        self.PB_back_0.setEnabled(True)
        self.PB_apply_qrange.setEnabled(True)
        
        # Page 2 states depend on current tracking state
        waiting_state = not self.auto_tracking and not self.manual_adjust and not self.check_peak_range_mode
        
        # During auto tracking, only Next and Stop buttons are enabled
        if self.auto_tracking and not self.manual_adjust:
            self.PB_next.setEnabled(True)
            self.PB_stop_auto_tracking.setEnabled(True)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(False)
        # In check peak range mode with peak selected, enable Next
        elif self.check_peak_range_mode and self.current_peak_name is not None:
            self.PB_next.setEnabled(True)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(True)  # Cancel 버튼으로 사용
            self.PB_check_peak_range.setText("Cancel")
        # In check peak range mode without peak selected
        elif self.check_peak_range_mode and self.current_peak_name is None:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(True)  # Cancel 버튼으로 사용
            self.PB_check_peak_range.setText("Cancel")
        # In adjustment mode with peak selected, enable Next
        elif self.manual_adjust and self.current_peak_name is not None:
            self.PB_next.setEnabled(True)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)  # Always enabled to toggle mode
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(False)
        # In adjustment mode without peak selected
        elif self.manual_adjust and self.current_peak_name is None:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)  # Always enabled to toggle mode
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(False)
        # In waiting state (tracking complete or stopped)
        elif waiting_state:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)
            self.PB_find_another_peak.setEnabled(True)
            self.PB_check_peak_range.setEnabled(
                len(self.main.PEAK_EXTRACT_DATA['found_peak_list']) > 0
            )
            self.PB_check_peak_range.setText("Check Peak Range")
        # Default state
        else:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
            self.PB_check_peak_range.setEnabled(False)
            
        # Print current state for debugging
        print(f"UI State - auto_tracking: {self.auto_tracking}, manual_adjust: {self.manual_adjust}, check_peak_range: {self.check_peak_range_mode}, current_peak: {self.current_peak_name}")

    
    def on_back_to_settings(self):
        """Handle back button to settings page"""
        # Show warning dialog
        reply = QtWidgets.QMessageBox.question(
            self.main,
            "Warning",
            "Going back will reset all tracking data. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Reset data and go back to settings page
            self.main.init_peak_data(self.contour_data)
            self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_apply_qrange(self):
        # Get selected q-range
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "Warning", 
                "Please select a valid q range first."
            )
            return
        
        # Determine if this is a new peak or continued tracking
        flag_start = self.current_peak_name is None
        flag_auto_tracking = True
        
        print(f"Applying q-range: {q_range} at index {self.current_index}, manual_adjust={self.manual_adjust}")
        
        # Find peak at current index
        result = find_peak_extraction(
            contour_data=self.contour_data,
            Index_number=self.current_index,
            input_range=q_range,
            fitting_function=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_auto_tracking=flag_auto_tracking,
            flag_manual_adjust=self.manual_adjust,
            flag_start=flag_start,
            start_index=0 if flag_start else None,
            current_peak_name=self.current_peak_name
        )
        
        # Handle result based on success/failure
        if isinstance(result, str):
            # Failed to find peak
            self.L_current_status_2.setText(f"Failed to find peak: {result}")
            self.auto_tracking = True  # Keep auto_tracking flag true for retry
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
            return
        
        # Unpack successful result
        peak_q, peak_intensity, output_range, fwhm, peak_name, fitting_function, fitting_params = result
        
        print(f"Found peak: {peak_name} at q={peak_q}, intensity={peak_intensity}")
        
        # Get current entry before using it
        current_entry = self.contour_data['Data'][self.current_index]
        
        # Check if we already have data for this frame and peak name
        # If so, replace it instead of adding a new entry
        found_existing = False
        for i, entry in enumerate(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
            if entry.get('frame_index') == self.current_index and entry.get('peak_name') == peak_name:
                # Replace existing entry
                print(f"Replacing existing entry for frame {self.current_index}, peak {peak_name}")
                self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][i] = {
                    "frame_index": self.current_index,
                    "Time": current_entry.get("Time", 0),
                    "Temperature": current_entry.get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity,
                    "fwhm": fwhm,
                    "peak_name": peak_name,
                    "output_range": output_range,
                    "fitting_function": fitting_function,
                    "fitting_params": fitting_params
                }
                found_existing = True
                break
        
        if not found_existing:
            # Save new result to tracked_peaks
            new_result = {
                "frame_index": self.current_index,
                "Time": current_entry.get("Time", 0),
                "Temperature": current_entry.get("Temperature", 0),
                "peak_q": peak_q,
                "peak_Intensity": peak_intensity,
                "fwhm": fwhm,
                "peak_name": peak_name,
                "output_range": output_range,
                "fitting_function": fitting_function,
                "fitting_params": fitting_params
            }
            self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'].append(new_result)
            
        # Remember last peak name and update states
        self.current_peak_name = peak_name
        
        # 수정: auto_tracking 중 에러 발생 후 manual_adjust=True인 경우 처리
        if self.manual_adjust and self.auto_tracking:
            # 자동 트래킹 중 수동 조정한 경우: 다음 프레임으로 이동하고 자동 트래킹 계속
            print(f"Manual adjustment successful for frame {self.current_index}, continuing to frame {self.current_index + 1}")
            
            # 다음 프레임으로 이동
            self.current_index += 1
            
            # manual_adjust 모드 해제하고 자동 트래킹 계속
            if self.current_index <= self.max_index:
                self.manual_adjust = False
                print(f"Switching back to automatic tracking from frame {self.current_index}")
                
                # 자동 트래킹 계속
                self.run_automatic_tracking(output_range)
                return
        # 그 외 일반적인 경우는 기존 로직 유지
        elif not self.manual_adjust:
            self.auto_tracking = True
        
        # If this is the first frame of a new peak, start automatic tracking
        if self.current_index == 0 and not self.manual_adjust:
            self.current_index += 1  # Move to next frame
            
            # Try automatic tracking for remaining frames
            self.run_automatic_tracking(output_range)  # Use output_range for better tracking
        else:
            # If not first frame or in manual_adjust mode, just update UI
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()

    def apply_q_range_modify(self):
        """
        Adjust 모드에서 사용: 선택된 프레임의 피크만 수정하고 컨투어 화면으로 돌아감
        """
        # Get selected q-range
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "Warning", 
                "Please select a valid q range first."
            )
            return
        
        print(f"Modifying peak: q-range={q_range} at index={self.current_index}, peak={self.current_peak_name}")
        
        # Find peak at current index
        result = find_peak_extraction(
            contour_data=self.contour_data,
            Index_number=self.current_index,
            input_range=q_range,
            fitting_function=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_auto_tracking=True,
            flag_manual_adjust=True,
            flag_start=False,
            start_index=None,
            current_peak_name=self.current_peak_name
        )
        
        # Handle result based on success/failure
        if isinstance(result, str):
            # Failed to find peak
            self.L_current_status_2.setText(f"Failed to adjust peak: {result}")
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
            return
        
        # Unpack successful result
        peak_q, peak_intensity, output_range, fwhm, peak_name, fitting_function, fitting_params = result
        
        print(f"Adjusted peak: {peak_name} at q={peak_q}, intensity={peak_intensity}")
        
        # Check if we already have data for this frame and peak name
        # If so, replace it instead of adding a new entry
        found_existing = False
        for i, entry in enumerate(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
            if entry.get('frame_index') == self.current_index and entry.get('peak_name') == peak_name:
                # Replace existing entry
                print(f"Replacing existing entry for frame {self.current_index}, peak {peak_name}")
                self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][i] = {
                    "frame_index": self.current_index,
                    "Time": self.contour_data['Data'][self.current_index].get("Time", 0),
                    "Temperature": self.contour_data['Data'][self.current_index].get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity,
                    "fwhm": fwhm,
                    "peak_name": peak_name,
                    "output_range": output_range,
                    "fitting_function": fitting_function,
                    "fitting_params": fitting_params
                }
                found_existing = True
                break
        
        if not found_existing:
            # Save new result to tracked_peaks
            current_entry = self.contour_data['Data'][self.current_index]
            new_result = {
                "frame_index": self.current_index,
                "Time": current_entry.get("Time", 0),
                "Temperature": current_entry.get("Temperature", 0),
                "peak_q": peak_q,
                "peak_Intensity": peak_intensity,
                "fwhm": fwhm,
                "peak_name": peak_name,
                "output_range": output_range,
                "fitting_function": fitting_function,
                "fitting_params": fitting_params
            }
            self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'].append(new_result)

        
        # 수정 성공 메시지 표시
        success_msg = f"Peak {peak_name} at frame {self.current_index} successfully adjusted"
        print(success_msg)
        
        # 중요: 피크 선택 상태 초기화
        modified_peak_name = peak_name  # 메시지용으로 임시 저장
        self.current_peak_name = None  # 선택된 피크 초기화
        
        # 컨투어 페이지로 돌아감
        self.L_current_status_2.setText(f"{success_msg}. Click on another peak to continue.")
        self.ui.stackedWidget.setCurrentIndex(2)  # 컨투어 페이지로 이동
        self.update_contour_plot()
        self.update_ui_state()
    
    def run_automatic_tracking(self, initial_q_range):
        """
        Run automatic tracking for remaining frames
        
        Parameters:
            initial_q_range (tuple): Initial q-range for tracking
        """
        print(f"Starting automatic tracking from index {self.current_index}")
        
        # Define callbacks for tracking updates
        def on_error(message):
            print(f"Tracking error: {message}")
            self.L_current_status_2.setText(message)
            # Keep auto_tracking flag true so we can retry
            self.auto_tracking = True
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
        
        def on_update(frame_index, peak_q, peak_intensity, fwhm, peak_name):
            print(f"Tracking update: frame {frame_index}, peak {peak_name}")
            self.L_current_status_1.setText(
                f"Tracking peak {peak_name}... ({frame_index} / {self.max_index})"
            )
            QtWidgets.QApplication.processEvents()  # Update UI during tracking
        
        def on_finish(message, last_peak_info):
            print(f"Tracking finished: {message}")
            print(f"Last peak info: {last_peak_info}")
            
            # If tracking is complete (either reached the end or explicitly finished)
            peak_name = last_peak_info.get("peak_name")
            
            # Add to found peak list if not already there
            if peak_name and peak_name not in self.main.PEAK_EXTRACT_DATA['found_peak_list']:
                print(f"Adding peak {peak_name} to found_peak_list")
                self.main.PEAK_EXTRACT_DATA['found_peak_list'].append(peak_name)
            
            # Reset tracking state to waiting state
            self.auto_tracking = False
            self.manual_adjust = False
            self.current_peak_name = None
            
            self.L_current_status_2.setText(f"Tracking complete for {peak_name}!")
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
        
        # Remember current state
        peak_info = {
            "peak_q": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['peak_q'],
            "peak_intensity": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['peak_Intensity'],
            "fwhm": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['fwhm'],
            "peak_name": self.current_peak_name,
            "fitting_function": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['fitting_function'],
            "fitting_params": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['fitting_params']
        }
        
        # Run tracking
        success, final_index, last_peak_info = run_automatic_tracking(
            contour_data=self.contour_data,
            tracked_peaks=self.main.PEAK_EXTRACT_DATA['tracked_peaks'],
            current_peak_info=peak_info,
            current_index=self.current_index - 1,  # Adjusting index since we've already processed the current frame
            max_index=self.max_index,
            get_q_range_func=lambda: initial_q_range,  # Use the initial q-range for all frames
            fitting_model=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_start=False,  # Never start a new peak in auto tracking
            flag_auto_tracking=True,  # Always set to True for proper tracking
            flag_manual_adjust=self.manual_adjust,
            on_error_callback=on_error,
            on_update_callback=on_update,
            on_finish_callback=on_finish
        )
        
        # Update current index for next operation
        self.current_index = final_index
    
    def update_contour_plot(self):
        """Update the contour plot with current tracked peaks"""
        if not self.contour_data:
            print("No contour data available")
            return
            
        if not self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
            print("No tracked peaks data available")
            return
            
        print(f"Updating contour plot with {len(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'])} tracked peaks")
        print(f"Found peak list: {self.main.PEAK_EXTRACT_DATA['found_peak_list']}")
        
        # 로딩 다이얼로그 추가
        loading = LoadingDialog(self.main, "피크 데이터 컨투어 처리중...")
        loading.progress.setMaximum(0)  # 불확정적 진행 표시(스피닝 인디케이터)
        loading.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Create or update the list of peaks to show
            # For auto-tracking, include the current_peak_name even if not in found_peak_list
            peaks_to_show = list(self.main.PEAK_EXTRACT_DATA['found_peak_list'])
            if self.current_peak_name and self.current_peak_name not in peaks_to_show:
                peaks_to_show.append(self.current_peak_name)
                
            # 피크 선택 콜백 함수 정의
            def on_peak_selected(peak_name, frame_index):
                if self.manual_adjust:
                    self.current_peak_name = peak_name
                    self.current_index = frame_index
                    self.L_current_status_2.setText(
                        f"Selected peak {peak_name} at frame {frame_index}. Click Next to adjust it."
                    )
                    self.update_ui_state()
                elif self.check_peak_range_mode:
                    self.current_peak_name = peak_name
                    self.current_index = frame_index
                    self.L_current_status_2.setText(
                        f"Selected peak {peak_name} at frame {frame_index}. Click Next to check q-ranges."
                    )
                    self.update_ui_state()
        
            # Create contour plot with callback
            self.contour_canvas = plot_contour_extraction(
                contour_data=self.contour_data,
                tracked_peaks=self.main.PEAK_EXTRACT_DATA['tracked_peaks'],
                found_peak_list=peaks_to_show,
                flag_adjust_mode=self.manual_adjust or self.check_peak_range_mode,  # 수정: 체크 모드에서도 피크 선택 가능
                on_peak_selected_callback=on_peak_selected  # 콜백 전달
            )
            
            if self.contour_canvas:
                print("Contour canvas created successfully")
                # Clear any existing layout
                if self.QGV_contour.layout():
                    QtWidgets.QWidget().setLayout(self.QGV_contour.layout())
                    
                # Set new layout with canvas
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(self.contour_canvas)
                self.QGV_contour.setLayout(layout)
                self.QGV_contour.update()  # Force update
            else:
                print("Failed to create contour canvas")
        finally:
            # 로딩 다이얼로그 닫기
            loading.close()
    
    def on_next(self):
        """
        Handle next button click
        
        Different behaviors depending on state:
        - If auto tracking failed: retry at current point with manual adjustment
        - If in adjustment mode: edit selected peak
        - If in check peak range mode: move to page 4 for peak range inspection
        """
        print(f"Next button clicked - current state: auto_tracking={self.auto_tracking}, manual_adjust={self.manual_adjust}, check_peak_range={self.check_peak_range_mode}")
        
        if self.check_peak_range_mode:
            if self.current_peak_name is None:
                # 피크가 선택되지 않은 경우 경고 메시지 표시
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "No Peak Selected",
                    "Please select a peak from the contour plot first.",
                    QtWidgets.QMessageBox.Ok
                )
                return
                
            # Check peak range mode - go to page 5 (index range page) - UI 인덱스는 4
            self.ui.stackedWidget.setCurrentIndex(4)  # 인덱스 4는 stackedWidget에서는 5번째 페이지
            
            # Initialize QRangeIndexPage with selected peak data
            if hasattr(self.main, 'qrange_index_page'):
                self.main.qrange_index_page.initialize_page(
                    self.contour_data,
                    self.current_peak_name,
                    self.current_index
                )
            
        elif self.manual_adjust:
            if self.current_peak_name is None:
                # 피크가 선택되지 않은 경우 경고 메시지 표시
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "No Peak Selected",
                    "Please select a peak from the contour plot first.",
                    QtWidgets.QMessageBox.Ok
                )
                return
            
            # In adjustment mode, edit selected peak at the current_index (which was set from the selection)
            self.ui.stackedWidget.setCurrentIndex(1)  # Go to q-range page
            self.setup_q_range_graph(mode="manual_adjust")  # 수동 조정 모드로 설정
            
            # Update status - 선택한 피크 수정
            self.L_current_status_1.setText(
                f"Adjusting peak {self.current_peak_name} at frame {self.current_index}"
            )
            
            # apply_q_range 버튼 연결 해제하고 apply_q_range_modify로 연결
            try:
                self.PB_apply_qrange.clicked.disconnect()
            except TypeError:
                # 이미 연결이 해제되었거나 연결된 적이 없는 경우
                pass
            
            self.PB_apply_qrange.clicked.connect(self.apply_q_range_modify)
            
        elif self.auto_tracking:
            # Set manual_adjust to True so threshold checks will be bypassed
            self.manual_adjust = True
            print(f"Setting manual_adjust=True for frame {self.current_index}")
            
            # Retry auto tracking from current point
            self.ui.stackedWidget.setCurrentIndex(1)  # Go to q-range page
            self.setup_q_range_graph(mode="auto_tracking_error")  # 오토트래킹 에러 모드로 설정
            
            # Update status - 직전 참조 피크를 사용하여 현재 프레임 수정
            # 직전 피크 정보 찾기
            prev_frame = None
            for entry in reversed(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
                if entry.get('peak_name') == self.current_peak_name:
                    prev_frame = entry.get('frame_index')
                    break
                    
            if prev_frame is not None:
                self.L_current_status_1.setText(
                    f"Manual adjustment for frame {self.current_index} using reference from frame {prev_frame}"
                )
            else:
                self.L_current_status_1.setText(
                    f"Manual adjustment for frame {self.current_index} / {self.max_index}"
                )
            
            # apply_q_range 버튼 연결 해제하고 일반 on_apply_qrange로 연결
            try:
                self.PB_apply_qrange.clicked.disconnect()
            except TypeError:
                # 이미 연결이 해제되었거나 연결된 적이 없는 경우
                pass
                    
            self.PB_apply_qrange.clicked.connect(self.on_apply_qrange)
        
        self.update_ui_state()


    def on_check_peak_range(self):
        """
        피크 범위 체크 모드 진입/취소
        """
        if self.check_peak_range_mode:
            # 이미 체크 모드인 경우, 취소 기능 수행
            self.check_peak_range_mode = False
            self.current_peak_name = None
            self.L_current_status_2.setText("Peak range check mode cancelled")
            self.PB_check_peak_range.setText("Check Peak Range")
            self.update_contour_plot()
        else:
            # 체크 모드 진입
            if not self.main.PEAK_EXTRACT_DATA['found_peak_list']:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "No Peaks Available",
                    "Please track at least one peak first."
                )
                return
                
            self.check_peak_range_mode = True
            self.current_peak_name = None
            self.L_current_status_2.setText("Click on a peak to check/modify its q-range at each frame")
            self.PB_check_peak_range.setText("Cancel")
            self.update_contour_plot()
            
        self.update_ui_state()


    def on_stop_tracking(self):
        """Stop current tracking and add to found_peak_list"""
        if not self.current_peak_name:
            return
            
        # Add current peak to found_peak_list if not already there
        if self.current_peak_name not in self.main.PEAK_EXTRACT_DATA['found_peak_list']:
            self.main.PEAK_EXTRACT_DATA['found_peak_list'].append(self.current_peak_name)
        
        # Reset tracking state
        self.auto_tracking = False
        self.manual_adjust = False
        self.current_peak_name = None
        
        # Update UI
        self.L_current_status_2.setText("Tracking stopped and saved.")
        self.update_ui_state()
        self.update_contour_plot()
    
    def on_enter_adjust_mode(self):
        """Enter or exit peak adjustment mode"""
        if self.manual_adjust:
            # Already in adjust mode, exit it
            print("Exiting adjustment mode")
            self.manual_adjust = False
            self.current_peak_name = None
            
            # Update button text
            self.PB_adjust_mode.setText("Adjust Mode")
            
            # Update UI
            self.L_current_status_2.setText("Adjustment mode disabled.")
            self.update_ui_state()
            self.update_contour_plot()
        else:
            # Enter adjustment mode
            print("Entering adjustment mode")
            self.manual_adjust = True
            
            # Update button text
            self.PB_adjust_mode.setText("Disable Adjust Mode")
            
            # Update UI
            self.L_current_status_2.setText("Click on a peak to select it for adjustment.")
            self.update_ui_state()
            
            # Redraw contour with selection enabled
            self.update_contour_plot()
    
    def on_find_another_peak(self):
        """Start tracking a new peak"""
        # Reset index and tracking state
        self.current_index = 0
        self.auto_tracking = True
        self.manual_adjust = False
        self.check_peak_range_mode = False  # 체크 모드도 해제
        self.current_peak_name = None
        
        # Go to q-range selection page
        self.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_graph()
        
        # Update status
        self.L_current_status_1.setText(
            f"Select q range for new peak at frame {self.current_index} / {self.max_index}"
        )
        
        self.update_ui_state()
