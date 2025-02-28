# asset/contour_page_sdd_1.py
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
from asset.contour_storage import DATA, PARAMS, FITTING_THRESHOLD
from asset.contour_util_gui import QRangeCorrectionHelper, plot_contour_with_peaks_gui 
from asset.fitting_util import find_peak

class SDDPeakTrackingPage(QtCore.QObject):
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # 현재 처리 중인 인덱스와 최대 프레임 수
        self.current_index = 0
        self.max_index = 0

        # contour 데이터와 검출 결과
        self.contour_data = None
        self.tracked_peaks = None

        # 전역 q range (사용자 선택 또는 재조정 시 저장)
        self.global_q_range = None

        # 현재 프레임의 피크 검출 성공 여부
        self.peak_found = None
        
        # 자동 추적 중인지 표시하는 플래그
        self.is_auto_tracking = False
        
        # 0번 인덱스 페이지에서 처음 시작했는지 표시하는 플래그
        self.started_from_index_0 = True

        # 1번 페이지용 QRangeCorrectionHelper
        self.q_correction_helper = None

        # 2번 페이지의 matplotlib 캔버스 (인터랙티브 선택용)
        self.adjust_canvas = None
        # 선택된 피크 인덱스 (tracked_peaks 내의 인덱스)
        self.selected_peak_index = None
        # 하이라이트 선 (수직/수평)
        self.highlight_marker_v = None
        self.highlight_marker_h = None

        # 조정 모드 플래그 (True이면, 해당 프레임만 재검출)
        self.in_adjustment_mode = False

        self.setup_ui_connections()

    def setup_ui_connections(self):
        """UI 위젯과 시그널 연결"""
        self.ui.PB_back_0.clicked.connect(
            lambda: self.main.ui.stackedWidget.setCurrentIndex(0)
        )
        self.ui.PB_apply_qrange.clicked.connect(self.apply_q_range)
        self.ui.PB_next.clicked.connect(self.retry_current_frame)
        self.ui.PB_sdd_correction_start.clicked.connect(self.finalize_peak_tracking)
        
        # 뒤로가기 버튼 초기 상태 설정
        self.update_back_button_state()

    def update_back_button_state(self):
        """뒤로가기 버튼 상태 업데이트"""
        self.ui.PB_back_0.setEnabled(self.started_from_index_0)
        
    def initialize_peak_tracking(self, contour_data):
        """피크 추적 프로세스 초기화 (인덱스 0부터 시작)"""
        self.contour_data = contour_data
        self.current_index = 0
        self.max_index = len(contour_data['Data']) - 1
        self.tracked_peaks = {
            "Series": self.contour_data["Series"],
            "Time-temp": self.contour_data["Time-temp"],
            "Data": []
        }
        self.global_q_range = None
        self.selected_peak_index = None
        self.highlight_marker_v = None
        self.highlight_marker_h = None
        self.in_adjustment_mode = False
        self.is_auto_tracking = False
        
        # 페이지 0에서 시작한 것으로 설정
        self.started_from_index_0 = True
        self.update_back_button_state()
        
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()

    def setup_q_range_selection(self, mode=None):
        """
        1번 페이지: QGV에 현재 프레임의 데이터를 띄우고 q range 선택
        
        Parameters:
            mode (str): 'auto_tracking_error' 또는 'manual_adjust' - 어떤 모드에서 호출되었는지 표시
        """
        if mode == "auto_tracking_error" or mode == "manual_adjust":
            # 페이지 2에서 왔으므로 뒤로가기 버튼 비활성화
            self.started_from_index_0 = False
            self.update_back_button_state()
        
        if not self.contour_data or self.current_index >= len(self.contour_data['Data']):
            QtWidgets.QMessageBox.warning(self.main, "Error", "Current frame index is out of range.")
            return
        if hasattr(self.ui, 'QGV_qrange'):
            if self.ui.QGV_qrange.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_qrange.layout())
            plot_widget = pg.PlotWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(plot_widget)
            self.ui.QGV_qrange.setLayout(layout)
            
            # 피크 정보와 범위 정보 가져오기
            peak_q = None
            peak_index = None
            custom_q_range = None
            
            if mode == "auto_tracking_error":
                # 오토트래킹 에러 케이스 - 직전 프레임 피크 참조
                for entry in reversed(self.tracked_peaks["Data"]):
                    if entry.get("frame_index") < self.current_index:
                        peak_q = entry.get("peak_q")
                        peak_index = entry.get("frame_index")
                        break
            elif mode == "manual_adjust":
                # manual adjust 케이스 - 현재 선택된 피크 사용
                for entry in self.tracked_peaks["Data"]:
                    if entry.get("frame_index") == self.current_index:
                        peak_q = entry.get("peak_q")
                        peak_index = entry.get("frame_index")
                        
                        # 피크의 실제 분석 범위 사용 (가능한 경우)
                        if "output_range" in entry and entry["output_range"] is not None:
                            custom_q_range = entry["output_range"]
                        # 없는 경우 피크 주변으로 범위 설정
                        elif peak_q is not None:
                            q_width = 0.1  # 적절한 범위 조정
                            custom_q_range = (peak_q - q_width, peak_q + q_width)
                        break
            else:
                # 기본 케이스 - 이전 로직 유지
                # 현재 인덱스의 피크 정보 찾기
                if len(self.tracked_peaks["Data"]) > 0:
                    for entry in self.tracked_peaks["Data"]:
                        if entry.get("frame_index") == self.current_index:
                            peak_q = entry.get("peak_q")
                            peak_index = entry.get("frame_index")
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
                q_range=custom_q_range if custom_q_range else self.global_q_range
            )
            
            self.q_correction_helper.add_selection_lines()
            
            # 상태 메시지 업데이트
            if mode == "auto_tracking_error" and peak_index is not None:
                self.ui.L_current_status_1.setText(
                    f"Manual adjustment for frame {self.current_index} using reference from frame {peak_index}"
                )
            elif mode == "manual_adjust":
                self.ui.L_current_status_1.setText(
                    f"Adjusting peak at frame {self.current_index}"
                )
            else:
                self.ui.L_current_status_1.setText(
                    f"Select q range for frame {self.current_index} / {self.max_index}"
                )
            
    def apply_q_range(self):
        """Process selected q range"""
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(self.main, "Warning", "Please select a q range first.")
            return
        self.global_q_range = q_range
        
        # Get current fitting model from PARAMS
        fitting_model = PARAMS.get('fitting_model', 'gaussian')
        
        # Find the peak with the selected q range
        result = find_peak(
            self.contour_data, 
            Index_number=self.current_index, 
            input_range=self.global_q_range,
            peak_info=None,
            fitting_function=fitting_model,
            threshold_config=FITTING_THRESHOLD
        )
        
        if isinstance(result, str):
            self.peak_found = False
            self.ui.L_current_status_2.setText(
                f"Peak not found for frame {self.current_index}: {result}. Please adjust q range."
            )
            return
        else:
            self.peak_found = True
            peak_q, peak_intensity, _, fwhm = result  # Now unpacking 4 values
            current_entry = self.contour_data['Data'][self.current_index]
            
            if self.in_adjustment_mode:
                # 기존 데이터 수정 모드 - 같은 프레임 인덱스의 항목 찾아서 업데이트
                found_existing = False
                for i, entry in enumerate(self.tracked_peaks["Data"]):
                    if entry.get("frame_index") == self.current_index:
                        # 기존 항목 업데이트
                        self.tracked_peaks["Data"][i] = {
                            "frame_index": self.current_index,
                            "Time": current_entry["Time"],
                            "Temperature": current_entry.get("Temperature", 0),
                            "peak_q": peak_q,
                            "peak_Intensity": peak_intensity,
                            "fwhm": fwhm
                        }
                        found_existing = True
                        break
                
                if not found_existing:
                    # 기존 항목이 없으면 새로 추가
                    self.tracked_peaks["Data"].append({
                        "frame_index": self.current_index,
                        "Time": current_entry["Time"],
                        "Temperature": current_entry.get("Temperature", 0),
                        "peak_q": peak_q,
                        "peak_Intensity": peak_intensity,
                        "fwhm": fwhm
                    })
                    # 프레임 인덱스 기준으로 정렬
                    self.tracked_peaks["Data"].sort(key=lambda x: x["frame_index"])
                
                self.in_adjustment_mode = False
                self.ui.L_current_status_2.setText(f"Peak at frame {self.current_index} successfully adjusted.")
                self.show_contour_page()
            else:
                # 자동 트래킹 모드
                self.tracked_peaks["Data"].append({
                    "frame_index": self.current_index,
                    "Time": current_entry["Time"],
                    "Temperature": current_entry.get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity,
                    "fwhm": fwhm
                })
                self.current_index += 1
                
                # 자동 추적 시작 전에 플래그 설정
                self.is_auto_tracking = True
                self.run_automatic_tracking()

    def run_automatic_tracking(self):
        """Automatic peak tracking with selected fitting model"""
        try:
            self.is_auto_tracking = True  # 자동 추적 시작
            fitting_model = PARAMS.get('fitting_model', 'gaussian')
            
            while self.current_index <= self.max_index:
                # 이전 프레임의 피크 정보 가져오기
                prev_peak_info = None
                
                # 직전 프레임 찾기
                for entry in reversed(self.tracked_peaks["Data"]):
                    if entry.get("frame_index") == self.current_index - 1:
                        prev_peak_info = {
                            "peak_q": entry.get("peak_q"),
                            "peak_intensity": entry.get("peak_Intensity"),
                            "fwhm": entry.get("fwhm"),
                            "peak_name": f"peak_{entry.get('frame_index')}_{entry.get('peak_q'):.4f}"
                        }
                        break
                
                # 직전 프레임의 피크 정보를 전달
                result = find_peak(
                    self.contour_data, 
                    Index_number=self.current_index, 
                    input_range=self.global_q_range,
                    peak_info=prev_peak_info,  # 이전 프레임 정보 전달
                    fitting_function=fitting_model,
                    threshold_config=FITTING_THRESHOLD
                )

                
                if isinstance(result, str):
                    self.peak_found = False
                    self.ui.L_current_status_2.setText(
                        f"Peak not found for frame {self.current_index}: {result}. Please adjust q range."
                    )
                    break
                else:
                    self.peak_found = True
                    peak_q, peak_intensity, _, fwhm = result  # Now unpacking 4 values
                    current_entry = self.contour_data['Data'][self.current_index]
                    self.tracked_peaks["Data"].append({
                        "frame_index": self.current_index,
                        "Time": current_entry["Time"],
                        "Temperature": current_entry.get("Temperature", 0),
                        "peak_q": peak_q,
                        "peak_Intensity": peak_intensity,
                        "fwhm": fwhm  # Adding FWHM to the tracked data
                    })
                    self.current_index += 1
                    
            if self.current_index > self.max_index:
                self.ui.L_current_status_2.setText("Peak find Completed")
            
            self.is_auto_tracking = False  # 자동 추적 종료
            self.show_contour_page()
            
        except Exception as e:
            self.is_auto_tracking = False  # 에러 발생 시에도 자동 추적 상태 해제
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Error during automatic tracking:\n{str(e)}"
            )

    def show_contour_page(self):
            """2번 페이지: 컨투어 플롯 및 현재까지의 피크 결과 표시"""
            canvas = plot_contour_with_peaks_gui(self.contour_data, self.tracked_peaks)
            if canvas is None:
                QtWidgets.QMessageBox.warning(
                    self.main, 
                    "Warning", 
                    "Could not create contour plot"
                )
                return
            
            if hasattr(self.ui, 'QGV_contour'):
                if self.ui.QGV_contour.layout():
                    QtWidgets.QWidget().setLayout(self.ui.QGV_contour.layout())
                layout = QtWidgets.QVBoxLayout()
                layout.addWidget(canvas)
                self.ui.QGV_contour.setLayout(layout)
            
            self.adjust_canvas = canvas
            self.setup_adjust_interaction()
            
            # -----------------------------------------------------
            # ★ 핵심: 찾은 피크 개수 vs 전체 프레임 수 확인
            # -----------------------------------------------------
            total_frames = self.max_index + 1
            found_peaks = len(self.tracked_peaks["Data"])

            if found_peaks == total_frames:
                # 모든 프레임 처리됨
                self.ui.L_current_status_2.setText("Peak find Completed")  
                
                # tracked_peaks를 DATA에 저장
                DATA['tracked_peaks'] = self.tracked_peaks
                
                # Next 버튼 → Adjust specific Peak
                try:
                    self.ui.PB_next.clicked.disconnect()
                except Exception:
                    pass
                self.ui.PB_next.setText("Adjust specific Peak")
                self.ui.PB_next.clicked.connect(self.adjust_specific_peak)

                # Final Apply 버튼과 SDD Correction 버튼 활성화
                self.ui.PB_final_apply.setEnabled(True)
                self.ui.PB_sdd_correction_start.setEnabled(True)

            else:
                # 아직 처리되지 않은 프레임이 존재
                self.ui.PB_next.setText("Next")
                try:
                    self.ui.PB_next.clicked.disconnect()
                except Exception:
                    pass
                self.ui.PB_next.clicked.connect(self.retry_current_frame)

                # 중간 단계에서는 Final Apply 버튼과 SDD Correction 버튼 비활성화
                self.ui.PB_final_apply.setEnabled(False)
                self.ui.PB_sdd_correction_start.setEnabled(False)

            self.main.ui.stackedWidget.setCurrentIndex(2)
            
            # 뒤로가기 버튼 상태 업데이트 (페이지 2로 이동했으므로 항상 비활성화)
            self.started_from_index_0 = False
            self.update_back_button_state()

    def setup_adjust_interaction(self):
        """
        2번 페이지의 캔버스에서, 사용자가 피크를 클릭하면 x, y축 실선(최상단, zorder=10)으로 하이라이트하고,
        선택된 피크 인덱스를 self.selected_peak_index에 저장합니다.
        """
        if self.adjust_canvas is None:
            return
        ax = self.adjust_canvas.figure.axes[0]
        self.selected_peak_index = None

        if self.highlight_marker_v is not None:
            try:
                self.highlight_marker_v.remove()
            except Exception:
                pass
            self.highlight_marker_v = None
        if self.highlight_marker_h is not None:
            try:
                self.highlight_marker_h.remove()
            except Exception:
                pass
            self.highlight_marker_h = None

        def on_click(event):
            # 자동 추적 중일 때는 피크 선택 비활성화
            if self.is_auto_tracking:
                return
                
            if event.inaxes != ax:
                return
            min_dist = float('inf')
            closest_index = None
            for i, entry in enumerate(self.tracked_peaks["Data"]):
                if "peak_q" in entry:
                    dx = event.xdata - entry["peak_q"]
                    dy = event.ydata - entry["Time"]
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = i
            if closest_index is not None:
                self.selected_peak_index = closest_index
                sel_entry = self.tracked_peaks["Data"][closest_index]
                if self.highlight_marker_v is not None:
                    try:
                        self.highlight_marker_v.remove()
                    except Exception:
                        pass
                if self.highlight_marker_h is not None:
                    try:
                        self.highlight_marker_h.remove()
                    except Exception:
                        pass
                self.highlight_marker_v = ax.axvline(x=sel_entry["peak_q"], color="green", linestyle="--", linewidth=2, zorder=10)
                self.highlight_marker_h = ax.axhline(y=sel_entry["Time"], color="green", linestyle="--", linewidth=2, zorder=10)
                self.adjust_canvas.draw()

        try:
            self.adjust_canvas.mpl_disconnect(getattr(self, "adjust_cid"))
        except Exception:
            pass
        self.adjust_cid = self.adjust_canvas.mpl_connect("button_press_event", on_click)

    def retry_current_frame(self):
        """현재 프레임을 수동으로 재처리"""
        self.main.ui.stackedWidget.setCurrentIndex(1)  # q-range 페이지로 이동
        
        # 이전에 이 프레임의 피크가 이미 있는지 체크
        existing_peak = False
        for entry in self.tracked_peaks["Data"]:
            if entry.get("frame_index") == self.current_index:
                existing_peak = True
                break
        
        # 모드 설정: 기존에 피크가 있으면 수동 조정 모드, 없으면 자동 트래킹 에러 모드
        mode = "manual_adjust" if existing_peak else "auto_tracking_error"
        
        # q-range 선택 화면 설정
        self.setup_q_range_selection(mode=mode)

    def adjust_specific_peak(self):
        """컨투어 플롯에서 선택한 피크를 수정"""
        # 선택된 피크가 있는지 확인
        if self.selected_peak_index is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "No Peak Selected", 
                "Please click on a peak in the contour plot to select it first."
            )
            return
            
        # 선택된 피크의 프레임 인덱스를 가져옴
        try:
            selected_entry = self.tracked_peaks["Data"][self.selected_peak_index]
            self.current_index = selected_entry["frame_index"]
            
            # 조정 모드 플래그 설정
            self.in_adjustment_mode = True
            
            # q-range 선택 화면으로 이동
            self.main.ui.stackedWidget.setCurrentIndex(1)
            self.setup_q_range_selection(mode="manual_adjust")
            
            # 상태 메시지 업데이트
            self.ui.L_current_status_1.setText(
                f"Adjusting peak at frame {self.current_index} (selected from contour)"
            )
        except (IndexError, KeyError) as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Selection Error",
                f"Error selecting peak: {str(e)}"
            )
            self.selected_peak_index = None

    def finalize_peak_tracking(self):
        """최종 완료: 모든 프레임이 처리되었을 때 후속 처리/저장 진행"""
        if self.tracked_peaks is None or len(self.tracked_peaks["Data"]) == 0:
            QtWidgets.QMessageBox.warning(self.main, "Warning", "No peaks have been tracked.")
            return

        # 1) frame_index를 기준으로, contour_data["Time-temp"]의 실제 온도값을 꺼내서 갱신
        times, temps = self.contour_data["Time-temp"]
        for entry in self.tracked_peaks["Data"]:
            fidx = entry["frame_index"]
            # 만약 이미 Temperature가 0이면, or 그냥 무조건 덮어쓸 수도 있음
            entry["Temperature"] = temps[fidx]  # 실제 온도 리스트에서 가져옴

        # 2) tracked_peaks를 전역 DATA에 반영
        print("Peak tracking completed")
        DATA['tracked_peaks'] = self.tracked_peaks

        # 3) 디버그 출력
        #print(f"DATA, tracked_peaks: {DATA['tracked_peaks']}")