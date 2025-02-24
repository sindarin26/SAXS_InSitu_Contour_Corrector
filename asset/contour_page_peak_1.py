import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from asset.contour_storage import DATA, PARAMS, FITTING_THRESHOLD
from asset.contour_util_gui import QRangeCorrectionHelper
from asset.fitting_util import find_peak_extraction, plot_contour_extraction

class PeakTrackingPage(QtCore.QObject):
    """1번 페이지: 피크 추적 기능을 담당하는 클래스"""
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui
        
        # 현재 처리 중인 인덱스와 최대 프레임 수
        self.current_index = 0
        self.max_index = 0
        
        # peak 검출 결과 및 상태 변수들
        self.peak_found = None
        self.current_peak_selection = None
        self.in_adjustment_mode = False  # 수정 모드 플래그 추가
        
        # GUI 헬퍼
        self.q_correction_helper = None
        
        # UI 요소 초기화
        self.setup_ui_connections()
        
        # 리사이즈 이벤트 필터 설치
        self.ui.QGV_contour.installEventFilter(self)
        
        # canvas 관련 변수
        self.canvas = None
        self.canvas_original_size = None
        self.selected_frame_index = None
        
    def eventFilter(self, obj, event):
        if obj == self.ui.QGV_contour and event.type() == QtCore.QEvent.Resize:
            self.resize_figure_to_view()
        return super().eventFilter(obj, event)
        
    def resize_figure_to_view(self):
        """피규어 크기를 view에 맞게 조정"""
        if not hasattr(self, 'canvas') or not self.canvas:
            return
            
        if not hasattr(self, 'canvas_original_size'):
            self.canvas_original_size = self.canvas.size()
            
        # view 크기 가져오기
        view_size = self.ui.QGV_contour.size()
        
        # 기본 figure 크기
        fig_size = (12, 8)
        fig_aspect = fig_size[0] / fig_size[1]
        
        # 새 크기 계산 (여백 고려)
        view_width = view_size.width() - 10
        view_height = view_size.height() - 10
        view_aspect = view_width / view_height
        
        if view_aspect > fig_aspect:
            new_height = view_height
            new_width = int(new_height * fig_aspect)
        else:
            new_width = view_width
            new_height = int(new_width / fig_aspect)
        
        # canvas 크기 조정
        self.canvas.setFixedSize(int(new_width), int(new_height))
        self.canvas.draw()
        
    def setup_ui_connections(self):
        """UI 시그널 연결"""
        self.ui.PB_back_0.clicked.connect(self.on_back)
        self.ui.PB_apply_qrange.clicked.connect(self.apply_q_range)
        
    def on_back(self):
        """Back 버튼 클릭 처리"""
        confirm = QtWidgets.QMessageBox.question(
            self.main,
            "Warning",
            "Going back will clear all current peak tracking progress. Continue?",
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        
        if confirm == QtWidgets.QMessageBox.Ok:
            # Peak 데이터 초기화
            self.main.PEAK_EXTRACT_DATA['PEAK'] = []
            self.main.PEAK_EXTRACT_DATA['NOTE'] = ''
            self.main.PEAK_EXTRACT_DATA['tracked_peaks'] = None
            # 0번 페이지로 이동
            self.main.ui.stackedWidget.setCurrentIndex(0)
    
    def initialize_peak_tracking(self, contour_data):
        """피크 추적 프로세스 초기화"""
        # 데이터 복사
        self.main.PEAK_EXTRACT_DATA['PEAK'] = contour_data.copy()
        
        # tracked_peaks 초기화
        self.main.PEAK_EXTRACT_DATA["tracked_peaks"] = {
            "Series": contour_data["Series"],
            "Time-temp": contour_data["Time-temp"],
            "Data": []
        }
        
        self.current_index = 0
        self.max_index = len(contour_data['Data']) - 1
        self.current_peak_selection = None
        
        # UI 페이지 변경 및 q range 선택 설정
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()
    
    def setup_q_range_selection(self):
        """q range 선택 화면 설정"""
        if hasattr(self.ui, 'QGV_qrange'):
            if self.ui.QGV_qrange.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_qrange.layout())
                
            plot_widget = pg.PlotWidget()
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(plot_widget)
            self.ui.QGV_qrange.setLayout(layout)
            
            self.q_correction_helper = QRangeCorrectionHelper(plot_widget)
            current_entry = self.main.PEAK_EXTRACT_DATA['PEAK']['Data'][self.current_index]
            self.q_correction_helper.set_data(current_entry['q'], current_entry['Intensity'])
            self.q_correction_helper.add_selection_lines()
            
            # 현재 피크 추적 상태에 따른 메시지
            if self.current_peak_selection and 'start_flag' in self.current_peak_selection:
                current_start_flag = self.current_peak_selection['start_flag']
                current_peaks = [p for p in self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"]
                            if p.get("peak_name", "").startswith(f"peak_{current_start_flag}_")]
                
                if current_peaks:
                    peak_q = current_peaks[0]["peak_q"]
                    self.ui.L_current_status_1.setText(
                        f"Select q range for frame {self.current_index} / {self.max_index} "
                        f"(Current peak: q = {peak_q:.4f})"
                    )
                else:
                    self.ui.L_current_status_1.setText(
                        f"Select q range for frame {self.current_index} / {self.max_index}"
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
                
        # 현재 피크 정보 저장용 변수
        current_result = None
            
        if self.in_adjustment_mode:
            # 단일 프레임 수정 모드
            result = find_peak_extraction(
                self.main.PEAK_EXTRACT_DATA['PEAK'],
                Index_number=self.current_index,
                input_range=q_range,
                peak_info=None,
                fitting_function=PARAMS.get('fitting_model', 'gaussian'),
                threshold_config=None,
                start_flag=self.current_peak_selection['start_flag']
            )
        else:
            # 신규 또는 연속 추적 모드
            start_flag = self.current_index if self.current_peak_selection is None else None
            peak_info = None if start_flag is not None else self.current_peak_selection
            threshold_config = None if start_flag is None else FITTING_THRESHOLD
                
            result = find_peak_extraction(
                self.main.PEAK_EXTRACT_DATA['PEAK'],
                Index_number=self.current_index,
                input_range=q_range,
                peak_info=peak_info,
                fitting_function=PARAMS.get('fitting_model', 'gaussian'),
                threshold_config=threshold_config,
                start_flag=start_flag
            )
            
        if isinstance(result, str):
            self.peak_found = False
            self.ui.L_current_status_1.setText(
                f"Peak not found for frame {self.current_index}: {result}"
            )
            return
            
        peak_q, peak_intensity, _, fwhm, peak_name = result
        current_entry = self.main.PEAK_EXTRACT_DATA['PEAK']['Data'][self.current_index]
            
        new_result = {
            "frame_index": self.current_index,
            "Time": current_entry["Time"],
            "Temperature": current_entry.get("Temperature", 0),
            "peak_q": peak_q,
            "peak_Intensity": peak_intensity,
            "fwhm": fwhm,
            "peak_name": peak_name
        }
            
        # 데이터 추가 및 정렬
        self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"].append(new_result)
        self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"].sort(
            key=lambda x: x["frame_index"]
        )
            
        # Debug: tracked_peaks 상태 출력
        self.print_tracked_peaks_status(self.main.PEAK_EXTRACT_DATA["tracked_peaks"])
            
        self.peak_found = True
        current_result = new_result  # 현재 찾은 피크 저장
            
        # 첫 피크면 current_peak_selection 설정하고 자동 추적 시작
        if not self.in_adjustment_mode and self.current_peak_selection is None:
            self.current_peak_selection = {
                "peak_q": peak_q,
                "peak_intensity": peak_intensity,
                "fwhm": fwhm,
                "peak_name": peak_name,
                "start_flag": int(peak_name.split('_')[1])  # peak_{start_flag}_{q} 형식에서 추출
            }
            print(f"Starting automatic tracking for peak at q = {peak_q:.4f}")
            self.run_automatic_tracking()  # 자동 추적 시작
        else:
            if not self.in_adjustment_mode:
                # 수정 모드가 아니면 다음 프레임으로
                self.current_index += 1
                if self.current_index <= self.max_index:
                    self.setup_q_range_selection()
                else:
                    self.show_contour_progress()
            else:
                # 수정 모드면 컨투어 플롯으로
                self.show_contour_progress()
    
    def print_tracked_peaks_status(self, tracked_peaks):
        """tracked_peaks의 현재 상태를 간단히 출력"""
        if not tracked_peaks or "Data" not in tracked_peaks:
            print("No tracked peaks data available")
            return

        # 피크 그룹별로 분류
        peaks_by_group = {}
        for entry in tracked_peaks["Data"]:
            peak_name = entry.get("peak_name", "")
            if peak_name.startswith("peak_"):
                parts = peak_name.split("_")
                if len(parts) >= 3:
                    group_id = int(parts[1])
                    if group_id not in peaks_by_group:
                        peaks_by_group[group_id] = []
                    peaks_by_group[group_id].append(entry)

        print("\n=== Tracked Peaks Status ===")
        print(f"Total peaks found: {len(tracked_peaks['Data'])}")
        print("Peak groups:")
        for group_id, peaks in peaks_by_group.items():
            first_peak = peaks[0]
            print(f"  Group {group_id}: {len(peaks)} peaks, q = {first_peak['peak_q']:.4f}")
        print("========================\n")

            
    def run_automatic_tracking(self):
        """현재 피크에 대한 자동 추적 실행"""
        try:
            self.current_index += 1  # 첫 번째 프레임은 이미 처리됨
            while self.current_index <= self.max_index:
                # 자동 추적에서는 threshold 체크 없이 진행
                result = find_peak_extraction(
                    self.main.PEAK_EXTRACT_DATA['PEAK'],
                    Index_number=self.current_index,
                    input_range=self.q_correction_helper.get_q_range(),
                    peak_info=None,  # threshold 체크 비활성화를 위해 None
                    fitting_function=PARAMS.get('fitting_model', 'gaussian'),
                    threshold_config=None,  # threshold 체크 비활성화
                    start_flag=self.current_peak_selection.get('start_flag')  # start_flag 유지
                )
                
                if isinstance(result, str):
                    self.peak_found = False
                    self.ui.L_current_status_1.setText(
                        f"Peak not found for frame {self.current_index}: {result}"
                    )
                    break
                    
                peak_q, peak_intensity, _, fwhm, peak_name = result
                current_entry = self.main.PEAK_EXTRACT_DATA['PEAK']['Data'][self.current_index]
                
                new_result = {
                    "frame_index": self.current_index,
                    "Time": current_entry["Time"],
                    "Temperature": current_entry.get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity,
                    "fwhm": fwhm,
                    "peak_name": peak_name
                }
                
                self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"].append(new_result)
                self.current_index += 1
                
            self.show_contour_progress()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Error during automatic tracking:\n{str(e)}"
            )
            
    def show_contour_progress(self):
        """현재까지의 진행상황을 contour plot으로 표시"""
        # 계산된 최소 크기
        fig_size = (12, 8)  # 기본 figure 크기
        dpi = 300  # 기본 DPI
        min_width = int(fig_size[0] * dpi / 4)  # 최소 크기는 원본의 1/4
        min_height = int(fig_size[1] * dpi / 4)
        
        # GraphicsView에 최소 크기 설정
        self.ui.QGV_contour.setMinimumSize(min_width, min_height)
        
        # 기존 레이아웃 초기화
        if self.ui.QGV_contour.layout():
            QtWidgets.QWidget().setLayout(self.ui.QGV_contour.layout())
            
        # 컨투어 플롯 생성
        canvas = plot_contour_extraction(
            self.main.PEAK_EXTRACT_DATA['PEAK'],
            self.main.PEAK_EXTRACT_DATA.get('tracked_peaks'),
            GUI=True
        )
        
        if canvas is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "Warning", 
                "Could not create contour plot"
            )
            return
            
        # 위젯 컨테이너 생성
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(canvas)
        
        # 크기 정책 설정
        canvas.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Ignored
        )
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding
        )
        
        # GraphicsView에 추가
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(container)
        self.ui.QGV_contour.setLayout(layout)
        
        # 클래스 속성으로 저장
        self.canvas = canvas
        self.canvas_original_size = canvas.size()
        
        # Contour plot 페이지로 이동
        self.main.ui.stackedWidget.setCurrentIndex(2)

        # -----------------------------------------------------
        # 상태 확인 및 UI 업데이트
        # -----------------------------------------------------
        if self.current_index > self.max_index:
            # 현재 추적 중인 피크의 완료 메시지 표시
            if self.current_peak_selection and 'start_flag' in self.current_peak_selection:
                current_start_flag = self.current_peak_selection['start_flag']
                # 현재 start_flag의 피크들 중 가장 최근 것을 찾음
                current_peaks = sorted(
                    [p for p in self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"]
                    if p.get("peak_name", "").startswith(f"peak_{current_start_flag}_")],
                    key=lambda x: x["frame_index"]
                )
                
                if current_peaks:
                    # 가장 최근 피크의 q 값으로 메시지 생성
                    latest_peak = current_peaks[-1]
                    peak_q = latest_peak["peak_q"]
                    self.ui.L_current_status_2.setText(f"Peak finding completed for q = {peak_q:.4f}")
                else:
                    self.ui.L_current_status_2.setText("Peak finding completed")
            else:
                self.ui.L_current_status_2.setText("Peak finding completed")


            # 버튼 연결 변경
            try:
                self.ui.PB_next.clicked.disconnect()
                self.ui.PB_sdd_correction_start.clicked.disconnect()
            except Exception:
                pass
            
            # Next 버튼 → Adjust specific Peak
            self.ui.PB_next.setText("Adjust selected peak")
            self.ui.PB_next.clicked.connect(self.modify_peak)
            self.ui.PB_sdd_correction_start.setText("Start new peak tracking")
            self.ui.PB_sdd_correction_start.clicked.connect(self.start_new_peak_tracking)
            
        else:  # 아직 처리되지 않은 프레임이 존재
            if self.peak_found:
                base_status = f"Processing frame {self.current_index + 1} / {self.max_index + 1}"
            else:
                base_status = f"Peak detection failed at frame {self.current_index}. Click Next to retry."
                
            # 현재 진행 중인 피크 상태 표시 (가장 최근 피크 사용)
            if self.current_peak_selection and 'start_flag' in self.current_peak_selection:
                current_start_flag = self.current_peak_selection['start_flag']
                current_peaks = sorted(
                    [p for p in self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"]
                    if p.get("peak_name", "").startswith(f"peak_{current_start_flag}_")],
                    key=lambda x: x["frame_index"]
                )
                
                if current_peaks:
                    # 가장 최근 피크의 q 값 표시
                    latest_peak = current_peaks[-1]
                    peak_q = latest_peak["peak_q"]
                    self.ui.L_current_status_2.setText(f"{base_status} (Current peak: q = {peak_q:.4f})")
                else:
                    self.ui.L_current_status_2.setText(base_status)
            else:
                self.ui.L_current_status_2.setText(base_status)

            
            # Next 버튼 설정
            try:
                self.ui.PB_next.clicked.disconnect()
            except Exception:
                pass
            self.ui.PB_next.setText("Next")
            self.ui.PB_next.clicked.connect(self.retry_current_frame)

            
    def setup_peak_adjustment(self):
        """컨투어 플롯에서 현재 피크 선택을 위한 설정"""
        if hasattr(self.ui, 'QGV_contour'):
            canvas = plot_contour_extraction(
                self.main.PEAK_EXTRACT_DATA['PEAK'],
                self.main.PEAK_EXTRACT_DATA.get('tracked_peaks'),
                GUI=True
            )
            
            if canvas is None:
                return
                
            if self.ui.QGV_contour.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_contour.layout())
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(canvas)
            self.ui.QGV_contour.setLayout(layout)
            
            self.canvas = canvas  # 캔버스 참조 저장
            self.setup_adjust_interaction()
            
    def setup_adjust_interaction(self):
        """피크 선택을 위한 마우스 이벤트 설정"""
        if not hasattr(self, 'canvas'):
            return
            
        self.selected_frame_index = None
        
        def on_click(event):
            if event.inaxes != self.canvas.figure.axes[0]:
                return
                
            # 현재 tracked_peaks의 데이터와 클릭 위치 비교
            min_dist = float('inf')
            closest_index = None
            
            tracked_data = self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"]
            start_flag = self.current_peak_selection.get('start_flag')
            
            for entry in tracked_data:
                # 현재 그룹의 피크만 선택 가능
                if entry["peak_name"].startswith(f"peak_{start_flag}"):
                    dx = event.xdata - entry["peak_q"]
                    dy = event.ydata - entry["Time"]
                    dist = np.sqrt(dx**2 + dy**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_index = entry["frame_index"]
            
            if closest_index is not None and min_dist < 0.1:  # 적절한 threshold
                self.selected_frame_index = closest_index
                self.ui.L_current_status_2.setText(f"Selected frame {closest_index}")
                
        self.canvas.mpl_connect('button_press_event', on_click)
        
    def modify_peak(self):
        """선택된 피크 수정"""
        if not hasattr(self.canvas, 'get_selected_peak_info'):
            QtWidgets.QMessageBox.warning(
                self.main,
                "Warning",
                "Please click on a peak point to select it first."
            )
            return
            
        group_id, selected_idx = self.canvas.get_selected_peak_info()
        if group_id is None or selected_idx is None:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Warning",
                "Please click on a peak point to select it first."
            )
            return
        
        try:
            # 선택된 피크 데이터 가져오기
            selected_data = self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"][selected_idx]
            frame_to_adjust = selected_data.get("frame_index")
            
            # 만약 frame_to_adjust가 전체 범위를 초과하면 최대 인덱스로 보정
            if frame_to_adjust is None or frame_to_adjust >= len(self.main.PEAK_EXTRACT_DATA['PEAK']['Data']):
                frame_to_adjust = len(self.main.PEAK_EXTRACT_DATA['PEAK']['Data']) - 1
                
            self.current_index = frame_to_adjust
            print(f"Selected peak from frame {self.current_index} for adjustment (Group {group_id})")
            
            # 해당 프레임의 기존 결과 삭제
            del self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"][selected_idx]
            
            # 조정 모드 설정 (피크 그룹 정보 유지)
            self.in_adjustment_mode = True
            self.current_peak_selection = {
                "start_flag": group_id  # 원래 그룹의 start_flag 유지
            }
            
            # q range 선택 페이지로 이동
            self.main.ui.stackedWidget.setCurrentIndex(1)
            self.setup_q_range_selection()
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Failed to modify peak: {str(e)}"
            )


    def retry_current_frame(self):
        """현재 프레임의 피크 재검출을 위해 q range 선택으로 돌아감"""
        # threshold 실패한 프레임의 peak_info 재설정
        if self.current_index > 0:
            # 이전 프레임의 피크 정보 가져오기
            prev_peaks = [p for p in self.main.PEAK_EXTRACT_DATA["tracked_peaks"]["Data"]
                        if p["frame_index"] == self.current_index - 1]
            if prev_peaks:
                # 같은 그룹의 이전 피크 찾기
                prev_peak = None
                for p in prev_peaks:
                    if p.get("peak_name", "").startswith(f"peak_{self.current_peak_selection['start_flag']}_"):
                        prev_peak = p
                        break
                
                if prev_peak:
                    # 이전 피크의 정보로 current_peak_selection 업데이트
                    self.current_peak_selection.update({
                        "peak_q": prev_peak["peak_q"],
                        "peak_intensity": prev_peak["peak_Intensity"],
                        "fwhm": prev_peak.get("fwhm")
                    })

        # q range 선택 페이지로 이동
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()


    def start_new_peak_tracking(self):
        """새로운 피크 추적 시작"""
        # 기존 current_peak_selection을 완전히 초기화
        self.current_peak_selection = None
        self.current_index = 0
        self.in_adjustment_mode = False
        self.peak_found = None  # peak_found도 초기화
        
        # q range 선택 페이지로 이동
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()
