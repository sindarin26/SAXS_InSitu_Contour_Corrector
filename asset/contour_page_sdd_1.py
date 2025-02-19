# asset/contour_page_sdd_1.py
from PyQt5 import QtWidgets, QtCore
import numpy as np
import pyqtgraph as pg
from asset.contour_storage import DATA, PARAMS
from asset.contour_util_gui import QRangeCorrectionHelper, plot_contour_with_peaks_gui 
from asset.contour_util import find_peak

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
        self.ui.PB_final_apply.clicked.connect(self.finalize_peak_tracking)

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
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()

    def setup_q_range_selection(self):
        """1번 페이지: QGV에 현재 프레임의 데이터를 띄우고 q range 선택"""
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
            self.q_correction_helper = QRangeCorrectionHelper(plot_widget)
            current_entry = self.contour_data['Data'][self.current_index]
            self.q_correction_helper.set_data(current_entry['q'], current_entry['Intensity'])
            self.q_correction_helper.add_selection_lines()
            self.ui.L_current_status_1.setText(
                f"Select q range for frame {self.current_index} / {self.max_index}"
            )

    def apply_q_range(self):
        """
        1번 페이지에서 사용자가 q range를 선택하면 그 범위를 저장하고,
        조정 모드 여부에 따라 단일 프레임(조정 모드) 또는 전체 프레임(자동 모드)에 대해 피크 검출을 진행합니다.
        """
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(self.main, "Warning", "Please select a q range first.")
            return
        self.global_q_range = q_range
        if self.in_adjustment_mode:
            # 조정 모드: 현재 프레임에 대해서만 피크 검출
            result = find_peak(self.contour_data, Index_number=self.current_index, input_range=self.global_q_range)
            if result is None:
                self.peak_found = False
                self.ui.L_current_status_2.setText(
                    f"Peak not found for frame {self.current_index}. Please adjust q range."
                )
            else:
                self.peak_found = True
                peak_q, peak_intensity, _ = result
                current_entry = self.contour_data['Data'][self.current_index]
                new_result = {
                    "frame_index": self.current_index,
                    "Time": current_entry["Time"],
                    "Temperature": current_entry.get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity
                }
                # 단일 프레임 재검출 결과는 append하고, frame_index 순으로 정렬
                self.tracked_peaks["Data"].append(new_result)
                self.tracked_peaks["Data"].sort(key=lambda x: x["frame_index"])
            # 조정 모드 종료
            self.in_adjustment_mode = False
            self.show_contour_page()
        else:
            self.run_automatic_tracking()

    def run_automatic_tracking(self):
        """
        전역 q range를 사용하여 현재 프레임부터 최대 프레임까지 자동으로 피크 검출을 진행합니다.
        검출 실패 시 해당 프레임에서 멈추고 2번 페이지로 전환하여 결과를 보여줍니다.
        """
        try:
            while self.current_index <= self.max_index:
                result = find_peak(self.contour_data, Index_number=self.current_index, input_range=self.global_q_range)
                if result is None:
                    self.peak_found = False
                    self.ui.L_current_status_2.setText(
                        f"Peak not found for frame {self.current_index}. Please adjust q range."
                    )
                    break
                else:
                    self.peak_found = True
                    peak_q, peak_intensity, _ = result
                    current_entry = self.contour_data['Data'][self.current_index]
                    self.tracked_peaks["Data"].append({
                        "frame_index": self.current_index,
                        "Time": current_entry["Time"],
                        "Temperature": current_entry.get("Temperature", 0),
                        "peak_q": peak_q,
                        "peak_Intensity": peak_intensity
                    })
                    self.current_index += 1
            if self.current_index > self.max_index:
                self.ui.L_current_status_2.setText("Peak find Completed")
            self.show_contour_page()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self.main, "Error", f"Error during automatic tracking:\n{str(e)}")

    def show_contour_page(self):
        """2번 페이지: 컨투어 플롯 및 현재까지의 피크 결과 표시"""
        canvas = plot_contour_with_peaks_gui(self.contour_data, self.tracked_peaks)
        if canvas is None:
            QtWidgets.QMessageBox.warning(self.main, "Warning", "Could not create contour plot.")
            return
        if hasattr(self.ui, 'QGV_contour'):
            if self.ui.QGV_contour.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_contour.layout())
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(canvas)
            self.ui.QGV_contour.setLayout(layout)
        self.adjust_canvas = canvas
        self.setup_adjust_interaction()
        # --- 여기서 버튼 텍스트 결정 ---
        if len(self.tracked_peaks["Data"]) == (self.max_index + 1):
            self.ui.L_current_status_2.setText("Peak find Completed")
            try:
                self.ui.PB_next.clicked.disconnect()
            except Exception:
                pass
            self.ui.PB_next.setText("Adjust specific Peak")
            self.ui.PB_next.clicked.connect(self.adjust_specific_peak)
        else:
            self.ui.PB_next.setText("Next")
            try:
                self.ui.PB_next.clicked.disconnect()
            except Exception:
                pass
            self.ui.PB_next.clicked.connect(self.retry_current_frame)
            self.ui.PB_final_apply.setEnabled(False)
        self.main.ui.stackedWidget.setCurrentIndex(2)

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
        """
        2번 페이지의 Next 버튼 동작:
        자동 검출 실패 시, 현재 프레임에 대해 1번 페이지로 돌아가 q range를 재조정합니다.
        """
        if self.current_index > self.max_index:
            QtWidgets.QMessageBox.information(self.main, "Info", "All frames processed.")
            return
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()

    def adjust_specific_peak(self):
        """
        2번 페이지에서 "Adjust specific Peak" 버튼을 누르면,
        만약 사용자가 캔버스에서 피크를 선택했다면, 해당 피크의 저장된 frame_index를 가져와 current_index로 설정하고,
        해당 프레임의 기존 피크 결과를 삭제한 후, 조정 모드로 전환하여 1번 페이지에서 단일 프레임에 대해 q range 재조정합니다.
        """
        if self.selected_peak_index is None:
            QtWidgets.QMessageBox.information(self.main, "Info", "Please click on a peak to select it first.")
            return
        selected_data = self.tracked_peaks["Data"][self.selected_peak_index]
        frame_to_adjust = selected_data.get("frame_index", None)
        # 만약 frame_to_adjust가 전체 범위와 같거나 초과하면 최대 인덱스로 보정
        if frame_to_adjust is None or frame_to_adjust >= len(self.contour_data['Data']):
            frame_to_adjust = len(self.contour_data['Data']) - 1
        self.current_index = frame_to_adjust
        print(f"Selected peak from frame {self.current_index} for adjustment.")
        # 해당 프레임의 기존 결과 삭제
        del self.tracked_peaks["Data"][self.selected_peak_index]
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
        # 조정 모드 활성화: 단일 프레임에 대해서만 피크 검출 진행
        self.in_adjustment_mode = True
        self.main.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_selection()

    def finalize_peak_tracking(self):
        """최종 완료: 모든 프레임이 처리되었을 때 후속 처리/저장 진행"""
        if self.tracked_peaks is None or len(self.tracked_peaks["Data"]) == 0:
            QtWidgets.QMessageBox.warning(self.main, "Warning", "No peaks have been tracked.")
            return
        print("Peak tracking completed")
        print(f"Tracked peaks: {self.tracked_peaks}")