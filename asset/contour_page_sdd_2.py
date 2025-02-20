#asset.contour_page_sdd_2.py
from PyQt5 import QtWidgets, QtCore
import numpy as np
from asset.contour_storage import DATA, PLOT_OPTIONS, PARAMS, PATH_INFO
from asset.contour_util_gui import IndexRangeSelectionHelper, PeakTempRangeHelper
from asset.contour_util import (plot_contour, fit_peak_vs_temp, calculate_corrected_sdd)
from asset.contour_util import theta_to_q, q_to_2theta, calculate_corrected_sdd
import pyqtgraph as pg
import traceback
import copy

class SDDFittingPage(QtCore.QObject):
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui

        # Initialize variables
        self.index_range = None
        self.peak_fit_temp_range = None
        self.fit_params = None
        self.current_step = 'index_range'
        self.temp_data = None  # 임시 데이터 저장용
        
        # Initialize helpers
        self.index_range_helper = None
        self.peak_temp_helper = None
        
        # Setup initial state
        self.setup_initial_state()
        self.connect_signals()

    def setup_initial_state(self):
        """Initialize UI state"""
        self.ui.PB_final_apply.setEnabled(False)
        self.L_current_status_3 = self.ui.L_current_status_3

    def recalc_q_list(self, q_list, original_sdd, corrected_sdd, energy_keV, converted_energy=8.042):
        """
        q 리스트를 새로운 SDD 값으로 재계산
        
        Parameters
        ----------
        q_list : array-like
            원래 데이터. 
            - converted_energy가 None이면 q 값 (Å⁻¹)
            - converted_energy가 주어지면 CuKα 기준의 2θ 값 (°)
        original_sdd : float
            원래 SDD (mm)
        corrected_sdd : float
            보정된 SDD (mm)
        energy_keV : float
            exp_energy (q 또는 2θ 변환 시 사용되는 에너지, keV)
        converted_energy : float, optional
            입력 데이터가 CuKα 기준일 경우의 에너지 (keV). 
            None이면 변환 없이 q 값으로 처리.
        
        Returns
        -------
        corrected_q_list : ndarray
            - converted_energy가 None이면: 보정 후 q 값 (Å⁻¹)
            - converted_energy가 주어지면: 보정 후 CuKα 기준의 2θ 값 (°)
        """

        # 만약 converted_energy가 주어졌다면, 입력 데이터는 2θ 값(°)이므로 q로 변환
        if converted_energy is None:
            q_conv = q_list
        else:
            q_conv = theta_to_q(q_list, converted_energy)
        
        # 1) exp_energy 기준의 2θ 값으로 변환
        exp_2theta = q_to_2theta(q_conv, energy_keV)
        
        # 2) 원래 SDD에서의 반경 R 계산
        R = original_sdd * np.tan(np.radians(exp_2theta))
        
        # 3) 보정 SDD 적용 시의 새로운 2θ (exp_energy 기준)
        corrected_2theta = np.degrees(np.arctan(R / corrected_sdd))
        
        # 4) 새로운 2θ를 exp_energy 기준의 q로 변환
        q_temp = theta_to_q(corrected_2theta, energy_keV)
        
        # 5) 만약 converted_energy가 주어졌다면, 최종 결과를 CuKα 기준의 2θ 값으로 변환
        if converted_energy is None:
            corrected_q_list = q_temp
        else:
            corrected_q_list = q_to_2theta(q_temp, converted_energy)
        
        return corrected_q_list

    def calculate_sdd_for_tracked_peaks_refactored(self, contour_data, tracked_peaks, original_sdd, experiment_energy, converted_energy=8.042):
        """각 타임프레임의 q 리스트를 SDD 보정"""
        # 입력 데이터 복사
        result_data = copy.deepcopy(contour_data)
        
        # contour_data의 각 항목에 대해 원래 q 리스트 백업
        for entry in result_data["Data"]:
            if "q_raw" not in entry:
                entry["q_raw"] = np.array(entry["q"])
        
        # index-wise로 처리
        for i, entry in enumerate(result_data["Data"]):
            try:
                peak_entry = tracked_peaks["Data"][i]
            except IndexError:
                print(f"DEBUG: No tracked peak data for frame {i}")
                continue
            
            # corrected_SDD와 corrected_peak_q 확인
            corrected_sdd = peak_entry.get("corrected_SDD")
            corrected_peak_q = peak_entry.get("corrected_peak_q")
            
            if corrected_sdd is not None and corrected_peak_q is not None:
                print(f"DEBUG: Frame {i}: Applying SDD correction")
                print(f"DEBUG: Original SDD: {original_sdd}, Corrected SDD: {corrected_sdd}")
                print(f"DEBUG: Original peak_q: {peak_entry['peak_q']}, Corrected peak_q: {corrected_peak_q}")
                
                # q_list 재계산
                new_q = self.recalc_q_list(
                    q_list=entry["q_raw"],
                    original_sdd=original_sdd,
                    corrected_sdd=corrected_sdd,
                    energy_keV=experiment_energy,
                    converted_energy=converted_energy
                )
                
                # 결과 확인
                print(f"DEBUG: Q range before correction: {min(entry['q_raw']):.4f}-{max(entry['q_raw']):.4f}")
                print(f"DEBUG: Q range after correction: {min(new_q):.4f}-{max(new_q):.4f}")
                
                # 재계산된 q 리스트로 덮어쓰기
                entry["q"] = new_q
            else:
                print(f"DEBUG: Frame {i}: No correction needed")
                entry["q"] = entry["q_raw"]
        
        return result_data

    def add_corrected_peak_q(
        self,
        tracked_peaks,
        fit_params,
        peak_fit_temp_range,
        original_SDD,
        experiment_energy,
        converted_energy
    ):
        """
        fit_params를 사용하여 corrected_peak_q 값을 계산하고 tracked_peaks에 추가
        ...
        """
        a, b = fit_params
        temp_max = peak_fit_temp_range[1]

        for entry in tracked_peaks["Data"]:
            T = entry["Temperature"]
            entry["original_SDD"] = original_SDD

            if T > temp_max:
                corrected_q = a * T + b
                entry["corrected_peak_q"] = corrected_q
                entry["corrected_SDD"] = round(
                    calculate_corrected_sdd(
                        entry["peak_q"],
                        original_SDD,
                        corrected_q,
                        experiment_energy,
                        converted_energy
                    ), 4
                )
            else:
                entry["corrected_peak_q"] = None
                entry["corrected_SDD"] = None

        return tracked_peaks

    def calculate_corrected_sdd(self, original_q, original_sdd, corrected_q, exp_energy, converted_energy):
        """Calculate corrected SDD for a single point"""
        return round(calculate_corrected_sdd(
            original_q, original_sdd, corrected_q,
            exp_energy, converted_energy
        ), 4)
        
    def initialize_fitting(self):
        """Start the fitting process"""
        self.current_step = 'index_range'
        self.setup_index_range_selection()
        self.L_current_status_3.setText("Select Index Range for Temperature Fitting")
        self.ui.PB_apply_final.setEnabled(True)
        self.ui.PB_final_apply.setEnabled(False)
        
    def setup_index_range_selection(self):
        """Setup the index range selection view"""
        if hasattr(self.ui, 'QGV_final'):
            if self.ui.QGV_final.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())
                
            layout = QtWidgets.QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            self.ui.QGV_final.setLayout(layout)
            
            # Initialize helper
            self.index_range_helper = IndexRangeSelectionHelper(plot_widget)
            
            if DATA['tracked_peaks'] is None:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    "No tracked peaks data available."
                )
                return
                
            temp_data = DATA['tracked_peaks']["Time-temp"][1]
            self.index_range_helper.set_data(temp_data)
            self.index_range_helper.add_selection_lines()
        
    def setup_peak_temp_range_selection(self):
        """Setup the peak vs temperature range selection view"""
        if hasattr(self.ui, 'QGV_final'):
            if self.ui.QGV_final.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())
                
            layout = QtWidgets.QVBoxLayout()
            plot_widget = pg.PlotWidget()
            layout.addWidget(plot_widget)
            self.ui.QGV_final.setLayout(layout)
            
            # Initialize helper
            self.peak_temp_helper = PeakTempRangeHelper(plot_widget)
            
            # Get data within selected index range
            tracked_peaks = DATA['tracked_peaks']
            start_idx, end_idx = self.index_range
            temp_data = np.array(tracked_peaks["Time-temp"][1])[start_idx:end_idx+1]
            peak_q_data = np.array([entry["peak_q"] for entry in tracked_peaks["Data"][start_idx:end_idx+1]])
            
            self.peak_temp_helper.set_data(temp_data, peak_q_data)
            self.peak_temp_helper.add_selection_lines()

    def show_preview_contour(self, contour_data=None):
        """
        *이미 보정된* contour_data를 받아서,
        QGV_final 위젯에 플롯(미리보기)만 담당하는 메서드.

        보정 계산(add_corrected_peak_q, calculate_sdd_for_tracked_peaks_refactored)
        등은 이 함수 호출 전에 다른 곳(예: on_apply_range)에서 끝낸다.
        """
        try:
            # 1) 기존 레이아웃 초기화 (로딩 메시지)
            if self.ui.QGV_final.layout():
                QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())

            loading = QtWidgets.QLabel("Preparing preview plot...")
            loading.setAlignment(QtCore.Qt.AlignCenter)
            self.ui.QGV_final.setLayout(QtWidgets.QVBoxLayout())
            self.ui.QGV_final.layout().addWidget(loading)
            QtWidgets.QApplication.processEvents()

            # 2) 매개변수로 contour_data를 직접 받지 않았다면,
            #    self.temp_data['contour_data']를 사용
            if contour_data is None:
                if not self.temp_data or 'contour_data' not in self.temp_data:
                    raise RuntimeError("No contour_data available for preview.")
                contour_data = self.temp_data['contour_data']

            preview_graph_option = PLOT_OPTIONS['graph_option'].copy()
            preview_graph_option['figure_title_enable'] = False
            preview_graph_option['contour_title_enable'] = False
            preview_graph_option['temp_title_enable'] = False
            preview_graph_option['legend'] = False
            preview_graph_option['temp'] = False
            preview_graph_option['contour_xlabel_enable'] = False
            preview_graph_option['contour_ylabel_enable'] = False

            # 3) 실제 Matplotlib 플롯 생성
            layout = QtWidgets.QVBoxLayout()
            canvas = plot_contour(
                contour_data,
                temp=False,
                legend=False,
                graph_option=PLOT_OPTIONS['graph_option'],
                GUI=True
            )

            # 4) canvas를 QGV_final 레이아웃에 연결
            if canvas:
                if self.ui.QGV_final.layout():
                    QtWidgets.QWidget().setLayout(self.ui.QGV_final.layout())
                layout.addWidget(canvas)
                self.ui.QGV_final.setLayout(layout)

                print("DEBUG: Preview plot created successfully")
            else:
                raise RuntimeError("Failed to create preview plot")

        except Exception as e:
            print(f"DEBUG: Error in show_preview_contour: {str(e)}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Failed to generate preview: {str(e)}"
            )

    def on_back(self):
        """Handle back button click"""
        self.index_range = None
        self.peak_fit_temp_range = None
        self.fit_params = None
        self.current_step = 'index_range'
        self.temp_data = None
        self.ui.stackedWidget.setCurrentIndex(2)
        
    def connect_signals(self):
        """Connect button signals"""
        self.ui.PB_back_3.clicked.connect(self.on_back)
        self.ui.PB_final_apply.clicked.connect(self.on_final_apply)
        self.ui.PB_apply_final.clicked.connect(self.on_apply_range)
        
    def on_apply_range(self):
        """Handle apply range button click"""
        if self.current_step == 'index_range':
            # 1) 인덱스 범위 가져오기
            self.index_range = self.index_range_helper.get_index_range()
            if self.index_range is None:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    "Please select a valid index range first."
                )
                return

            print(f"[Debug] Index Range = {self.index_range}")
            
            # 2) 스텝 전환: 'temp_range'
            self.current_step = 'temp_range'
            self.L_current_status_3.setText("Select Temperature Range for Peak Fitting")

            # 3) 온도 범위 선택 화면 세팅
            self.setup_peak_temp_range_selection()

            # 4) 버튼 상태 업데이트
            #    (여기서는 temp_range 스텝에서도 같은 버튼을 쓰므로, 그대로 둬도 됨)
            self.ui.PB_apply_final.setEnabled(True)
            self.ui.PB_final_apply.setEnabled(False)

        elif self.current_step == 'temp_range':
            # 1) 온도 범위 가져오기
            self.peak_fit_temp_range = self.peak_temp_helper.get_temp_range()
            if self.peak_fit_temp_range is None:
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    "Please select a valid temperature range first."
                )
                return

            print("DEBUG: Calculating fit parameters")
            # 2) (a, b) 피팅
            self.fit_params = fit_peak_vs_temp(DATA['tracked_peaks'], self.peak_fit_temp_range)
            print(f"DEBUG: Fit parameters: {self.fit_params}")

            # 3) 보정
            corrected_peaks = self.add_corrected_peak_q(
                tracked_peaks=DATA['tracked_peaks'],
                fit_params=self.fit_params,
                peak_fit_temp_range=self.peak_fit_temp_range,
                original_SDD=PARAMS['original_sdd'],
                experiment_energy=PARAMS['experiment_energy'],
                converted_energy=PARAMS['converted_energy']
            )

            print(f"DEBUG: Corrected peaks: {corrected_peaks}")

            # 4) contour_data 재계산
            corrected_contour = self.calculate_sdd_for_tracked_peaks_refactored(
                DATA['contour_data'],
                corrected_peaks,
                PARAMS['original_sdd'],
                PARAMS['experiment_energy'],
                PARAMS['converted_energy']
            )

            # 5) 결과 저장 및 미리보기
            self.temp_data = {
                'tracked_peaks': corrected_peaks,
                'contour_data': corrected_contour
            }
            self.show_preview_contour(corrected_contour)

            # 6) 버튼 상태 업데이트
            self.ui.PB_apply_final.setEnabled(False)
            self.ui.PB_final_apply.setEnabled(True)
            self.current_step = 'preview'
            self.L_current_status_3.setText("Preview of Corrected Contour")

    def on_final_apply(self):
        """Apply temporary data to global DATA"""
        if self.temp_data is None:
            print("DEBUG: No temporary data available")
            return
            
        try:
            # Update the global data
            DATA['tracked_peaks'] = self.temp_data['tracked_peaks']
            DATA['contour_data'] = self.temp_data['contour_data']
            
            print("DEBUG: Global data updated with temporary data")

            # Trigger contour plot update in the main window
            self.main.parent().contour_plot_page.create_contour_plot()
            
            # Show success message and return to previous page
            QtWidgets.QMessageBox.information(
                self.main,
                "Success",
                "SDD correction has been applied successfully."
            )
            self.on_back()
            
        except Exception as e:
            print(f"DEBUG: Error in on_final_apply: {str(e)}")
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Failed to apply correction: {str(e)}"
            )