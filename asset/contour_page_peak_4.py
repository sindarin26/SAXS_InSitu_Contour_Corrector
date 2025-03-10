#asset/contour_page_peak_4.py
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from asset.contour_storage import PARAMS, FITTING_THRESHOLD
from asset.fitting_util import find_peak_extraction
from asset.contour_util_gui import QRangeCorrectionHelper
from asset.page_asset import LoadingDialog

class QRangeIndexPage(QtCore.QObject):
    """
    페이지 4: 선택된 피크의 모든 인덱스에 대한 q-range를 확인하고 수정할 수 있는 페이지
    """
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui
        
        # UI 요소
        self.CB_index = self.ui.CB_index
        self.QGV_qrange_index = self.ui.QGV_qrange_index
        self.PB_back_to_contourpage_2 = self.ui.PB_back_to_contourpage_2
        self.PB_apply_qrange_index = self.ui.PB_apply_qrange_index
        
        # 상태 변수
        self.contour_data = None
        self.selected_peak_data = []  # 선택된 피크의 모든 데이터 포인트
        self.current_peak_name = None
        self.current_frame_index = None
        self.q_correction_helper = None
        
        # 시그널 연결
        self.connect_signals()
        
    def connect_signals(self):
        """UI 버튼 및 콤보박스 시그널 연결"""
        self.PB_back_to_contourpage_2.clicked.connect(self.on_back_to_contour)
        self.PB_apply_qrange_index.clicked.connect(self.on_apply_qrange)
        self.CB_index.currentIndexChanged.connect(self.on_index_changed)
        
    def initialize_page(self, contour_data, peak_name, frame_index):
        """
        페이지 초기화 및 데이터 설정
        
        Parameters:
            contour_data (dict): 컨투어 데이터
            peak_name (str): 선택된 피크 이름
            frame_index (int): 선택된 프레임 인덱스
        """
        self.contour_data = contour_data
        self.current_peak_name = peak_name
        self.current_frame_index = frame_index
        
        # 선택된 피크 이름으로 모든 데이터 포인트 수집
        self.selected_peak_data = []
        for entry in self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
            if entry.get('peak_name') == peak_name:
                self.selected_peak_data.append(entry)
        
        # 데이터가 없으면 오류 메시지 표시 후 이전 페이지로 돌아감
        if not self.selected_peak_data:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"No data found for peak: {peak_name}"
            )
            self.on_back_to_contour()
            return
        
        # 프레임 인덱스로 정렬
        self.selected_peak_data.sort(key=lambda x: x.get('frame_index', 0))
        
        # 콤보박스 아이템 설정
        self.setup_index_combobox()
        
        # 현재 프레임 인덱스에 해당하는 아이템 선택
        self.select_current_frame()
        
    def setup_index_combobox(self):
        """콤보박스에 인덱스 항목 추가"""
        self.CB_index.clear()
        
        # 피크 이름에서 시작 인덱스 추출
        start_index = 0
        if self.current_peak_name:
            # "peak_0_1.2345" 형식에서 0 추출
            parts = self.current_peak_name.split('_')
            if len(parts) >= 2 and parts[0] == 'peak':
                try:
                    start_index = int(parts[1])
                except ValueError:
                    start_index = 0
        
        # 콤보박스에 항목 추가
        for entry in self.selected_peak_data:
            frame_idx = entry.get('frame_index', 0)
            peak_q = entry.get('peak_q', 0)
            item_text = f"{start_index}_{self.current_peak_name}_{frame_idx} (q={peak_q:.4f})"
            self.CB_index.addItem(item_text, frame_idx)  # userData로 frame_index 저장
    
    def select_current_frame(self):
        """콤보박스에서 현재 프레임 인덱스에 해당하는 항목 선택"""
        for i in range(self.CB_index.count()):
            if self.CB_index.itemData(i) == self.current_frame_index:
                self.CB_index.setCurrentIndex(i)
                return
    
    def on_index_changed(self, index):
        """콤보박스 선택 변경 시 호출되는 함수"""
        if index < 0:
            return
            
        # 선택된 프레임 인덱스 가져오기
        self.current_frame_index = self.CB_index.itemData(index)
        
        # 해당 프레임의 피크 데이터 찾기
        peak_entry = None
        for entry in self.selected_peak_data:
            if entry.get('frame_index') == self.current_frame_index:
                peak_entry = entry
                break
                
        if peak_entry:
            # q-range 그래프 설정
            self.setup_qrange_graph(peak_entry)
        
    def setup_qrange_graph(self, peak_entry):
        """
        선택된 피크의 q-range 그래프 설정
        
        Parameters:
            peak_entry (dict): 선택된 피크 데이터
        """
        if self.QGV_qrange_index.layout():
            QtWidgets.QWidget().setLayout(self.QGV_qrange_index.layout())
            
        # 그래프 위젯 생성
        plot_widget = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(plot_widget)
        self.QGV_qrange_index.setLayout(layout)
        
        # 현재 프레임 데이터 가져오기
        frame_index = peak_entry.get('frame_index', 0)
        peak_q = peak_entry.get('peak_q')
        
        # 중요: 여기서 정확한 output_range를 사용
        output_range = peak_entry.get('output_range')
        
        # 피팅에 사용된 범위가 없으면 peak_q 주변으로 고정 너비 범위 생성
        if output_range is None:
            q_width = 0.1  # 임의의 적절한 범위
            output_range = (peak_q - q_width, peak_q + q_width)
            print(f"Warning: No output_range found for frame {frame_index}, using default width around peak")
        else:
            # 디버그 출력: 실제 사용된 범위 확인
            print(f"DEBUG: Frame {frame_index}, peak_q={peak_q:.4f}, range=({output_range[0]:.4f}, {output_range[1]:.4f})")
        
        try:
            # 현재 프레임의 q, Intensity 데이터
            current_entry = self.contour_data['Data'][frame_index]
            q_data = current_entry['q']
            intensity_data = current_entry['Intensity']
            
            # QRangeCorrectionHelper 생성 및 데이터 설정
            self.q_correction_helper = QRangeCorrectionHelper(plot_widget)
            self.q_correction_helper.set_data(
                q_data, 
                intensity_data,
                peak=peak_q,
                index=frame_index,
                current_index=frame_index,
                q_range=output_range  # 정확한 범위 사용
            )
            
            self.q_correction_helper.add_selection_lines()
            
        except (IndexError, KeyError) as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Error loading data for frame {frame_index}: {str(e)}"
            )
    
    def on_apply_qrange(self):
        """q-range 변경 적용"""
        if not self.q_correction_helper:
            return
            
        # 현재 선택된 q-range 가져오기
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "Warning", 
                "Please select a valid q range first."
            )
            return
        
        # 현재 프레임 인덱스 및 피크 이름 확인
        frame_index = self.current_frame_index
        peak_name = self.current_peak_name
        
        if frame_index is None or peak_name is None:
            return
            
        # 피크 추출 실행
        result = find_peak_extraction(
            contour_data=self.contour_data,
            Index_number=frame_index,
            input_range=q_range,
            fitting_function=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_auto_tracking=False,
            flag_manual_adjust=True,  # threshold 체크 우회
            flag_start=False,  # 새 피크가 아님
            current_peak_name=peak_name
        )
        
        if isinstance(result, str):
            # 실패 시 에러 메시지 표시
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to extract peak: {result}"
            )
            return
            
        # 성공 시 결과 처리 - raw peak 데이터 포함해서 9개 값 언팩
        peak_q, peak_intensity, output_range, fwhm, peak_name, fitting_function, fitting_params, peak_q_max_raw, peak_intensity_max_raw = result
        
        # 업데이트된 데이터를 저장할 변수
        updated_entry = None
        
        # 기존 데이터 업데이트
        for i, entry in enumerate(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
            if entry.get('frame_index') == frame_index and entry.get('peak_name') == peak_name:
                # 기존 엔트리 업데이트 - raw peak 데이터 포함
                self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][i].update({
                    'peak_q': peak_q,
                    'peak_Intensity': peak_intensity,
                    'fwhm': fwhm,
                    'output_range': output_range,
                    'fitting_function': fitting_function,
                    'fitting_params': fitting_params,
                    'peak_q_max_raw': peak_q_max_raw,
                    'peak_intensity_max_raw': peak_intensity_max_raw
                })
                
                # 참조 저장 (나중에 사용)
                updated_entry = self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][i]
                
                # 성공 메시지 표시
                QtWidgets.QMessageBox.information(
                    self.main,
                    "Success",
                    f"Peak updated: q = {peak_q:.4f}, intensity = {peak_intensity:.2f}\n"
                    f"New q-range: ({output_range[0]:.4f}, {output_range[1]:.4f})"
                )
                
                # selected_peak_data 완전히 새로고침 (참조 문제 방지)
                self.selected_peak_data = []
                for global_entry in self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
                    if global_entry.get('peak_name') == peak_name:
                        # 딕셔너리 복사하여 추가 (새로운 참조 생성)
                        self.selected_peak_data.append(dict(global_entry))
                
                # 프레임 인덱스로 정렬
                self.selected_peak_data.sort(key=lambda x: x.get('frame_index', 0))
                
                # 업데이트된 데이터로 콤보박스 새로고침
                current_index = self.CB_index.currentIndex()
                self.setup_index_combobox()
                self.CB_index.setCurrentIndex(current_index)  # 현재 선택 유지
                
                # q-range 그래프 업데이트
                # 업데이트된 데이터를 기반으로 그래프 새로 그리기
                if updated_entry:
                    self.setup_qrange_graph(dict(updated_entry))  # 딕셔너리 복사하여 전달
                
                return
        
        # 해당 피크를 찾지 못한 경우
        QtWidgets.QMessageBox.warning(
            self.main,
            "Error",
            f"Could not find peak entry for frame {frame_index} and peak {peak_name}"
        )
    
    def on_back_to_contour(self):
        """
        컨투어 페이지로 돌아가고 모든 상태 초기화 (완전 대기 상태로 만듦)
        """
        # 현재 페이지의 상태 변수 초기화
        self.selected_peak_data = []
        self.current_peak_name = None
        self.current_frame_index = None
        self.q_correction_helper = None
        
        # 콤보박스 초기화
        self.CB_index.clear()
        
        # QGraphicsView 내용 초기화
        if self.QGV_qrange_index.layout():
            QtWidgets.QWidget().setLayout(self.QGV_qrange_index.layout())
        
        # 페이지 2(컨투어 페이지)로 이동
        self.ui.stackedWidget.setCurrentIndex(2)
        
        # 피크 트래킹 페이지의 상태도 초기화
        if hasattr(self.main, 'peak_tracking_page'):
            peak_page = self.main.peak_tracking_page
            
            # 모든 모드 및 선택 상태 초기화
            peak_page.check_peak_range_mode = False
            peak_page.manual_adjust = False
            peak_page.current_peak_name = None
            
            # 선택 관련 상태 초기화 (있을 경우)
            if hasattr(peak_page, 'selected_peak_index'):
                peak_page.selected_peak_index = None
            
            # UI 상태 업데이트
            peak_page.update_ui_state()
            
            # 상태 메시지 업데이트
            if hasattr(peak_page, 'L_current_status_2'):
                peak_page.L_current_status_2.setText("Ready.")
            
            # 컨투어 플롯 업데이트 (대기 상태로)
            peak_page.update_contour_plot()