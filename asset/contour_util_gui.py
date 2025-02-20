#asset.contour_util_gui.py
import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from asset.contour_storage import PLOT_OPTIONS

class DraggableLine(pg.InfiniteLine):
    """Draggable vertical line with label"""
    def __init__(self, pos=0, angle=90, movable=True, bounds=None):
        super().__init__(pos=pos, angle=angle, movable=movable)
        
        # bounds 설정
        if bounds is not None:
            self.setBounds(bounds)

        # 이벤트 연결
        self.sigPositionChangeFinished.connect(self.on_position_changed)
        self.label = None
    
    def set_label(self, label_item):
        self.label = label_item
    
    def on_position_changed(self):
        """Update label position when line is moved"""
        if self.label:
            self.label.setPos(self.value(), self.label.pos().y())

class TempCorrectionHelper:
    """Helper class for temperature correction operations"""
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.steady_lines = []
        self.adjust_lines = []
        self.temp_data = None
        self.selection_mode = "steady"
        
    def set_data(self, temp_data):
        """Set temperature data"""
        self.temp_data = temp_data
        self.plot_data()
        
    def plot_data(self):
        """Plot temperature data"""
        if self.temp_data is None:
            return
            
        self.plot_widget.clear()
        indices = np.arange(len(self.temp_data))
        self.plot_widget.plot(
            indices, 
            self.temp_data,
            pen=pg.mkPen(color='k', width=1),  # 검정색 실선
            symbol='o',                         # 동그라미 마커
            symbolSize=5,                       # 마커 크기
            symbolPen=pg.mkPen('k'),           # 마커 테두리 검정
            symbolBrush=pg.mkBrush('w')        # 마커 내부 흰색
        )

        
    def add_selection_lines(self):
        if self.temp_data is None:
            return

        # 플롯 초기화 및 전체 데이터 플롯
        self.plot_widget.clear()
        indices = np.arange(len(self.temp_data))
        self.plot_widget.plot(
            indices, 
            self.temp_data,
            pen=pg.mkPen(color='k', width=1),  # 검정색 실선
            symbol='o',                         # 동그라미 마커
            symbolSize=5,                       # 마커 크기
            symbolPen=pg.mkPen('k'),           # 마커 테두리 검정
            symbolBrush=pg.mkBrush('w')        # 마커 내부 흰색
        )

        # y좌표를 어느 정도 위로 설정 (라벨을 보기 좋게)
        y_offset = (np.max(self.temp_data) - np.min(self.temp_data)) * 0.1
        label_ypos = np.max(self.temp_data) + y_offset

        if self.selection_mode == "steady":
            pos1, pos2 = len(indices)//4, len(indices)*3//4

            line1 = DraggableLine(pos=pos1, bounds=(0, len(indices)-1))
            line2 = DraggableLine(pos=pos2, bounds=(0, len(indices)-1))

            line1.setPen(pg.mkPen(color='b', width=1.5))  # 파란색 실선
            line2.setPen(pg.mkPen(color='b', width=1.5))


            # 라벨 생성 (초기 텍스트: "Start\n(인덱스, 온도°C)")
            label1 = pg.TextItem(color=(0, 0, 0))
            label2 = pg.TextItem(color=(0, 0, 0))

            # 우선 임시 위치에 라벨 배치
            label1.setPos(pos1, label_ypos)
            label2.setPos(pos2, label_ypos)

            # 라벨을 라인에 연결
            line1.set_label(label1)
            line2.set_label(label2)

            # 실제 콜백(시그널) 연결 - 드래그 시 라벨 업데이트
            line1.sigPositionChanged.connect(lambda: self.update_line_label(line1, "Start", label_ypos))
            line2.sigPositionChanged.connect(lambda: self.update_line_label(line2, "End", label_ypos))

            # 초기 한 번 업데이트
            self.update_line_label(line1, "Start", label_ypos)
            self.update_line_label(line2, "End", label_ypos)

            # 플롯에 추가
            self.plot_widget.addItem(line1)
            self.plot_widget.addItem(line2)
            self.plot_widget.addItem(label1)
            self.plot_widget.addItem(label2)

            self.steady_lines = [line1, line2]

        elif self.selection_mode == "adjust":
            pos1, pos2 = len(indices)//4, len(indices)*3//4

            line1 = DraggableLine(pos=pos1, bounds=(0, len(indices)-1))
            line2 = DraggableLine(pos=pos2, bounds=(0, len(indices)-1))

            line1.setPen(pg.mkPen(color='r', width=1.5))
            line2.setPen(pg.mkPen(color='r', width=1.5))


            label1 = pg.TextItem(color=(0, 0, 0))
            label2 = pg.TextItem(color=(0, 0, 0))

            label1.setPos(pos1, label_ypos)
            label2.setPos(pos2, label_ypos)

            line1.set_label(label1)
            line2.set_label(label2)

            # adjust 모드에서는 "Adjust Start", "Adjust End" 등으로 표시 가능
            line1.sigPositionChanged.connect(lambda: self.update_line_label(line1, "Adjust Start", label_ypos))
            line2.sigPositionChanged.connect(lambda: self.update_line_label(line2, "Adjust End", label_ypos))
            
            self.update_line_label(line1, "Adjust Start", label_ypos)
            self.update_line_label(line2, "Adjust End", label_ypos)

            self.plot_widget.addItem(line1)
            self.plot_widget.addItem(line2)
            self.plot_widget.addItem(label1)
            self.plot_widget.addItem(label2)

            self.adjust_lines = [line1, line2]

        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)


    def get_steady_range(self):
        """Return (start, end) index of the steady range."""
        if len(self.steady_lines) == 2:
            start = int(self.steady_lines[0].value())
            end = int(self.steady_lines[1].value())
            return (min(start, end), max(start, end))
        return None
    
    def get_adjust_range(self):
        """Return (start, end) index of the adjust range."""
        if len(self.adjust_lines) == 2:
            start = int(self.adjust_lines[0].value())
            end = int(self.adjust_lines[1].value())
            return (min(start, end), max(start, end))
        return None
    
    def apply_temp_correction(self, steady_range, adjust_range):
        """
        steady_range의 데이터로 선형 피팅을 수행하고,
        adjust_range 내에서 변화량이 0인 구간의 온도를 보정합니다.
        
        Parameters:
            steady_range: (start, end) - 정상 구간의 인덱스 범위
            adjust_range: (start, end) - 보정할 구간의 인덱스 범위
        
        Returns:
            numpy.ndarray: 보정된 온도 데이터, 실패 시 None
        """
        if not steady_range or not adjust_range or self.temp_data is None:
            return None
            
        try:
            indices = np.arange(len(self.temp_data))
            
            # 정상 구간에서 선형 피팅 수행
            steady_mask = (indices >= steady_range[0]) & (indices <= steady_range[1])
            steady_indices = indices[steady_mask]
            steady_temps = self.temp_data[steady_mask]
            
            if len(steady_indices) < 2:
                return None
                
            # 선형 피팅 계수 계산
            a, b = np.polyfit(steady_indices, steady_temps, 1)
            
            # 보정할 구간 선택
            adjust_mask = (indices >= adjust_range[0]) & (indices <= adjust_range[1])
            adjust_indices = np.where(adjust_mask)[0]
            
            # 변화량이 0인 지점 찾기
            temps_in_range = self.temp_data[adjust_mask]
            diffs = np.diff(temps_in_range)
            zero_diff_indices = np.where(diffs == 0)[0]
            zero_diff_global = adjust_indices[zero_diff_indices + 1]
            
            # 피팅된 온도 계산
            fitted_temps = a * indices + b
            
            # 새로운 온도 데이터 생성
            new_temps = self.temp_data.copy()
            
            # 변화량이 0인 지점의 온도를 피팅된 값으로 교체
            for idx in zero_diff_global:
                new_temps[idx] = fitted_temps[idx]
                
            return new_temps
            
        except Exception as e:
            print(f"Error in temperature correction: {str(e)}")
            return None
        

    def update_line_label(self, line, prefix, fixed_y):
        """
        수직선 라벨을 '(인덱스, 온도°C)' 형태로 업데이트하는 메서드.
        prefix: "Start", "End", "Adjust Start" 등 라벨 맨 위 줄 텍스트
        fixed_y: 라벨을 놓을 y좌표(고정)
        """
        if self.temp_data is None:
            return

        # 1) 현재 라인 x좌표(=인덱스) 가져오기
        x_pos = line.value()

        # 2) 배열 인덱스는 정수로 처리 (범위 밖이면 클램프)
        idx = int(round(x_pos))
        if idx < 0:
            idx = 0
        elif idx >= len(self.temp_data):
            idx = len(self.temp_data) - 1

        # 3) 해당 인덱스의 온도값 찾기
        temp_val = self.temp_data[idx]

        # 4) 라벨 텍스트 갱신
        #    prefix 줄바꿈 후 (인덱스, 온도)
        line.label.setText(f"{prefix}\n({idx}, {temp_val:.2f}°C)")

        # 5) 라벨 위치는 x좌표 = line.value(), y좌표 = fixed_y (고정)
        line.label.setPos(x_pos, fixed_y)

def plot_contour_with_peaks_gui(contour_data, tracked_peaks, graph_option=None):
    """
    컨투어 플롯과 피크 위치를 QtGraphics View에 맞게 canvas로 반환하는 함수.
    피크는 시간 순서대로 선으로 연결되어 표시됨.
    """
    
    default_graph_option = {
        "figure_size": PLOT_OPTIONS["graph_option"]["figure_size"],
        "figure_dpi": 150,
        "contour_levels": PLOT_OPTIONS["graph_option"]["contour_levels"],
        "contour_cmap": PLOT_OPTIONS["graph_option"]["contour_cmap"],
        "contour_lower_percentile": PLOT_OPTIONS["graph_option"]["contour_lower_percentile"],
        "contour_upper_percentile": PLOT_OPTIONS["graph_option"]["contour_upper_percentile"],
        "global_ylim": PLOT_OPTIONS["graph_option"]["global_ylim"],
        "contour_xlim": PLOT_OPTIONS["graph_option"]["contour_xlim"]
    }
    if graph_option is None:
        graph_option = {}
    final_opt = {**default_graph_option, **graph_option}

    # Prepare contour data
    times, _ = contour_data["Time-temp"]
    q_all = np.concatenate([entry["q"] for entry in contour_data["Data"]])
    intensity_all = np.concatenate([entry["Intensity"] for entry in contour_data["Data"]])
    time_all = np.concatenate([[entry["Time"]] * len(entry["q"]) for entry in contour_data["Data"]])
    
    q_common = np.linspace(np.min(q_all), np.max(q_all), len(np.unique(q_all)))
    time_common = np.unique(time_all)
    grid_q, grid_time = np.meshgrid(q_common, time_common)
    grid_intensity = griddata((q_all, time_all), intensity_all, (grid_q, grid_time), method='nearest')
    
    if grid_intensity is None or np.isnan(grid_intensity).all():
        print("Warning: All interpolated intensity values are NaN. Check input data.")
        return None

    lower_bound = np.nanpercentile(grid_intensity, final_opt["contour_lower_percentile"])
    upper_bound = np.nanpercentile(grid_intensity, final_opt["contour_upper_percentile"])

    # Create figure and plot contour
    fig, ax = plt.subplots(figsize=final_opt["figure_size"], dpi=final_opt["figure_dpi"])
    cp = ax.contourf(grid_q, grid_time, grid_intensity,
                     levels=final_opt["contour_levels"],
                     cmap=final_opt["contour_cmap"],
                     vmin=lower_bound,
                     vmax=upper_bound)
    
    # Set limits if provided
    if final_opt["contour_xlim"] is not None:
        ax.set_xlim(final_opt["contour_xlim"])
    if final_opt["global_ylim"] is not None:
        ax.set_ylim(final_opt["global_ylim"])
    
    # Set labels and title
    ax.set_title("Contour Plot with Peak Positions", fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Plot peaks as a line
    peak_data = []
    for entry in tracked_peaks["Data"]:
        peak_q = entry.get("peak_q")
        time_val = entry.get("Time")
        if peak_q is not None and not np.isnan(peak_q):
            peak_data.append((peak_q, time_val))
    
    if peak_data:
        # Sort by time to ensure proper line order
        peak_data.sort(key=lambda x: x[1])
        peak_q_vals, peak_times = zip(*peak_data)
        
        # Plot line connecting peaks
        ax.plot(peak_q_vals, peak_times, 'b-', linewidth=1.5, label='Peak Trajectory', zorder=10)
    
    ax.legend()
    
    # Convert to canvas for Qt
    canvas = FigureCanvas(fig)
    canvas.draw()
    return canvas

class QRangeCorrectionHelper:
    """Q 범위 선택을 위한 헬퍼 클래스"""
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.q_data = None
        self.intensity_data = None
        self.selection_lines = []
        
    def set_data(self, q_data, intensity_data):
        """데이터 설정"""
        # 입력 데이터를 NumPy 배열로 명시적 변환
        self.q_data = np.array(q_data)
        self.intensity_data = np.array(intensity_data)
        self.plot_data()
        
    def plot_data(self):
        """데이터 플롯"""
        if self.q_data is None or self.intensity_data is None:
            return
        
        self.plot_widget.clear()
        self.plot_widget.plot(
            self.q_data, 
            self.intensity_data,
            pen=pg.mkPen(color='k', width=1),
            symbol='o',
            symbolSize=5,
            symbolPen=pg.mkPen('k'),
            symbolBrush=pg.mkBrush('w')
        )
        
    def add_selection_lines(self):
        """선택 라인 추가"""
        if self.q_data is None:
            return
        
        self.plot_widget.clear()
        self.plot_data()
        
        # y좌표를 어느 정도 위로 설정 (라벨을 보기 좋게)
        y_offset = (np.max(self.intensity_data) - np.min(self.intensity_data)) * 0.1
        label_ypos = np.max(self.intensity_data) + y_offset
        
        # 두 개의 선택 라인 생성
        pos1, pos2 = self.q_data[len(self.q_data)//4], self.q_data[len(self.q_data)*3//4]
        
        line1 = DraggableLine(pos=pos1, bounds=(min(self.q_data), max(self.q_data)))
        line2 = DraggableLine(pos=pos2, bounds=(min(self.q_data), max(self.q_data)))
        
        line1.setPen(pg.mkPen(color='b', width=1.5))
        line2.setPen(pg.mkPen(color='b', width=1.5))
        
        # 라벨 생성
        label1 = pg.TextItem(color=(0, 0, 0))
        label2 = pg.TextItem(color=(0, 0, 0))
        
        # 라벨 위치 설정
        label1.setPos(pos1, label_ypos)
        label2.setPos(pos2, label_ypos)
        
        # 라인에 라벨 연결
        line1.set_label(label1)
        line2.set_label(label2)
        
        # 라인 이동 시 라벨 업데이트 시그널 연결
        line1.sigPositionChanged.connect(lambda: self.update_line_label(line1, "Start", label_ypos))
        line2.sigPositionChanged.connect(lambda: self.update_line_label(line2, "End", label_ypos))
        
        # 초기 라벨 업데이트
        self.update_line_label(line1, "Start", label_ypos)
        self.update_line_label(line2, "End", label_ypos)
        
        # 플롯에 추가
        self.plot_widget.addItem(line1)
        self.plot_widget.addItem(line2)
        self.plot_widget.addItem(label1)
        self.plot_widget.addItem(label2)
        
        self.selection_lines = [line1, line2]
        
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
    
    def update_line_label(self, line, prefix, fixed_y):
        """라인 라벨 업데이트"""
        if self.q_data is None:
            return
        
        # 현재 라인 x좌표(=q값) 가져오기
        x_pos = line.value()
        
        # 가장 가까운 인덱스 찾기
        idx = np.abs(self.q_data - x_pos).argmin()
        
        # 해당 인덱스의 q값과 intensity 찾기
        q_val = self.q_data[idx]
        intensity_val = self.intensity_data[idx]
        
        # 라벨 텍스트 갱신
        line.label.setText(f"{prefix}\n(q = {q_val:.4f}, I = {intensity_val:.4f})")
        
        # 라벨 위치는 x좌표 = line.value(), y좌표 = fixed_y (고정)
        line.label.setPos(x_pos, fixed_y)
    
    def get_q_range(self):
        """선택된 q 범위 반환"""
        if len(self.selection_lines) == 2:
            q1 = self.selection_lines[0].value()
            q2 = self.selection_lines[1].value()
            return (min(q1, q2), max(q1, q2))
        return None

# In contour_util_gui.py, add:

class IndexRangeSelectionHelper:
    """Helper class for index range selection"""
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.temp_data = None
        self.index_lines = []
        
    def set_data(self, temp_data):
        """Set temperature data"""
        self.temp_data = temp_data
        self.plot_data()
        
    def plot_data(self):
        """Plot temperature vs index data"""
        if self.temp_data is None:
            return
            
        self.plot_widget.clear()
        indices = np.arange(len(self.temp_data))
        self.plot_widget.plot(
            indices, 
            self.temp_data,
            pen=pg.mkPen(color='k', width=1),
            symbol='o',
            symbolSize=5,
            symbolPen=pg.mkPen('k'),
            symbolBrush=pg.mkBrush('w')
        )
        
    def add_selection_lines(self):
        """Add vertical lines for selection"""
        if self.temp_data is None:
            return

        # Plot data
        self.plot_widget.clear()
        indices = np.arange(len(self.temp_data))
        self.plot_widget.plot(
            indices, 
            self.temp_data,
            pen=pg.mkPen(color='k', width=1),
            symbol='o',
            symbolSize=5,
            symbolPen=pg.mkPen('k'),
            symbolBrush=pg.mkBrush('w')
        )

        # y offset for labels
        y_offset = (np.max(self.temp_data) - np.min(self.temp_data)) * 0.1
        label_ypos = np.max(self.temp_data) + y_offset

        # Add selection lines
        pos1, pos2 = len(indices)//4, len(indices)*3//4
        
        line1 = DraggableLine(pos=pos1, bounds=(0, len(indices)-1))
        line2 = DraggableLine(pos=pos2, bounds=(0, len(indices)-1))
        
        line1.setPen(pg.mkPen(color='b', width=1.5))
        line2.setPen(pg.mkPen(color='b', width=1.5))

        # Create labels
        label1 = pg.TextItem(color=(0, 0, 0))
        label2 = pg.TextItem(color=(0, 0, 0))
        
        label1.setPos(pos1, label_ypos)
        label2.setPos(pos2, label_ypos)
        
        line1.set_label(label1)
        line2.set_label(label2)
        
        line1.sigPositionChanged.connect(lambda: self.update_line_label(line1, "Start", label_ypos))
        line2.sigPositionChanged.connect(lambda: self.update_line_label(line2, "End", label_ypos))
        
        self.update_line_label(line1, "Start", label_ypos)
        self.update_line_label(line2, "End", label_ypos)
        
        self.plot_widget.addItem(line1)
        self.plot_widget.addItem(line2)
        self.plot_widget.addItem(label1)
        self.plot_widget.addItem(label2)
        
        self.index_lines = [line1, line2]
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
    def update_line_label(self, line, prefix, fixed_y):
        """Update label text and position"""
        if self.temp_data is None:
            return
            
        x_pos = line.value()
        idx = int(round(x_pos))
        if idx < 0:
            idx = 0
        elif idx >= len(self.temp_data):
            idx = len(self.temp_data) - 1
            
        temp_val = self.temp_data[idx]
        line.label.setText(f"{prefix}\n(Index: {idx}, {temp_val:.2f}°C)")
        line.label.setPos(x_pos, fixed_y)
        
    def get_index_range(self):
        """Return selected index range"""
        if len(self.index_lines) == 2:
            start = int(self.index_lines[0].value())
            end = int(self.index_lines[1].value())
            return (min(start, end), max(start, end))
        return None

class PeakTempRangeHelper:
    """Helper class for temperature range selection in peak fitting"""
    def __init__(self, plot_widget):
        self.plot_widget = plot_widget
        self.temp_data = None
        self.peak_q_data = None
        self.temp_lines = []
        
    def set_data(self, temp_data, peak_q_data):
        """Set temperature and peak q data"""
        self.temp_data = temp_data
        self.peak_q_data = peak_q_data
        self.plot_data()
        
    def plot_data(self):
        """Plot peak q vs temperature data"""
        if self.temp_data is None or self.peak_q_data is None:
            return
            
        self.plot_widget.clear()
        self.plot_widget.plot(
            self.temp_data, 
            self.peak_q_data,
            pen=pg.mkPen(color='k', width=1),
            symbol='o',
            symbolSize=5,
            symbolPen=pg.mkPen('k'),
            symbolBrush=pg.mkBrush('w')
        )
        
    def add_selection_lines(self):
        """Add vertical lines for temperature range selection"""
        if self.temp_data is None or self.peak_q_data is None:
            return

        self.plot_widget.clear()
        self.plot_widget.plot(
            self.temp_data, 
            self.peak_q_data,
            pen=pg.mkPen(color='k', width=1),
            symbol='o',
            symbolSize=5,
            symbolPen=pg.mkPen('k'),
            symbolBrush=pg.mkBrush('w')
        )

        y_offset = (np.max(self.peak_q_data) - np.min(self.peak_q_data)) * 0.1
        label_ypos = np.max(self.peak_q_data) + y_offset

        min_temp, max_temp = np.min(self.temp_data), np.max(self.temp_data)
        pos1 = min_temp + (max_temp - min_temp) / 4
        pos2 = min_temp + 3 * (max_temp - min_temp) / 4
        
        line1 = DraggableLine(pos=pos1, bounds=(min_temp, max_temp))
        line2 = DraggableLine(pos=pos2, bounds=(min_temp, max_temp))
        
        line1.setPen(pg.mkPen(color='r', width=1.5))
        line2.setPen(pg.mkPen(color='r', width=1.5))

        label1 = pg.TextItem(color=(0, 0, 0))
        label2 = pg.TextItem(color=(0, 0, 0))
        
        label1.setPos(pos1, label_ypos)
        label2.setPos(pos2, label_ypos)
        
        line1.set_label(label1)
        line2.set_label(label2)
        
        line1.sigPositionChanged.connect(lambda: self.update_line_label(line1, "Start", label_ypos))
        line2.sigPositionChanged.connect(lambda: self.update_line_label(line2, "End", label_ypos))
        
        self.update_line_label(line1, "Start", label_ypos)
        self.update_line_label(line2, "End", label_ypos)
        
        self.plot_widget.addItem(line1)
        self.plot_widget.addItem(line2)
        self.plot_widget.addItem(label1)
        self.plot_widget.addItem(label2)
        
        self.temp_lines = [line1, line2]
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
    def update_line_label(self, line, prefix, fixed_y):
        """Update label text and position"""
        if self.temp_data is None:
            return
            
        x_pos = line.value()
        line.label.setText(f"{prefix}\nTemp: {x_pos:.2f}°C")
        line.label.setPos(x_pos, fixed_y)
        
    def get_temp_range(self):
        """Return selected temperature range"""
        if len(self.temp_lines) == 2:
            temp1 = self.temp_lines[0].value()
            temp2 = self.temp_lines[1].value()
            return (min(temp1, temp2), max(temp1, temp2))
        return None

def create_plot_widget():
    """Create a preconfigured plot widget"""
    plot = pg.PlotWidget()
    plot.setBackground('w')
    plot.setLabel('left', 'Temperature')
    plot.setLabel('bottom', 'Index')
    plot.showGrid(x=True, y=True)
    plot.getAxis('bottom').setPen('k')
    plot.getAxis('left').setPen('k')
    return plot