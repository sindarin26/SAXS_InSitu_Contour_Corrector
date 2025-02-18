#asset.contour_util_gui.py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

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
        self.plot_widget.plot(indices, self.temp_data, pen='b')
        
    def add_selection_lines(self):
        if self.temp_data is None:
            return

        # 1) 플롯 초기화 및 전체 데이터 플롯
        self.plot_widget.clear()
        indices = np.arange(len(self.temp_data))
        self.plot_widget.plot(indices, self.temp_data, pen='b')

        # y좌표를 어느 정도 위로 설정 (라벨을 보기 좋게)
        y_offset = (np.max(self.temp_data) - np.min(self.temp_data)) * 0.1
        label_ypos = np.max(self.temp_data) + y_offset

        if self.selection_mode == "steady":
            pos1, pos2 = len(indices)//4, len(indices)*3//4

            line1 = DraggableLine(pos=pos1, bounds=(0, len(indices)-1))
            line2 = DraggableLine(pos=pos2, bounds=(0, len(indices)-1))

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