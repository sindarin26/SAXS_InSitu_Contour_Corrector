#asset.contour_page_2.py
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from asset.contour_storage import DATA, PATH_INFO, PARAMS, PLOT_OPTIONS, PROCESS_STATUS
from asset.page_asset import LoadingDialog, DragDropLineEdit, normalize_path
from asset.contour_util import extract_contour_data, plot_contour
from asset.contour_settings_dialog import ContourSettingsDialog
import os
import datetime
import pandas as pd


class ContourPlotPage(QtCore.QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        
        # Store UI elements
        self.GV_contour = self.main.ui.GV_contour
        self.L_current_series = self.main.ui.L_current_series
        self.L_current_output_dir = self.main.ui.L_current_output_dir
        self.PB_back_1 = self.main.ui.PB_back_1
        self.PB_export = self.main.ui.PB_export
        self.PB_contour_setting = self.main.ui.PB_contour_setting
        self.PB_SDD_correction = self.main.ui.PB_SDD_correction
        
        # 추가: 온도, 데이터 내보내기 토글 버튼
        self.PB_export_temp_on = self.main.ui.PB_export_temp_on
        self.PB_export_data_on = self.main.ui.PB_export_data_on
        
        # Setup output directory widgets
        self.setup_output_widgets()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize variables
        self.current_series = None
        self.canvas = None
        
        # Add resize event filter to GV_contour
        self.GV_contour.installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.GV_contour and event.type() == QtCore.QEvent.Resize:
            self.resize_figure_to_view()
        return super().eventFilter(obj, event)

    def setup_output_widgets(self):
        """Initialize output directory widgets with drag and drop support"""
        # Create new DragDropLineEdit
        self.LE_output_dir_2 = DragDropLineEdit(self.main)
        
        # Replace existing line edit
        old_widget = self.main.ui.LE_output_dir_2
        layout = old_widget.parent().layout()
        layout.replaceWidget(old_widget, self.LE_output_dir_2)
        old_widget.deleteLater()
        
        # Store button references
        self.PB_output_browse_2 = self.main.ui.PB_output_browse_2
        self.PB_output_apply_2 = self.main.ui.PB_output_apply_2
        
        # If there's an existing output directory, display it
        if PATH_INFO.get('output_dir'):
            self.LE_output_dir_2.setText(PATH_INFO['output_dir'])

    def connect_signals(self):
        """Connect all UI signals"""
        self.PB_back_1.clicked.connect(lambda: self.main.SW_Main_page.setCurrentIndex(1))
        self.PB_export.clicked.connect(self.export_data)
        self.PB_contour_setting.clicked.connect(self.open_contour_settings)
        self.PB_SDD_correction.clicked.connect(self.start_sdd_correction)
        
        # Connect output directory signals
        self.PB_output_browse_2.clicked.connect(self.browse_output_dir)
        self.PB_output_apply_2.clicked.connect(self.apply_output_dir)
        self.LE_output_dir_2.returnPressed.connect(self.apply_output_dir)

    def on_page_entered(self):
        """Handle initial setup when page is displayed"""
        self.update_output_dir_display()
        self.prepare_and_create_contour()

    def update_output_dir_display(self):
        """Update the display of current output directory"""
        if PATH_INFO.get('output_dir'):
            self.L_current_output_dir.setText(PATH_INFO['output_dir'])
        else:
            self.L_current_output_dir.setText("No output directory selected")

    def prepare_and_create_contour(self):
        """Prepare contour data and create plot"""
        try:
            if not DATA.get('contour_data'):
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    "No contour data available. Please go back and process the data first."
                )
                return
                
            # Update series label
            self.L_current_series.setText(f"Current Series: {PROCESS_STATUS['selected_series']}")
            
            # Create and display the contour plot
            self.create_contour_plot()
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Failed to prepare contour data: {str(e)}"
            )

    def create_contour_plot(self):
        """Create and display the contour plot"""
        if not DATA.get('contour_data'):
            print("No contour data available")
            return
            
        # Calculate minimum size needed for the figure
        fig_size = PLOT_OPTIONS['graph_option']['figure_size']
        dpi = PLOT_OPTIONS['graph_option']['figure_dpi']
        min_width = int(fig_size[0] * dpi / 4)  # 최소 크기는 원본의 1/4
        min_height = int(fig_size[1] * dpi / 4)
        
        # Set minimum size for GraphicsView
        self.GV_contour.setMinimumSize(min_width, min_height)
            
        # Clear existing layout
        if self.GV_contour.layout():
            QtWidgets.QWidget().setLayout(self.GV_contour.layout())
            
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Create figure with original size from PLOT_OPTIONS
        canvas = plot_contour(
            DATA['contour_data'],
            temp=PLOT_OPTIONS['temp'],
            legend=PLOT_OPTIONS['legend'],
            graph_option=PLOT_OPTIONS['graph_option'],
            GUI=True
        )
        
        if canvas:
            # Create a container widget
            container = QtWidgets.QWidget()
            container_layout = QtWidgets.QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(canvas)
            
            # Set policies
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Ignored
            )
            container.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            
            # Add to main layout
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(container)
            
            # Set up GraphicsView
            self.GV_contour.setLayout(layout)
            self.canvas = canvas
            
            # Initial resize
            self.resize_figure_to_view()

    def resize_figure_to_view(self):
        """Resize the figure to fit the view while maintaining aspect ratio"""
        if not hasattr(self, 'canvas') or not self.canvas:
            return
            
        if not hasattr(self, 'canvas_original_size'):
            self.canvas_original_size = self.canvas.size()
            
        # Get the current size of the GraphicsView
        view_size = self.GV_contour.size()
        
        # Get the figure size from PLOT_OPTIONS
        fig_size = PLOT_OPTIONS['graph_option']['figure_size']
        fig_aspect = fig_size[0] / fig_size[1]
        
        # Calculate new size maintaining aspect ratio
        view_width = view_size.width() - 10  # 여백 고려
        view_height = view_size.height() - 10  # 여백 고려
        view_aspect = view_width / view_height
        
        if view_aspect > fig_aspect:
            # View is wider than figure
            new_height = view_height
            new_width = int(new_height * fig_aspect)
        else:
            # View is taller than figure
            new_width = view_width
            new_height = int(new_width / fig_aspect)
        
        # Resize the canvas
        self.canvas.setFixedSize(int(new_width), int(new_height))
        self.canvas.draw()

    def browse_output_dir(self):
        """Open file dialog to browse for output directory"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            "Select Output Directory"
        )
        if dir_path:
            self.LE_output_dir_2.setText(normalize_path(dir_path))
            self.apply_output_dir()

    def apply_output_dir(self):
        """Set and create output directory"""
        path = normalize_path(self.LE_output_dir_2.text().strip())
        if not path:
            return

        try:
            os.makedirs(path, exist_ok=True)
            PATH_INFO['output_dir'] = path
            self.update_output_dir_display()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to create output directory:\n{str(e)}"
            )

    def export_data(self):
        """Export contour image and optionally temperature and extracted data to files"""
        if not hasattr(self, 'canvas_original_size'):
            self.canvas_original_size = self.canvas.size()
        output_dir = PATH_INFO.get('output_dir') or self.LE_output_dir_2.text().strip()
        if not output_dir:
            QtWidgets.QMessageBox.warning(self.main, "Export", "No output directory selected.")
            return

        if not self.canvas:
            QtWidgets.QMessageBox.warning(self.main, "Export", "No contour plot available to export.")
            return

        # 저장할 파일 수 계산
        total_files = 1  # 기본 컨투어 플롯
        if self.PB_export_temp_on.isChecked():
            total_files += 1
        if self.PB_export_data_on.isChecked():
            total_files += 1

        # 프로그레스 다이얼로그 생성
        progress = LoadingDialog(parent=self.main, message="Exporting files...")
        progress.progress.setMaximum(total_files)
        progress.progress.setValue(0)
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            # 파일명 기본 정보 설정
            series = PROCESS_STATUS.get('selected_series') or "Series"
            xlim = PLOT_OPTIONS['graph_option'].get("contour_xlim")
            ylim = PLOT_OPTIONS['graph_option'].get("global_ylim")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 컨투어 플롯 이미지 저장
            progress.label.setText("Saving contour plot...")
            QtWidgets.QApplication.processEvents()
            
            # 현재 크기 저장
            current_size = self.canvas.size()
            
            # 원본 figure size로 복원
            fig = self.canvas.figure
            fig_size = PLOT_OPTIONS['graph_option']['figure_size']
            fig.set_size_inches(fig_size[0], fig_size[1])
            
            name_parts = [series]
            if xlim is not None:
                name_parts.append(f"xlim{str(xlim)}")
            if ylim is not None:
                name_parts.append(f"ylim{str(ylim)}")
            name_parts.append(timestamp)
            filename = "_".join(name_parts) + ".png"
            output_path = os.path.join(output_dir, filename)
            
            # 원본 크기와 DPI로 저장
            fig.savefig(output_path, format='png', dpi=PLOT_OPTIONS['graph_option']['figure_dpi'])
            
            # 크기 복원 및 다시 그리기
            self.canvas.setFixedSize(self.canvas_original_size)
            self.resize_figure_to_view()
            progress.progress.setValue(1)

            # 2. 온도 데이터 저장
            if self.PB_export_temp_on.isChecked():
                progress.label.setText("Saving temperature data...")
                QtWidgets.QApplication.processEvents()
                contour_data = DATA.get('contour_data')
                if contour_data:
                    times, temperatures = contour_data.get("Time-temp", ([], []))
                    if times and temperatures:
                        temp_data = {
                            'Time': times,
                            'Temperature': temperatures
                        }
                        df_temp = pd.DataFrame(temp_data)
                        temp_filename = "_".join([series, "temperature", timestamp]) + ".xlsx"
                        temp_output_path = os.path.join(output_dir, temp_filename)
                        df_temp.to_excel(temp_output_path, index=False)
                progress.progress.setValue(2)

            # 3. q/Intensity 데이터 저장
            if self.PB_export_data_on.isChecked():
                progress.label.setText("Saving intensity data...")
                QtWidgets.QApplication.processEvents()
                contour_data = DATA.get('contour_data')
                if contour_data:
                    data_entries = contour_data.get("Data", [])
                    if data_entries:
                        data_dict = {}
                        for idx, entry in enumerate(data_entries, 1):
                            q_key = f"q_{idx}"
                            intensity_key = f"intensity_{idx}"
                            data_dict[q_key] = pd.Series(entry.get("q", []))
                            data_dict[intensity_key] = pd.Series(entry.get("Intensity", []))
                        
                        df_all = pd.DataFrame(data_dict)
                        data_filename = "_".join([series, "raw_data", timestamp]) + ".xlsx"
                        data_output_path = os.path.join(output_dir, data_filename)
                        df_all.to_excel(data_output_path, index=False, sheet_name='Raw Data')
                progress.progress.setValue(3)

            # 모든 작업이 완료되면 성공 메시지 표시 후 다이얼로그 닫기
            progress.label.setText("Export completed successfully!")
            QtCore.QTimer.singleShot(1000, progress.close)  # 1초 후 자동으로 닫힘

        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self.main, "Export Error", str(e))

    def open_contour_settings(self):
        """컨투어 설정 다이얼로그 열기"""
        self.contour_settings_dialog = ContourSettingsDialog(
            callback=self.create_contour_plot,
            parent=self.main
        )
        self.contour_settings_dialog.show()

    def start_sdd_correction(self):
        """Start SDD correction process"""
        # Implement SDD correction
        pass