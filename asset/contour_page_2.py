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
            # Clear any existing contour data to ensure fresh plotting
            DATA['contour_data'] = None
            
            if not DATA.get('contour_data'):
                if not PROCESS_STATUS['selected_series']:
                    QtWidgets.QMessageBox.warning(
                        self.main,
                        "Warning",
                        "No series selected. Please go back and select a series."
                    )
                    return
                    
                # Extract contour data using the selected series
                contour_data = extract_contour_data(PROCESS_STATUS['selected_series'], DATA['extracted_data'])
                
                # Store the contour data
                DATA['contour_data'] = contour_data
                
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
            
        # Clear existing layout
        if self.GV_contour.layout():
            QtWidgets.QWidget().setLayout(self.GV_contour.layout())
            
        # Create layout
        layout = QtWidgets.QVBoxLayout()
        
        # Use plot options directly from PLOT_OPTIONS
        canvas = plot_contour(
            DATA['contour_data'],
            temp=PLOT_OPTIONS['temp'],
            legend=PLOT_OPTIONS['legend'],
            graph_option=PLOT_OPTIONS['graph_option'],
            GUI=True
        )
        
        if canvas:
            # Set size policy
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding
            )
            
            # Configure the layout
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(canvas)
            
            # Configure the GraphicsView
            self.GV_contour.setViewportUpdateMode(
                QtWidgets.QGraphicsView.FullViewportUpdate
            )
            self.GV_contour.setRenderHints(
                QtGui.QPainter.Antialiasing |
                QtGui.QPainter.SmoothPixmapTransform
            )
            
            self.GV_contour.setLayout(layout)
            self.canvas = canvas
            
            # Make sure the canvas updates properly
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

        # 프로그레스 다이얼로그 생성 (먼저 띄운 후 UI 업데이트)
        progress = LoadingDialog(parent=self.main, message="Exporting files...")
        progress.progress.setMaximum(total_files)
        progress.progress.setValue(0)
        progress.show()

        # UI 업데이트 강제 실행 (다이얼로그가 즉시 표시되도록)
        QtWidgets.QApplication.processEvents()

        try:
            # 파일명 기본 정보 설정
            series = PROCESS_STATUS.get('selected_series') or "Series"
            xlim = PLOT_OPTIONS['graph_option'].get("contour_xlim")
            ylim = PLOT_OPTIONS['graph_option'].get("global_ylim")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. 컨투어 플롯 이미지 저장
            progress.label.setText("Saving contour plot...")
            QtWidgets.QApplication.processEvents()  # UI 업데이트
            name_parts = [series]
            if xlim is not None:
                name_parts.append(f"xlim{str(xlim)}")
            if ylim is not None:
                name_parts.append(f"ylim{str(ylim)}")
            name_parts.append(timestamp)
            filename = "_".join(name_parts) + ".png"
            output_path = os.path.join(output_dir, filename)
            self.canvas.figure.savefig(output_path, format='png')
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
        """컨투어 설정 다이얼로그 열기: 값이 바뀌면 즉시 plot_contour 재생성 호출"""
        self.contour_settings_dialog = ContourSettingsDialog(
            callback=self.create_contour_plot,
            parent=self.main
        )
        self.contour_settings_dialog.show()

    def start_sdd_correction(self):
        """Start SDD correction process"""
        # Implement SDD correction
        pass