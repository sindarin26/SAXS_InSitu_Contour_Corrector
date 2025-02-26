#asset/contour_page_peak_3.py
from PyQt5 import QtWidgets, QtCore, QtGui
import pandas as pd
import numpy as np
import os
import datetime
from asset.contour_storage import PROCESS_STATUS
from asset.page_asset import LoadingDialog

class DataExportPage(QtCore.QObject):
    """
    Page for displaying and exporting peak data in table format.
    This is the third page (index 3) in the peak export dialog.
    """
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui
        
        # UI references
        self.TW_table = self.ui.TW_table
        self.L_current_status_3 = self.ui.L_current_status_3
        self.PB_export = self.ui.PB_export
        self.PB_open_output_folder = self.ui.PB_open_output_folder
        self.PB_back_to_contourpage_1 = self.ui.PB_back_to_contourpage_1
        
        # Data storage - will be populated when entering the page
        self.table_data = {}  # Dictionary to store formatted data for each sheet
        self.table_views = {}  # Dictionary to store table views for each sheet
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to handlers"""
        self.PB_export.clicked.connect(self.export_data)
        self.PB_open_output_folder.clicked.connect(self.open_output_folder)
        self.PB_back_to_contourpage_1.clicked.connect(self.on_back_to_contour)
    
    def initialize_page(self):
        """Initialize the page when it is entered"""
        # Show loading dialog
        loading = LoadingDialog(self.main, "Converting to table...")
        loading.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Clear existing tabs
            self.clear_tables()
            
            # Format data for tables
            self.format_data_for_tables()
            
            # Create table views
            self.create_table_views()
            
            # Update status
            self.L_current_status_3.setText(f"Peak Extraction Result: {len(self.table_data)} peaks found")
        finally:
            loading.close()
    
    def clear_tables(self):
        """Clear all tabs and table views"""
        self.table_data = {}
        self.table_views = {}
        
        # Remove all tabs
        while self.TW_table.count() > 0:
            self.TW_table.removeTab(0)
    
    def format_data_for_tables(self):
        """
        Format peak data from PEAK_EXTRACT_DATA for display in tables.
        Creates DataFrames for each peak with all related data.
        """
        if not hasattr(self.main, 'PEAK_EXTRACT_DATA') or not self.main.PEAK_EXTRACT_DATA:
            return
        
        peak_data = self.main.PEAK_EXTRACT_DATA
        tracked_peaks = peak_data.get('tracked_peaks', {})
        found_peak_list = peak_data.get('found_peak_list', [])
        
        if not tracked_peaks or not found_peak_list:
            return
        
        # Get series name
        series_name = tracked_peaks.get('Series', "Unknown Series")
        
        # Get note
        note = peak_data.get('NOTE', "")
        
        # 피크별로 처리하기 때문에 각 피크 데이터마다 fitting_function을 확인해야 함
        # 일단 기본값을 설정해두고 각 피크별로 업데이트
        
        # Extract peak data by peak name
        peak_data_by_name = {}
        for peak_name in found_peak_list:
            peak_entries = []
            for entry in tracked_peaks.get('Data', []):
                if entry.get('peak_name') == peak_name:
                    peak_entries.append(entry)
            
            if peak_entries:
                # Sort by frame_index
                peak_entries.sort(key=lambda x: x.get('frame_index', 0))
                peak_data_by_name[peak_name] = peak_entries
        
        # Create a table for each peak
        for peak_name, entries in peak_data_by_name.items():
            # Create sheet name - just use the peak_name
            sheet_name = peak_name
            
            # 각 피크의 fitting_function 가져오기
            # 첫 번째 엔트리에서 fitting_function 정보를 찾음
            if not entries:
                continue
                
            # 모든 엔트리에서 fitting_function 찾기
            fitting_function = None
            for entry in entries:
                if entry.get('fitting_function'):
                    fitting_function = entry['fitting_function']
                    break
                    
            if not fitting_function:
                # fitting_function이 없으면 에러 메시지 표시
                QtWidgets.QMessageBox.warning(
                    self.main,
                    "Warning",
                    f"No fitting function found for peak {peak_name}. This data may be incomplete."
                )
                fitting_function = "unknown"  # 기본값 설정
            
            # fitting_function에 따라 모델 방정식과 파라미터 설정
            if fitting_function == "gaussian":
                model_equation = "f(x) = a * exp(-((x-mu)^2)/(2*sigma^2)) + offset"
                model_params = ["a", "mu", "sigma", "offset"]
            elif fitting_function == "lorentzian":
                model_equation = "f(x) = a * gamma^2 / ((x-x0)^2 + gamma^2) + offset"
                model_params = ["a", "x0", "gamma", "offset"]
            elif fitting_function == "voigt":
                model_equation = "f(x) = a * voigt_profile(x-mu, sigma, gamma) + offset"
                model_params = ["a", "mu", "sigma", "gamma", "offset"]
            else:
                model_equation = f"Unknown model: {fitting_function}"
                # 첫 번째 엔트리의 fitting_params에서 파라미터 이름 추출
                if entries[0].get('fitting_params'):
                    model_params = list(entries[0]['fitting_params'].keys())
                else:
                    model_params = ["unknown_params"]
            
            # Create DataFrame for header (rows 1-10)
            header_df = pd.DataFrame(index=range(1, 11), columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            header_df.iloc[0, 0] = series_name
            header_df.iloc[1, 0] = "NOTE"
            header_df.iloc[1, 1] = note
            header_df.iloc[2, 0] = "Fitting Model"
            header_df.iloc[2, 1] = fitting_function
            header_df.iloc[3, 0] = model_equation
            
            # 파라미터 이름 추가
            header_df.iloc[4, 0] = "Parameters"
            for i, param in enumerate(model_params):
                header_df.iloc[4, i+1] = param
            
            # 데이터 헤더와 데이터를 위한 빈 DataFrame 생성
            data_headers = ["Elapsed Time", "Temperature", "Index", "peak_q", "peak_Intensity", "FWHM"]
            data_headers.extend(model_params)
            
            # 데이터 추출
            times = [entry.get('Time', 0) for entry in entries]
            temps = [entry.get('Temperature', 0) for entry in entries]
            indices = [int(entry.get('frame_index', 0)) for entry in entries]
            peak_qs = [entry.get('peak_q', 0) for entry in entries]
            intensities = [entry.get('peak_Intensity', 0) for entry in entries]
            fwhms = [entry.get('fwhm', 0) for entry in entries]
            
            # 수정된 부분: 명시적으로 열 이름을 알파벳으로 지정
            col_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:len(data_headers)]
            
            # 데이터 행 생성 (헤더 행 먼저)
            data_rows = [data_headers]
            
            # 데이터 행 추가
            for i in range(len(entries)):
                row = [
                    times[i], 
                    temps[i],
                    indices[i],
                    peak_qs[i],
                    intensities[i],
                    fwhms[i]
                ]
                
                # 파라미터 값 추가
                for param in model_params:
                    if (entries[i].get('fitting_params') and
                        param in entries[i]['fitting_params']):
                        row.append(entries[i]['fitting_params'][param])
                    else:
                        row.append("")
                
                data_rows.append(row)
            
            # 수정된 부분: 열 이름을 명시적으로 지정하여 DataFrame 생성
            data_df = pd.DataFrame(data_rows)
            data_df.columns = col_names + list(range(len(col_names), data_df.shape[1]))
            
            # 빈 행 생성 (헤더와 데이터 구분용)
            empty_row = pd.DataFrame([[""] * data_df.shape[1]], columns=data_df.columns)
            
            # 헤더 DataFrame의 열 수 조정
            header_df = header_df.iloc[:, :data_df.shape[1]]
            
            # 결합
            combined_df = pd.concat([header_df, empty_row, data_df], ignore_index=True)
            self.table_data[sheet_name] = combined_df
    
    def create_table_views(self):
        """Create table views for each sheet and add them to tabs"""
        for sheet_name, df in self.table_data.items():
            # Create a new tab widget
            tab = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout(tab)
            
            # Create table view
            table_view = QtWidgets.QTableView()
            model = TableModel(df)
            table_view.setModel(model)
            
            # Style the table
            table_view.horizontalHeader().setVisible(False)
            table_view.verticalHeader().setVisible(False)
            table_view.setAlternatingRowColors(True)
            
            # Auto resize columns to content
            table_view.resizeColumnsToContents()
            
            # Store reference to table view
            self.table_views[sheet_name] = table_view
            
            # Add to layout
            layout.addWidget(table_view)
            
            # Add tab
            self.TW_table.addTab(tab, sheet_name)
    
    def export_data(self):
        """Export table data to Excel file"""
        if not self.table_data:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Warning",
                "No data available to export."
            )
            return
        
        # Get series name
        series_name = self.main.PEAK_EXTRACT_DATA.get('tracked_peaks', {}).get('Series', "Unknown")
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{series_name}_extracted_peaks_{timestamp}.xlsx"
        
        # Get output directory
        output_dir = self.get_output_directory()
        if not output_dir:
            return
        
        # Create full path
        file_path = os.path.join(output_dir, filename)
        
        # Show loading dialog
        loading = LoadingDialog(self.main, "Exporting data to Excel...")
        loading.show()
        QtWidgets.QApplication.processEvents()
        
        try:
            # Create Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write each sheet
                for sheet_name, df in self.table_data.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
            QtWidgets.QMessageBox.information(
                self.main,
                "Success",
                f"Data exported successfully to:\n{file_path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self.main,
                "Error",
                f"Failed to export data: {str(e)}"
            )
        finally:
            loading.close()
    
    def get_output_directory(self):
        """Get output directory from parent application"""
        from asset.contour_storage import PATH_INFO
        output_dir = PATH_INFO.get('output_dir')
        
        if not output_dir:
            # Ask user for directory
            output_dir = QtWidgets.QFileDialog.getExistingDirectory(
                self.main,
                "Select Output Directory"
            )
            
            if output_dir:
                PATH_INFO['output_dir'] = output_dir
        
        return output_dir
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        output_dir = self.get_output_directory()
        if not output_dir:
            return
        
        try:
            import os
            if os.name == 'nt':  # Windows
                os.startfile(output_dir)
            elif os.name == 'posix':  # macOS and Linux
                import subprocess
                if os.path.exists('/usr/bin/open'):  # macOS
                    subprocess.Popen(['open', output_dir])
                else:  # Linux
                    subprocess.Popen(['xdg-open', output_dir])
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to open folder: {str(e)}"
            )
    
    def on_back_to_contour(self):
        """Return to contour page (index 2) and reset to waiting state"""
        # Clear all data
        self.clear_tables()
        
        # Switch to contour page
        self.ui.stackedWidget.setCurrentIndex(2)
        
        # Reset peak tracking page state to waiting state
        if hasattr(self.main, 'peak_tracking_page'):
            peak_page = self.main.peak_tracking_page
            
            # Reset all modes and selection states
            peak_page.check_peak_range_mode = False
            peak_page.manual_adjust = False
            peak_page.current_peak_name = None
            
            # Reset selection state if it exists
            if hasattr(peak_page, 'selected_peak_index'):
                peak_page.selected_peak_index = None
            
            # Update UI state
            peak_page.update_ui_state()
            
            # Update status message
            if hasattr(peak_page, 'L_current_status_2'):
                peak_page.L_current_status_2.setText("Ready.")
            
            # Update contour plot
            peak_page.update_contour_plot()


class TableModel(QtCore.QAbstractTableModel):
    """
    Custom table model for displaying DataFrame data with proper formatting.
    """
    def __init__(self, data):
        super().__init__()
        self._data = data
    
    def rowCount(self, parent=None):
        return len(self._data)
    
    def columnCount(self, parent=None):
        return len(self._data.columns)
    
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
        
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            # Handle NaN and empty values
            if pd.isna(value) or value == 'nan' or value == '':
                return ""
            elif isinstance(value, (int, float)):
                # Format integers without decimal point
                if value == int(value):
                    return str(int(value))
                # Format floats with appropriate precision
                return str(value)
            return str(value)
        
        if role == QtCore.Qt.TextAlignmentRole:
            # Center align everything
            return QtCore.Qt.AlignCenter
        
        if role == QtCore.Qt.BackgroundRole:
            # Highlight header region (rows 0-9)
            if index.row() < 10:
                return QtGui.QColor(240, 240, 240)
            
            # Highlight column headers (row 11)
            if index.row() == 11:
                return QtGui.QColor(220, 220, 220)
        
        return None