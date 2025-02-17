import os
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt
from asset.storage import DATA, PATH_INFO, PARAMS
from asset.page_asset import DragDropLineEdit, normalize_path
from asset.spec_log_extractor import parse_log_file
from asset.dat_extractor import process_dat_files

class BrowsePage(QtCore.QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main = main_window
        self.setupPageWidgets()
        self.connectSignals()
        self.debug_init()

    def debug_init(self):
        self.LE_spec_log_dir.setText("/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/log241219-HS")
        self.LE_dat_dir.setText("/Users/yhk/Library/CloudStorage/OneDrive-postech.ac.kr/Seminar/2024-12-19 whole/2024-12-19/HS/Averaged/")
        self.LE_output_dir.setText("./")


    def setupPageWidgets(self):
        """Initialize all widgets for the browse page"""
        # Get widgets from UI
        self.LE_spec_log_dir = DragDropLineEdit(self.main)
        self.LE_dat_dir = DragDropLineEdit(self.main)
        self.LE_output_dir = DragDropLineEdit(self.main)
        
        # Replace existing line edits with drag-drop enabled ones
        for old_widget, new_widget in [
            (self.main.ui.LE_spec_log_dir, self.LE_spec_log_dir),
            (self.main.ui.LE_dat_dir, self.LE_dat_dir),
            (self.main.ui.LE_output_dir, self.LE_output_dir)
        ]:
            layout = old_widget.parent().layout()
            layout.replaceWidget(old_widget, new_widget)
            old_widget.deleteLater()

        # Initialize references to other widgets
        self.TW_Process_List = self.main.ui.TW_Process_List
        self.PB_spec_log_browse = self.main.ui.PB_spec_log_browse
        self.PB_dat_browse = self.main.ui.PB_dat_browse
        self.PB_output_browse = self.main.ui.PB_output_browse
        self.PB_spec_apply = self.main.ui.PB_spec_apply
        self.PB_dat_apply = self.main.ui.PB_dat_apply
        self.PB_output_apply = self.main.ui.PB_output_apply
        self.PB_next_1 = self.main.ui.PB_next_1
        self.PB_reset = self.main.ui.PB_reset
        
        # Setup table
        self.TW_Process_List.setColumnCount(4)
        self.TW_Process_List.setHorizontalHeaderLabels(["Filename", "Index", "Temperature", "Dat"])
        
        header = self.TW_Process_List.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

        # Initial button states
        self.PB_next_1.setEnabled(False)

    def connectSignals(self):
        """Connect all signals for the browse page"""
        # Browse buttons
        self.PB_spec_log_browse.clicked.connect(self.browseSpecLog)
        self.PB_dat_browse.clicked.connect(self.browseDatDir)
        self.PB_output_browse.clicked.connect(self.browseOutputDir)

        # Apply buttons
        self.PB_spec_apply.clicked.connect(self.applySpecLog)
        self.PB_dat_apply.clicked.connect(self.applyDatDir)
        self.PB_output_apply.clicked.connect(self.applyOutputDir)

        # Line edit return pressed
        self.LE_spec_log_dir.returnPressed.connect(self.applySpecLog)
        self.LE_dat_dir.returnPressed.connect(self.applyDatDir)
        self.LE_output_dir.returnPressed.connect(self.applyOutputDir)

        # Navigation buttons
        self.PB_reset.clicked.connect(self.resetAll)
        self.PB_next_1.clicked.connect(self.gotoNextPage)

    def browseSpecLog(self):
        """Open file dialog to browse for spec log file"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.main,
            "Select SPEC Log File",
            "",
            "All Files (*.*)"
        )
        if file_path:
            self.LE_spec_log_dir.setText(normalize_path(file_path))
            self.applySpecLog()

    def browseDatDir(self):
        """Open file dialog to browse for dat directory"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            "Select DAT Files Directory"
        )
        if dir_path:
            self.LE_dat_dir.setText(normalize_path(dir_path))
            self.applyDatDir()

    def browseOutputDir(self):
        """Open file dialog to browse for output directory"""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self.main,
            "Select Output Directory"
        )
        if dir_path:
            self.LE_output_dir.setText(normalize_path(dir_path))
            self.applyOutputDir()

    def applySpecLog(self):
        """Process spec log file"""
        path = normalize_path(self.LE_spec_log_dir.text().strip())
        if not path or not os.path.isfile(path):
            return

        try:
            log_data = parse_log_file(path)
            if log_data:
                # Save data
                DATA['log_data'] = log_data
                PATH_INFO['spec_log'] = path

                # Update table with initial data
                self.TW_Process_List.setRowCount(0)  # Clear current rows
                entries = [(k, v) for k, v in log_data.items() 
                        if k != "Error Log" and isinstance(v, dict)]
                
                self.TW_Process_List.setRowCount(len(entries))
                
                for row, (filename, data) in enumerate(entries):
                    # Filename
                    self.TW_Process_List.setItem(row, 0, 
                        QtWidgets.QTableWidgetItem(filename))
                    
                    # Index
                    index_item = QtWidgets.QTableWidgetItem(str(data.get('Index', '')))
                    index_item.setTextAlignment(Qt.AlignCenter)
                    self.TW_Process_List.setItem(row, 1, index_item)
                    
                    # Temperature
                    temp_item = QtWidgets.QTableWidgetItem(
                        f"{data.get('Temperature', ''):.2f}")
                    temp_item.setTextAlignment(Qt.AlignCenter)
                    self.TW_Process_List.setItem(row, 2, temp_item)
                    
                    # Dat status (initially all False)
                    dat_item = QtWidgets.QTableWidgetItem('✗')
                    dat_item.setTextAlignment(Qt.AlignCenter)
                    self.TW_Process_List.setItem(row, 3, dat_item)

                self.checkNextButtonState()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to process spec log file:\n{str(e)}"
            )

    def applyDatDir(self):
        """Process dat directory"""
        path = normalize_path(self.LE_dat_dir.text().strip())
        if not path or not os.path.isdir(path):
            return

        try:
            if DATA['log_data']:
                extracted_data = process_dat_files(path, DATA['log_data'])
                if extracted_data:
                    DATA['extracted_data'] = extracted_data
                    PATH_INFO['dat_dir'] = path
                    self.updateProcessList()
                    self.checkNextButtonState()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to process dat files:\n{str(e)}"
            )

    def applyOutputDir(self):
        """Set and create output directory"""
        path = normalize_path(self.LE_output_dir.text().strip())
        if not path:
            return

        try:
            os.makedirs(path, exist_ok=True)
            PATH_INFO['output_dir'] = path
            self.checkNextButtonState()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to create output directory:\n{str(e)}"
            )

    def updateProcessList(self):
        """Update process list table with current data"""
        self.TW_Process_List.setRowCount(0)
        
        if not DATA['log_data']:
            return

        log_data = DATA['log_data']
        extracted_data = DATA['extracted_data']

        # Filter out Error Log and get entries
        entries = [(k, v) for k, v in log_data.items() 
                  if k != "Error Log" and isinstance(v, dict)]
        
        self.TW_Process_List.setRowCount(len(entries))
        
        for row, (filename, data) in enumerate(entries):
            # Filename
            self.TW_Process_List.setItem(row, 0, 
                QtWidgets.QTableWidgetItem(filename))
            
            # Index
            index_item = QtWidgets.QTableWidgetItem(str(data.get('Index', '')))
            index_item.setTextAlignment(Qt.AlignCenter)
            self.TW_Process_List.setItem(row, 1, index_item)
            
            # Temperature
            temp_item = QtWidgets.QTableWidgetItem(
                f"{data.get('Temperature', ''):.2f}")
            temp_item.setTextAlignment(Qt.AlignCenter)
            self.TW_Process_List.setItem(row, 2, temp_item)
            
            # Dat status
            has_dat = (extracted_data is not None and 
                      filename in extracted_data and 
                      'q' in extracted_data[filename])
            dat_item = QtWidgets.QTableWidgetItem('✓' if has_dat else '✗')
            dat_item.setTextAlignment(Qt.AlignCenter)
            self.TW_Process_List.setItem(row, 3, dat_item)

    def checkNextButtonState(self):
        """Enable next button if all required data is present"""
        has_log = DATA['log_data'] is not None
        has_dat = DATA['extracted_data'] is not None
        has_output = PATH_INFO['output_dir'] is not None
        
        self.PB_next_1.setEnabled(has_log and has_dat and has_output)

    def resetAll(self):
        """Reset all settings and clear data"""
        # Clear storages
        DATA.clear()
        DATA.update({
            'log_data': None,
            'extracted_data': None,
            'contour_data': None,
            'tracked_peaks': None,
        })
        
        PATH_INFO.clear()
        PATH_INFO.update({
            'spec_log': None,
            'dat_dir': None,
            'output_dir': None,
        })
        
        # Clear UI
        self.LE_spec_log_dir.clear()
        self.LE_dat_dir.clear()
        self.LE_output_dir.clear()
        self.TW_Process_List.setRowCount(0)
        self.PB_next_1.setEnabled(False)

    def gotoNextPage(self):
        """Prepare data and move to next page"""
        try:
            from contour_util import select_series, extract_contour_data
            
            # Get selected series and extract contour data
            selected_series = select_series(DATA['extracted_data'])
            DATA['contour_data'] = extract_contour_data(selected_series, DATA['extracted_data'])
            
            # Move to next page
            self.main.SW_Main_page.setCurrentIndex(1)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self.main,
                "Error",
                f"Failed to prepare data for next page:\n{str(e)}"
            )