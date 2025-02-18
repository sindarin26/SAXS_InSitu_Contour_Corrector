from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from asset.contour_storage import DATA, PATH_INFO, PARAMS, PLOT_OPTIONS, PROCESS_STATUS
from asset.page_asset import LoadingDialog
from asset.contour_util import select_series, extract_contour_data, plot_contour

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
        
        # Connect signals
        self.connect_signals()
        
        # Initialize variables
        self.current_series = None
        self.canvas = None

    def connect_signals(self):
        """Connect all UI signals"""
        self.PB_back_1.clicked.connect(lambda: self.main.SW_Main_page.setCurrentIndex(1))
        self.PB_export.clicked.connect(self.export_data)
        self.PB_contour_setting.clicked.connect(self.open_contour_settings)
        self.PB_SDD_correction.clicked.connect(self.start_sdd_correction)

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
        
        # Create plot
        canvas = plot_contour(
            DATA['contour_data'],
            temp=PLOT_OPTIONS['temp'],
            legend=PLOT_OPTIONS['legend'],
            graph_option=PLOT_OPTIONS['graph_option'],
            GUI=True
        )
        
        if canvas:
            layout.addWidget(canvas)
            self.GV_contour.setLayout(layout)
            self.canvas = canvas

    def export_data(self):
        """Export contour data"""
        # Implement export functionality
        pass

    def open_contour_settings(self):
        """Open contour plot settings dialog"""
        # Implement settings dialog
        pass

    def start_sdd_correction(self):
        """Start SDD correction process"""
        # Implement SDD correction
        pass