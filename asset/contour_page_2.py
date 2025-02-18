from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from asset.contour_storage import DATA, PATH_INFO, PARAMS, PLOT_OPTIONS, PROCESS_STATUS
from asset.page_asset import LoadingDialog, DragDropLineEdit, normalize_path
from asset.contour_util import extract_contour_data, plot_contour
import os

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