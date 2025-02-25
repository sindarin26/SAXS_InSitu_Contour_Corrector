#asset/contour_page_peak_1.py
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import pyqtgraph as pg
from asset.contour_storage import PARAMS, FITTING_THRESHOLD
from asset.fitting_util import find_peak_extraction, run_automatic_tracking, plot_contour_extraction
from asset.contour_util_gui import QRangeCorrectionHelper

class PeakTrackingPage(QtCore.QObject):
    """
    Page for tracking peaks in contour data. Includes q-range selection,
    automatic tracking, and peak adjustment capabilities.
    """
    
    def __init__(self, main_dialog):
        super().__init__()
        self.main = main_dialog
        self.ui = main_dialog.ui
        
        # State variables
        self.contour_data = None
        self.current_index = 0
        self.max_index = 0
        self.auto_tracking = False
        self.manual_adjust = False
        self.current_peak_name = None
        
        # UI elements from page 1 (index 1)
        self.QGV_qrange = self.ui.QGV_qrange
        self.L_current_status_1 = self.ui.L_current_status_1
        self.PB_back_0 = self.ui.PB_back_0
        self.PB_apply_qrange = self.ui.PB_apply_qrange
        
        # UI elements from page 2 (index 2)
        self.QGV_contour = self.ui.QGV_contour
        self.L_current_status_2 = self.ui.L_current_status_2
        self.PB_next = self.ui.PB_next
        self.PB_stop_auto_tracking = self.ui.PB_stop_auto_tracking
        self.PB_adjust_mode = self.ui.PB_adjust_mode
        self.PB_find_another_peak = self.ui.PB_find_another_peak
        
        # Set initial button text
        self.PB_adjust_mode.setText("Adjust Mode")
        
        # Helper for q-range selection
        self.q_correction_helper = None
        
        # Canvas for contour visualization
        self.contour_canvas = None
        
        # Timer for peak selection
        self.selection_timer = None
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to handlers"""
        # Page 1 buttons
        self.PB_back_0.clicked.connect(self.on_back_to_settings)
        self.PB_apply_qrange.clicked.connect(self.on_apply_qrange)
        
        # Page 2 buttons
        self.PB_next.clicked.connect(self.on_next)
        self.PB_stop_auto_tracking.clicked.connect(self.on_stop_tracking)
        self.PB_adjust_mode.clicked.connect(self.on_enter_adjust_mode)
        self.PB_find_another_peak.clicked.connect(self.on_find_another_peak)
    
    def initialize_peak_tracking(self, contour_data):
        """
        Initialize page with contour data and prepare for peak tracking
        
        Parameters:
            contour_data (dict): The contour data to analyze
        """
        self.contour_data = contour_data
        self.current_index = 0
        self.max_index = len(contour_data['Data']) - 1
        
        # Reset state
        self.auto_tracking = False
        self.manual_adjust = False
        self.current_peak_name = None
        
        # Setup q-range selection graph
        self.setup_q_range_graph()
        
        # Update status
        self.L_current_status_1.setText(
            f"Select q range for frame {self.current_index} / {self.max_index}"
        )
        
        # Set initial button states
        self.update_ui_state()
    
    def setup_q_range_graph(self):
        """Setup QGV_qrange with current frame's data"""
        if not self.contour_data or self.current_index >= len(self.contour_data['Data']):
            return
        
        # Clear any existing content
        if self.QGV_qrange.layout():
            QtWidgets.QWidget().setLayout(self.QGV_qrange.layout())
        
        # Create plot widget
        plot_widget = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(plot_widget)
        self.QGV_qrange.setLayout(layout)
        
        # Initialize helper
        self.q_correction_helper = QRangeCorrectionHelper(plot_widget)
        
        # Set data from current frame
        current_entry = self.contour_data['Data'][self.current_index]
        self.q_correction_helper.set_data(current_entry['q'], current_entry['Intensity'])
        self.q_correction_helper.add_selection_lines()
    
    def update_ui_state(self):
        """
        Update button states based on current tracking state
        
        There are several states:
        1. Selecting q-range (Page 1)
        2. Auto tracking in progress (Page 2)
        3. Tracking failed (Page 2)
        4. Waiting state after tracking completes/stops (Page 2)
        5. Adjustment mode (Page 2)
        """
        # Page 1 state
        self.PB_back_0.setEnabled(True)
        self.PB_apply_qrange.setEnabled(True)
        
        # Page 2 states depend on current tracking state
        waiting_state = not self.auto_tracking and not self.manual_adjust
        
        # During auto tracking, only Next and Stop buttons are enabled
        if self.auto_tracking and not self.manual_adjust:
            self.PB_next.setEnabled(True)
            self.PB_stop_auto_tracking.setEnabled(True)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
        # In adjustment mode with peak selected, enable Next
        elif self.manual_adjust and self.current_peak_name is not None:
            self.PB_next.setEnabled(True)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)  # Always enabled to toggle mode
            self.PB_find_another_peak.setEnabled(False)
        # In adjustment mode without peak selected
        elif self.manual_adjust and self.current_peak_name is None:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)  # Always enabled to toggle mode
            self.PB_find_another_peak.setEnabled(False)
        # In waiting state (tracking complete or stopped)
        elif waiting_state:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(True)
            self.PB_find_another_peak.setEnabled(True)
        # Default state
        else:
            self.PB_next.setEnabled(False)
            self.PB_stop_auto_tracking.setEnabled(False)
            self.PB_adjust_mode.setEnabled(False)
            self.PB_find_another_peak.setEnabled(False)
            
        # Print current state for debugging
        print(f"UI State - auto_tracking: {self.auto_tracking}, manual_adjust: {self.manual_adjust}, current_peak: {self.current_peak_name}")
    
    def on_back_to_settings(self):
        """Handle back button to settings page"""
        # Show warning dialog
        reply = QtWidgets.QMessageBox.question(
            self.main,
            "Warning",
            "Going back will reset all tracking data. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Reset data and go back to settings page
            self.main.init_peak_data(self.contour_data)
            self.ui.stackedWidget.setCurrentIndex(0)
    
    def on_apply_qrange(self):
        """Apply selected q-range and start peak tracking"""
        # Get selected q-range
        q_range = self.q_correction_helper.get_q_range()
        if q_range is None:
            QtWidgets.QMessageBox.warning(
                self.main, 
                "Warning", 
                "Please select a valid q range first."
            )
            return
        
        # Determine if this is a new peak or continued tracking
        flag_start = self.current_peak_name is None
        flag_auto_tracking = True
        
        print(f"Applying q-range: {q_range} at index {self.current_index}, manual_adjust={self.manual_adjust}")
        
        # Find peak at current index
        result = find_peak_extraction(
            contour_data=self.contour_data,
            Index_number=self.current_index,
            input_range=q_range,
            fitting_function=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_auto_tracking=flag_auto_tracking,
            flag_manual_adjust=self.manual_adjust,
            flag_start=flag_start,
            start_index=0 if flag_start else None,
            current_peak_name=self.current_peak_name
        )
        
        # Handle result based on success/failure
        if isinstance(result, str):
            # Failed to find peak
            self.L_current_status_2.setText(f"Failed to find peak: {result}")
            self.auto_tracking = True  # Keep auto_tracking flag true for retry
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
            return
        
        # Unpack successful result
        peak_q, peak_intensity, output_range, fwhm, peak_name = result
        
        print(f"Found peak: {peak_name} at q={peak_q}, intensity={peak_intensity}")
        
        # Check if we already have data for this frame and peak name
        # If so, replace it instead of adding a new entry
        found_existing = False
        for i, entry in enumerate(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']):
            if entry.get('frame_index') == self.current_index and entry.get('peak_name') == peak_name:
                # Replace existing entry
                print(f"Replacing existing entry for frame {self.current_index}, peak {peak_name}")
                self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][i] = {
                    "frame_index": self.current_index,
                    "Time": self.contour_data['Data'][self.current_index].get("Time", 0),
                    "Temperature": self.contour_data['Data'][self.current_index].get("Temperature", 0),
                    "peak_q": peak_q,
                    "peak_Intensity": peak_intensity,
                    "fwhm": fwhm,
                    "peak_name": peak_name,
                    "output_range": output_range
                }
                found_existing = True
                break
        
        if not found_existing:
            # Save new result to tracked_peaks
            current_entry = self.contour_data['Data'][self.current_index]
            new_result = {
                "frame_index": self.current_index,
                "Time": current_entry.get("Time", 0),
                "Temperature": current_entry.get("Temperature", 0),
                "peak_q": peak_q,
                "peak_Intensity": peak_intensity,
                "fwhm": fwhm,
                "peak_name": peak_name,
                "output_range": output_range
            }
            self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'].append(new_result)
        
        # Remember last peak name and update states
        self.current_peak_name = peak_name
        self.auto_tracking = True
        
        # Update status
        self.L_current_status_1.setText(
            f"Tracking peak {peak_name}... ({self.current_index} / {self.max_index})"
        )
        
        # If manually adjusting and successful, continue to next frame or auto-tracking
        if self.manual_adjust:
            # Move to next frame
            next_index = self.current_index + 1
            if next_index <= self.max_index:
                self.current_index = next_index
                print(f"Manual adjustment successful for frame {self.current_index-1}, continuing to frame {self.current_index}")
                
                # Return to automatic tracking mode after manual fix
                self.manual_adjust = False
                print(f"Switching back to automatic tracking from frame {self.current_index}")
                
                # Try automatic tracking for remaining frames
                self.run_automatic_tracking(output_range)  # Use the new output_range for better tracking
            else:
                # We've reached the end, complete the tracking
                print(f"Manual adjustment completed the tracking")
                self.auto_tracking = False
                self.manual_adjust = False
                
                # Add to found peak list if not already there
                if peak_name not in self.main.PEAK_EXTRACT_DATA['found_peak_list']:
                    self.main.PEAK_EXTRACT_DATA['found_peak_list'].append(peak_name)
                
                self.L_current_status_2.setText(f"Tracking complete for {peak_name}!")
                self.ui.stackedWidget.setCurrentIndex(2)
                self.update_ui_state()
                self.update_contour_plot()
        # If this is the first frame, start automatic tracking
        elif self.current_index == 0:
            self.current_index += 1  # Move to next frame
            
            # Try automatic tracking for remaining frames
            self.run_automatic_tracking(output_range)  # Use output_range for better tracking
        else:
            # If not first frame (unusual case), just update UI
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
    
    def run_automatic_tracking(self, initial_q_range):
        """
        Run automatic tracking for remaining frames
        
        Parameters:
            initial_q_range (tuple): Initial q-range for tracking
        """
        print(f"Starting automatic tracking from index {self.current_index}")
        
        # Define callbacks for tracking updates
        def on_error(message):
            print(f"Tracking error: {message}")
            self.L_current_status_2.setText(message)
            # Keep auto_tracking flag true so we can retry
            self.auto_tracking = True
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
        
        def on_update(frame_index, peak_q, peak_intensity, fwhm, peak_name):
            print(f"Tracking update: frame {frame_index}, peak {peak_name}")
            self.L_current_status_1.setText(
                f"Tracking peak {peak_name}... ({frame_index} / {self.max_index})"
            )
            QtWidgets.QApplication.processEvents()  # Update UI during tracking
        
        def on_finish(message, last_peak_info):
            print(f"Tracking finished: {message}")
            print(f"Last peak info: {last_peak_info}")
            
            # If tracking is complete (either reached the end or explicitly finished)
            peak_name = last_peak_info.get("peak_name")
            
            # Add to found peak list if not already there
            if peak_name and peak_name not in self.main.PEAK_EXTRACT_DATA['found_peak_list']:
                print(f"Adding peak {peak_name} to found_peak_list")
                self.main.PEAK_EXTRACT_DATA['found_peak_list'].append(peak_name)
            
            # Reset tracking state to waiting state
            self.auto_tracking = False
            self.manual_adjust = False
            self.current_peak_name = None
            
            self.L_current_status_2.setText(f"Tracking complete for {peak_name}!")
            self.ui.stackedWidget.setCurrentIndex(2)  # Go to contour page
            self.update_ui_state()
            self.update_contour_plot()
        
        # Remember current state
        peak_info = {
            "peak_q": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['peak_q'],
            "peak_intensity": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['peak_Intensity'],
            "fwhm": self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'][-1]['fwhm'],
            "peak_name": self.current_peak_name
        }
        
        # Run tracking
        success, final_index, last_peak_info = run_automatic_tracking(
            contour_data=self.contour_data,
            tracked_peaks=self.main.PEAK_EXTRACT_DATA['tracked_peaks'],
            current_peak_info=peak_info,
            current_index=self.current_index - 1,  # Adjusting index since we've already processed the current frame
            max_index=self.max_index,
            get_q_range_func=lambda: initial_q_range,  # Use the initial q-range for all frames
            fitting_model=PARAMS.get('fitting_model', 'gaussian'),
            threshold_config=FITTING_THRESHOLD,
            flag_start=False,  # Never start a new peak in auto tracking
            flag_auto_tracking=True,  # Always set to True for proper tracking
            flag_manual_adjust=self.manual_adjust,
            on_error_callback=on_error,
            on_update_callback=on_update,
            on_finish_callback=on_finish
        )
        
        # Update current index for next operation
        self.current_index = final_index
    
    def update_contour_plot(self):
        """Update the contour plot with current tracked peaks"""
        if not self.contour_data:
            print("No contour data available")
            return
            
        if not self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
            print("No tracked peaks data available")
            return
            
        print(f"Updating contour plot with {len(self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data'])} tracked peaks")
        print(f"Found peak list: {self.main.PEAK_EXTRACT_DATA['found_peak_list']}")
        
        # Create or update the list of peaks to show
        # For auto-tracking, include the current_peak_name even if not in found_peak_list
        peaks_to_show = list(self.main.PEAK_EXTRACT_DATA['found_peak_list'])
        if self.current_peak_name and self.current_peak_name not in peaks_to_show:
            peaks_to_show.append(self.current_peak_name)
            
        # Create contour plot
        self.contour_canvas = plot_contour_extraction(
            contour_data=self.contour_data,
            tracked_peaks=self.main.PEAK_EXTRACT_DATA['tracked_peaks'],
            found_peak_list=peaks_to_show,  # Show all relevant peaks
            flag_adjust_mode=self.manual_adjust
        )
        
        if self.contour_canvas:
            print("Contour canvas created successfully")
            # Clear any existing layout
            if self.QGV_contour.layout():
                QtWidgets.QWidget().setLayout(self.QGV_contour.layout())
                
            # Set new layout with canvas
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.contour_canvas)
            self.QGV_contour.setLayout(layout)
            self.QGV_contour.update()  # Force update
        else:
            print("Failed to create contour canvas")
    
    def on_next(self):
        """
        Handle next button click
        
        Different behaviors depending on state:
        - If auto tracking failed: retry at current point with manual adjustment
        - If in adjustment mode: edit selected peak
        """
        print(f"Next button clicked - current state: auto_tracking={self.auto_tracking}, manual_adjust={self.manual_adjust}")
        
        if self.manual_adjust and self.current_peak_name:
            # In adjustment mode, edit selected peak
            self.ui.stackedWidget.setCurrentIndex(1)  # Go to q-range page
            self.setup_q_range_graph()  # Setup graph for current frame
            
            # Update status
            self.L_current_status_1.setText(
                f"Adjusting peak {self.current_peak_name} at frame {self.current_index}"
            )
        elif self.auto_tracking:
            # Set manual_adjust to True so threshold checks will be bypassed
            self.manual_adjust = True
            print(f"Setting manual_adjust=True for frame {self.current_index}")
            
            # Retry auto tracking from current point
            self.ui.stackedWidget.setCurrentIndex(1)  # Go to q-range page
            self.setup_q_range_graph()  # Setup graph for current frame
            
            # Update status
            self.L_current_status_1.setText(
                f"Manual adjustment for frame {self.current_index} / {self.max_index}"
            )
        
        self.update_ui_state()
    
    def on_stop_tracking(self):
        """Stop current tracking and add to found_peak_list"""
        if not self.current_peak_name:
            return
            
        # Add current peak to found_peak_list if not already there
        if self.current_peak_name not in self.main.PEAK_EXTRACT_DATA['found_peak_list']:
            self.main.PEAK_EXTRACT_DATA['found_peak_list'].append(self.current_peak_name)
        
        # Reset tracking state
        self.auto_tracking = False
        self.manual_adjust = False
        self.current_peak_name = None
        
        # Update UI
        self.L_current_status_2.setText("Tracking stopped and saved.")
        self.update_ui_state()
        self.update_contour_plot()
    
    def on_enter_adjust_mode(self):
        """Enter or exit peak adjustment mode"""
        if self.manual_adjust:
            # Already in adjust mode, exit it
            print("Exiting adjustment mode")
            self.manual_adjust = False
            self.current_peak_name = None
            
            # Update button text
            self.PB_adjust_mode.setText("Adjust Mode")
            
            # Update UI
            self.L_current_status_2.setText("Adjustment mode disabled.")
            self.update_ui_state()
            self.update_contour_plot()
        else:
            # Enter adjustment mode
            print("Entering adjustment mode")
            self.manual_adjust = True
            
            # Update button text
            self.PB_adjust_mode.setText("Disable Adjust Mode")
            
            # Update UI
            self.L_current_status_2.setText("Click on a peak to select it for adjustment.")
            self.update_ui_state()
            
            # Redraw contour with selection enabled
            self.update_contour_plot()
            
            # Setup selection handler
            if self.contour_canvas and hasattr(self.contour_canvas, 'get_selected_peak_name'):
                # Create a timer to check for peak selection
                self.selection_timer = QtCore.QTimer()
                self.selection_timer.timeout.connect(self.check_peak_selection)
                self.selection_timer.start(500)  # Check every 500 ms
    
    def check_peak_selection(self):
        """Check if a peak is selected in the contour plot"""
        if not self.manual_adjust or not self.contour_canvas:
            if hasattr(self, 'selection_timer') and self.selection_timer.isActive():
                self.selection_timer.stop()
            return
            
        # Get selected peak name from canvas
        if hasattr(self.contour_canvas, 'get_selected_peak_name'):
            selected = self.contour_canvas.get_selected_peak_name()
            
            if selected and selected != self.current_peak_name:
                print(f"Peak selected: {selected}")
                # New peak selected, update state
                self.current_peak_name = selected
                
                # Find frame index for this peak
                found_frame = False
                for entry in self.main.PEAK_EXTRACT_DATA['tracked_peaks']['Data']:
                    if entry.get('peak_name') == selected:
                        self.current_index = entry.get('frame_index', 0)
                        found_frame = True
                        break
                
                if not found_frame:
                    print(f"Warning: Could not find frame index for peak {selected}")
                    self.current_index = 0
                
                # Update UI
                self.L_current_status_2.setText(
                    f"Selected peak {selected} at frame {self.current_index}. Click Next to adjust it."
                )
                self.update_ui_state()
                self.PB_next.setEnabled(True)  # Ensure Next button is enabled
    
    def on_find_another_peak(self):
        """Start tracking a new peak"""
        # Reset index and tracking state
        self.current_index = 0
        self.auto_tracking = True
        self.manual_adjust = False
        self.current_peak_name = None
        
        # Go to q-range selection page
        self.ui.stackedWidget.setCurrentIndex(1)
        self.setup_q_range_graph()
        
        # Update status
        self.L_current_status_1.setText(
            f"Select q range for new peak at frame {self.current_index} / {self.max_index}"
        )
        
        self.update_ui_state()