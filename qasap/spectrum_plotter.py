# qasap Spectrum Plotter --- v0.11
"""
Main spectrum plotter widget for interactive spectral analysis

KEYBOARD SHORTCUTS:

Navigation & View Controls:
  [ / ]                - Pan left/right through spectrum
  \ (backslash)        - Reset spectrum view to starting bounds
  x                    - Center on mouse wavelength position
  u / i                - Set lower/upper x-bounds (wavelength bounds)
  t / T                - Zoom in/out horizontally (narrow/widen x-range)
  y / Y                - Zoom in/out vertically (narrow/widen y-range)
  O / P                - Set lower/upper y-bounds (flux bounds)
  l                    - Toggle log y-axis
  L                    - Toggle log x-axis
  f                    - Enter fullscreen mode

Spectrum Processing:
  1-9                  - Apply Gaussian smoothing with different kernel sizes
  0                    - Remove smoothing (restore original spectrum)
  ~ (tilde)            - Toggle between step plot and line plot
  ` (backtick)         - Save screenshot of plot

Fitting Modes:
  m                    - Enter continuum fitting mode (define regions with SPACE)
  M                    - Remove a continuum region
  ENTER (in continuum mode) - Fit polynomial continuum to defined regions
  g                    - Enter Single Mode Gaussian fit (click to fit, SPACE to select bounds)
  | (pipe)             - Enter Multi-Gaussian fit mode (fit multiple Gaussians simultaneously)
  n                    - Single mode Voigt profile fitting
  e                    - Open line list selector window (new line list system)
  k                    - Open Listfit window for composite fitting

Measurement & Analysis:
  v                    - Calculate equivalent width of fitted line
  S                    - Save all fits (Gaussian, Voigt, Continuum, Listfit) to files
  a                    - Load saved fits from file (or use Load Fit button)
  ;                    - Show/toggle total line for Single Mode fitted lines
  w                    - Remove fitted profile under cursor
  ,                    - Add a line tag to fitted profile under cursor
  <                    - Remove tag from fitted profile under cursor
  r                    - Toggle residual panel
  j                    - Toggle Item Tracker window
  ?                    - Show keyboard shortcuts help window

Redshift & Velocity:
  z                    - Enter redshift mode (select already fitted line under cursor with SPACE)
  escape               - Exit redshift mode
  SPACE (in velocity mode) - Toggle between wavelength and velocity space
  d                    - Activate velocity mode (set rest-frame wavelength)
  SPACE (in mask mode) - Select bounds to mask out regions
  RETURN (in mask mode) - Finish masking

Instrument Filters & Bands:
  ! through ) (Shift+1-0) - Toggle instrument bandpass overlays (press Shift+number)
  - / _ / = / +        - Show filter bandpasses (requires downloaded filter files)

Item Management:
  * (asterisk)         - Toggle Item Tracker window visibility

Help:
  ?                    - Show keyboard shortcuts help window

Display Options (Line Lists):
  Line lists from the selector (e key) automatically display with current pan position
  Redshift adjustments update line displays immediately

File Storage:
  All saved screenshots, redshifts, and profile info are stored in the directory
  where QASAP was launched from.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.interpolate import interp1d
import lmfit
from lmfit import Model, Parameters, conf_interval, minimize
from lmfit.models import PolynomialModel
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon, QKeyEvent
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import ast
import re
from qasap.spectrum_io import SpectrumIO

# Suppress numexpr pandas UserWarning
warnings.filterwarnings('ignore', category=UserWarning, module='pandas.core.computation.expressions')

# Import LineListWindow from sibling module
from .linelist_window import LineListWindow
from .listfit_window import ListfitWindow
from .item_tracker import ItemTracker
from .fit_information_window import FitInformationWindow
from .linelist import get_available_line_lists
from .linelist_selector_window import LineListSelector
from .action_history import ActionHistory
from .action_history_window import ActionHistoryWindow


class OutputStreamCapture:
    """Helper class to capture stdout/stderr and emit to output panel."""
    def __init__(self, output_panel):
        self.output_panel = output_panel
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
    
    def write(self, message):
        """Write message to output panel."""
        if message and message != '\n':
            self.output_panel.append_text(message)
        # Also print to original stdout
        self.original_stdout.write(message)
    
    def flush(self):
        """Flush the stream."""
        self.original_stdout.flush()


class OutputPanel(QtWidgets.QWidget):
    """Panel for displaying output (stdout/stderr) from the application."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # Set up output capture
        self.stream_capture = OutputStreamCapture(self)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Redirect stdout and stderr
        sys.stdout = self.stream_capture
        sys.stderr = self.stream_capture
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(0)
        
        # Create text display area
        self.text_edit = QtWidgets.QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setMaximumHeight(100)  # Set reasonable height
        self.text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f5f5f5;
                color: #333333;
                font-family: 'Courier New', monospace;
                font-size: 12pt;
                border: 1px solid #cccccc;
            }
        """)
        
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
    
    def append_text(self, text):
        """Append text to the output panel."""
        self.text_edit.appendPlainText(text.rstrip('\n'))
        # Auto-scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_output(self):
        """Clear the output panel."""
        self.text_edit.clear()
    
    def restore_streams(self):
        """Restore original stdout/stderr."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class HelpWindow(QtWidgets.QDialog):
    """Help window displaying all keyboard shortcuts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        from qasap.ui_utils import get_qasap_icon
        self.setWindowTitle("QASAP - Keyboard Shortcuts Help")
        self.setWindowIcon(get_qasap_icon())
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        # Create text display
        text_edit = QtWidgets.QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setMarkdown(self.get_help_text())
        layout.addWidget(text_edit)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setLayout(layout)
    
    def get_help_text(self):
        return """# QASAP Keyboard Shortcuts

## Navigation & View Controls
- **[** / **]** - Pan left/right through spectrum
- **\\\\** (backslash) - Reset spectrum view to starting bounds
- **x** - Center on wavelength position under cursor
- **u** / **i** - Set lower/upper x-bounds (wavelength bounds)
- **T** / **t** - Zoom in/out horizontally (narrow/widen x-range)
- **Y** / **y** - Zoom in/out vertically (narrow/widen y-range)
- **O** / **P** - Set lower/upper y-bounds (flux bounds)
- **l** - Toggle log y-axis
- **L** - Toggle log x-axis
- **f** - Enter fullscreen mode

## Spectrum Processing
- **1**-**9** - Apply Gaussian smoothing with different kernel sizes
- **0** - Remove smoothing (restore original spectrum)
- **~** (tilde) - Toggle between step plot and line plot
- **`** (backtick) - Save screenshot of plot

## Fitting Modes
- **m** - Enter continuum fitting mode (define regions with SPACE, then ENTER)
- **M** - Remove a continuum region
- **d** - Enter Single Mode Gaussian fit (SPACE to select bounds)
- **|** (pipe) - Enter Multi-Gaussian fit mode (define bounds with SPACE and ENTER)
- **n** - Single mode Voigt profile fitting
- **H** - Enter Listfit window for composite fitting
- **r** - Toggle residual panel

## Line List
- **e** - Open Line List window

## Line Profiles
- **w** - Remove fitted profile under cursor
- **,** (comma) - Add a line tag to fitted profile under cursor
- **<** (less than) - Remove tag from fitted profile under cursor
- **S** - Save all fitted profiles (Gaussian, Voigt, Continuum, Listfit) to files
- **;** (semicolon) - Show/toggle total line for Single Mode fitted lines
- **v** - Calculate equivalent width of fitted line. In progress. Use with caution.

## Redshift & Velocity
- **z** - Enter redshift mode (select already fitted line under cursor with SPACE, hit ESC to exit redshift mode)
- **b** - Activate velocity mode (set rest-frame wavelength). In progress.

## Instrument Filters & Bands
- **!** through **)** (Shift+1-0) - Toggle instrument bandpass overlays
- **-** / **_** / **=** / **+** - Show filter bandpasses

## Item Management
- **j** - Toggle Item Tracker window (shows summary of all fitted profiles)
- **Z** - Toggle Fit Information window (shows detailed parameters for all fitted profiles)

## Application Control
- **q** or **Q** - Quit QASAP

## Help
- **?** - Show this help window

## File Storage
All saved screenshots, redshifts, and profile info are stored in the directory where QASAP was launched from.
"""
    
    def keyPressEvent(self, event):
        """Handle key press events in help window - forward Q to parent"""
        if event.key() in (Qt.Key_Q, Qt.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)

class SpectrumPlotter(QtWidgets.QMainWindow):
    def __init__(self, fits_file, redshift=0.0, zoom_factor=0.1, file_flag=0, lsf="10",):
        super().__init__()

        self.fits_file = fits_file
        self.redshift = redshift
        self.zoom_factor = zoom_factor
        self.file_flag = file_flag
        self.lsf = lsf
        self.lsf_kernel_x = None
        self.lsf_kernel_y = None

        # Process LSF
        self.process_lsf(lsf)

        # Create central widget for the control panel
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        # Initialize the control panel (will be added to central widget)
        # This must be done BEFORE create_menu_bar() since the menu bar references windows created here
        self.init_controlpanel()
        
        # Initialize window creation attempt flags and resources BEFORE create_menu_bar()
        self.help_window = None
        self.help_window_attempted = False  # Track if we've already tried to create it
        resources_dir = str(Path(__file__).parent.parent / 'resources')
        self.line_list_selector = None  # Will be created on demand
        self.line_list_selector_attempted = False  # Track if we've already tried to create it
        self.resources_dir = resources_dir
        
        # Create the menu bar (after control panel is initialized)
        self.create_menu_bar()

        self.wav = []
        self.spec = []
        self.err = []
        self.spec_line = []
        self.err_line = []
        self.spec_step = []
        self.err_step = []
        self.x_data = []
        self.original_spec = []
        self.data_loaded_from_gui = False  # Track if data was loaded via GUI
        self.fig = None  # Will be created on first plot_spectrum() call
        self.ax = None  # Will be created on first plot_spectrum() call

        self.gaussian_mode = False
        self.multi_gaussian_mode = False
        self.multi_gaussian_mode_old = False # Placeholder for previous multi Gaussian fit functionality
        self.bounds = []
        self.bound_lines = []
        self.gaussian_line = None
        self.gaussian_fits = []  # Stores each Gaussian's line, x-bounds, and parameters

        self.continuum_mode = False
        self.continuum_regions = []  # Stores regions for continuum fitting
        self.continuum_patches = []
        self.continuum_fits = []
        self.continuum_params = []

        self.listfit_mode = False
        self.listfit_bounds = []
        self.listfit_bound_lines = []
        self.listfit_components = []
        self.listfit_fits = []  # Stores completed listfit results
        self.listfit_component_lines = {}  # Store plotted component lines by component ID
        self.deleted_listfit_polynomials = set()  # Track deleted polynomial item_ids for residual calculation

        self.redshift_estimation_mode = False
        self.rest_wavelength = None  # Set to `None` if no initial rest wavelength
        self.rest_id = None
        self.wavelength_unit = "Å"  # Current wavelength unit (Å, nm, or µm)

        self.x_upper_bound = None
        self.x_lower_bound = None
        self.y_upper_bound = None
        self.y_lower_bound = None
        self.original_xlim = None
        self.original_ylim = None

        self.is_step_plot = False
        self.spectrum_line = None
        self.error_line = None
        self.is_residual_shown = False # Residual panel is initially hidden
        self.linelist_plots = []
        self.is_velocity_mode = False
        self.velocities = []
        self.residual_line = None

        self.residual_ax = None  # Placeholder for the residual axis
        self.residuals = []

        self.line_ids = []
        self.line_wavelengths = []
        self.band_ranges = []
        self.current_gaussian_plot = None
        self.current_voigt_plot = None

        self.ew_fill = None

        self.comp_x = []
        self.comp_xs = []
        self.continuum_subtracted_y = []
        self.continuum_subtracted_ys = []

        self.voigt_mode = False
        self.voigt_fits = []
        self.voigt_comps = []
        self.gaussian_comps = []
        self.multi_voigt_mode = False
        self.fit_id = 0
        self.component_id = 0

        self.markers = []
        self.labels = []
        self.selected_gaussian = None
        self.selected_voigt = None

        self.selected_line_id = None
        self.selected_line_wavelength = None

        self.show_total_line = False

        self.show_filters = False
        self.filter_lines = []

        self.osc_ids = []
        self.osc_wavelengths = []
        self.osc_strengths = []

        self.show_bands = False
        self.band_areas = []
        self.band_labels = []

        # MCMC
        self.bayes_bounds = []
        self.bayes_mode = False
        self.bayes_bound_lines = []
        self.mask_bounds = []
        self.mask_bound_lines = []
        self.mask_mode = False
        self.mask_patches = []

        # Item Tracker
        self.item_tracker = ItemTracker()
        self.fit_information_window = FitInformationWindow()
        self.item_id_counter = 0
        self.item_id_map = {}  # Maps item_id to {'type': 'gaussian', 'fit': fit_dict, ...}
        self.highlighted_item_ids = set()  # Track all currently highlighted items
        self.redshift_selected_line = None  # Track the line object selected for redshift
        
        # Connect items_changed signal to update total line if displayed
        self.item_tracker.items_changed.connect(self.update_total_line_if_shown)
        
        # Action History for Undo/Redo
        self.action_history = ActionHistory()
        self.action_history_window = ActionHistoryWindow()
        self.action_history_window.set_action_history(self.action_history)
        self.action_history_window.action_selected.connect(self.on_action_selected)
        
        # Track if this is the first spectrum load (for initial action recording)
        self.is_first_load = True
        self.initial_spectrum_file = fits_file
        
        # Active line lists tracking (resources_dir and line_list_selector already initialized earlier)
        self.active_line_lists = []  # {linelist: LineList, color: str}
        self.current_linelist_lines = []  # Store plotted linelist lines for removal

    def create_menu_bar(self):
        """Create the menu bar with Qasap, File, Edit, and View menus"""
        menubar = self.menuBar()
        
        # Create Qasap menu (application menu on macOS)
        self.qasap_menu = menubar.addMenu("Qasap")
        
        # Add Quit action to Qasap menu
        quit_action = self.qasap_menu.addAction("Quit QASAP")
        quit_action.setShortcut("Cmd+Q")
        quit_action.triggered.connect(self.quit_application)
        
        # Create File menu
        self.file_menu = menubar.addMenu("File")
        
        # Create Edit menu
        self.edit_menu = menubar.addMenu("Edit")
        
        # Add Undo action
        undo_action = self.edit_menu.addAction("Undo")
        undo_action.setShortcut("Cmd+Z")
        undo_action.triggered.connect(self.on_undo)
        
        # Add Redo action
        redo_action = self.edit_menu.addAction("Redo")
        redo_action.setShortcut("Cmd+Shift+Z")
        redo_action.triggered.connect(self.on_redo)
        
        # Create View menu
        self.view_menu = menubar.addMenu("View")
        self.update_view_menu()

    def update_view_menu(self):
        """Update the View menu with all available windows"""
        self.view_menu.clear()
        
        # Get windows to display with their visibility status
        windows = []
        
        # Always add Control Panel (self)
        is_visible = self.isVisible()
        windows.append(("Control Panel", self, is_visible))
        
        # Add Spectrum Plotter (matplotlib window)
        # Check if figure exists and has a visible canvas
        if hasattr(self, 'fig') and self.fig is not None:
            if hasattr(self.fig, 'canvas'):
                try:
                    # The spectrum plotter is always visible by default since it's shown with plt.show()
                    is_visible = True
                    windows.append(("Spectrum Plotter", self.fig.canvas, is_visible))
                except:
                    pass
        
        # Add Item Tracker (always exists after initialization)
        if hasattr(self, 'item_tracker') and self.item_tracker is not None:
            is_visible = self.item_tracker.isVisible()
            windows.append(("Item Tracker", self.item_tracker, is_visible))
        
        # Add Fit Information (always exists after initialization)
        if hasattr(self, 'fit_information_window') and self.fit_information_window is not None:
            is_visible = self.fit_information_window.isVisible()
            windows.append(("Fit Information", self.fit_information_window, is_visible))
        
        # Add Action History (always exists after initialization)
        if hasattr(self, 'action_history_window') and self.action_history_window is not None:
            is_visible = self.action_history_window.isVisible()
            windows.append(("Action History", self.action_history_window, is_visible))
        
        # Add Help window (create if doesn't exist, but only try once)
        if not self.help_window_attempted:
            self._create_help_window()
            self.help_window_attempted = True
        if self.help_window is not None:
            is_visible = self.help_window.isVisible()
            windows.append(("Help", self.help_window, is_visible))
        
        # Add Line List window (create if doesn't exist, but only try once)
        if not self.line_list_selector_attempted:
            self._create_line_list_selector()
            self.line_list_selector_attempted = True
        if self.line_list_selector is not None:
            is_visible = self.line_list_selector.isVisible()
            windows.append(("Line List", self.line_list_selector, is_visible))
        
        # Add Listfit window (create if doesn't exist)
        if hasattr(self, 'listfit_window'):
            if self.listfit_window is None:
                self._create_listfit_window()
            if self.listfit_window is not None:
                is_visible = self.listfit_window.isVisible()
                windows.append(("Listfit", self.listfit_window, is_visible))
        
        # Add each window as a checkable menu item with visual indicator
        for window_name, window_obj, is_visible in windows:
            action = self.view_menu.addAction(window_name)
            action.setCheckable(True)
            action.setChecked(is_visible)
            # Store the window object as data so we can retrieve it in the handler
            action.window_obj = window_obj
            action.triggered.connect(lambda checked, obj=window_obj: self.toggle_window(obj))
        
        # If no windows were added, add a placeholder so the menu isn't empty
        if len(self.view_menu.actions()) == 0:
            placeholder = self.view_menu.addAction("(No windows available)")
            placeholder.setEnabled(False)
    
    def _create_help_window(self):
        """Create the Help window if it doesn't exist"""
        try:
            from qasap.help_window import HelpWindow
            self.help_window = HelpWindow()
        except Exception as e:
            print(f"Could not create Help window: {e}")
    
    def _create_line_list_selector(self):
        """Create the Line List Selector if it doesn't exist"""
        try:
            from qasap.line_list_selector import LineListSelector
            self.line_list_selector = LineListSelector(self.resources_dir)
            self.line_list_selector.line_lists_changed.connect(self.on_line_lists_changed)
        except Exception as e:
            print(f"Could not create Line List Selector: {e}")
    
    def _create_listfit_window(self):
        """Create the Listfit window if it doesn't exist"""
        try:
            # Listfit window initialization - may need special setup
            # For now, we'll skip auto-creation as it may have dependencies
            pass
        except Exception as e:
            print(f"Could not create Listfit window: {e}")

    def toggle_window(self, window_obj):
        """Toggle the visibility of a window"""
        if window_obj is None:
            return
        
        # Handle matplotlib canvas specially
        if hasattr(window_obj, 'figure'):
            # It's a matplotlib canvas
            if window_obj.isVisible():
                window_obj.hide()
            else:
                window_obj.show()
        elif hasattr(window_obj, 'isVisible'):
            # It's a QWidget or similar
            if window_obj.isVisible():
                window_obj.hide()
            else:
                window_obj.show()
        
        # Refresh the View menu after toggling to update checkmarks
        self.update_view_menu()

    def refresh_view_menu(self):
        """Refresh the View menu - called after windows are created or shown/hidden"""
        self.update_view_menu()

    def init_controlpanel(self):
        # Set main window title and geometry
        from qasap.ui_utils import get_qasap_icon
        self.setWindowTitle("QASAP - Control Panel")
        self.setWindowIcon(get_qasap_icon())
        self.setGeometry(100, 100, 440, 230)

        # Create redshift input field
        self.label_redshift = QLabel("Redshift:", self)
        self.label_redshift.move(20, 30)
        self.input_redshift = QLineEdit(self)
        self.input_redshift.move(105, 25)
        self.input_redshift.resize(100, 30)
        self.input_redshift.setText(str(self.redshift))  # Set initial redshift value

        # Create fine-tuning buttons for the redshift
        self.button_redshift_decrease_01 = QPushButton("↓ 0.1", self)
        self.button_redshift_decrease_01.move(210, 40)
        self.button_redshift_decrease_01.resize(65, 30)
        self.button_redshift_decrease_01.clicked.connect(lambda: self.adjust_redshift(-0.1))

        self.button_redshift_decrease_001 = QPushButton("↓ 0.01", self)
        self.button_redshift_decrease_001.move(275, 40)
        self.button_redshift_decrease_001.resize(65, 30)
        self.button_redshift_decrease_001.clicked.connect(lambda: self.adjust_redshift(-0.01))

        self.button_redshift_decrease_0001 = QPushButton("↓ 0.001", self)
        self.button_redshift_decrease_0001.move(345, 40)
        self.button_redshift_decrease_0001.resize(75, 30)
        self.button_redshift_decrease_0001.clicked.connect(lambda: self.adjust_redshift(-0.001))

        self.button_redshift_increase_01 = QPushButton("↑ 0.1", self)
        self.button_redshift_increase_01.move(210, 10)
        self.button_redshift_increase_01.resize(65, 30)
        self.button_redshift_increase_01.clicked.connect(lambda: self.adjust_redshift(0.1))

        self.button_redshift_increase_001 = QPushButton("↑ 0.01", self)
        self.button_redshift_increase_001.move(275, 10)
        self.button_redshift_increase_001.resize(65, 30)
        self.button_redshift_increase_001.clicked.connect(lambda: self.adjust_redshift(0.01))

        self.button_redshift_increase_0001 = QPushButton("↑ 0.001", self)
        self.button_redshift_increase_0001.move(345, 10)
        self.button_redshift_increase_0001.resize(75, 30)
        self.button_redshift_increase_0001.clicked.connect(lambda: self.adjust_redshift(0.001))

        # Create zoom factor input field
        self.label_zoom = QLabel("Zoom Factor:", self)
        self.label_zoom.move(20, 80)
        self.input_zoom = QLineEdit(self)
        self.input_zoom.move(105, 75)
        self.input_zoom.resize(100, 30)
        self.input_zoom.setText(str(self.zoom_factor))  # Set initial zoom factor
        zoom_validator = QDoubleValidator(0.001, 0.4, 3, self)  # (min, max, decimals)
        self.input_zoom.setValidator(zoom_validator)

        # Connect pressing "Enter" in the input field to apply changes
        self.input_redshift.returnPressed.connect(self.apply_changes)
        self.input_zoom.returnPressed.connect(self.apply_changes)

        # Create an "Apply" button
        self.apply_button = QPushButton("Enter", self)
        self.apply_button.move(20, 120)
        self.apply_button.resize(75, 30)
        self.apply_button.clicked.connect(self.apply_changes)
        
        # Create a "Load Spectrum..." button to load a new spectrum
        self.open_button = QPushButton("Load Spectrum...", self)
        self.open_button.move(120, 120)
        self.open_button.resize(143, 30)
        self.open_button.clicked.connect(self.open_spectrum_file)
        
        # Create a "Load Fit..." button to load saved fits
        self.load_fit_button = QPushButton("Load Fit...", self)
        self.load_fit_button.move(270, 120)
        self.load_fit_button.resize(130, 30)
        self.load_fit_button.clicked.connect(self.load_fit_file)
        
        # Create a separator line
        self.separator_line = QtWidgets.QFrame(self)
        self.separator_line.setGeometry(20, 155, 380, 2)
        self.separator_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        
        # Create a "Quit" button (positioned below Enter with spacing)
        self.quit_button = QPushButton("Quit", self)
        self.quit_button.move(20, 165)
        self.quit_button.clicked.connect(self.quit_application)
        
        # Create "Undo" button (to the right of Quit with spacing)
        self.undo_button = QPushButton("← Undo", self)
        self.undo_button.move(120, 165)
        self.undo_button.resize(75, 30)
        self.undo_button.clicked.connect(self.on_undo)
        
        # Create "Redo" button (to the right of Undo)
        self.redo_button = QPushButton("Redo →", self)
        self.redo_button.move(220, 165)
        self.redo_button.resize(75, 30)
        self.redo_button.clicked.connect(self.on_redo)
        
        # Expand window height to accommodate new layout
        self.setGeometry(100, 100, 550, 220)

        # Create a separator line between Quit and Poly Order (hidden by default)
        self.separator_line_poly = QtWidgets.QFrame(self)
        self.separator_line_poly.setGeometry(20, 200, 380, 2)
        self.separator_line_poly.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator_line_poly.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.separator_line_poly.hide()

        # Create polynomial order field (for continuum fitting, hidden by default)
        self.label_poly_order = QLabel("Poly Order:", self)
        self.label_poly_order.move(20, 210)
        self.label_poly_order.hide()
        self.input_poly_order = QLineEdit(self)
        self.input_poly_order.move(105, 205)
        self.input_poly_order.resize(100, 30)
        self.input_poly_order.setText("1")  # Default to first-order polynomial
        self.input_poly_order.hide()
        poly_validator = QIntValidator(0, 10, self)
        self.input_poly_order.setValidator(poly_validator)
        
        self.poly_order = 1  # Store the current polynomial order
        
        # Expand window height to accommodate poly order field when shown
        self.setGeometry(100, 100, 440, 250)

        # Show the window
        self.show()

    def adjust_redshift(self, delta):
        """Adjust redshift by a specified delta value and update the input field."""
        self.redshift += delta
        # Format redshift without trailing zeros
        redshift_str = f"{self.redshift:.6f}".rstrip('0').rstrip('.')
        self.input_redshift.setText(redshift_str)  # Update input field with new redshift
        
        # Redisplay active line lists with new redshift
        if self.active_line_lists:
            self.display_linelist()
        
        # Also handle legacy linelist_plots for backwards compatibility
        if self.linelist_plots:
            self.clear_linelist()
            self.display_linelist()
        
        self.fig.canvas.draw_idle()  # Redraw the figure to update the display

    def apply_changes(self):
        """Apply changes based on user input for redshift and zoom factor."""
        try:
            # Update redshift and zoom factor from input fields
            self.redshift = float(self.input_redshift.text())
            self.zoom_factor = float(self.input_zoom.text())
            
            # Format redshift without trailing zeros for display
            redshift_str = f"{self.redshift:.6f}".rstrip('0').rstrip('.')
            self.input_redshift.setText(redshift_str)
            
            # Update polynomial order if visible
            if not self.input_poly_order.isHidden():
                self.poly_order = int(self.input_poly_order.text())
                print(f"Polynomial order set to: {self.poly_order}")
            
            print(f"Applied Redshift: {self.redshift}, Zoom Factor: {self.zoom_factor}")
            
            # Redisplay active line lists with new redshift
            if self.active_line_lists:
                self.display_linelist()
            
            # Also handle legacy linelist_plots for backwards compatibility
            if self.linelist_plots:
                self.clear_linelist()
                self.display_linelist()
            
            self.fig.canvas.draw_idle()  # Redraw the figure to update the display
        except ValueError:
            print("Invalid input for redshift, zoom factor, or polynomial order. Please enter numerical values.")

    def open_spectrum_file(self):
        """Open a file dialog to select and load a new spectrum file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from qasap.format_picker_dialog import FormatPickerDialog
        
        dialog = QFileDialog(
            self,
            "Open Spectrum File",
            "",
            "All Files (*);;FITS Files (*.fits *.fit);;ASCII Files (*.txt *.dat)"
        )
        dialog.setFileMode(QFileDialog.ExistingFiles)
        
        if dialog.exec_() != QFileDialog.Accepted:
            return  # User cancelled
        
        files = dialog.selectedFiles()
        if not files:
            return
        
        file_path = files[0]
        
        try:
            # If this is not the first load and we have fits, ask if user wants to clear them
            if not self.is_first_load and (self.gaussian_fits or self.voigt_fits or self.continuum_fits or self.listfit_fits):
                reply = QMessageBox.question(
                    self,
                    "Clear Existing Fits?",
                    "You have existing fits from the previous spectrum. Do you want to clear them before loading the new spectrum?\n\n"
                    "Click 'Yes' to clear all fits and load cleanly.\n"
                    "Click 'No' to keep existing fits (may cause issues).",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    self.clear_all_fits()
            
            # Auto-detect format
            candidates = SpectrumIO.detect_spectrum_format(file_path)
            if not candidates:
                print("Error: Could not auto-detect format for the selected file")
                return
            
            # Show format picker dialog
            dialog = FormatPickerDialog(file_path, candidates, parent=self)
            result = dialog.exec_()
            
            if result != QtWidgets.QDialog.Accepted:
                return  # User cancelled
            
            selection = dialog.get_selection()
            if not selection:
                return
            
            fmt, options = selection
            
            # Load the spectrum
            wav, spec, err, meta = SpectrumIO.read_spectrum(file_path, fmt=fmt, options=options)
            
            # Apply scaling factor if provided
            if "scaling_factor" in options:
                scaling_factor = options["scaling_factor"]
                if scaling_factor != 1.0:
                    spec = spec * scaling_factor
                    if err is not None:
                        err = err * abs(scaling_factor)
                    print(f"Applied scaling factor: {scaling_factor}")
            
            # Load the data
            self.load_spectrum_data(wav, spec, err, meta, file_path)
            
            # Redraw the plot
            self.clear_plot_and_reset()
            self.plot_spectrum()
            
            print(f"Loaded: {file_path}")
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            import traceback
            traceback.print_exc()

    def load_fit_file(self):
        """Load previously saved fits from consolidated CSV file"""
        from PyQt5.QtWidgets import QFileDialog
        import pandas as pd
        
        dialog = QFileDialog(
            self,
            "Load Fit File",
            "",
            "QASAP Fit Files (qasap_fits_*.csv gaussian_fits_*.csv voigt_fits_*.csv continuum_fits_*.csv listfit_polynomials_*.csv);;CSV Files (*.csv);;All Files (*)"
        )
        dialog.setFileMode(QFileDialog.ExistingFile)
        
        if dialog.exec_() != QFileDialog.Accepted:
            return
        
        file_path = dialog.selectedFiles()[0]
        
        try:
            basename = os.path.basename(file_path)
            df = pd.read_csv(file_path)
            
            # Check if this is a consolidated file (has 'type' column)
            if 'type' in df.columns:
                # New consolidated format
                self._load_consolidated_fits_from_dataframe(df)
                print(f"Loaded {len(df)} fits from consolidated file {file_path}")
            # Legacy support: auto-detect from old format filenames
            elif 'gaussian_fits' in basename:
                self._load_gaussian_fits_from_dataframe(df)
                print(f"Loaded {len(df)} Gaussian fits from {file_path}")
            elif 'voigt_fits' in basename:
                self._load_voigt_fits_from_dataframe(df)
                print(f"Loaded {len(df)} Voigt fits from {file_path}")
            elif 'continuum_fits' in basename:
                self._load_continuum_fits_from_dataframe(df)
                print(f"Loaded {len(df)} continuum fits from {file_path}")
            elif 'listfit_polynomials' in basename:
                self._load_listfit_polynomials_from_dataframe(df)
                print(f"Loaded listfit polynomials from {file_path}")
            else:
                # Try to detect by column names
                self._load_by_column_detection(df)
            
            # Update plot and redraw all loaded fits
            self.plot_spectrum()
            self._redraw_loaded_fits()
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error loading fit file: {e}")
            import traceback
            traceback.print_exc()
    
    def _redraw_loaded_fits(self):
        """Redraw all loaded fit lines on the plot and register with tracker"""
        # Redraw Gaussian fits
        for idx, fit in enumerate(self.gaussian_fits):
            if fit.get('line') is None:  # Only if line hasn't been created yet
                try:
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                    _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
                    if a is not None and b is not None:
                        existing_continuum = self.continuum_model(x_plot, a, b)
                    else:
                        existing_continuum = np.zeros_like(x_plot)
                    y_plot = self.gaussian(x_plot, fit['amp'], fit['mean'], fit['stddev']) + existing_continuum
                    fit['line'], = self.ax.plot(x_plot, y_plot, color="red", linestyle='--')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Gaussian (μ={fit['mean']:.1f}, σ={fit['stddev']:.1f})"
                    self.register_item('gaussian', name, fit_dict=fit, line_obj=fit['line'], 
                                     color='red', bounds=fit['bounds'])
                except Exception as e:
                    print(f"Error redrawing Gaussian fit: {e}")
        
        # Redraw Voigt fits
        for idx, fit in enumerate(self.voigt_fits):
            if fit.get('line') is None:
                try:
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                    _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
                    if a is not None and b is not None:
                        existing_continuum = self.continuum_model(x_plot, a, b)
                    else:
                        existing_continuum = np.zeros_like(x_plot)
                    y_plot = self.voigt(x_plot, fit['amp'], fit['center'], fit['sigma'], fit['gamma']) + existing_continuum
                    fit['line'], = self.ax.plot(x_plot, y_plot, color="orange", linestyle='--')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Voigt (c={fit['center']:.1f}, σ={fit['sigma']:.1f}, γ={fit['gamma']:.1f})"
                    self.register_item('voigt', name, fit_dict=fit, line_obj=fit['line'],
                                     color='orange', bounds=fit['bounds'])
                except Exception as e:
                    print(f"Error redrawing Voigt fit: {e}")
        
        # Redraw Continuum fits
        for idx, fit in enumerate(self.continuum_fits):
            if fit.get('line') is None:
                try:
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                    y_plot = fit['a'] * x_plot + fit['b']
                    fit['line'], = self.ax.plot(x_plot, y_plot, color="magenta", linestyle='--')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Continuum (a={fit['a']:.2e}, b={fit['b']:.2f})"
                    self.register_item('continuum', name, fit_dict=fit, line_obj=fit['line'],
                                     color='magenta', bounds=fit['bounds'])
                except Exception as e:
                    print(f"Error redrawing continuum fit: {e}")
        
        # Redraw Listfit composites
        for idx, listfit in enumerate(self.listfit_fits):
            try:
                bounds = listfit.get('bounds', (None, None))
                # Handle invalid bounds
                if bounds[0] is None or bounds[1] is None or np.isnan(bounds[0]) or np.isnan(bounds[1]):
                    bounds = (self.wav.min(), self.wav.max())
                
                param_values = listfit.get('param_values', {})
                
                # If we have parameters, rebuild and evaluate the model
                if param_values:
                    x_plot = np.linspace(bounds[0], bounds[1], 200)
                    
                    # Rebuild the composite model from components
                    components = listfit.get('components', [])
                    composite = None
                    param_counter = {}  # Track parameter indices for naming
                    
                    for comp in components:
                        comp_type = comp.get('type', 'polynomial')
                        
                        if comp_type == 'polynomial':
                            order = comp.get('order', 1)
                            # Create polynomial model
                            from lmfit.models import PolynomialModel
                            poly_model = PolynomialModel(order)
                            
                            # Set parameters to saved values - try different naming conventions
                            params_set = False
                            for i in range(order + 1):
                                # Try different parameter name formats
                                possible_names = [
                                    f'poly_c{i}',
                                    f'p0_c{i}',
                                    f'c{i}',
                                    f'p_c{i}'
                                ]
                                for name in possible_names:
                                    if name in param_values:
                                        poly_model.set_param_hint(f'c{i}', value=param_values[name])
                                        params_set = True
                                        break
                            
                            if composite is None:
                                composite = poly_model
                            else:
                                composite = composite + poly_model
                        
                        elif comp_type == 'gaussian':
                            from lmfit.models import GaussianModel
                            g_idx = param_counter.get('gaussian', 0)
                            param_counter['gaussian'] = g_idx + 1
                            
                            prefix = f'g{g_idx}_'
                            g_model = GaussianModel(prefix=prefix)
                            
                            # Set Gaussian parameters from saved values
                            if f'{prefix}amplitude' in param_values:
                                g_model.set_param_hint(f'{prefix}amplitude', value=param_values[f'{prefix}amplitude'])
                            if f'{prefix}center' in param_values:
                                g_model.set_param_hint(f'{prefix}center', value=param_values[f'{prefix}center'])
                            if f'{prefix}sigma' in param_values:
                                g_model.set_param_hint(f'{prefix}sigma', value=param_values[f'{prefix}sigma'])
                            
                            if composite is None:
                                composite = g_model
                            else:
                                composite = composite + g_model
                        
                        elif comp_type == 'voigt':
                            from lmfit.models import VoigtModel
                            v_idx = param_counter.get('voigt', 0)
                            param_counter['voigt'] = v_idx + 1
                            
                            prefix = f'v{v_idx}_'
                            v_model = VoigtModel(prefix=prefix)
                            
                            # Set Voigt parameters
                            if f'{prefix}amplitude' in param_values:
                                v_model.set_param_hint(f'{prefix}amplitude', value=param_values[f'{prefix}amplitude'])
                            if f'{prefix}center' in param_values:
                                v_model.set_param_hint(f'{prefix}center', value=param_values[f'{prefix}center'])
                            if f'{prefix}sigma' in param_values:
                                v_model.set_param_hint(f'{prefix}sigma', value=param_values[f'{prefix}sigma'])
                            if f'{prefix}gamma' in param_values:
                                v_model.set_param_hint(f'{prefix}gamma', value=param_values[f'{prefix}gamma'])
                            
                            if composite is None:
                                composite = v_model
                            else:
                                composite = composite + v_model
                    
                    # Evaluate the reconstructed model
                    if composite is not None:
                        # Create parameters dict with all saved values
                        eval_params = composite.make_params()
                        for param_name, param_value in param_values.items():
                            # Try to match parameter names
                            if param_name in eval_params:
                                eval_params[param_name].value = param_value
                        
                        y_plot = composite.eval(eval_params, x=x_plot)
                        # Use the standard listfit color: #003d7a (dark blue)
                        listfit_line, = self.ax.plot(x_plot, y_plot, label='Total Listfit', color='#003d7a', linestyle='-', linewidth=2)
                        listfit['line'] = listfit_line
                        
                        # Register with item tracker - use saved name if available
                        n_components = len(listfit.get('components', []))
                        chi2 = listfit.get('quality_metrics', {}).get('chisqr', 0)
                        name = listfit.get('_tracker_name') or f"Total Listfit ({n_components} components, χ²={chi2:.2f})"
                        self.register_item('listfit_total', name, fit_dict=listfit, line_obj=listfit_line,
                                         color='#003d7a', bounds=bounds)
            except Exception as e:
                print(f"Error redrawing listfit composite: {e}")
                import traceback
                traceback.print_exc()
    
    def _load_consolidated_fits_from_dataframe(self, df):
        """Load fits from consolidated DataFrame with 'type' column"""
        gaussian_count = 0
        voigt_count = 0
        continuum_count = 0
        listfit_count = 0
        
        for idx, row in df.iterrows():
            fit_type = row.get('type')
            fit_dict = row.to_dict()
            # Extract and remove tracker_name before cleaning NaN
            tracker_name = fit_dict.pop('tracker_name', None) if 'tracker_name' in fit_dict else None
            # Remove NaN and type column
            fit_dict = {k: v for k, v in fit_dict.items() if pd.notna(v) and k != 'type'}
            
            # Reconstruct bounds tuple from min/max if present
            if 'bounds_min' in fit_dict and 'bounds_max' in fit_dict:
                fit_dict['bounds'] = (fit_dict.pop('bounds_min'), fit_dict.pop('bounds_max'))
            
            if fit_type == 'gaussian':
                self.gaussian_fits.append(fit_dict)
                gaussian_count += 1
            elif fit_type == 'voigt':
                self.voigt_fits.append(fit_dict)
                voigt_count += 1
            elif fit_type == 'continuum':
                self.continuum_fits.append(fit_dict)
                continuum_count += 1
            elif fit_type == 'listfit':
                self._load_single_listfit_entry(fit_dict)
                listfit_count += 1
            
            # Store tracker name in fit_dict for use during redraw
            fit_dict['_tracker_name'] = tracker_name
        
        print(f"  - Gaussian: {gaussian_count}")
        print(f"  - Voigt: {voigt_count}")
        print(f"  - Continuum: {continuum_count}")
        print(f"  - Listfit: {listfit_count}")
    
    def _load_single_listfit_entry(self, fit_dict):
        """Load a single listfit entry from consolidated format"""
        from lmfit import CompositeModel
        
        # Reconstruct listfit structure
        listfit_entry = {
            'bounds': (fit_dict.get('bounds_min'), fit_dict.get('bounds_max')),
            'components': self._parse_string_repr(fit_dict.get('components', '[]')),
            'initial_guesses': self._parse_string_repr(fit_dict.get('initial_guesses', '{}')),
            'constraints': self._parse_string_repr(fit_dict.get('constraints', '{}')),
        }
        
        # Store parameter values (for later evaluation)
        param_values = {k.replace('param_', ''): v for k, v in fit_dict.items() if k.startswith('param_')}
        listfit_entry['param_values'] = param_values
        
        # Store quality metrics
        listfit_entry['quality_metrics'] = {
            'chisqr': fit_dict.get('chi_squared'),
            'redchi': fit_dict.get('redchi'),
            'aic': fit_dict.get('aic'),
            'bic': fit_dict.get('bic'),
        }
        
        self.listfit_fits.append(listfit_entry)
    
    def _parse_string_repr(self, s):
        """Safely parse string representation of list/dict"""
        if not s or s == 'nan' or s != s:  # Check for NaN
            return [] if s == '[]' else {}
        try:
            import ast
            return ast.literal_eval(str(s))
        except (ValueError, SyntaxError):
            return [] if isinstance(s, str) and s.startswith('[') else {}
    
    def _load_by_column_detection(self, df):
        """Detect fit type by analyzing DataFrame columns (legacy support)"""
        # Check what columns are present to infer fit type
        if {'amp', 'mean', 'stddev'}.issubset(df.columns):
            self._load_gaussian_fits_from_dataframe(df)
            print(f"Auto-detected Gaussian fits: {len(df)} records")
        elif {'a', 'b'}.issubset(df.columns):
            self._load_continuum_fits_from_dataframe(df)
            print(f"Auto-detected continuum fits: {len(df)} records")
        elif {'sigma', 'gamma'}.issubset(df.columns):
            self._load_voigt_fits_from_dataframe(df)
            print(f"Auto-detected Voigt fits: {len(df)} records")
        else:
            print(f"Could not auto-detect fit type. Columns: {list(df.columns)}")

    
    def _load_gaussian_fits_from_dataframe(self, df):
        """Load Gaussian fits from DataFrame"""
        for idx, row in df.iterrows():
            fit_dict = row.to_dict()
            # Handle NaN values
            fit_dict = {k: v for k, v in fit_dict.items() if pd.notna(v)}
            self.gaussian_fits.append(fit_dict)
    
    def _load_voigt_fits_from_dataframe(self, df):
        """Load Voigt fits from DataFrame"""
        for idx, row in df.iterrows():
            fit_dict = row.to_dict()
            fit_dict = {k: v for k, v in fit_dict.items() if pd.notna(v)}
            self.voigt_fits.append(fit_dict)
    
    def _load_continuum_fits_from_dataframe(self, df):
        """Load continuum fits from DataFrame"""
        for idx, row in df.iterrows():
            fit_dict = row.to_dict()
            fit_dict = {k: v for k, v in fit_dict.items() if pd.notna(v)}
            self.continuum_fits.append(fit_dict)
    
    def _load_listfit_polynomials_from_dataframe(self, df):
        """Load listfit polynomials from DataFrame (simplified - stores as metadata)"""
        print("[INFO] Listfit polynomial loading from CSV is limited. Consider saving full listfit results for complete reload.")
        # For now, just load the polynomial data
        for idx, row in df.iterrows():
            poly_dict = row.to_dict()
            poly_dict = {k: v for k, v in poly_dict.items() if pd.notna(v)}
            print(f"  Polynomial: bounds=[{poly_dict.get('bounds_min')}, {poly_dict.get('bounds_max')}], order={poly_dict.get('polynomial_order')}")

    def load_spectrum_data(self, wav, spec, err, meta, fits_file):
        """Load spectrum data into the plotter."""
        # If in velocity mode, exit it first before loading new spectrum
        if self.is_velocity_mode:
            print("Exiting velocity mode to load new spectrum...")
            self.exit_velocity_mode()
        
        # Record initial load action only on first load
        if self.is_first_load:
            if self.initial_spectrum_file:
                import os
                filename = os.path.basename(self.initial_spectrum_file)
                self.record_action('load_spectrum', f'Load Spectrum: {filename}')
            else:
                self.record_action('open_qasap', 'Open qasap')
            self.is_first_load = False
        else:
            # Record subsequent spectrum loads from GUI
            import os
            filename = os.path.basename(fits_file)
            self.record_action('load_spectrum', f'Load Spectrum: {filename}')
        
        self.wav = wav
        self.spec = spec
        self.err = err
        self.fits_file = fits_file
        self.data_loaded_from_gui = True  # Mark that data was loaded from GUI
        
        # Extract and store wavelength unit from metadata
        if meta and 'wave_unit' in meta:
            self.wavelength_unit = meta['wave_unit']
        else:
            self.wavelength_unit = "Å"  # Default to Angstroms
        
        # Reset smoothing and other processing
        self.smoothing_kernel = None
        self.smoothed_spectrum = None
        
        print(f"Loaded {len(wav)} wavelength points")
        print(f"Wavelength: {wav[0]:.2f} - {wav[-1]:.2f} {self.wavelength_unit}")
        print(f"Flux range: {np.min(spec):.2e} - {np.max(spec):.2e}")

    def clear_all_fits(self):
        """Clear all fits (Gaussian, Voigt, continuum, listfit) and remove them from plot."""
        # Remove all fit lines and continuum patches from the plot
        for fit in self.gaussian_fits:
            if 'line' in fit and fit['line']:
                try:
                    fit['line'].remove()
                except (ValueError, AttributeError):
                    pass
        
        for fit in self.voigt_fits:
            if 'line' in fit and fit['line']:
                try:
                    fit['line'].remove()
                except (ValueError, AttributeError):
                    pass
        
        for fit in self.listfit_fits:
            if 'line' in fit and fit['line']:
                try:
                    fit['line'].remove()
                except (ValueError, AttributeError):
                    pass
        
        # Remove continuum patches
        for patch in self.continuum_patches:
            if 'patch_obj' in patch and patch['patch_obj']:
                try:
                    patch['patch_obj'].remove()
                except (ValueError, AttributeError):
                    pass
        
        # Clear all fit lists
        self.gaussian_fits.clear()
        self.voigt_fits.clear()
        self.continuum_fits.clear()
        self.listfit_fits.clear()
        self.continuum_patches.clear()
        self.deleted_listfit_polynomials.clear()
        
        # Clear item tracker
        self.item_tracker.clear_all()
        self.item_id_map.clear()
        self.highlighted_item_ids.clear()
        self.fit_information_window.clear_all()
        
        # Reset plotting state
        self.gaussian_fit_display = None
        self.voigt_fit_display = None
        self.residuals = None
        self.is_residual_shown = False
        if self.residual_ax:
            self.residual_ax.clear()
            self.residual_ax.set_visible(False)
        
        plt.draw()

    def _convert_wavelength_from_angstrom(self, wavelength_angstrom):
        """Convert wavelength from Angstroms to current display unit"""
        if self.wavelength_unit == "nm":
            return wavelength_angstrom / 10.0
        elif self.wavelength_unit == "µm" or self.wavelength_unit == "um":
            return wavelength_angstrom / 1e4
        else:  # Angstrom (default)
            return wavelength_angstrom
    
    def _convert_wavelength_to_angstrom(self, wavelength_display):
        """Convert wavelength from current display unit to Angstroms"""
        if self.wavelength_unit == "nm":
            return wavelength_display * 10.0
        elif self.wavelength_unit == "µm" or self.wavelength_unit == "um":
            return wavelength_display * 1e4
        else:  # Angstrom (default)
            return wavelength_display
    
    def _get_wavelength_unit_label(self):
        """Get the x-axis label with current wavelength unit"""
        return f"Wavelength ({self.wavelength_unit})"

    def quit_application(self):
        """Quit the QASAP application"""
        QtWidgets.QApplication.quit()

    def clear_plot_and_reset(self):
        """Clear the current plot and reset all fitting data and item tracker."""
        # Clear the axis if it exists
        if self.ax is not None:
            self.ax.clear()
        
        # Reset all fitting data
        self.fitted_gaussians = []
        self.fitted_voigts = []
        self.fitted_continuum_coeffs = None
        self.continuum_points_x = []
        self.continuum_points_y = []
        self.item_tracker.clear_all()
        self.fit_information_window.clear_all()
        
        # Reset item selection tracking
        self.highlighted_item_ids.clear()
        
        # Reset other states
        self.smoothing_kernel = None
        self.smoothed_spectrum = None
        self.residuals = None
        self.show_residuals = False
        self.is_step_plot = False
        self.redshift = 0.0
        self.input_redshift.setText(str(self.redshift))

    def plot_spectrum(self):
        # Read lines and instrument bands
        self.line_wavelengths, self.line_ids = self.read_lines()
        self.osc_wavelengths, self.osc_ids, self.osc_strengths = self.read_osc()
        self.read_instrument_bands()

        # Find title from file name
        if self.fits_file:
            title = Path(self.fits_file).name
        else:
            title = "QASAP - Load a Spectrum to Begin"

        # File handling based on flag - only read from file if not already loaded via GUI
        if self.fits_file and not self.data_loaded_from_gui:
            if self.file_flag == 1:
                data = np.genfromtxt(self.fits_file, comments='#', delimiter='\t')
                self.wav, self.spec, self.err = data[:, 0], data[:, 1], data[:, 2]
            elif self.file_flag == 2:
                with fits.open(self.fits_file) as hdul:
                    self.spec = hdul[0].data.flatten()
                    header = hdul[0].header
                    crpix1, crval1, cdelt1 = header.get('CRPIX1'), header.get('CRVAL1'), header.get('CDELT1')
                    self.wav = crval1 + (np.arange(len(self.spec)) - (crpix1 - 1)) * cdelt1
                    self.err = None  # No error spectrum
            elif self.file_flag == 3:
                print("Reading file flag 3") # [DEBUG]
                print("self.fits_file:", self.fits_file) # [DEBUG]
                data = np.loadtxt(self.fits_file)
                print(data)
                self.wav, self.spec, self.err = data[:, 0], data[:, 1], data[:, 2] if data.shape[1] > 2 else None
            elif self.file_flag == 4:
                data = np.loadtxt(self.fits_file)
                self.wav, self.spec, self.err = data[:, 0], data[:, 2], data[:, 3]
            elif self.file_flag == 5:
                with fits.open(self.fits_file) as hdul:
                    data = hdul[1].data
                    self.wav = data['wave']
                    self.spec = data['flux']
                    self.err = None
            elif self.file_flag == 6:
                with fits.open(self.fits_file) as hdul:
                    data = hdul[1].data
                    self.wav = data[0][0]
                    self.spec = data[0][1]
                    self.err = None
            elif self.file_flag == 7:
                with fits.open(self.fits_file) as hdul:
                    hdul.info()
                    data = hdul['SPECTRUM'].data
                    self.wav = np.nan_to_num(data['wave'], nan=0.0)
                    wav_mid = np.nan_to_num(data['wave_grid_mid'], nan=0.0)
                    self.spec = np.nan_to_num(data['flux'], nan=0.0)
                    ivar = np.nan_to_num(data['ivar'], nan=0.0)
                    mask = np.nan_to_num(data['mask'], nan=0)
                    self.err = np.sqrt(np.where(ivar > 0, 1 / ivar, 0))
            elif self.file_flag == 8:
                with fits.open(self.fits_file) as hdul:
                    data = hdul['SPECTRUM'].data
                    self.wav = data['WAVE'][0]
                    self.spec = data['FLUX'][0]
                    self.err  = data['ERR'][0]
            elif self.file_flag == 9:
                with fits.open(self.fits_file) as hdul:
                    data = hdul[1].data
                    self.wav = data['WAVE'][0]
                    self.spec = data['FLUX'][0]
                    self.err  = data['ERR'][0]
            elif self.file_flag == 10:
                # Reading .sed or .txt format: 2-column ASCII with comment lines in nm
                wavelengths = []
                fluxes = []
                with open(self.fits_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            wavelengths.append(float(parts[0]))  # nm
                            fluxes.append(float(parts[1]))       # arbitrary units

                self.wav = np.array(wavelengths)
                self.spec = np.array(fluxes)
                self.err = None  # No error spectrum
            else:
                # Default FITS format
                try:
                    with fits.open(self.fits_file) as hdul:
                        self.wav, self.spec, self.err = hdul[0].data, hdul[1].data, hdul[2].data
                except (OSError, IndexError):
                    # If read fails, data should have been loaded already
                    pass
        elif not self.fits_file:
            # No file specified - initialize empty spectrum
            self.wav = np.array([])
            self.spec = np.array([])
            self.err = None

        # print("wav:",self.wav)
        # print("spec:",self.spec)

        # Define initial plot limits
        if len(self.wav) > 0 and len(self.spec) > 0:
            xlim = (self.wav.min(), self.wav.max())
            # Handle NaN values in spectrum
            valid_spec = self.spec[~np.isnan(self.spec)]
            if len(valid_spec) > 0:
                ylim = (valid_spec.min(), valid_spec.max())
            else:
                # All spectrum values are NaN - use default range
                ylim = (-1, 1)
        else:
            xlim = (0, 1)
            ylim = (0, 1)
        redshift = self.redshift
        zoom_factor = self.zoom_factor

        # Store original bounds at the beginning
        self.original_xlim = xlim
        self.original_ylim = ylim

        # Store original spectrum
        self.original_spec = self.spec

        # Store regions and axis bounds
        self.continuum_regions = []
        self.line_region = []
        self.x_upper_bound = xlim[1]
        self.x_lower_bound = xlim[0]
        self.y_upper_bound = ylim[1]
        self.y_lower_bound = ylim[0]

        # To track axvspan objects
        self.continuum_patches = []
        self.line_patches = []

        # Set up x_data variable to store the currently plotted data (wavelength vs. velocity)
        self.x_data = self.wav # Default is wavelength

        # Set up plot - reuse existing figure if available, otherwise create new
        is_first_plot = not hasattr(self, 'fig') or self.fig is None
        if is_first_plot:
            # Create new figure for first time
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            # Adjust plot
            self.fig.subplots_adjust(bottom=0.35)
            
            # Create output panel for first plot only
            self.output_panel = OutputPanel()
        else:
            # Reuse existing figure - clear axes
            self.ax.clear()
        
        # Plot the spectrum
        self.step_spec, = self.ax.step(self.x_data, self.spec, label='Data', color='black', where='mid')
        self.line_spec, = self.ax.plot(self.x_data, self.spec, color='black', visible=False)
        
        # Only plot error if errors exist
        if self.err is not None:
            self.step_error, = self.ax.step(self.x_data, self.err, color='red', linestyle='--', alpha=0.4, label='Error', where='mid')
            self.line_error, = self.ax.plot(self.x_data, self.err, color='red', linestyle='--', alpha=0.4, visible=False)
            self.error_line = self.step_error if self.is_step_plot else self.line_error
        else:
            self.step_error = None
            self.line_error = None
            self.error_line = None
        
        self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
        self.ax.plot(self.x_data, [0] * len(self.x_data), color='gray', linestyle='--', linewidth=1) # Add horizontal line at y=0
        self.ax.set_xlabel(self._get_wavelength_unit_label())
        self.ax.set_ylabel(r'Flux (arbitrary units)') # Use arbitrary units instead
        # self.ax.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)')
        self.ax.set_title(title)
        # Set initial plot limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Determine the range of xlim
        self.x_range = xlim[1] - xlim[0]

        # Update the x-ticks
        self.update_ticks(self.ax)

        # Set the custom window title
        self.fig.canvas.manager.set_window_title("QASAP - Quick Analysis of Spectra and Profiles (v0.11)")

        self.ax.legend(loc='upper right')

        # Connect the key press and mouse move event
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        # Connect x-bounds update to the main plot's x-axis
        self.ax.callbacks.connect('xlim_changed', self.update_residual_xbounds)
        # self.fig.canvas.setFocus() # Removed this because it was not needed
        # self.fig.canvas.draw() # Removed this because it was not needed

        # Show the Item Tracker window (in background)
        self.item_tracker.item_deleted.connect(self.on_item_deleted_from_tracker)
        self.item_tracker.item_selected.connect(self.on_item_selected_from_tracker)
        self.item_tracker.item_individually_deselected.connect(self.on_item_individually_deselected_from_tracker)
        self.item_tracker.item_deselected.connect(self.on_item_deselected_from_tracker)
        self.item_tracker.estimate_redshift.connect(self.on_estimate_redshift_from_tracker)
        self.item_tracker.setGeometry(1100, 700, 650, 350)  # Position on the right side
        self.item_tracker.show()

        # Connect Fit Information window signals
        self.fit_information_window.item_selected.connect(self.on_fit_info_item_selected)
        self.fit_information_window.item_deselected.connect(self.on_fit_info_item_deselected)
        self.fit_information_window.setGeometry(100, 500, 1200, 400)  # Position below main window

        # Set icon for the matplotlib figure window
        if hasattr(self.fig, 'canvas') and hasattr(self.fig.canvas, 'manager'):
            if hasattr(self.fig.canvas.manager, 'window'):
                logo_path = Path(__file__).parent.parent / 'logo' / 'qasap_logo.png'
                if logo_path.exists():
                    self.fig.canvas.manager.window.setWindowIcon(QIcon(str(logo_path)))
                self.fig.canvas.manager.window.setWindowTitle("QASAP - Spectrum Viewer")

        # Update the View menu now that the spectrum plotter figure has been created
        self.update_view_menu()

        # Only show the figure window on first plot creation
        if is_first_plot:
            # Show the plot without blocking
            plt.show(block=False)
            
            if hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'window'):
                mpl_window = self.fig.canvas.manager.window
                mpl_window.setWindowTitle("QASAP - Spectrum Viewer")
                
                # Integrate output panel with matplotlib window
                import time
                time.sleep(0.1)  # Give Qt time to fully create the window
                
                # Get the current central widget (matplotlib canvas)
                original_canvas = mpl_window.centralWidget()
                
                # Create wrapper widget with vertical layout
                wrapper_widget = QtWidgets.QWidget()
                wrapper_layout = QVBoxLayout()
                wrapper_layout.setContentsMargins(0, 0, 0, 0)
                wrapper_layout.setSpacing(0)
                
                # Add matplotlib canvas to top (with more space)
                wrapper_layout.addWidget(original_canvas, stretch=1)
                
                # Add output panel to bottom
                wrapper_layout.addWidget(self.output_panel, stretch=0)
                
                # Set wrapper as central widget
                wrapper_widget.setLayout(wrapper_layout)
                mpl_window.setCentralWidget(wrapper_widget)

        self.fig.canvas.setFocus() # Removed this because it was not needed
        self.fig.canvas.draw() # Removed this because it was not needed

    def read_lines(self):
        # Read spectral lines from file
        line_file = str(Path(self.resources_dir) / 'linelist' / 'emlines.txt')
        line_ids = []
        line_wavelengths = []
        with open(line_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    line_ids.append(parts[1].strip())
                    line_wavelengths.append(float(parts[0].strip()))
        line_wavelengths = np.array(line_wavelengths)
        return line_wavelengths, line_ids
    
    def get_all_available_line_lists(self):
        """Get all available line lists from the resources directory."""
        return get_available_line_lists(self.resources_dir)
    
    def read_osc(self):
        # Read spectral lines from file
        line_file = str(Path(self.resources_dir) / 'linelist' / 'emlines_osc.txt')
        line_ids = []
        line_wavelengths = []
        line_osc = []
        with open(line_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    line_ids.append(parts[1].strip())
                    line_wavelengths.append(float(parts[0].strip()))
                    line_osc.append(float(parts[2].strip()))
        line_wavelengths = np.array(line_wavelengths)
        line_osc = np.array(line_osc)
        return line_wavelengths, line_ids, line_osc

    def read_instrument_bands(self):
        # Read instrument bands from file
        self.band_ranges = []  # Reset to empty list
        bands_file = str(Path(self.resources_dir) / 'bands' / 'instrument_bands.txt')
        with open(bands_file, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    self.band_ranges.append((parts[0], float(parts[1]), float(parts[2])))
        self.band_ranges = np.array(self.band_ranges)

    def process_lsf(self, lsf):
        try:
            # Check if `lsf` is a float (interpreted as kernel width in km/s)
            lsf_width = float(lsf) # FWHM
            # Construct a Gaussian LSF kernel with the specified FWHM
            sigma = lsf_width / (2*np.sqrt(2*np.log(2)))  # Convert FWHM to standard deviation (approximation)
            kernel_size = int(8 * sigma)  # Define kernel size based on sigma
            self.lsf_kernel_x = np.linspace(-4 * sigma, 4 * sigma, kernel_size)
            self.lsf_kernel_y = np.exp(-0.5 * (self.lsf_kernel_x / sigma)**2)
            self.lsf_kernel_y /= np.sum(self.lsf_kernel_y)  # Normalize to sum to 1
            print(f"Using Gaussian LSF with FWHM {lsf_width} km/s.")
        
        except ValueError:
            # Otherwise, assume `lsf` is a file path
            try:
                data = np.loadtxt(lsf)  # Load the LSF file
                self.lsf_kernel_x = data[:, 0] # Velocity units
                self.lsf_kernel_y = data[:, 1] / np.sum(data[:, 1])  # Flux normalized to 1
                print(f"Using custom LSF from file: {lsf}")
            except Exception as e:
                print(f"Error loading LSF from file {lsf}: {e}")
                sys.exit(1)

    def apply_lsf(self, profile):
        """Convolves the input y_data with the LSF kernel."""
        if self.lsf_kernel_x is not None and self.lsf_kernel_y is not None and self.is_velocity_mode:
            # Interpolate LSF kernel to match the profile shape
            lsf_interp = interp1d(self.lsf_kernel_x, self.lsf_kernel_y, bounds_error=False, fill_value=0)
            # Create x-values that match the profile's x-values
            # profile_x = np.linspace(np.min(self.x_data), np.max(self.x_data), len(profile)) # DON'T USE THIS LINE!!!
            new_lsf_kernel_x = np.linspace(np.min(self.lsf_kernel_x), np.max(self.lsf_kernel_x), len(profile)) # Create x-array with same size as the fitted profile
            lsf_interp_y = lsf_interp(new_lsf_kernel_x) # Interpolated LSF values
            lsf_interp_y /= np.sum(lsf_interp_y) # Normalize
            # print("new_lsf_kernel_x:",new_lsf_kernel_x) # [DEBUG]
            # print("lsf_interp_y:",lsf_interp_y)
            # print("Plotting convolution kernel")
            # self.ax.step(self.lsf_kernel_x, self.lsf_kernel_y, color='purple', linestyle=':')
            # Convolve the profile with the interpolated LSF
            convolved_profile = np.convolve(profile, lsf_interp_y, mode='same')  # Use 'same' to maintain the size
            # print("Plotting convolved profile")
            # self.ax.step(new_lsf_kernel_x, convolved_profile, color='blue', linestyle=':')
            return convolved_profile
        else:
            return profile  # Return original if no LSF kernel

    # Update plot to reflect new axis bounds
    def update_bounds(self):
        self.ax.set_xlim(self.x_lower_bound, self.x_upper_bound)
        if self.is_residual_shown:
            self.residual_ax.set_xlim(self.x_lower_bound, self.x_upper_bound)
        self.ax.set_ylim(self.y_lower_bound, self.y_upper_bound)
        self.fig.canvas.draw_idle()

    def update_ticks(self, ax):
        # Set initial plot limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Determine the range of xlim
        x_range = xlim[1] - xlim[0]
        # Conditional tick settings based on the range of xlim
        if x_range >= 10000:
            major_ticks = np.arange(np.floor(xlim[0] / 10000) * 10000, np.ceil(xlim[1] / 10000) * 10000 + 1, 10000)
            major_ticks = major_ticks[(major_ticks >= xlim[0]) & (major_ticks <= xlim[1])]  # Filter to stay within xlim
            minor_ticks = np.arange(np.floor(xlim[0] / 1000) * 1000, np.ceil(xlim[1] / 1000) * 1000 + 1, 1000)
            minor_ticks = minor_ticks[(minor_ticks >= xlim[0]) & (minor_ticks <= xlim[1])]  # Filter to stay within xlim
        elif 1000 <= x_range < 10000:
            major_ticks = np.arange(np.floor(xlim[0] / 1000) * 1000, np.ceil(xlim[1] / 1000) * 1000 + 1, 1000)
            major_ticks = major_ticks[(major_ticks >= xlim[0]) & (major_ticks <= xlim[1])]  # Filter to stay within xlim
            minor_ticks = np.arange(np.floor(xlim[0] / 100) * 100, np.ceil(xlim[1] / 100) * 100 + 1, 100)
            minor_ticks = minor_ticks[(minor_ticks >= xlim[0]) & (minor_ticks <= xlim[1])]  # Filter to stay within xlim
        elif 100 <= x_range < 1000:
            major_ticks = np.arange(np.floor(xlim[0] / 100) * 100, np.ceil(xlim[1] / 100) * 100 + 1, 100)
            major_ticks = major_ticks[(major_ticks >= xlim[0]) & (major_ticks <= xlim[1])]  # Filter to stay within xlim
            minor_ticks = np.arange(np.floor(xlim[0] / 10) * 10, np.ceil(xlim[1] / 10) * 10 + 1, 10)
            minor_ticks = minor_ticks[(minor_ticks >= xlim[0]) & (minor_ticks <= xlim[1])]  # Filter to stay within xlim
        elif 10 <= x_range < 100:
            major_ticks = np.arange(np.floor(xlim[0] / 10) * 10, np.ceil(xlim[1] / 10) * 10 + 1, 10)
            major_ticks = major_ticks[(major_ticks >= xlim[0]) & (major_ticks <= xlim[1])]  # Filter to stay within xlim
            minor_ticks = np.arange(np.floor(xlim[0] / 1) * 1, np.ceil(xlim[1] / 1) * 1 + 1, 1)
            minor_ticks = minor_ticks[(minor_ticks >= xlim[0]) & (minor_ticks <= xlim[1])]  # Filter to stay within xlim
        elif 1 <= x_range < 10:
            major_ticks = np.arange(np.floor(xlim[0] / 1) * 1, np.ceil(xlim[1] / 1) * 1 + 1, 1)
            major_ticks = major_ticks[(major_ticks >= xlim[0]) & (major_ticks <= xlim[1])]  # Filter to stay within xlim
            minor_ticks = np.arange(np.floor(xlim[0] / 0.1) * 0.1, np.ceil(xlim[1] / 0.1) * 0.1 + 1, 0.1)
            minor_ticks = minor_ticks[(minor_ticks >= xlim[0]) & (minor_ticks <= xlim[1])]  # Filter to stay within xlim
        else:
            # Use default ticks if range is too small or too large
            major_ticks = ax.get_xticks()
            minor_ticks = []
        # Set major and minor ticks
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        # Move ticks to the inside
        ax.tick_params(axis='x', direction='in', which='minor', length=3, top=True, bottom=True)  # Adjust the length as needed
        ax.tick_params(axis='x', direction='in', which='major', length=6, top=True, bottom=True)  # Adjust the length as needed
        ax.tick_params(axis='y', direction='in', which='minor', length=3, left=True, right=True)  # Adjust the length as needed
        ax.tick_params(axis='y', direction='in', which='major', length=6, left=True, right=True)  # Move y-ticks to the inside as well
        self.fig.canvas.draw_idle()

    def update_residual_ticks(self):
        if self.residual_ax is not None:
            self.update_ticks(self.residual_ax)
            self.ax.set_xticks([])
            plt.draw()

    def update_residual_ybounds(self):
        # Get the current x-limits of the residual axis
        x_min, x_max = self.residual_ax.get_xlim()

        # Mask residual data within the current x-limits
        mask = (self.x_data >= x_min) & (self.x_data <= x_max)
        visible_residual = self.residuals[mask]

        if visible_residual.size > 0:
            # Calculate the min and max of the visible residual data
            min_residual = visible_residual.min()
            max_residual = visible_residual.max()

            # Add a 10% margin to both top and bottom for display purposes
            margin = 0.1 * (max_residual - min_residual)
            self.residual_ax.set_ylim(min_residual - margin, max_residual + margin)

        # Redraw the canvas to apply changes
        self.fig.canvas.draw_idle()

    # Update the plot with new redshift
    def update_redshift(self, new_redshift):
        self.redshift = new_redshift # Update the global redshift variable
        
        # Redisplay line lists with new redshift
        if self.active_line_lists:
            self.display_linelist()
        
        self.fig.canvas.draw_idle()

    # Smooth the spectrum with a Gaussian kernel
    def smooth_spectrum(self, kernel_width):
        from scipy.ndimage import gaussian_filter1d  # Import here to avoid global imports

        # [DEBUG] Ensure original spectrum data is present
        if self.original_spec is None:
            print("Error: original_spec is not defined.")
            return

        print("Applying Gaussian smoothing with kernel width:", kernel_width)
        self.smoothed_spec = gaussian_filter1d(self.original_spec, sigma=kernel_width)
        
        # [DEBUG] Verify smoothing result
        print("Smoothed spectrum (first 5 values):", self.smoothed_spec[:5])

        if self.smoothed_spec is not None:  # Check if the spectrum_line is defined
            print("Spectrum line updated with smoothed data.")
        else:
            print("Error: spectrum_line is not defined.")

    # Check if there is an existing fitted continuum covering the current bounds
    def get_existing_continuum(self, left_bound, right_bound):
        try:
            for continuum_fit in self.continuum_fits:
                if continuum_fit['bounds'][0] <= left_bound and continuum_fit['bounds'][1] >= right_bound:
                    x_range = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                    if 'coeffs' in continuum_fit:
                        # New format with polynomial coefficients
                        continuum_vals = np.polyval(continuum_fit['coeffs'], x_range)
                        return continuum_vals, continuum_fit['coeffs'][0], continuum_fit['coeffs'][-1]
                    else:
                        # Old format with a, b parameters (backwards compatibility)
                        continuum_vals = self.continuum_model(x_range, continuum_fit['a'], continuum_fit['b'])
                        return continuum_vals, continuum_fit['a'], continuum_fit['b']
            # No matching continuum found
            return None, None, None
        except (ValueError, KeyError):
            print("No existing continuum within bounds")
            return None, None, None
    
    def check_continuum_partial_overlap(self, left_bound, right_bound):
        """Check if fit bounds partially overlap with any continuum region.
        Returns (has_partial_overlap, overlap_message)
        """
        for continuum_fit in self.continuum_fits:
            cont_left, cont_right = continuum_fit['bounds']
            # Check if bounds are partially outside the continuum
            if (left_bound < cont_left and right_bound > cont_left and right_bound <= cont_right) or \
               (left_bound >= cont_left and left_bound < cont_right and right_bound > cont_right) or \
               (left_bound < cont_left and right_bound > cont_right):
                # Partial overlap - one or both bounds are outside continuum region
                if not (cont_left <= left_bound and cont_right >= right_bound):
                    return True, f"Fit bounds [{left_bound:.2f}, {right_bound:.2f}] partially overlap continuum region [{cont_left:.2f}, {cont_right:.2f}]"
        return False, None

    def get_bounds(self, fit):
        if self.is_velocity_mode:
            if fit['is_velocity_mode']:
                left_bound, right_bound = fit['bounds'] # May need to change this to convert to the current self.rest_wavelength
            else:
                left_bound, right_bound = self.wav_to_vel(left_bound, self.rest_wavelength, z=self.redshift), self.wav_to_vel(right_bound, self.rest_wavelength, z=self.redshift)
        else:
            if fit['is_velocity_mode']:
                left_bound, right_bound = self.vel_to_wav(left_bound, self.rest_wavelength, z=self.redshift), self.wav_to_vel(right_bound, self.rest_wavelength, z=self.redshift)
            else:
                left_bound, right_bound = fit['bounds']

        return left_bound, right_bound

    # Code for EW from Gaussian
    def calculate_equivalent_width(self, profile_function, continuum_params, x_bounds):
        c_in_km_per_s = 2.9979246e5
        # Generate x values over the specified bounds for integration
        x_values = np.linspace(x_bounds[0], x_bounds[1], 100)
        if self.is_velocity_mode:
            x_values = self.rest_wavelength * (1 + x_values/c_in_km_per_s)
        
        # Calculate profile and continuum values
        profile_values = profile_function(x_values)
        continuum_values = self.continuum_model(x_values, *continuum_params)
        delta_lambda = np.diff(x_values)
        delta_lambda = np.append(delta_lambda, delta_lambda[-1])
        
        # Calculate the equivalent width (EW) using the trapezoidal rule
        if self.is_velocity_mode:
            x_values
            ew = (self.rest_wavelength/c_in_km_per_s) * np.trapz(1 - (profile_values + continuum_values) / continuum_values, x_values)
        else:
            ew = np.trapz(1 - (profile_values + continuum_values) / continuum_values, x_values)
        ew_r = ew / (1 + self.redshift)

        # Calculate the equivalent width (EW) using summation
        # ew = np.sum((1 - profile_values / continuum_values) * delta_lambda)

        # [DEBUG] Print debug information
        print(f"Avg. continuum level: {np.mean(continuum_values):.2f}")
        print(f"Avg. delta_lambda: {np.mean(delta_lambda):.2f}")
        print(f"Equivalent Width (observed): {ew:.2f} Å")
        print(f"Equivalent Width (rest): {ew_r:.2f} Å")
        return ew
    
    # Code for EW from Gaussian
    def expr_ew(self, comp_x, cont_y, model_y, redshift):
        ew = np.trapz(1 - (model_y + cont_y) / cont_y, comp_x)
        ew_r = ew / (1 + redshift)
        return ew, ew_r

    def calculate_and_plot_residuals(self):
        self.residuals = self.calculate_residuals()  # Custom function to calculate residuals
        # Clear previous lines before plotting new ones
        self.residual_ax.clear()
        # Update plot
        self.residual_line, = self.residual_ax.step(self.x_data, self.residuals, color='royalblue', where='mid')
        self.residual_ax.plot(self.x_data, [0] * len(self.x_data), color='gray', linestyle='--', linewidth=1) # Add horizontal line at y=0
        # Restore labels after clear
        if self.is_velocity_mode:
            self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)")
        else:
            self.residual_ax.set_xlabel(self._get_wavelength_unit_label())
        self.residual_ax.set_ylabel("Residuals")
        self.update_bounds()
        self.update_residual_ticks()  # Update ticks to look nice
        self.update_residual_ybounds()  # Update y-bounds to look nice
        if self.is_velocity_mode:
            self.residual_line.set_xdata(self.velocities)
        
    def toggle_residual_panel(self):
        if not self.is_residual_shown:
            # Create residual panel only if it doesn't exist
            if self.residual_ax is None:
                # Create a new axis for residuals below the spectrum
                self.residual_ax = self.fig.add_axes([0.125, 0.2, 0.775, 0.15])  # Adjusted position and height and position below the main plot
            if self.is_velocity_mode:
                self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)")
            else:
                self.residual_ax.set_xlabel(self._get_wavelength_unit_label())
            self.residual_ax.set_ylabel("Residuals")

            # Calculate and plot residuals
            self.calculate_and_plot_residuals()
            self.residual_ax.set_visible(True)  # Show the residual panel
            self.ax.set_xticks([])  # Hide x-ticks of the main plot
            self.residual_ax.set_xlim(self.ax.get_xlim())  # Match x-bounds with the main plot
            self.update_residual_ybounds()
            self.is_residual_shown = True
        else:
            # Hide the residual panel
            self.residual_ax.clear()
            self.residual_ax.set_visible(False)
            self.is_residual_shown = False

            self.ax.set_xticks(self.ax.get_xticks())  # Restore the x-ticks based on current limits

        plt.draw()  # Refresh plot to show/hide residual panel

    def calculate_residuals(self):
        # Calculate total fitted Gaussian, Voigt, and continuum values
        gaussian_sum = np.zeros_like(self.spec)
        for fit in self.gaussian_fits:
            left_bound, right_bound = fit['bounds']
            # Check if bounds are within current spectrum range
            if right_bound < self.x_data.min() or left_bound > self.x_data.max():
                continue  # Skip fits outside current spectrum
            mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
            if not np.any(mask):
                continue  # No data points in this range
            comp_x = self.x_data[mask]
            amp = fit['amp']
            mean = fit['mean']
            stddev = fit['stddev']
            gaussian_sum[mask] += self.gaussian(comp_x, amp, mean, stddev)

        voigt_sum = np.zeros_like(self.spec)
        for fit in self.voigt_fits:  # Loop through each fit that's stored
            amp = fit['amp']
            center = fit['center']
            gamma = fit['gamma']
            sigma = fit['sigma']
            left_bound, right_bound = fit['bounds']  # Get bounds from fit
            
            # Check if bounds are within current spectrum range
            if right_bound < self.x_data.min() or left_bound > self.x_data.max():
                continue  # Skip fits outside current spectrum
            mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
            if not np.any(mask):
                continue  # No data points in this range
            comp_x = self.x_data[mask]

            # Update voigt_sum for the valid range
            voigt_sum[mask] += self.voigt(comp_x, amp=amp, center=center, gamma=gamma, sigma=sigma)
        
        continuum_sum = np.zeros_like(self.spec)
        if self.continuum_fits:
            for continuum_fit in self.continuum_fits:
                left_bound, right_bound = continuum_fit['bounds']
                # Check if bounds are within current spectrum range
                if right_bound < self.x_data.min() or left_bound > self.x_data.max():
                    continue  # Skip fits outside current spectrum
                mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
                if not np.any(mask):
                    continue  # No data points in this range
                comp_x = self.x_data[mask]
                coeffs = continuum_fit['coeffs']
                continuum_sum[mask] = np.polyval(coeffs, comp_x)

        # Add Listfit polynomial components to residuals (skip deleted polynomials)
        listfit_poly_sum = np.zeros_like(self.spec)
        if self.listfit_fits:
            for listfit in self.listfit_fits:
                left_bound, right_bound = listfit['bounds']
                # Check if bounds are within current spectrum range
                if right_bound < self.x_data.min() or left_bound > self.x_data.max():
                    continue  # Skip listfit outside current spectrum
                mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
                if not np.any(mask):
                    continue  # No data points in this range
                comp_x = self.x_data[mask]
                components = listfit['components']
                result = listfit['result']
                
                # Find and add polynomial components from the listfit (skip deleted ones)
                poly_count = 0
                for comp in components:
                    if comp['type'] == 'polynomial':
                        # Check if this polynomial was deleted
                        is_deleted = False
                        for item_id, item_info in self.item_id_map.items():
                            if (item_info.get('type') == 'polynomial' and 
                                item_info.get('fit_dict', {}).get('listfit_bounds') == (left_bound, right_bound) and
                                item_info.get('fit_dict', {}).get('poly_index') == poly_count):
                                if item_id in self.deleted_listfit_polynomials:
                                    is_deleted = True
                                break
                        
                        if not is_deleted:
                            order = comp.get('order', 1)
                            prefix = f'p{poly_count}_'
                            poly_coeffs = []
                            for i in range(order + 1):
                                coeff_val = result.params[f'{prefix}c{i}'].value
                                poly_coeffs.append(coeff_val)
                            # Reverse coefficients for np.polyval (expects highest order first)
                            poly_coeffs = poly_coeffs[::-1]
                            y_poly = np.polyval(poly_coeffs, comp_x)
                            listfit_poly_sum[mask] += y_poly
                        poly_count += 1

        # Calculate residual as (spectrum - fitted Gaussians - Voigts - continuum - listfit polynomials)
        return self.spec - gaussian_sum - voigt_sum - continuum_sum - listfit_poly_sum

    def update_residual_xbounds(self, event):
        if self.is_residual_shown and self.residual_ax is not None:
            self.residual_ax.set_xlim(self.ax.get_xlim())
            plt.draw()

    # Function to save the current plot as a PDF
    def save_plot_as_pdf(self):
        """
        Saves the current figure as a PDF, hiding specific text boxes temporarily.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.pdf"

        self.fig.savefig(filename, format='pdf', bbox_inches='tight')

        print(f"Plot saved as {filename}")

    def wav_to_vel(self, lam_obs, lam_0, z):
        c_in_km_per_s = 2.9979246e5
        lam_rest = lam_obs * (1 + z) ** (-1)  # lam_rest is wavelength in the rest frame (object not moving)
        v = c_in_km_per_s * (lam_rest - lam_0) / lam_0
        return v

    def vel_to_wav(self, v, lam_0, z):
        c_in_km_per_s = 2.9979246e5
        lam_rest = lam_0 * (1 + v / c_in_km_per_s)
        # Convert back to observed wavelength if redshifted
        lam_obs = lam_rest * (1 + z)
        return lam_obs

    def convert_continuum_to_velocity(self, rest_wavelength):
        # Convert all defined continuum regions to velocities
        velocity_regions = []
        for start, end in self.continuum_regions:
            start_vel = self.wav_to_vel(start, rest_wavelength, z=self.redshift)
            end_vel = self.wav_to_vel(end, rest_wavelength, z=self.redshift)
            velocity_regions.append((start_vel, end_vel))
        return velocity_regions

    def convert_to_velocity(self, line):
    # def convert_gaussian_to_velocity(self, mean, bounds, line):
        """Convert Gaussian mean and bounds from wavelength to velocity."""
        # velocity_mean = self.wav_to_vel(mean, self.rest_wavelength, z=self.redshift)
        # velocity_bounds = (
        #     self.wav_to_vel(bounds[0], self.rest_wavelength, z=self.redshift),
        #     self.wav_to_vel(bounds[1], self.rest_wavelength, z=self.redshift)
        # )
        # Extract the x-data (wavelength) from line for conversion
        wavelength_line_data = line.get_xdata()
        velocity_line_data = self.wav_to_vel(wavelength_line_data, self.rest_wavelength, z=self.redshift)
        # return velocity_mean, velocity_bounds, velocity_line
        return velocity_line_data

    def convert_to_wavelength(self, line):
    # def convert_gaussian_to_wavelength(self, velocity_mean, bounds, line):
        """Convert Gaussian mean and bounds from velocity back to wavelength."""
        # wavelength_mean = self.vel_to_wav(velocity_mean, self.rest_wavelength, z=self.redshift)
        # wavelength_bounds = (
        #     self.vel_to_wav(bounds[0], self.rest_wavelength, z=self.redshift),
        #     self.vel_to_wav(bounds[1], self.rest_wavelength, z=self.redshift)
        # )
        # wavelength_bounds = bounds
        # Extract the velocity data from line for conversion
        velocity_line_data = line.get_xdata()
        wavelength_line_data = self.vel_to_wav(velocity_line_data, self.rest_wavelength, z=self.redshift)
        # return wavelength_mean, wavelength_bounds, wavelength_line
        return wavelength_line_data

    def convert_voigt_to_velocity(self, line):
        """Convert Voigt parameters from wavelength to velocity."""
        # velocity_params = {
        #     'amp': fit['amp'],
        #     'center': self.wav_to_vel(fit['center'], self.rest_wavelength, z=self.redshift),
        #     'sigma': fit['sigma'],
        #     'gamma': fit['gamma']
        # }
        # return velocity_params
        wavelength_line_data = line.get_xdata()
        velocity_line_data = self.wav_to_vel(wavelength_line_data, self.rest_wavelength, z=self.redshift)
        # return wavelength_mean, wavelength_bounds, wavelength_line
        return velocity_line_data


    def convert_voigt_to_wavelength(self, line):
        """Convert Voigt parameters from velocity back to wavelength."""
        # wavelength_params = {
        #     'amp': velocity_params['amp'],
        #     'center': velocity_params['center'],
        #     'sigma': velocity_params['sigma'],
        #     'gamma': velocity_params['gamma']
        # }
        # return wavelength_params
        velocity_line_data = line.get_xdata()
        wavelength_line_data = self.vel_to_wav(velocity_line_data, self.rest_wavelength, z=self.redshift)
        # return wavelength_mean, wavelength_bounds, wavelength_line
        return wavelength_line_data

    def activate_velocity_mode(self):
        print("Entering velocity mode. Please enter the wavelength to set as rest-frame (in Å):")
        
        # Request rest-frame wavelength input from the user
        try:
            # Get all available line lists for selection
            available_line_lists = self.get_all_available_line_lists()
            self.line_list_window = LineListWindow(available_line_lists=available_line_lists)
            self.line_list_window.selected_line.connect(self.set_rest_wavelength)
            self.line_list_window.show()
        except ValueError:
            print("Invalid wavelength input. Please enter a numeric value.")
            return

    def set_rest_wavelength(self, line_id, line_wavelength):
        # Set rest_wavelength and perform the conversion
        self.rest_id, self.rest_wavelength = line_id, line_wavelength
        print(f"Selected rest wavelength: {self.rest_wavelength:.2f} Å")
        
        # Convert spectrum wavelengths to Angstroms for velocity calculation
        wav_in_angstrom = self._convert_wavelength_to_angstrom(self.wav)
        
        # Calculate velocity for each wavelength point in the spectrum
        self.velocities = self.wav_to_vel(wav_in_angstrom, self.rest_wavelength, z=self.redshift)
        self.x_data = self.velocities  # Set x_data to velocities

        # Update x-axis labels and limits for the main plot
        self.spectrum_line.set_xdata(self.x_data)
        self.step_spec.set_xdata(self.x_data)
        # Only update error lines if they exist
        if self.step_error is not None:
            self.step_error.set_xdata(self.x_data)
        if self.line_error is not None:
            self.line_error.set_xdata(self.x_data)
        self.line_spec.set_xdata(self.x_data)
        self.ax.set_xlabel(r"Velocity (km s$^{-1}$)")
        self.ax.plot(self.x_data, [0] * len(self.x_data), color='gray', linestyle='--', linewidth=1)

        # Update continuum fits to velocity space
        for fit in self.continuum_fits:
            velocity_line_data = self.convert_to_velocity(fit['line'])
            fit['line'].set_xdata(velocity_line_data)

            for patch_data in fit['patches']:
                if fit['is_velocity_mode']:
                    vel_start, vel_end = patch_data['bounds']
                else:
                    # Bounds stored in current display unit, convert to Angstrom then to velocity
                    wav_start_angstrom = self._convert_wavelength_to_angstrom(patch_data['bounds'][0])
                    wav_end_angstrom = self._convert_wavelength_to_angstrom(patch_data['bounds'][1])
                    vel_start = self.wav_to_vel(wav_start_angstrom, self.rest_wavelength, z=self.redshift)
                    vel_end = self.wav_to_vel(wav_end_angstrom, self.rest_wavelength, z=self.redshift)
                patch_data['patch'].remove()
                new_patch = self.ax.axvspan(vel_start, vel_end, color='magenta', alpha=0.3, hatch='//')
                patch_data['patch'] = new_patch

        # Convert Gaussian fits to velocity space
        for fit in self.gaussian_fits:
            velocity_line_data = self.convert_to_velocity(fit['line'])
            fit['line'].set_xdata(velocity_line_data)

        # Convert Voigt fits to velocity space
        for fit in self.voigt_fits:
            velocity_line_data = self.convert_to_velocity(fit['line'])
            fit['line'].set_xdata(velocity_line_data)

        # Convert Listfit components to velocity space
        for listfit in self.listfit_fits:
            component_lines = self.listfit_component_lines.get(listfit.get('id'))
            if component_lines:
                for line in component_lines.values():
                    velocity_line_data = self.convert_to_velocity(line)
                    line.set_xdata(velocity_line_data)

        # Update residual plot if shown
        if self.is_residual_shown:
            self.residual_line.set_xdata(self.x_data)
            self.residual_ax.set_xlim(-3000, 3000)
            self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)")
            self.update_residual_ticks()
            self.update_residual_ybounds()

        # Update main axis ticks and redraw the plot
        self.update_ticks(self.ax)
        self.ax.set_xlim(-3000, 3000)

        self.is_velocity_mode = True
        self.fig.canvas.draw()  # Force immediate redraw for tick labels
        print(f"Velocity mode activated with rest wavelength {self.rest_wavelength:.2f} Å and redshift {self.redshift:.3f}.")

    def exit_velocity_mode(self):
        
        # Revert x-axis data to wavelength for main plot elements
        # Note: self.wav is still in original units, so just use it directly
        self.x_data = self.wav
        self.spectrum_line.set_xdata(self.x_data)
        self.step_spec.set_xdata(self.x_data)
        # Only update error lines if they exist
        if self.step_error is not None:
            self.step_error.set_xdata(self.x_data)
        if self.line_error is not None:
            self.line_error.set_xdata(self.x_data)
        self.line_spec.set_xdata(self.x_data)
        
        # Set x-axis labels back to wavelength
        self.ax.set_xlabel(self._get_wavelength_unit_label())
        self.ax.plot(self.x_data, [0] * len(self.x_data), color='gray', linestyle='--', linewidth=1)

        # Convert continuum fits back to wavelength space
        for fit in self.continuum_fits:
            wavelength_line_data = self.convert_to_wavelength(fit['line'])
            fit['line'].set_xdata(wavelength_line_data)

            for patch_data in fit['patches']:
                if fit['is_velocity_mode']:
                    # These bounds are in velocity, convert to wavelength then to display unit
                    vel_start, vel_end = patch_data['bounds']
                    wav_start_angstrom = self.vel_to_wav(vel_start, self.rest_wavelength, z=self.redshift)
                    wav_end_angstrom = self.vel_to_wav(vel_end, self.rest_wavelength, z=self.redshift)
                    wav_start = self._convert_wavelength_from_angstrom(wav_start_angstrom)
                    wav_end = self._convert_wavelength_from_angstrom(wav_end_angstrom)
                else:
                    # These bounds are already in display unit
                    wav_start, wav_end = patch_data['bounds']
                patch_data['patch'].remove()
                new_patch = self.ax.axvspan(wav_start, wav_end, color='magenta', alpha=0.3, hatch='//')
                patch_data['patch'] = new_patch

        # Convert Gaussian fits back to wavelength space
        for fit in self.gaussian_fits:
            wavelength_line_data = self.convert_to_wavelength(fit['line'])
            fit['line'].set_xdata(wavelength_line_data)

        # Convert Voigt fits back to wavelength space
        for fit in self.voigt_fits:
            wavelength_line_data = self.convert_to_wavelength(fit['line'])
            fit['line'].set_xdata(wavelength_line_data)

        # Convert Listfit components back to wavelength space
        for listfit in self.listfit_fits:
            component_lines = self.listfit_component_lines.get(listfit.get('id'))
            if component_lines:
                for line in component_lines.values():
                    wavelength_line_data = self.convert_to_wavelength(line)
                    line.set_xdata(wavelength_line_data)

        # Update residual plot if shown
        if self.is_residual_shown:
            self.residual_line.set_xdata(self.x_data)
            self.residual_ax.set_xlim(self.x_data.min(), self.x_data.max())
            self.residual_ax.set_xlabel(self._get_wavelength_unit_label())
            self.update_residual_ticks()
            self.update_residual_ybounds()

        # Update main axis ticks and redraw plot
        self.update_ticks(self.ax)
        self.ax.set_xlim(self.x_data.min(), self.x_data.max())

        self.rest_wavelength = None
        self.rest_id = None
        self.is_velocity_mode = False
        
        self.fig.canvas.draw()  # Force immediate redraw for tick labels
        print("Exited velocity mode and reverted to wavelength space.")

    # Define fitted functions
    def gaussian(self, x, amp, mean, stddev):
        y = amp * np.exp(-(x - mean)**2 / (2 * stddev**2))
        # return y
        return self.apply_lsf(y)

    def multi_gaussian(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, mean, stddev = params[i:i+3]
            y += self.gaussian(x, amp, mean, stddev)
        return y

    def multi_gaussian_sharedsigma(self, x, *params):
        y = np.zeros_like(x)
        stddev = params[-1]
        for i in range(0, len(params)-1, 2):
            amp, mean = params[i:i+2]
            y += self.gaussian(x, amp, mean, stddev)
        return y

    def voigt(self, x, amp, center, sigma, gamma):
        # z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
        # return amplitude * np.real(wofz(z))
        tiny = np.finfo(float).eps
        s2 = np.sqrt(2)
        s2pi = np.sqrt(2*np.pi)
        if gamma is None:
            gamma = sigma
        z = (x-center + 1j*gamma) / max(tiny, (sigma*s2))
        y = amp*np.real(wofz(z)) / max(tiny, (sigma*s2pi))
        return self.apply_lsf(y)

    def multi_voigt(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 4):  # Iterate over params in sets of four
            amp, center, sigma, gamma = params[i:i+4]
            y += self.voigt(x, amp, center, sigma, gamma)
        return y

    def continuum_model(self, x, *params):
        """
        Polynomial continuum model of order determined by number of parameters.
        params are coefficients for polynomial from highest to lowest order.
        """
        return np.polyval(params, x)

    # Define a function to fit the continuum
    def fit_continuum(self, x, y, err, sigma_threshold=2, max_iterations=10, tolerance=1e-4, poly_order=None):
        """
        Fits a polynomial continuum model to the provided data using iterative sigma-clipping.
        
        Parameters:
        -----------
        x, y, err : arrays or None
            Data points and errors (err can be None)
        sigma_threshold : float
            Sigma clipping threshold
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        poly_order : int, optional
            Polynomial order. If None, uses self.poly_order
        """
        if poly_order is None:
            poly_order = self.poly_order
        
        try:
            # Start with all values
            mask = np.ones_like(y, dtype=bool)
            prev_num_inliers = 0

            for it in range(max_iterations):
                # Apply the current mask to filter x and y arrays
                x_filtered = x[mask]
                y_filtered = y[mask]
                if err is not None:
                    err_filtered = err[mask]
                else:
                    err_filtered = None

                # Fit polynomial of specified order
                coeffs = np.polyfit(x_filtered, y_filtered, poly_order)
                continuum = np.polyval(coeffs, x)

                # Calculate residuals and updated mean and standard deviation
                residuals = y - continuum
                std_residuals = np.std(residuals)

                # Update bounds for the current sigma threshold
                lower_bound = -sigma_threshold * std_residuals
                upper_bound = sigma_threshold * std_residuals

                # Update mask to exclude outliers based on the new bounds
                mask = (residuals >= lower_bound) & (residuals <= upper_bound)

                # Check for convergence: stop if the number of inliers hasn't changed significantly
                num_inliers = mask.sum()
                if abs(num_inliers - prev_num_inliers) / num_inliers < tolerance:
                    print(f"Convergence reached after {it + 1} iterations.")
                    break

                prev_num_inliers = num_inliers

            else:
                print("Warning: Maximum iterations reached without full convergence.")

            # Calculate errors on coefficients
            coeffs_with_cov = np.polyfit(x_filtered, y_filtered, poly_order, cov=True)
            if isinstance(coeffs_with_cov, tuple):
                coeffs = coeffs_with_cov[0]
                pcov = coeffs_with_cov[1]
                perr = np.sqrt(np.diag(pcov))
            else:
                coeffs = coeffs_with_cov
                perr = np.ones(poly_order + 1)

            # Return the final continuum and parameters
            return continuum, coeffs, perr
        except RuntimeError as e:
            print(f"Error in fitting continuum: {e}")
            return None, None, None

    # Function to clear all continuum regions
    def clear_continuum_regions(self):
        """
        Clears all previously defined continuum regions from the plot.
        """
        for patch in self.continuum_patches:
            patch.remove()
        self.continuum_patches.clear()

    # Function to clear the line region
    def clear_line_region():
        """
        Clears all previously defined line regions from the plot.
        """
        for patch in self.line_patches:
            patch.remove()
        self.line_patches.clear()

    def prompt_user_for_file_list(self):
        """Prompt the user to choose which set of file suffixes to load"""
        print("Choose the file suffix list to display:")
        print("1. Show filter throughputs ending with 'w.txt'")
        print("2. Show filter throughputs ending with 'lp.txt'")
        print("3. Show filter throughputs ending with 'm.txt'")
        print("4. Show filter throughputs ending with 'n.txt'")
        print("5. Show filter throughputs ending with 'p.txt'")
        print("6. Show filter throughputs ending with 'x.txt'")
        print("7. Show all filter throughputs")
        choice = input("Enter a number (1-7) to choose: ")

        if choice == '1':
            suffixes = ('w.txt',)
        elif choice == '2':
            suffixes = ('lp.txt',)
        elif choice == '3':
            suffixes = ('m.txt',)
        elif choice == '4':
            suffixes = ('n.txt',)
        elif choice == '5':
            suffixes = ('p.txt',)
        elif choice == '6':
            suffixes = ('x.txt',)
        elif choice == '7':
            suffixes = ('w.txt', 'lp.txt', 'm.txt', 'n.txt', 'p.txt', 'x.txt')
        else:
            print("Invalid choice. Defaulting to 'w.txt'.")
            suffixes = ('w.txt',)
        
        return suffixes

    def toggle_filter_bands(self, index):
        """Toggle the display of filter bands and their labels on the plot."""
        base_dir = Path(__file__).parent
        directories = [
            str(base_dir / 'throughputs/WFC3_UVIS/'),
            str(base_dir / 'throughputs/WFC3_IR/'),
            str(base_dir / 'throughputs/ACS/'),
            str(base_dir / 'throughputs/NIRCam/')
        ]

        # Ensure index is within bounds
        if index >= len(directories):
            print("Invalid directory index.")
            return

        # Set the chosen directory based on key press
        throughput_dir = directories[index]

        # Ensure x-axis limits are correctly defined as floats
        x_limits = self.ax.get_xlim()
        x_lower_bound = float(x_limits[0])
        x_upper_bound = float(x_limits[1])

        # If filter lines are already visible, remove them and toggle 'show_filters'
        if self.show_filters:
            for line in getattr(self, "filter_lines", []):
                line.remove()  # Remove each Line2D object from the axes
            self.filter_lines = []  # Clear the list after removal

            # Remove each label as well
            for label in getattr(self, "filter_labels", []):
                label.remove()  # Remove each Text object from the axes
            self.filter_labels = []  # Clear the list after removal
        else:

            # Get the file suffixes based on user input
            suffixes = self.prompt_user_for_file_list()

            # List all .txt files ending with 'w.txt'
            throughput_files = [f for f in os.listdir(throughput_dir) if f.endswith(suffixes)]

            # Get the y-axis limits for scaling the throughput curves
            y_limits = self.ax.get_ylim()
            y_min, y_max = y_limits

            # Use a color map
            colormap = cm.get_cmap('jet', len(throughput_files))

            # Initialize lists to keep track of plotted lines and labels for clearing later
            self.filter_lines = []
            self.filter_labels = []

            # Plot each filter band from the files with unique colors from the colormap
            for i, file_name in enumerate(throughput_files):
                # Extract the filter name, e.g., 'f814w' from 'f814w.txt'
                filter_name = file_name.split('.')[0]
                
                # Load data from the file
                file_path = str(Path(throughput_dir) / file_name)
                data = np.loadtxt(file_path, skiprows=1)
                
                # Assuming the first column is wavelength and the second column is throughput
                wavelengths = data[:, 0]
                throughputs = data[:, 1]
                
                # Scale throughput to match the plot's y-axis limits
                # scaled_throughputs = (throughputs * (y_max - y_min)) + y_min
                scaled_throughputs = (throughputs * y_max)
                
                # Plot throughput curve with a unique color from the colormap
                color = colormap(i / len(throughput_files))
                line, = self.ax.plot(wavelengths, scaled_throughputs, color=color, linestyle='-')
                self.filter_lines.append(line)  # Store the line for later removal
                
                # Place text label at the peak of each throughput curve
                peak_index = np.argmax(scaled_throughputs)
                peak_wavelength = wavelengths[peak_index]
                peak_throughput = scaled_throughputs[peak_index]
                label = self.ax.text(peak_wavelength, peak_throughput, filter_name, color=color, ha='center', va='bottom')
                self.filter_labels.append(label)  # Store the label for later removal

        # Reset x-axis limits after toggling bands
        self.ax.set_xlim(x_lower_bound, x_upper_bound)
        self.fig.canvas.draw_idle()  # Redraw the figure to update the display
        self.show_filters = not self.show_filters  # Toggle the state

    def toggle_instrument_bands(self, index):
        """Toggle the display of instrument bands on the plot."""
        # Ensure x-axis limits are correctly defined as floats
        x_limits = self.ax.get_xlim()
        x_lower_bound = float(x_limits[0])
        x_upper_bound = float(x_limits[1])

        if self.show_bands:
            self.clear_band_areas()
        else:
            self.add_band_area(index)

        # Reset x-axis limits after toggling bands
        self.ax.set_xlim(x_lower_bound, x_upper_bound)
        self.fig.canvas.draw_idle()  # Redraw the figure to update the display
        self.show_bands = not self.show_bands  # Toggle the state

    def clear_band_areas(self):
        """Remove existing band areas and labels from the plot."""
        for area in self.band_areas:
            area.remove()  # Remove the filled area from the plot
        for label in self.band_labels:
            label.remove()  # Remove the label from the plot
        self.band_areas.clear()  # Clear the list of band areas
        self.band_labels.clear()

    def add_band_area(self, index):
        """Add a band area to the plot based on the provided index."""
        band_range = self.band_ranges[index]
        band_id = band_range[0]
        start = float(band_range[1])  # Ensure type consistency
        end = float(band_range[2])
        
        band_area = self.ax.axvspan(start, end, color='orange', alpha=0.3)
        band_label = self.ax.text(start, 0, f'{band_id} {start}-{end}',
                                   rotation=90, verticalalignment='bottom', color='orange', fontsize=8)
        self.band_areas.append(band_area)  # Store the area for later removal
        self.band_labels.append(band_label)

        # Check if the area is within the x-limits
        if start < self.x_lower_bound or end > self.x_upper_bound:
            print("Warning: Band area exceeds x-limits.")

    def plot_redshift_gaussian(self, fit):
        left_bound, right_bound = fit['bounds']
        x = np.linspace(left_bound, right_bound, 100)
        print(f"Amplitude: {fit['amp']}, Mean: {fit['mean']}, Sigma: {fit['stddev']}")  # [DEBUG]

        gaussian_curve = fit['amp'] * np.exp(-0.5 * ((x - fit['mean']) / fit['stddev']) ** 2)
        continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
        
        # Handle case where no continuum exists
        if continuum_vals is None:
            plot_data = gaussian_curve
        else:
            existing_continuum = self.continuum_model(x, a, b) # Make continuum with same dimensions as gaussian curve
            plot_data = gaussian_curve + existing_continuum
            
        if hasattr(self, 'current_gaussian_plot') and self.current_gaussian_plot:
            self.current_gaussian_plot.remove() # Remove any previous plot of the selected gaussian
        self.current_gaussian_plot = self.ax.plot(x, plot_data, color='lime', linestyle='-', linewidth=2)[0]  # Store the first element (line object)
        
        # Highlight the original fit line in neon green for redshift mode
        if 'line' in fit and fit['line']:
            fit['line'].set_color('lime')
            fit['line'].set_linewidth(2.5)
            self.redshift_selected_line = fit['line']
        
        plt.draw()  # Refresh the plot

    def plot_redshift_voigt(self, fit):
        left_bound, right_bound = fit['bounds']
        x = np.linspace(left_bound, right_bound, 100)
        
        # Retrieve Voigt parameters
        amp = fit['amp']
        center = fit['center']
        gamma = fit['gamma']
        sigma = fit['sigma']
        print(f"Amplitude: {amp}, Center: {center}, Sigma: {sigma}, Gamma: {gamma}")

        voigt_curve = self.voigt(x, amp, center, sigma, gamma)
        continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
        
        # Handle case where no continuum exists
        if continuum_vals is None:
            plot_data = voigt_curve
        else:
            existing_continuum = self.continuum_model(x, a, b)
            plot_data = voigt_curve + existing_continuum
            
        if hasattr(self, 'current_voigt_plot') and self.current_voigt_plot:
            self.current_voigt_plot.remove()
        self.current_voigt_plot = self.ax.plot(x, plot_data, color='lime', linestyle='-', linewidth=2)[0]
        
        # Highlight the original fit line in neon green for redshift mode
        if 'line' in fit and fit['line']:
            fit['line'].set_color('lime')
            fit['line'].set_linewidth(2.5)
            self.redshift_selected_line = fit['line']
        
        plt.draw()  # Refresh the plot to show the updated Voigt profile

    def display_linelist(self):
        """Display all active line lists on the spectrum"""
        # Clear previous lines
        self.clear_linelist()
        
        # Use the stored bounds instead of getting them from the axis
        # This ensures we use the correct bounds even if they haven't been applied to the canvas yet
        xlim = (self.x_lower_bound, self.x_upper_bound)
        
        # Display each active line list
        for linelist_info in self.active_line_lists:
            linelist = linelist_info['linelist']
            color = linelist_info['color']
            
            for line in linelist.lines:
                # Apply redshift to the line wavelength (linelist is in Angstroms)
                shifted_wl = line.wave * (1 + self.redshift)
                
                # Convert from Angstroms to current display unit
                shifted_wl_display = self._convert_wavelength_from_angstrom(shifted_wl)
                
                # Check if wavelength is within current x-limits
                if xlim[0] <= shifted_wl_display <= xlim[1]:
                    # Draw the vertical line
                    vline = self.ax.axvline(shifted_wl_display, color=color, linestyle='--', alpha=0.7)
                    label = self.ax.text(shifted_wl_display, 0, line.name,
                                        rotation=90, verticalalignment='bottom', 
                                        color=color, fontsize=8)
                    self.current_linelist_lines.append((vline, label))
        
        plt.draw()

    def clear_linelist(self):
        """Remove all displayed line lists from the plot"""
        for line, label in self.current_linelist_lines:
            line.remove()
            label.remove()
        self.current_linelist_lines = []  # Clear the list of plotted lines

    def open_linelist_window(self):
        """Open the line list window for redshift estimation."""
        available_line_lists = self.get_all_available_line_lists()
        self.ll_window = LineListWindow(available_line_lists=available_line_lists)
        self.ll_window.selected_line.connect(self.estimate_redshift)
        self.ll_window.closed.connect(self.on_close_linelist)
        self.ll_window.setWindowFlags(self.ll_window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.ll_window.show()
        self.ll_window.raise_()
        self.ll_window.activateWindow()
    
    def show_line_list_selector(self):
        """Show the line list selector window"""
        if self.line_list_selector is None:
            self.line_list_selector = LineListSelector(self.resources_dir)
            self.line_list_selector.line_lists_changed.connect(self.on_line_lists_changed)
        
        self.line_list_selector.show()
        self.line_list_selector.raise_()
        self.line_list_selector.activateWindow()
    
    def on_line_lists_changed(self, line_lists_with_colors):
        """Handle change in selected line lists"""
        self.active_line_lists = line_lists_with_colors
        
        # Redisplay line lists
        if self.active_line_lists:
            self.display_linelist()
        else:
            self.clear_linelist()
            plt.draw()

    def on_close_linelist(self):
        """Remove the Gaussian plot when LineListWindow is closed."""
        if self.current_gaussian_plot is not None:
            self.current_gaussian_plot.remove()
            self.current_gaussian_plot = None
            plt.draw()  # Refresh the plot to reflect the removal
            self.restore_redshift_highlight()
            self.redshift_estimation_mode = False
            print('Exiting redshift estimation mode.')
        elif self.current_voigt_plot is not None:
            self.current_voigt_plot.remove()
            self.current_voigt_plot = None
            plt.draw()  # Refresh the plot to reflect the removal
            self.restore_redshift_highlight()
            self.redshift_estimation_mode = False
            print('Exiting redshift estimation mode.')
    
    def restore_redshift_highlight(self):
        """Restore the original color and linewidth of the line selected for redshift mode"""
        if self.redshift_selected_line:
            # Restore to original color and linewidth
            # Check the color and linewidth from the item_id_map to get the true originals
            for item_info in self.item_id_map.values():
                if item_info.get('line_obj') == self.redshift_selected_line:
                    original_color = item_info.get('color', 'red')
                    original_linewidth = item_info.get('original_linewidth', 1)
                    self.redshift_selected_line.set_color(original_color)
                    self.redshift_selected_line.set_linewidth(original_linewidth)
                    break
            self.redshift_selected_line = None
            plt.draw()
        
    def register_item(self, item_type, name, fit_dict=None, line_obj=None, patch_obj=None, position='', color='gray', bounds=None):
        """Register an item with the tracker"""
        item_id = f"{item_type}_{self.item_id_counter}"
        self.item_id_counter += 1
        
        # Store original linewidth for restoration on deselection
        original_linewidth = None
        if line_obj:
            original_linewidth = line_obj.get_linewidth()
        elif patch_obj:
            original_linewidth = patch_obj.get_linewidth()
        
        self.item_id_map[item_id] = {
            'type': item_type,
            'fit_dict': fit_dict,
            'line_obj': line_obj,
            'patch_obj': patch_obj,
            'name': name,
            'position': position,
            'color': color,
            'bounds': bounds,
            'original_linewidth': original_linewidth
        }
        self.item_tracker.add_item(item_id, item_type, name, position=position, color=color, line_obj=line_obj)
        # Also add to Fit Information window
        self.fit_information_window.add_fit(item_id, item_type, fit_dict, name)
        return item_id
    
    def unregister_item(self, item_id):
        """Remove item from tracker"""
        if item_id in self.item_id_map:
            del self.item_id_map[item_id]
            self.item_tracker.remove_item(item_id)
            self.fit_information_window.remove_fit(item_id)
        
        # Clean up highlighting tracking if this item was highlighted
        if item_id in self.highlighted_item_ids:
            self.highlighted_item_ids.discard(item_id)
    
    def show_item_tracker(self):
        """Show the item tracker window"""
        self.item_tracker.item_deleted.connect(self.on_item_deleted_from_tracker)
        self.item_tracker.show()
    
    def on_item_deleted_from_tracker(self, item_id):
        """Handle item deletion from tracker"""
        if item_id not in self.item_id_map:
            return
        
        item_info = self.item_id_map[item_id]
        item_type = item_info.get('type')
        
        # Capture state before deletion for undo/redo
        state_before = self.capture_state()
        
        # Remove from internal storage lists based on item type
        fit_dict = item_info.get('fit_dict')
        if fit_dict:
            if item_type == 'gaussian':
                # Remove from gaussian_fits list
                self.gaussian_fits = [f for f in self.gaussian_fits if f is not fit_dict]
            elif item_type == 'voigt':
                # Remove from voigt_fits list
                self.voigt_fits = [f for f in self.voigt_fits if f is not fit_dict]
            elif item_type == 'continuum':
                # Remove from continuum_fits list
                self.continuum_fits = [f for f in self.continuum_fits if f is not fit_dict]
            elif item_type == 'listfit_total':
                # Remove the Total Listfit - this removes the entire listfit fit
                # Find and remove the corresponding listfit from listfit_fits
                listfit_bounds = fit_dict.get('listfit_bounds')
                if listfit_bounds:
                    self.listfit_fits = [f for f in self.listfit_fits if f.get('bounds') != listfit_bounds]
                    # Also remove any Gaussian/Voigt/Polynomial items that belonged to this listfit
                    items_to_remove = []
                    for check_id, check_info in self.item_id_map.items():
                        check_bounds = check_info.get('fit_dict', {}).get('bounds')
                        if check_bounds == listfit_bounds:
                            items_to_remove.append(check_id)
                    for remove_id in items_to_remove:
                        if remove_id in self.item_id_map:
                            # Remove from tracker (this will recursively call on_item_deleted_from_tracker)
                            self.item_tracker.unregister_item(remove_id)
        
        # Handle polynomial deletion - mark it as deleted for residual calculation
        if item_type == 'polynomial':
            self.deleted_listfit_polynomials.add(item_id)
        
        # Handle line objects (gaussian, voigt, continuum, polynomial, listfit_total)
        line_obj = item_info.get('line_obj')
        if line_obj:
            try:
                line_obj.remove()
            except (ValueError, NotImplementedError):
                # Object may have already been removed or cannot be removed
                pass
        
        # Handle patches (continuum regions, masks)
        patch_obj = item_info.get('patch_obj')
        if patch_obj:
            try:
                patch_obj.remove()
            except (ValueError, NotImplementedError):
                # Object may have already been removed or cannot be removed
                pass
        
        # If it's a continuum region, also remove from continuum_patches list
        if item_type == 'continuum_region' and 'bounds' in item_info:
            bounds = item_info['bounds']
            self.continuum_patches = [p for p in self.continuum_patches if p.get('bounds') != bounds]
        
        # Update residual display if shown
        if self.is_residual_shown:
            self.calculate_and_plot_residuals()
        
        # Redraw the figure immediately
        if self.fig is not None:
            self.fig.canvas.draw()
        
        self.unregister_item(item_id)
        
        # Record action for undo/redo
        self.record_action('delete_item', f'Delete {item_type}: {item_info.get("name", item_id)}')
    
    def on_item_selected_from_tracker(self, item_id):
        """Handle item selection from tracker - highlight with royal blue color"""
        if item_id not in self.item_id_map:
            return
        
        # Add to highlighted items set (support multiple selections)
        self.highlighted_item_ids.add(item_id)
        
        # Highlight the selected item
        item_info = self.item_id_map[item_id]
        
        # Handle line objects (Gaussians, Voigts)
        line_obj = item_info.get('line_obj')
        if line_obj:
            # Change to royal blue (original color stored in item_id_map)
            line_obj.set_color('royalblue')
            line_obj.set_linewidth(2.5)
        
        # Handle patch objects (continuum regions)
        patch_obj = item_info.get('patch_obj')
        if patch_obj:
            # Change edge color to royal blue (original color stored in item_id_map)
            patch_obj.set_edgecolor('royalblue')
            patch_obj.set_linewidth(2.5)
        
        self.fig.canvas.draw_idle()
    
    def on_item_deselected_from_tracker(self):
        """Handle item deselection from tracker - restore original color and linewidth for all items"""
        # Restore all highlighted items to their original colors and linewidths
        for item_id in self.highlighted_item_ids:
            if item_id in self.item_id_map:
                item_info = self.item_id_map[item_id]
                # Get original color and linewidth from item_id_map
                original_color = item_info.get('color', 'gray')
                original_linewidth = item_info.get('original_linewidth', 1)
                
                # Restore line objects (Gaussians, Voigts)
                line_obj = item_info.get('line_obj')
                if line_obj:
                    line_obj.set_color(original_color)
                    line_obj.set_linewidth(original_linewidth)
                
                # Restore patch objects (continuum regions)
                patch_obj = item_info.get('patch_obj')
                if patch_obj:
                    patch_obj.set_edgecolor(original_color)
                    patch_obj.set_linewidth(original_linewidth)
        
        # Clear tracking variables
        self.highlighted_item_ids.clear()
        self.fig.canvas.draw_idle()
    
    def on_item_individually_deselected_from_tracker(self, item_id):
        """Handle individual item deselection - restore original color and linewidth for that item"""
        if item_id in self.highlighted_item_ids:
            self.highlighted_item_ids.remove(item_id)
        
        if item_id not in self.item_id_map:
            return
        
        item_info = self.item_id_map[item_id]
        # Get original color and linewidth from item_id_map
        original_color = item_info.get('color', 'gray')
        original_linewidth = item_info.get('original_linewidth', 1)
        
        # Restore line objects (Gaussians, Voigts)
        line_obj = item_info.get('line_obj')
        if line_obj:
            line_obj.set_color(original_color)
            line_obj.set_linewidth(original_linewidth)
        
        # Restore patch objects (continuum regions)
        patch_obj = item_info.get('patch_obj')
        if patch_obj:
            patch_obj.set_edgecolor(original_color)
            patch_obj.set_linewidth(original_linewidth)
        
        self.fig.canvas.draw_idle()
    
    def on_estimate_redshift_from_tracker(self, item_id):
        """Handle estimate redshift action from ItemTracker context menu"""
        if item_id not in self.item_id_map:
            return
        
        item_info = self.item_id_map[item_id]
        item_type = item_info.get('type')
        
        # Activate redshift mode for Gaussians and Voigts
        if item_type == 'gaussian':
            fit = item_info.get('fit_dict')
            if fit:
                self.selected_gaussian = fit
                self.selected_voigt = None
                self.redshift_estimation_mode = True
                # IMPORTANT: Clear center_profile so estimate_redshift will extract from the fit dict
                self.center_profile = None
                self.center_profile_err = None
                self.plot_redshift_gaussian(fit)
                print("Redshift estimation mode activated for Gaussian")
                # Open line list window for line selection
                self.open_linelist_window()
        elif item_type == 'voigt':
            fit = item_info.get('fit_dict')
            if fit:
                self.selected_voigt = fit
                self.selected_gaussian = None
                self.redshift_estimation_mode = True
                # IMPORTANT: Clear center_profile so estimate_redshift will extract from the fit dict
                self.center_profile = None
                self.center_profile_err = None
                self.plot_redshift_voigt(fit)
                print("Redshift estimation mode activated for Voigt")
                # Open line list window for line selection
                self.open_linelist_window()
        
    def estimate_redshift(self, selected_id, selected_wavelength):
        self.selected_id, self.selected_rest_wavelength = selected_id, selected_wavelength  # Store the selected wavelength
        if self.selected_id is not None and self.selected_rest_wavelength is not None:
            # Extract center from selected fit if not already set
            if not hasattr(self, 'center_profile') or self.center_profile is None:
                # Try to get from selected Gaussian
                if self.selected_gaussian:
                    self.center_profile = self.selected_gaussian.get('mean')
                    self.center_profile_err = self.selected_gaussian.get('mean_err')
                    print(f"[DEBUG] Gaussian mean_err from fit dict: {self.center_profile_err}")
                    # Print the fitted parameters like in redshift mode
                    amp = self.selected_gaussian.get('amp')
                    mean = self.selected_gaussian.get('mean')
                    stddev = self.selected_gaussian.get('stddev')
                    print(f"Amplitude: {amp}, Mean: {mean}, Sigma: {stddev}")
                # Try to get from selected Voigt
                elif self.selected_voigt:
                    self.center_profile = self.selected_voigt.get('center')
                    self.center_profile_err = self.selected_voigt.get('center_err')
                    print(f"[DEBUG] Voigt center_err from fit dict: {self.center_profile_err}")
                    # Print the fitted parameters like in redshift mode
                    amp = self.selected_voigt.get('amp')
                    center = self.selected_voigt.get('center')
                    sigma = self.selected_voigt.get('sigma')
                    gamma = self.selected_voigt.get('gamma')
                    print(f"Amplitude: {amp}, Center: {center}, Sigma: {sigma}, Gamma: {gamma}")
                else:
                    print("Error: No Gaussian or Voigt selected for redshift estimation")
                    return
            
            # Print center info like in redshift mode
            print(f"Center of selected Gaussian: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
            
            est_redshift = (self.center_profile - self.selected_rest_wavelength) / self.selected_rest_wavelength
            est_redshift_err = self.center_profile_err / self.selected_rest_wavelength
            print(f"Estimated Redshift: {est_redshift:.6f}+-{est_redshift_err:.6f}")
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            data = {
                'z': [est_redshift],
                'z_err': [est_redshift_err],
                'line_id': [self.selected_id],
                'lambda_rest': [self.selected_rest_wavelength],
                'lambda_obs': [self.center_profile],
                'lambda_obs_err': [self.center_profile_err],
                'timestamp': [timestamp]
            }
            df = pd.DataFrame(data)
            filename = f"z_{est_redshift:.3f}_from{self.selected_rest_wavelength:.2f}_{timestamp}.csv"
            df.to_csv(filename, sep='\t', index=False, float_format='%.6f')
        else:
            print("No line selected.")
    
    def on_fit_info_item_selected(self, item_id):
        """Handle item selection from Fit Information window - sync with item tracker"""
        if item_id not in self.item_id_map:
            return
        
        # Call the same handler as item tracker selection
        self.on_item_selected_from_tracker(item_id)
        # Also highlight the corresponding item in the tracker
        self.item_tracker.highlight_item(item_id)
    
    def on_fit_info_item_deselected(self):
        """Handle item deselection from Fit Information window - sync with item tracker"""
        # Call the same handler as item tracker deselection
        self.on_item_deselected_from_tracker()

    def select_line_from_list():
        """
        Opens the line list window and returns the selected line's ID and wavelength.

        Returns:
            line_id (str): The ID of the selected line.
            line_wavelength (float): The wavelength of the selected line.
        """
        # Open the line list selection window
        linelist_window = self.open_linelist_window()

    def receive_voigt(self, selected_line_id, selected_wavelength):
        # Assign the selected line ID and wavelength to the selected Gaussian or Voigt profile
        self.selected_line_id = selected_line_id
        self.selected_line_wavelength = selected_wavelength
        print(f"Assigned line ID '{selected_line_id}' with wavelength {selected_wavelength} Å to profile.")
        left_bound, right_bound = self.current_bounds
        # Store bounds for component
        comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
        comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]

        # Use existing continuum if available, otherwise fit new continuum
        existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
        continuum_subtracted_y = comp_y - existing_continuum

        # Calculate distance from first line - to be used for relative wavelength constraint
        if self.voigt_comps:  # If this is not the first component
            distance = (selected_wavelength - self.voigt_comps[0]['line_wavelength']) * (1 + self.redshift) # Take the redshift into account when calculating the wavelength separation of the lines
        else:
            distance = 0

        # Get oscillator strength
        # Find the index where self.osc_id matches line_id and save this index to variable osc_idx
        osc_idx = [i for i, x in enumerate(self.osc_ids) if x == self.selected_line_id]
        if len(osc_idx) != 1:
            print("WARNING: The line list does not contain oscillator strengths for the line",self.selected_line_id)
        # Get the oscillator strength from self.osc_strength by getting self.osc_strength[osc_idx]
        osc_strength = self.osc_strengths[osc_idx][0]

        # Store parameters for each fit
        self.voigt_comps.append({
            'line_id': self.selected_line_id,
            'line_wavelength': self.selected_line_wavelength,
            'osc_strength': osc_strength,
            'distance': distance,
            'bounds': (left_bound, right_bound),
            'comp_id': len(self.voigt_comps) + 1,
            'comp_x': comp_x,
            'comp_y': comp_y,
            'existing_continuum': existing_continuum,
            'continuum_subtracted_y': continuum_subtracted_y
        })

    def receive_gaussian(self, selected_line_id, selected_wavelength):
        # Assign the selected line ID and wavelength to the selected Gaussian or Voigt profile
        self.selected_line_id = selected_line_id
        self.selected_line_wavelength = selected_wavelength
        print(f"Assigned line ID '{selected_line_id}' with wavelength {selected_wavelength} Å to profile.")
        left_bound, right_bound = self.current_bounds
        # Store bounds for component
        comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
        comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]

        # Use existing continuum if available, otherwise fit new continuum
        existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
        continuum_subtracted_y = comp_y - existing_continuum

        # Calculate distance from first line - to be used for relative wavelength constraint
        if self.gaussian_comps:  # If this is not the first component
            distance = (selected_wavelength - self.gaussian_comps[0]['line_wavelength']) * (1 + self.redshift) # Take the redshift into account when calculating the wavelength separation of the lines
        else:
            distance = 0

        # Get oscillator strength
        # Find the index where self.osc_id matches line_id and save this index to variable osc_idx
        osc_idx = [i for i, x in enumerate(self.osc_ids) if x == self.selected_line_id]
        if len(osc_idx) != 1:
            print("WARNING: The line list does not contain oscillator strengths for the line",self.selected_line_id)
        # Get the oscillator strength from self.osc_strength by getting self.osc_strength[osc_idx]
        osc_strength = self.osc_strengths[osc_idx][0]

        # Store parameters for each fit
        self.gaussian_comps.append({
            'line_id': self.selected_line_id,
            'line_wavelength': self.selected_line_wavelength,
            'osc_strength': osc_strength,
            'distance': distance,
            'bounds': (left_bound, right_bound),
            'comp_id': len(self.gaussian_comps) + 1,
            'comp_x': comp_x,
            'comp_y': comp_y,
            'existing_continuum': existing_continuum,
            'continuum_subtracted_y': continuum_subtracted_y
        })

    def plot_marker_and_label(self, profile_type, center_or_mean, line_id, bounds):
        # Get current y-axis limits and calculate the y-position at 3/4 of the plot height
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_pos_add = (x_max - x_min) * 0.02
        y_pos = y_min + 0.925 * (y_max - y_min)
        
        # Determine the marker color based on profile type
        marker_color = '#eca829' if profile_type == 'Voigt' else 'red'
        
        # Draw a vertical line as a marker at the specified position
        marker, = self.ax.plot(
            [center_or_mean, center_or_mean],
            [y_pos, y_pos + 0.05 * (y_max - y_min)],
            color=marker_color,
            lw=2
        )
        # Attach the bounds to the marker as a custom attribute
        setattr(marker, 'bounds', bounds)
        setattr(marker, 'center', center_or_mean)
        setattr(marker, 'line_id', line_id)
        self.markers.append(marker)  # Append marker to the list
        
        # Add a vertically oriented textbox for the line ID
        label = self.ax.text(
            center_or_mean + x_pos_add, y_pos - 0.10 * (y_max - y_min),
            line_id,
            color=marker_color,
            verticalalignment='center',
            horizontalalignment='center',
            rotation='vertical',
            # bbox=dict(facecolor='white', edgecolor=marker_color, boxstyle='round,pad=0.3')
        )
        # Attach the bounds to the label as a custom attribute
        setattr(label, 'bounds', bounds)
        setattr(label, 'center', center_or_mean)
        setattr(label, 'marker', marker)  # Link label to marker for removal
        self.labels.append(label)  # Append label to the list
        
        # Add marker to item tracker
        marker_id = f"marker_{len(self.markers)-1}_{line_id}"
        self.item_tracker.add_item(marker_id, 'marker', f'Marker: {line_id}', position=f'{center_or_mean:.2f} Å', color=marker_color, line_obj=marker)
        
        # Redraw plot to ensure the new marker and label are visible
        plt.draw()

    def assign_line_to_fit(self, selected_line_id, selected_wavelength):
        # Assign the selected line ID and wavelength to the selected Gaussian or Voigt profile
        if self.selected_gaussian:
            self.selected_gaussian['line_id'] = selected_line_id
            self.selected_gaussian['line_wavelength'] = selected_wavelength
            print(f"Assigned line ID '{selected_line_id}' with wavelength {selected_wavelength} Å to Gaussian.")
        if self.selected_voigt:
            self.selected_voigt['line_id'] = selected_line_id
            self.selected_voigt['line_wavelength'] = selected_wavelength
            print(f"Assigned line ID '{selected_line_id}' with wavelength {selected_wavelength} Å to Voigt.")
        
        # Plot and store the marker and label for the assigned line
        selected_profile = self.selected_gaussian if self.selected_gaussian else self.selected_voigt
        profile_type = 'Gaussian' if self.selected_gaussian else 'Voigt'
        center_or_mean = selected_profile['mean'] if self.selected_gaussian else selected_profile['center']
        bounds = selected_profile['bounds']
        line_id = selected_profile['line_id']
        self.plot_marker_and_label(profile_type, center_or_mean, line_id, bounds)
        
        # Record action
        self.record_action('add_marker', f'Add Marker: {line_id} at λ={selected_wavelength:.2f} Å')
        
        # Clear selection references
        self.selected_gaussian = None
        self.selected_voigt = None

    def update_marker_and_label_positions(self):
        # Get the updated y-axis limits
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_pos_add = (x_max - x_min) * 0.02
        y_pos = y_min + 0.925 * (y_max - y_min)  # Calculate new y position at 3/4 of the range

        # Update the y positions of each marker and label in the lists
        for marker in self.markers:
            x = getattr(marker, 'center')  # x position remains the same
            marker.set_ydata([y_pos, y_pos + 0.05 * (y_max - y_min)])  # Update y-data range for the marker

        for label in self.labels:
            x = getattr(label, 'center')  # x position remains the same
            label.set_position((x + x_pos_add, y_pos - 0.10 * (y_max - y_min)))  # Update y position of the label

        # Redraw plot to reflect the changes
        plt.draw()

    # Define a residuals function for lmfit.Minimizer
    def voigt_residuals(self, params, x_data, y_data, bound_pairs):
        model_total = np.zeros_like(x_data)
        
        # Loop through each Voigt component based on the bounds provided
        for idx, (left_bound, right_bound) in enumerate(bound_pairs):
            prefix = f"p{idx + 1}_"
            
            # Retrieve the component's parameters from params with the prefix
            amp = params[f"{prefix}amp"]
            center = params[f"{prefix}center"]
            sigma = params[f"{prefix}sigma"]
            gamma = params[f"{prefix}gamma"]
            
            # Apply bounds to get x and y data within the component’s range
            mask = (x_data >= left_bound) & (x_data <= right_bound)
            x_comp = x_data[mask]
            
            # Calculate the Voigt model values for this component
            y_model = self.voigt(x_comp, amp, center, sigma, gamma)
            
            # Add to the total model within the component’s range
            model_total[mask] += y_model
        
        # Calculate residuals
        return model_total - y_data

    # Main function to perform the fit
    def fit_voigt_profiles(self, x_data, y_data, bound_pairs):
        params = lmfit.Parameters()
        
        # Set up parameters for each Voigt profile component
        for idx, (left_bound, right_bound) in enumerate(bound_pairs):
            prefix = f"p{idx + 1}_"
            
            # Prepare data for fitting within the current bound
            comp_x = x_data[(x_data >= left_bound) & (x_data <= right_bound)]
            comp_y = y_data[(x_data >= left_bound) & (x_data <= right_bound)]
            
            # Define initial parameters for each Voigt profile component
            initial_amp = max(comp_y) - min(comp_y)
            initial_center = np.mean(comp_x)
            initial_sigma = np.std(comp_x) / 10
            initial_gamma = np.std(comp_x) / 10

            # Add parameters with unique prefixes
            params.add(f'{prefix}amp', value=initial_amp, min=0)
            params.add(f'{prefix}center', value=initial_center, min=min(comp_x), max=max(comp_x))
            params.add(f'{prefix}sigma', value=initial_sigma, min=0)
            params.add(f'{prefix}gamma', value=initial_gamma, min=0)
        
        # Initialize the Minimizer with the residuals function
        minimizer = lmfit.Minimizer(self.voigt_residuals, params, fcn_args=(x_data, y_data, bound_pairs))
        
        # Perform the minimization
        result = minimizer.minimize()
        print(result)
        
        # Access and print the confidence intervals for each parameter
        for idx, (left_bound, right_bound) in enumerate(bound_pairs):
            prefix = f"p{idx + 1}_"
        
        return result

    def column_density(self, flux_continuum, flux_line, f, lam, velocities):
        from scipy.integrate import simps
        c_in_km_per_s = 2.9979246e5 # Speed of light in km/s
        pie2_mec = 2.654e-15 # pi * e^2 / m_e * c (in cgs units)
        log_flux_ratio = np.log(flux_continuum / flux_line)
        integrated_flux = simps(log_flux_ratio, velocities)
        col_dens = (1 / (pie2_mec * f * lam)) * integrated_flux
        return col_dens # cm^{-2}

    def T_eff(self, b, m):
        k_in_km2_g_per_K_s2 = 1.380649e-26
        T_eff = b**2 * m / (2 * k_in_km2_g_per_K_s2)
        return T_eff

    def prompt_mask_ranges(self):
        print("Mask mode activated: press SPACE to select bounds to mask out regions.")
        print("Press RETURN when done masking.")
        self.mask_bounds = []
        self.mask_bound_lines = []
        self.mask_mode = True
        self.mask_temp = []  # Temporarily hold one pair
        self.mask_patches = []

        self.fig.canvas.mpl_connect('key_press_event', self.on_mask_keypress)

    def on_mask_keypress(self, event):
        if not self.mask_mode:
            return

        if event.key == ' ':
            if event.xdata is None:
                print("Click inside the plot area to define a mask region.")
                return

            if len(self.mask_temp) == 0 or self.mask_temp[-1][1] is not None:
                # Start a new mask region
                self.mask_temp.append([event.xdata, None])

                # Draw vertical line to indicate start
                line = self.ax.axvline(event.xdata, color='gray', linestyle='--')
                self.mask_bound_lines.append(line)
                self.fig.canvas.draw()

                print(f"Mask region start defined at: {event.xdata:.2f}. Press space again to set end.")
            else:
                # Complete the current region
                self.mask_temp[-1][1] = event.xdata
                line = self.ax.axvline(event.xdata, color='gray', linestyle='--')
                self.mask_bound_lines.append(line)
                self.fig.canvas.draw()
                x0, x1 = sorted(self.mask_temp[-1])
                left, right = self.bayes_bounds

                if x0 in self.bayes_bounds or x1 in self.bayes_bounds:
                    print("Skipping region that matches Bayesian fit bounds.")
                else:
                    self.mask_bounds.append((x0, x1))
                    patch = self.ax.axvspan(x0, x1, color='gray', alpha=0.3)
                    self.mask_patches.append({'patch': patch, 'bounds': (x0, x1)})
                    self.fig.canvas.draw()

                print(f"Mask region end defined at: {event.xdata:.2f}. Region: [{x0:.2f}, {x1:.2f}].")

        elif event.key == 'enter':
            self.mask_mode = False
            print(f"Finalized {len(self.mask_bounds)} mask regions.")

            # Apply the mask and continue
            mask_full = np.ones_like(self.wav, dtype=bool)
            for x0, x1 in self.mask_bounds:
                mask_full &= ~((self.wav > x0) & (self.wav < x1))

            left, right = self.bayes_bounds
            in_bounds = (self.wav > left) & (self.wav < right)
            final_mask = mask_full & in_bounds

            x = self.wav[final_mask]
            y = self.spec[final_mask]
            # Handle optional error spectrum in Bayesian fitting
            if self.err is not None:
                yerr = self.err[final_mask]
            else:
                yerr = np.ones_like(y) * np.std(y)

            _, _, _, poly_order, poly_guess = self._bayes_fit_args
            self.prompt_gaussian_selection(x, y, yerr, poly_order, poly_guess)



    # MCMC Bayesian Posterior Functionality
    def prompt_bayes_fit(self):
        left, right = self.bayes_bounds
        mask = (self.wav > left) & (self.wav < right)
        x = self.wav[mask]
        y = self.spec[mask]
        # Handle optional error spectrum in Bayesian fitting
        if self.err is not None:
            yerr = self.err[mask]
        else:
            yerr = np.ones_like(y) * np.std(y) 

        # Ask user for polynomial order using Qt dialog
        poly_order_str, ok = QtWidgets.QInputDialog.getText(
            self, 'Polynomial Order', 'Enter polynomial order for continuum:'
        )
        if not ok or not poly_order_str.strip():
            return
        try:
            poly_order = int(poly_order_str)
        except ValueError:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Please enter a valid integer')
            return

        # Get initial polynomial guess from existing continuum
        _, slope, intercept = self.get_existing_continuum(left, right)
        if poly_order == 1:
            poly_guess = [slope, intercept] # Guess for order 1
        else:
            poly_guess = [0.0] * (poly_order + 1)  # Initialize all coefficients to 0
            poly_guess[-1] = np.mean(y)      # Set the intercept

        # Ask user for mask region(s) using Qt dialog
        reply = QtWidgets.QMessageBox.question(
            self, 'Mask Regions?',
            'Do you want to mask out any regions before fitting?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            print("Click and press spacebar to define mask region(s), then press enter to finalize.")
            # Store inputs and delay fitting until after masking
            self._bayes_fit_args = (x, y, yerr, poly_order, poly_guess)
            self.prompt_mask_ranges()
            return  # Exit early — wait for user to finish masking
        else:
            # Proceed directly to Gaussian selection and fitting
            self.prompt_gaussian_selection(x, y, yerr, poly_order, poly_guess)

    def prompt_gaussian_selection(self, x, y, yerr, poly_order, poly_guess):
        reply = QtWidgets.QMessageBox.question(
            self, 'Manual Gaussian Guess?',
            'Do you want to enter a manual Gaussian guess?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            while True:
                # Get mean
                mean_str, ok = QtWidgets.QInputDialog.getText(
                    self, 'Gaussian Parameters', 'Enter central wavelength (mean):'
                )
                if not ok:
                    return
                try:
                    mean = float(mean_str)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid input. Please enter a number.')
                    continue

                # Get sigma
                sigma_str, ok = QtWidgets.QInputDialog.getText(
                    self, 'Gaussian Parameters', 'Enter sigma (stddev):'
                )
                if not ok:
                    return
                try:
                    sigma = float(sigma_str)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid input. Please enter a number.')
                    continue

                # Get amplitude
                amp_str, ok = QtWidgets.QInputDialog.getText(
                    self, 'Gaussian Parameters', 'Enter amplitude:'
                )
                if not ok:
                    return
                try:
                    amp = float(amp_str)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, 'Error', 'Invalid input. Please enter a number.')
                    continue

                # Plot the manual Gaussian guess
                # Compute Gaussian guess using self.wav
                gauss_full = amp * np.exp(-(self.wav - mean)**2 / (2 * sigma**2))

                # Apply bounds
                left, right = self.bayes_bounds
                in_bounds = (self.wav > left) & (self.wav < right)

                # Apply mask only if defined
                if hasattr(self, 'mask_bounds') and self.mask_bounds:
                    mask_full = np.ones_like(self.wav, dtype=bool)
                    for x0, x1 in self.mask_bounds:
                        mask_full &= ~((self.wav > x0) & (self.wav < x1))
                    final_mask = mask_full & in_bounds
                else:
                    final_mask = in_bounds

                # Use masked data for plotting
                x_plot = self.wav[final_mask]
                gauss_plot = gauss_full[final_mask]
                continuum_, slope, intercept = self.get_existing_continuum(left, right)
                continuum_full = self.continuum_model(self.wav, slope, intercept)
                continuum_plot = continuum_full[final_mask]

                # Back to plot
                temp_line, = self.ax.plot(x_plot, gauss_plot + continuum_plot, color='lightcoral')
                self.fig.canvas.draw()
                plt.pause(0.001)

                reply = QtWidgets.QMessageBox.question(
                    self, 'Confirm Gaussian',
                    'Proceed with this manual guess?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel
                )
                if reply == QtWidgets.QMessageBox.Yes:
                    gauss_guess = [amp, mean, sigma]
                    self.run_bayes_fit(x, y, yerr, poly_order, gauss_guess, poly_guess)
                    return
                elif reply == QtWidgets.QMessageBox.Cancel:
                    print("Manual input cancelled. Returning to click-based selection.")
                    temp_line.remove()
                    self.fig.canvas.draw()
                    break
                else:
                    print("Manual guess discarded. Please enter a new guess.")
                    temp_line.remove()
                    self.fig.canvas.draw()
        
        # Click-based selection if no manual guess provided
        print("Click on a known Gaussian profile within the bounds to select as an initial guess.")

        def on_click(event):
            x_pos = event.xdata
            for fit in self.gaussian_fits:
                l, r = fit['bounds']
                if l <= x_pos <= r:
                    print(f"Selected Gaussian: amp={fit['amp']}, mean={fit['mean']}, stddev={fit['stddev']}")
                    gauss_guess = [fit['amp'], fit['mean'], fit['stddev']]
                    self.fig.canvas.mpl_disconnect(cid)
                    self.run_bayes_fit(x, y, yerr, poly_order, gauss_guess, poly_guess)
                    for line in self.bayes_bound_lines:
                        line.remove()
                    self.bayes_bound_lines.clear()
                    self.mask_bound_lines.clear()
                    return
            print("No Gaussian found at clicked location.")

        cid = self.fig.canvas.mpl_connect('button_press_event', on_click)

    def run_bayes_fit(self, x, y, yerr, poly_order, gauss_guess, poly_guess):
        from datetime import datetime
        import emcee
        import corner

        def calculate_ew(x, model_flux, continuum_flux):
            """
            Calculate the Equivalent Width (EW) for a given model flux and continuum flux.
            EW = integral (1 - model_flux / continuum_flux) dx
            """
            continuum_flux = np.maximum(continuum_flux, 1e-10) # Ensure continuum flux is non-zero to avoid division by zero
            flux_diff = 1 - (model_flux / continuum_flux)
            # Compute the equivalent width by integrating over the wavelength range
            ew = np.trapz(flux_diff, x)
            return ew

        def model(x, amp, mu, sigma, *poly_coeffs):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + np.polyval(poly_coeffs, x)

        def log_likelihood(theta, x, y, yerr):
            amp, mu, sigma = theta[:3]
            poly = theta[3:]
            model_y = model(x, amp, mu, sigma, *poly)
            return -0.5 * np.sum(((y - model_y) / yerr) ** 2)

        def log_prior(theta):
            amp, mu, sigma = theta[:3]
            if not (-1e3 < amp < 1e3 and 0 < sigma < 100 and np.isfinite(mu)):
                return -np.inf
            return 0.0

        def log_prob(theta, x, y, yerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)

        # Mask?
        if self.mask_bounds:
            mask = np.ones_like(x, dtype=bool)
            for x0, x1 in self.mask_bounds:
                mask &= ~((x >= x0) & (x <= x1))
            x = x[mask]
            y = y[mask]
            yerr = yerr[mask]

        # Initial setup
        initial = gauss_guess + poly_guess
        ndim = len(initial)
        nwalkers = 50
        nsteps = 2000
        pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=(x, y, yerr))
        print("Running MCMC...")
        sampler.run_mcmc(pos, nsteps, progress=True)
        samples = sampler.get_chain(discard=int(0.2 * nsteps), flat=True)

        # Output
        mean_params = np.mean(samples, axis=0)
        amp, mu, sigma = mean_params[:3]
        poly = mean_params[3:]
        gauss = amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        poly_y = np.polyval(poly, x)
        total = gauss + poly_y
        # Chi2
        residuals = y - total
        chi2 = np.sum((residuals / yerr)**2)
        num_params = len(initial)
        dof = len(y) - num_params  # replace num_params with your model's number of free parameters
        chi2_nu = chi2 / dof

        now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        fname = f"bayes_{now}.txt"
        with open(fname, 'w') as f:
            f.write("# MCMC Posterior Means:\n")
            f.write(f"Amplitude: {amp:.4f}\nMean: {mu:.4f}\nSigma: {sigma:.4f}\n")
            f.write(f"Polynomial Coefficients: {poly.tolist()}\n")
            f.write(f"Chi2: {chi2:.4e}\n")
            f.write(f"Chi2_nu: {chi2_nu:.4e}\n")
            f.write("\n# Wavelength  TotalFit  Gaussian  Continuum\n")
            for xi, ti, gi, pi in zip(x, total, gauss, poly_y):
                f.write(f"{xi:.6f}  {ti:.6f}  {gi:.6f}  {pi:.6f}\n")
            print(f"Fit saved as {fname}.")

        # Print chi2 to terminal
        print(f"Chi2: {chi2:.4e}")
        print(f"Chi2_nu: {chi2_nu:.4e}\n")

        # Calculate equivalent width for each sample
        ew_samples = []
        for sample in samples:
            amp, mu, sigma = sample[:3]
            poly_coeffs = sample[3:]
            
            # Recompute the Gaussian and the total model for each sample
            gauss_sample = amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
            total_sample = gauss_sample + np.polyval(poly_coeffs, x)

            # Calculate the EW for the Gaussian and total model
            ew_gauss = calculate_ew(x, gauss_sample, poly_y)  # Gaussian EW
            ew_total = calculate_ew(x, total_sample, poly_y)  # Total model EW
            
            ew_samples.append((ew_gauss, ew_total))

        ew_samples = np.array(ew_samples)
        
        # Calculate the 2-sigma confidence intervals (16th, 50th, and 84th percentiles)
        ew_gauss_16th = np.percentile(ew_samples[:, 0], 16)
        ew_gauss_50th = np.percentile(ew_samples[:, 0], 50)
        ew_gauss_84th = np.percentile(ew_samples[:, 0], 84)
        # Rest EW
        z = self.redshift
        ew_gauss_50th_rest   = ew_gauss_50th / (1 + z)
        ew_gauss_16th_rest   = ew_gauss_16th / (1 + z)
        ew_gauss_84th_rest   = ew_gauss_84th / (1 + z)
        
        ew_total_16th = np.percentile(ew_samples[:, 1], 16)
        ew_total_50th = np.percentile(ew_samples[:, 1], 50)
        ew_total_84th = np.percentile(ew_samples[:, 1], 84)
        # Rest EW
        ew_total_50th_rest   = ew_total_50th / (1 + z)
        ew_total_16th_rest   = ew_total_16th / (1 + z)
        ew_total_84th_rest   = ew_total_84th / (1 + z)

        print(f"Gaussian EW 2σ confidence interval (obs): [{ew_gauss_16th:.4f}, {ew_gauss_50th:.4f}, {ew_gauss_84th:.4f}]")
        print(f"Total EW 2σ confidence interval (obs): [{ew_total_16th:.4f}, {ew_total_50th:.4f}, {ew_total_84th:.4f}]")
        print(f"Gaussian EW 2σ confidence interval (rest): [{ew_gauss_16th_rest:.4f}, {ew_gauss_50th_rest:.4f}, {ew_gauss_84th_rest:.4f}]")
        print(f"Total EW 2σ confidence interval (rest): [{ew_total_16th_rest:.4f}, {ew_total_50th_rest:.4f}, {ew_total_84th_rest:.4f}]")

        # Error bars
        ew_gauss_lo = ew_gauss_50th - ew_gauss_16th
        ew_gauss_hi = ew_gauss_84th - ew_gauss_50th

        ew_total_lo = ew_total_50th - ew_total_16th
        ew_total_hi = ew_total_84th - ew_total_50th
        print(f"Gaussian EW: {ew_gauss_50th:.4f} (+{ew_gauss_hi:.4f} / -{ew_gauss_lo:.4f})")
        print(f"Total EW: {ew_total_50th:.4f} (+{ew_total_hi:.4f} / -{ew_total_lo:.4f})")

        # More restframe EW calculations
        ew_gauss_50th_rest = ew_gauss_50th / (1 + z)
        ew_gauss_lo_rest   = (ew_gauss_50th - ew_gauss_16th) / (1 + z)
        ew_gauss_hi_rest   = (ew_gauss_84th - ew_gauss_50th) / (1 + z)

        ew_total_50th_rest = ew_total_50th / (1 + z)
        ew_total_lo_rest   = (ew_total_50th - ew_total_16th) / (1 + z)
        ew_total_hi_rest   = (ew_total_84th - ew_total_50th) / (1 + z)
        print(f"Gaussian Rest-frame EW: {ew_gauss_50th_rest:.4f} (+{ew_gauss_hi_rest:.4f} / -{ew_gauss_lo_rest:.4f})")
        print(f"Total Rest-frame EW: {ew_total_50th_rest:.4f} (+{ew_total_hi_rest:.4f} / -{ew_total_lo_rest:.4f})")

        # Optionally save results to file
        now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        fname = f"bayes_{now}_ew.txt"
        with open(fname, 'w') as f:
            f.write("# Equivalent Width Posterior Medians (Observed and Rest-frame)\n")
            f.write("Type,Frame,16th,50th,84th\n")
            f.write(f"Gaussian,Observed,{ew_gauss_16th:.4f},{ew_gauss_50th:.4f},{ew_gauss_84th:.4f}\n")
            f.write(f"Total,Observed,{ew_total_16th:.4f},{ew_total_50th:.4f},{ew_total_84th:.4f}\n")
            f.write(f"Gaussian,Rest-frame,{ew_gauss_16th_rest:.4f},{ew_gauss_50th_rest:.4f},{ew_gauss_84th_rest:.4f}\n")
            f.write(f"Total,Rest-frame,{ew_total_16th_rest:.4f},{ew_total_50th_rest:.4f},{ew_total_84th_rest:.4f}\n")

        print(f"MCMC fit complete. Results written to {fname}.")

        # Plot to see what was fitted
        n_plot = 100
        inds = np.random.choice(len(samples), size=n_plot, replace=False)
        for i in inds:
            amp_i, mu_i, sigma_i = samples[i][:3]
            poly_i = samples[i][3:]
            model_i = model(x, amp_i, mu_i, sigma_i, *poly_i)
            self.ax.plot(x, model_i, color='orange', alpha=0.1, lw=0.5)
        spec_line, = self.ax.step(x, y, color='lightblue', linestyle='-', where='mid')
        profile_line, = self.ax.plot(x, gauss, color='lightgreen', linestyle=':')
        poly_line, = self.ax.plot(x, poly_y, color='lightgreen', linestyle=':')
        total_line, = self.ax.plot(x, total, color='green', linestyle=':')
        self.fig.canvas.draw_idle()

        # Dynamically construct parameter labels
        raw_labels = []

        n_gaussians = 1  # For future support of multiple Gaussians
        for i in range(n_gaussians):
            suffix = f"_{i+1}" if n_gaussians > 1 else ""
            raw_labels += [f"amp{suffix}", f"mu{suffix}", f"sigma{suffix}"]

        # Polynomial coefficients (highest degree first)
        for i in reversed(range(len(poly_guess))):
            raw_labels.append(f"coeff_{i}")

        # Mapping raw parameter names to LaTeX-formatted labels
        latex_labels = []
        for label in raw_labels:
            if label.startswith("amp"):
                idx = label[3:]  # get suffix
                latex_labels.append(rf"$A{idx}$" if idx else r"$A$")
            elif label.startswith("mu"):
                idx = label[2:]
                latex_labels.append(rf"$\mu{idx}$" if idx else r"$\mu$")
            elif label.startswith("sigma"):
                idx = label[5:]
                latex_labels.append(rf"$\sigma{idx}$" if idx else r"$\sigma$")
            elif label.startswith("coeff_"):
                degree = int(label.split("_")[1])
                latex_labels.append(rf"$c_{{{degree}}}$")
            else:
                latex_labels.append(label)  # fallback

        # Make corner plot
        low_thresh, high_thresh = 1e-2, None # Thresholds for switching to scientific notation
        # flat_samples = sampler.get_chain(discard=int(0.2 * nsteps), flat=True)
        intervals = np.percentile(samples, [5, 16, 50, 84, 95], axis=0)
        best_fit = intervals[2]  # 50th percentile as best fit
        fig = corner.corner(
            samples,
            truths=truths if 'truths' in locals() else None,
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            quantiles=[0.05, 0.16, 0.5, 0.84, 0.95],  # 1-sigma, 2-sigma contours
            plot_density=True,  # Show the density contours
            levels=[0.68, 0.95],  # 1-sigma, 2-sigma confidence levels
            labels=latex_labels
        )
        # Change to only get "helpful" labels - not 0.00 +- 0.00 - so change to scientific notation
        axes = np.array(fig.axes).reshape((ndim, ndim))
        for i in range(ndim):
            ax = axes[i, i]
            mean = best_fit[i]
            lower = best_fit[i] - intervals[1][i]  # 16th percentile
            upper = intervals[3][i] - best_fit[i]  # 84th percentile
            # Decide whether to use scientific notation
            use_sci = (low_thresh is not None and abs(mean) < low_thresh) or \
                    (high_thresh is not None and abs(mean) > high_thresh)
            if use_sci:
                mean_str = f"{mean:.2e}"
                upper_str = f"{upper:.2e}"
                lower_str = f"{lower:.2e}"
            else:
                mean_str = f"{mean:.2f}"
                upper_str = f"{upper:.2f}"
                lower_str = f"{lower:.2f}"
            title_text = f"{mean_str}$^{{+{upper_str}}}_{{-{lower_str}}}$"
            ax.set_title(title_text, fontsize=12)
        # Customize
        ndim = samples.shape[1]  # Number of parameters
        for row in range(ndim):
            for col in range(ndim):
                ax_idx = row * ndim + col
                ax = fig.axes[ax_idx]

                # Get the contour collections
                contours = ax.collections
                for contour in contours:
                    paths = contour.get_paths()
                    if not paths:
                        continue  # Skip if there are no paths
                    y_mean = paths[0].vertices[:, 1].mean()
                    if np.isclose(y_mean, 0.68, atol=0.01):  # 1-sigma
                        contour.set_color('green')
                    elif np.isclose(y_mean, 0.95, atol=0.01):  # 2-sigma
                        contour.set_color('lightblue')

                # Add vertical/horizontal lines at the mean
                if col < ndim and row >= col:
                    ax.axvline(intervals[2][col], color='red', linestyle='--', lw=1.5)
                if row < ndim and row != col and row >= col:
                    ax.axhline(intervals[2][row], color='red', linestyle='--', lw=1.5)

                # Add best-fit point (only on off-diagonal plots)
                if row != col and row > col:
                    ax.scatter(
                        best_fit[col], best_fit[row],
                        color='darkred', marker='s', s=100, edgecolor='black', zorder=5
                    )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"corner_{timestamp}.pdf"
        fig.subplots_adjust(hspace=0.6, wspace=0.6)
        plt.tight_layout()
        fig.savefig(filename, bbox_inches='tight')
        print(f"Corner plot saved to {filename}.")
        plt.show()
        
        # Record action for undo/redo
        self.record_action('perform_bayesian_fit', f'Perform Bayesian MCMC Fit')
        
        self.bayes_mode = False
        print("Exiting Bayes fit mode")

    # Interactive functions
    def command_listener(self):
        while True:
            command = input("Enter command (type 'quit' to exit): ")
            if command.strip().lower() == "quit":
                print("Quitting the application...")
                plt.close(self.fig)
                sys.exit()
            elif command.strip().lower() == "save":
                self.save_current_plot()
            else:
                print(f"Unknown command: {command}")

    def on_mouse_move(self, event):
        # Check if the cursor is within the axes bounds
        if event.inaxes == self.ax:
            self.x_lower_bound, self.x_upper_bound = self.ax.get_xlim()
            self.y_lower_bound, self.y_upper_bound = self.ax.get_ylim()
            if not (self.x_lower_bound <= event.xdata <= self.x_upper_bound and
                    self.y_lower_bound <= event.ydata <= self.y_upper_bound):
                return  # Exit if the cursor is outside the plot area

    def capture_state(self):
        """Capture current state for undo/redo"""
        # Extract only essential data from listfit_fits (exclude unpicklable result objects)
        listfit_fits_simplified = []
        for fit in self.listfit_fits:
            simplified_fit = {
                'bounds': fit.get('bounds'),
                'x_data': deepcopy(fit.get('x_data')) if fit.get('x_data') is not None else None,
                'y_data': deepcopy(fit.get('y_data')) if fit.get('y_data') is not None else None,
                'err_data': deepcopy(fit.get('err_data')) if fit.get('err_data') is not None else None,
            }
            listfit_fits_simplified.append(simplified_fit)
        
        # Extract only bounds from continuum_patches (can't deepcopy matplotlib patches)
        continuum_patches_simplified = []
        for patch_info in self.continuum_patches:
            continuum_patches_simplified.append({
                'bounds': patch_info.get('bounds')
            })
        
        return {
            'gaussian_fits': deepcopy(self.gaussian_fits),
            'voigt_fits': deepcopy(self.voigt_fits),
            'continuum_fits': deepcopy(self.continuum_fits),
            'continuum_regions': deepcopy(self.continuum_regions),
            'continuum_patches': continuum_patches_simplified,
            'listfit_fits': listfit_fits_simplified,
            'redshift': self.redshift,
            'fit_id': self.fit_id,
            'component_id': self.component_id,
        }
    
    def record_action(self, action_type, description):
        """Record an action in the history"""
        state = self.capture_state()
        self.action_history.record_action(action_type, description, state)
        self.action_history_window.refresh_display()
        self.update_undo_redo_buttons()
    
    def restore_state(self, state):
        """Restore a previously captured state"""
        if not state:
            return
        
        # Save current view bounds BEFORE any changes
        current_xlim = self.ax.get_xlim() if self.ax is not None else None
        current_ylim = self.ax.get_ylim() if self.ax is not None else None
        
        # Remove only fit lines from the axes, NOT the spectrum line
        if self.ax is not None:
            # Get all lines and identify which ones to keep
            lines_to_remove = []
            for line in self.ax.get_lines():
                # Keep the spectrum line (stored in self.spectrum_line and self.line_spec/self.step_spec)
                if line not in [self.spectrum_line, getattr(self, 'line_spec', None), getattr(self, 'step_spec', None)]:
                    # Keep error lines too
                    if line not in [getattr(self, 'line_error', None), getattr(self, 'step_error', None)]:
                        lines_to_remove.append(line)
            
            # Remove only the fit lines
            for line in lines_to_remove:
                line.remove()
            
            # Remove all continuum patches
            for patch in self.ax.patches[:]:
                patch.remove()
        
        # Restore state
        self.gaussian_fits = deepcopy(state.get('gaussian_fits', []))
        self.voigt_fits = deepcopy(state.get('voigt_fits', []))
        self.continuum_fits = deepcopy(state.get('continuum_fits', []))
        self.continuum_regions = deepcopy(state.get('continuum_regions', []))
        self.continuum_patches = state.get('continuum_patches', [])  # Don't deepcopy - will recreate patches below
        self.listfit_fits = deepcopy(state.get('listfit_fits', []))
        self.redshift = state.get('redshift', 0.0)
        self.fit_id = state.get('fit_id', 0)
        self.component_id = state.get('component_id', 0)
        
        # Clear tracking sets
        self.deleted_listfit_polynomials.clear()
        
        # Clear item tracker
        self.item_tracker.clear_all()
        self.fit_information_window.clear_all()
        self.item_id_map.clear()
        
        # Redraw all fits on existing axes
        if self.ax is not None:
            # Re-draw continuum region patches with proper styling and register with item tracker
            for patch_info in self.continuum_patches:
                bounds = patch_info.get('bounds')
                if bounds:
                    # Recreate the axvspan patch with magenta color and hatching (same as original)
                    patch = self.ax.axvspan(bounds[0], bounds[1], color='magenta', alpha=0.3, hatch='//')
                    # Store the patch object back in the patch_info dict
                    patch_info['patch'] = patch
                    # Register the region patch with ItemTracker
                    position_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å"
                    self.register_item('continuum_region', f'Continuum Region', patch_obj=patch, 
                                     position=position_str, color='magenta', bounds=bounds)
            
            # Re-plot all continuum fits
            for fit in self.continuum_fits:
                if 'line' in fit and fit['line'] is not None:
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color='magenta', linestyle='--', linewidth=1.5, label='Continuum')
                    # Re-register with tracker
                    bounds = fit.get('bounds')
                    bounds_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å" if bounds else "Continuum"
                    self.register_item('continuum', f'Continuum (order {fit.get("poly_order", 1)})', 
                                     fit_dict=fit, line_obj=fit['line'], 
                                     position=bounds_str, color='magenta')
            
            # Re-plot all Gaussian fits
            for fit in self.gaussian_fits:
                if 'line' in fit and fit['line'] is not None:
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color='red', linestyle='--', linewidth=1.5, label='Gaussian')
                    # Re-register with tracker
                    mean = fit.get('mean', 0)
                    position_str = f"λ: {mean:.2f} Å"
                    self.register_item('gaussian', 'Gaussian', fit_dict=fit, 
                                     line_obj=fit['line'], position=position_str, color='red')
            
            # Re-plot all Voigt fits
            for fit in self.voigt_fits:
                if 'line' in fit and fit['line'] is not None:
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color='orange', linestyle='--', linewidth=1.5, label='Voigt')
                    # Re-register with tracker
                    center = fit.get('center', fit.get('mean', 0))
                    position_str = f"λ: {center:.2f} Å"
                    self.register_item('voigt', 'Voigt', fit_dict=fit, 
                                     line_obj=fit['line'], position=position_str, color='orange')
            
            # Re-plot total line if it was shown
            if self.show_total_line and (self.gaussian_fits or self.voigt_fits or self.continuum_fits or self.listfit_fits):
                self.draw_total_line()
            
            # Restore view bounds - preserve the current zoom level
            if current_xlim is not None:
                self.ax.set_xlim(current_xlim)
            if current_ylim is not None:
                self.ax.set_ylim(current_ylim)
            
            # Redraw canvas
            self.ax.figure.canvas.draw_idle()
        
        self.update_undo_redo_buttons()
    
    def update_undo_redo_buttons(self):
        """Update the enabled state of undo/redo buttons"""
        if hasattr(self, 'undo_button'):
            self.undo_button.setEnabled(self.action_history.can_undo())
        if hasattr(self, 'redo_button'):
            self.redo_button.setEnabled(self.action_history.can_redo())
    
    def on_undo(self):
        """Perform undo action"""
        state = self.action_history.undo()
        if state:
            self.restore_state(state)
            self.action_history_window.refresh_display()
    
    def on_redo(self):
        """Perform redo action"""
        state = self.action_history.redo()
        if state:
            self.restore_state(state)
            self.action_history_window.refresh_display()
    
    def on_action_selected(self, index):
        """User selected an action from history window"""
        state = self.action_history.goto_action(index)
        if state:
            self.restore_state(state)
            self.action_history_window.refresh_display()

    def keyPressEvent(self, event):
        """Handle key press events - check for undo/redo shortcuts and quit first"""
        if isinstance(event, QKeyEvent):
            # Check for Ctrl+Z (undo) or Cmd+Z (undo) on macOS
            if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.MetaModifier):
                # Check if Shift is also pressed (for Redo)
                if event.modifiers() & Qt.ShiftModifier:
                    self.on_redo()
                else:
                    self.on_undo()
                return
            
            # Check for 'q' or 'Q' to quit
            if event.key() == Qt.Key_Q:
                self.quit_application()
                return
        
        self.on_key(event)

    def update_total_line_if_shown(self):
        """Redraw the total line if it's currently displayed. Called when fits change."""
        if self.show_total_line and (self.continuum_fits or self.voigt_fits or self.gaussian_fits or self.listfit_fits):
            # Remove existing total line
            total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total Line"]
            for line in total_lines:
                line.remove()
            
            # Redraw the total line
            self.draw_total_line()
            self.ax.figure.canvas.draw()
    
    def draw_total_line(self):
        """Draw the total line from all fitted profiles."""
        # Generate x values for plotting
        x_plot = np.linspace(self.x_data.min(), self.x_data.max(), 10000)

        # Combine continuum fits
        total_continuum = np.zeros_like(x_plot)
        for fit in self.continuum_fits:
            left_bound, right_bound = fit['bounds']
            mask = (x_plot >= left_bound) & (x_plot <= right_bound)
            if mask.any():
                coeffs = fit['coeffs']
                total_continuum[mask] += np.polyval(coeffs, x_plot[mask])
        
        # Combine Gaussian fits
        total_gaussian = np.zeros_like(x_plot)
        for fit in self.gaussian_fits:
            fit_x = fit['line'].get_xdata()
            fit_y = fit['line'].get_ydata()
            left_bound, right_bound = min(fit_x), max(fit_x)
            
            # Get continuum coefficients and interpolate to fit line's x values
            existing_continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
            if existing_continuum_vals is not None:
                # Interpolate continuum to fit line's x values
                continuum_interp = interp1d(self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)], 
                                           existing_continuum_vals, bounds_error=False, fill_value='extrapolate')
                existing_continuum = continuum_interp(fit_x)
            else:
                existing_continuum = np.zeros_like(fit_y)
            
            if fit['is_velocity_mode'] and self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit['is_velocity_mode'] and not self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit['is_velocity_mode'] and self.is_velocity_mode:
                profile_interp = interp1d(self.wav_to_vel(fit_x, fit['rest_wavelength'], z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif fit['is_velocity_mode'] and not self.is_velocity_mode:
                profile_interp = interp1d(self.vel_to_wav(fit_x, fit['rest_wavelength'], z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            total_gaussian += profile_interp(x_plot)

        # Combine Voigt fits
        total_voigt = np.zeros_like(x_plot)
        for fit in self.voigt_fits:
            fit_x = fit['line'].get_xdata()
            fit_y = fit['line'].get_ydata()
            left_bound, right_bound = min(fit_x), max(fit_x)
            
            # Get continuum coefficients and interpolate to fit line's x values
            existing_continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
            if existing_continuum_vals is not None:
                # Interpolate continuum to fit line's x values
                continuum_interp = interp1d(self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)], 
                                           existing_continuum_vals, bounds_error=False, fill_value='extrapolate')
                existing_continuum = continuum_interp(fit_x)
            else:
                existing_continuum = np.zeros_like(fit_y)
            
            if fit['is_velocity_mode'] and self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit['is_velocity_mode'] and not self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit['is_velocity_mode'] and self.is_velocity_mode:
                profile_interp = interp1d(self.wav_to_vel(fit_x, fit['rest_wavelength'], z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif fit['is_velocity_mode'] and not self.is_velocity_mode:
                profile_interp = interp1d(self.vel_to_wav(fit_x, fit['rest_wavelength'], z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            total_voigt += profile_interp(x_plot)

        # Combine Listfit fits - extract and add all components including polynomials
        total_listfit = np.zeros_like(x_plot)
        for fit in self.listfit_fits:
            left_bound, right_bound = fit.get('bounds', (0, 0))
            mask = (x_plot >= left_bound) & (x_plot <= right_bound)
            if mask.any():
                components = fit.get('components', [])
                result = fit.get('result')
                
                # Calculate Gaussian components from the listfit
                gauss_count = 0
                for comp in components:
                    if comp['type'] == 'gaussian':
                        prefix = f'g{gauss_count}_'
                        g_amp = result.params[f'{prefix}amp'].value
                        g_mean = result.params[f'{prefix}mean'].value
                        g_stddev = result.params[f'{prefix}stddev'].value
                        y_gauss = self.gaussian(x_plot[mask], g_amp, g_mean, g_stddev)
                        total_listfit[mask] += y_gauss
                        gauss_count += 1
                
                # Calculate Voigt components from the listfit
                voigt_count = 0
                for comp in components:
                    if comp['type'] == 'voigt':
                        prefix = f'v{voigt_count}_'
                        v_amp = result.params[f'{prefix}amp'].value
                        v_center = result.params[f'{prefix}center'].value
                        v_sigma = result.params[f'{prefix}sigma'].value
                        v_gamma = result.params[f'{prefix}gamma'].value
                        y_voigt = self.voigt(x_plot[mask], v_amp, v_center, v_sigma, v_gamma)
                        total_listfit[mask] += y_voigt
                        voigt_count += 1
                
                # Calculate polynomial components from the listfit
                poly_count = 0
                for comp in components:
                    if comp['type'] == 'polynomial':
                        order = comp.get('order', 1)
                        prefix = f'p{poly_count}_'
                        poly_coeffs = []
                        for i in range(order + 1):
                            coeff_key = f'{prefix}c{i}'
                            if coeff_key in result.params:
                                coeff_val = result.params[coeff_key].value
                                poly_coeffs.append(coeff_val)
                        # Reverse coefficients for np.polyval (expects highest order first)
                        if poly_coeffs:
                            poly_coeffs = poly_coeffs[::-1]
                            y_poly = np.polyval(poly_coeffs, x_plot[mask])
                            total_listfit[mask] += y_poly
                        poly_count += 1

        total_y = total_continuum + total_gaussian + total_voigt + total_listfit

        # Plot the total line
        self.ax.plot(x_plot, total_y, label="Total Line", color='#0ed8ca', linestyle='-')
        self.ax.legend(loc='upper right')

    # Function for handling key events
    def on_key(self, event):

        # Check for undo/redo shortcuts (Cmd+Z for undo, Cmd+Shift+Z for redo)
        if hasattr(event, 'key') and event.key is not None:
            # Check for undo: Cmd+Z (macOS) or Ctrl+Z (all platforms)
            if event.key == 'ctrl+z' or event.key == 'cmd+z':
                self.on_undo()
                return
            # Check for redo: Cmd+Shift+Z (macOS) or Ctrl+Shift+Z (all platforms)
            if event.key == 'ctrl+shift+z' or event.key == 'cmd+shift+z':
                self.on_redo()
                return
            
            # Quit application with 'q' or 'Q' key - handle this FIRST to override matplotlib's default
            if event.key == 'q' or event.key == 'Q':
                self.quit_application()
                return

        # Check if the cursor is within the axes bounds
        if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
            if event.xdata is not None and event.ydata is not None:
                if not (self.x_lower_bound <= event.xdata <= self.x_upper_bound and self.y_lower_bound <= event.ydata <= self.y_upper_bound):
                    return  # Exit if the cursor is outside the plot area

        # Show/hide item tracker with 'j' key
        if event.key == 'j':
            if self.item_tracker.isVisible():
                self.item_tracker.hide()
            else:
                self.show_item_tracker()

        # Show/hide fit information window with 'K' key (uppercase)
        if event.key == 'K':
            if self.fit_information_window.isVisible():
                self.fit_information_window.hide()
            else:
                self.fit_information_window.show()

        # Show help window with '?' key
        if event.key == '?':
            if self.help_window is None:
                self.help_window = HelpWindow(self)
            self.help_window.show()
            self.help_window.raise_()
            self.help_window.activateWindow()

        # Toggle residual panel
        if event.key == 'r':
            self.toggle_residual_panel()

        # Toggle plot style with '~' key
        if event.key == '~':
            self.is_step_plot = not self.is_step_plot
            self.step_spec.set_visible(self.is_step_plot)
            self.line_spec.set_visible(not self.is_step_plot)
            # Only toggle error visibility if error lines exist
            if self.step_error is not None:
                self.step_error.set_visible(self.is_step_plot)
            if self.line_error is not None:
                self.line_error.set_visible(not self.is_step_plot)

            # Update references to the currently visible lines
            self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
            if self.step_error is not None:
                self.error_line = self.step_error if self.is_step_plot else self.line_error

            plt.draw()  # Redraw to reflect changes
            print("Plot style toggled:", "Step plot" if self.is_step_plot else "Line plot")

        if event.key == 'v':  # Use 'v' key to calculate equivalent width
            x_pos = event.xdata  # Get x position of mouse click

            # Find the Gaussian fit corresponding to the selected x position
            selected_gaussian = None
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    selected_gaussian = fit
                    print(f"Gaussian with parameters amp:{selected_gaussian['amp']}, mean: {selected_gaussian['mean']}, stddev: {selected_gaussian['stddev']} selected.")
                    break  # Select only the first Gaussian fit found within the bounds

            # OR find the Voigt fit corresponding to the selected x position
            selected_voigt = None
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    selected_voigt = fit
                    print(f"Voigt with parameters amp: {selected_voigt['amp']}, center: {selected_voigt['center']}, sigma: {selected_voigt['sigma']}, gamma: {selected_voigt['gamma']}  selected.")
                    break  # Select only the first Gaussian fit found within the bounds
            
            if selected_gaussian or selected_voigt:
                if selected_gaussian:
                    # Get Gaussian parameters for the selected fit
                    bounds = selected_gaussian['bounds']
                    amp = selected_gaussian['amp']
                    mean = selected_gaussian['mean']
                    stddev = selected_gaussian['stddev']
                    
                    # Define the Gaussian function based on selected parameters
                    gaussian_function = lambda x: self.gaussian(x, amp, mean, stddev)
                if selected_voigt:
                    # Get Voigt parameters for the selected fit
                    bounds = selected_voigt['bounds']
                    amp = selected_voigt['amp']
                    center = selected_voigt['center']
                    gamma = selected_voigt['gamma']
                    sigma = selected_voigt['sigma']
                    
                    # Define the Voigt function based on selected parameters
                    voigt_function = lambda x: self.voigt(x, amp, center, sigma, gamma)

                # Retrieve the fitted continuum over the Gaussian's bounds
                continuum_within_bounds = self.get_existing_continuum(bounds[0], bounds[1])

                if continuum_within_bounds is not None:
                    # Extract continuum values and parameters from returned data
                    _, a, b = continuum_within_bounds

                    # Calculate Equivalent Width
                    if selected_gaussian:
                        ew = self.calculate_equivalent_width(gaussian_function, (a, b), bounds)
                        # Plot the filled area between Gaussian and continuum
                        x_fill = np.linspace(bounds[0], bounds[1], 100)
                        y_gaussian = gaussian_function(x_fill)
                        y_continuum = self.continuum_model(x_fill, a, b)
                    if selected_voigt:
                        ew = self.calculate_equivalent_width(voigt_function, (a, b), bounds)
                        # Plot the filled area between Gaussian and continuum
                        x_fill = np.linspace(bounds[0], bounds[1], 100)
                        y_gaussian = voigt_function(x_fill)
                        y_continuum = self.continuum_model(x_fill, a, b)
                    
                    # Remove the previous fill region if it exists
                    if self.ew_fill:
                        self.ew_fill.remove()
                    
                    # Create a new fill region and store its reference
                    self.ew_fill = self.ax.fill_between(x_fill, y_gaussian + y_continuum, y_continuum, color='cyan', alpha=0.7)
                    plt.draw()  # Redraw plot to show the filled area
                else:
                    print("No continuum specified. Unable to calculate EW.")

        if event.key == 'm':
            if self.continuum_mode:
                # Already in continuum mode, so pressing 'm' again exits it
                self.continuum_mode = False
                self.label_poly_order.hide()
                self.input_poly_order.hide()
                self.separator_line_poly.hide()
                self.setGeometry(100, 100, 440, 200)  # Restore original window size
                self.continuum_regions = []  # Clear any defined regions
                print('Exiting continuum mode.')
            else:
                # Enter continuum mode
                self.continuum_mode = True
                self.label_poly_order.show()
                self.input_poly_order.show()
                self.separator_line_poly.show()
                self.setGeometry(100, 100, 440, 250)  # Expand window to fit poly order field
                print("Continuum fitting mode: Use the spacebar to define regions.")
                print(f"Current polynomial order: {self.poly_order}")

        if event.key == 'enter' and self.continuum_mode:
            # Update polynomial order from input field
            try:
                self.poly_order = int(self.input_poly_order.text())
                print(f"Using polynomial order: {self.poly_order}")
            except ValueError:
                print("Invalid polynomial order, using default order 1")
                self.poly_order = 1
            
            # Combine all defined regions into a single dataset for fitting
            combined_wav = []
            combined_spec = []
            combined_err = []

            for region in self.continuum_regions:
                start, end = region
                # Find indices for the selected range
                mask = (self.x_data >= start) & (self.x_data <= end)
                if np.any(mask):
                    combined_wav.extend(self.x_data[mask])
                    combined_spec.extend(self.spec[mask])
                    if self.err is not None:
                        combined_err.extend(self.err[mask])

            # Make combined region to define the outer x-bounds of the continuum fit
            region_bounds = (min(region[0] for region in self.continuum_regions), max(region[1] for region in self.continuum_regions))

            # Convert to numpy arrays
            combined_wav = np.array(combined_wav)
            combined_spec = np.array(combined_spec)
            combined_err = np.array(combined_err) if combined_err else None

            # Fit the continuum using the combined data
            continuum, coeffs, perr = self.fit_continuum(combined_wav, combined_spec, combined_err, poly_order=self.poly_order)
            
            # Print fit parameters
            poly_str = f"Continuum fit (order {self.poly_order}):"
            for i, (coeff, err) in enumerate(zip(coeffs, perr)):
                poly_str += f" c{i}={coeff:.6e}±{err:.6e}"
            print(poly_str)

            # Plot the fitted continuum only between the minimum and maximum continuum region wavelengths
            # Create wavelength array between region bounds for plotting
            x_plot = np.linspace(region_bounds[0], region_bounds[1], 500)
            continuum_full = np.polyval(coeffs, x_plot)
            continuum_line, = self.ax.plot(x_plot, continuum_full, color='magenta', linestyle='--', alpha=0.8)
            if self.is_residual_shown:
                self.calculate_and_plot_residuals()
            plt.legend()
            # Force immediate redraw of the canvas
            self.ax.figure.canvas.draw()
            QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
            print("Fitting completed for defined regions.")
            # Add continuum fit
            continuum_fit = {
                'bounds': region_bounds,  # Tuple (left_bound, right_bound)
                'coeffs': coeffs,               # Polynomial coefficients
                'coeffs_err': perr,             # Coefficient errors
                'poly_order': self.poly_order,  # Polynomial order
                'patches': self.continuum_patches,    # The patches object is for plotting
                'line': continuum_line,            
                'is_velocity_mode': self.is_velocity_mode
            }
            self.continuum_fits.append(continuum_fit)
            # Register with ItemTracker
            bounds_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
            self.register_item('continuum', f'Continuum (order {self.poly_order})', fit_dict=continuum_fit,
                             line_obj=continuum_line, position=bounds_str, color='magenta')
            
            # Record action for undo/redo
            self.record_action('fit_continuum', f'Fit Continuum (order {self.poly_order})')
            
            self.continuum_regions = [] # Clear continuum_regions
            self.continuum_mode = False # Exit continuum mode
            self.label_poly_order.hide()
            self.input_poly_order.hide()
            self.setGeometry(100, 100, 440, 200)  # Restore original window size
            print('Exiting continuum mode.')
        elif event.key == ' ' and self.continuum_mode:
            # Capture current mouse x-coordinate for regions
            if len(self.continuum_regions) == 0 or self.continuum_regions[-1][1] is not None:
                # Start a new region
                self.continuum_regions.append((event.xdata, None))  # Add start point
                print(f"Region start defined at: {event.xdata:.2f}. Press space again to set end or enter to finalize.")
            else:
                # Set the end point for the last region
                self.continuum_regions[-1] = (self.continuum_regions[-1][0], event.xdata)  # Update end point
                print(f"Region end defined at: {event.xdata:.2f}. Press space to define another region or enter to finalize.")
                # Plot the region as a shaded patch
                patch = self.ax.axvspan(self.continuum_regions[-1][0], event.xdata, color='magenta', alpha=0.3, hatch='//')
                region_bounds = (self.continuum_regions[-1][0], event.xdata)
                self.continuum_patches.append({'patch': patch, 'bounds': region_bounds}) # Store the patch
                # Register the region patch with ItemTracker
                position_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
                self.register_item('continuum_region', f'Continuum Region', patch_obj=patch, 
                                 position=position_str, color='magenta', bounds=region_bounds)
                # Record action for defining a continuum region
                self.record_action('define_continuum_region', f'Define Continuum Region λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å')
                # self.continuum_patches.append(patch) # Store the patch
                self.fig.canvas.draw_idle()  # Update plot with the new region
        # Remove continuum region
        if event.key == 'M':
            # Check each continuum fit to see if the mouse is over it
            for fit in self.continuum_fits:
                region_bounds = fit['bounds']
                left_bound, right_bound = region_bounds
                if left_bound <= event.xdata <= right_bound:
                    continuum_line = fit.get('line')
                    if continuum_line:
                        continuum_line.remove()
                        print(f"Removed continuum line {continuum_line} in range: {region_bounds}")
                    
                    # Remove from item tracker - find items matching this line
                    item_ids_to_remove = []
                    for item_id, item_info in self.item_id_map.items():
                        if item_info.get('line_obj') == continuum_line or item_info.get('fit_dict') == fit:
                            item_ids_to_remove.append(item_id)
                    for item_id in item_ids_to_remove:
                        self.unregister_item(item_id)
                    
                    # Remove the fit from the list
                    self.continuum_fits.remove(fit)
                    plt.draw()  # Redraw the plot after removal
                    break  # Exit the loop after the first match
            for patch_info in self.continuum_patches:
                region_bounds = patch_info['bounds']
                left_bound, right_bound = region_bounds
                if left_bound <= event.xdata <= right_bound:
                    # Remove the corresponding continuum patch
                    patch = patch_info['patch']
                    if patch in self.ax.patches:
                        patch.remove()  # Remove the patch from the plot
                        print(f"Removed continuum patch {patch} in range: {region_bounds}")
                    
                    # Remove from item tracker - find items matching this patch
                    item_ids_to_remove = []
                    for item_id, item_info in self.item_id_map.items():
                        if item_info.get('patch_obj') == patch:
                            item_ids_to_remove.append(item_id)
                    for item_id in item_ids_to_remove:
                        self.unregister_item(item_id)
                    
                    # Remove the entry from the list of patches
                    self.continuum_patches.remove(patch_info)
                    plt.draw()  # Redraw the plot after removal
                    break  # Exit the loop after the first match

        # Listfit mode - activate with 'H' key
        if event.key == 'H':
            self.listfit_mode = True
            self.listfit_bounds = []
            self.listfit_bound_lines = []
            self.listfit_components = []
            print("Listfit mode: Use the spacebar to define left and right boundaries.")

        # Set Listfit bounds with space bar
        if event.key == ' ' and self.listfit_mode:
            if len(self.listfit_bounds) == 0:
                # Start a new region
                # If in velocity mode, convert velocity back to wavelength for processing
                bound_value = event.xdata
                if self.is_velocity_mode and self.rest_wavelength is not None:
                    # Convert velocity back to wavelength
                    bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
                
                self.listfit_bounds.append(bound_value)
                line = self.ax.axvline(event.xdata, color='green', linestyle='--')
                self.listfit_bound_lines.append(line)
                print(f"Listfit bound start: {event.xdata:.2f}. Press space again to set end bound.")
                # Record action for setting Listfit lower bound
                self.record_action('set_listfit_bound_1', f'Set Listfit lower bound at λ={bound_value:.2f} Å')
                self.fig.canvas.draw_idle()
            elif len(self.listfit_bounds) == 1:
                # Set the end bound
                # If in velocity mode, convert velocity back to wavelength for processing
                bound_value = event.xdata
                if self.is_velocity_mode and self.rest_wavelength is not None:
                    # Convert velocity back to wavelength
                    bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
                
                self.listfit_bounds.append(bound_value)
                line = self.ax.axvline(event.xdata, color='green', linestyle='--')
                self.listfit_bound_lines.append(line)
                print(f"Listfit bound end: {event.xdata:.2f}. Opening component selection dialog...")
                # Record action for setting Listfit upper bound
                self.record_action('set_listfit_bound_2', f'Set Listfit upper bound at λ={bound_value:.2f} Å')
                self.fig.canvas.draw_idle()
                
                # Show the listfit window
                self.show_listfit_window()

        # Exit listfit mode with Escape
        if event.key == 'escape' and self.listfit_mode:
            self.listfit_mode = False
            for line in self.listfit_bound_lines:
                line.remove()
            self.listfit_bound_lines.clear()
            self.listfit_bounds = []
            plt.draw()
            print("Exiting listfit mode.")

        # Set Gaussian bounds with space bar
        if event.key == ' ' and (self.gaussian_mode or self.multi_gaussian_mode_old):
            line_id = None
            line_wavelength = None
            # Register the bound at the current cursor position
            # If in velocity mode, convert velocity back to wavelength for processing
            bound_value = event.xdata
            if self.is_velocity_mode and self.rest_wavelength is not None:
                # Convert velocity back to wavelength
                bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
            
            self.bounds.append(bound_value)
            line = self.ax.axvline(event.xdata, color='red', linestyle='--')  # Plot bound line using displayed coords
            self.bound_lines.append(line)  # Store the line object
            print(f"Bound set at x = {event.xdata}")
            # Record action for setting a bound
            if len(self.bounds) == 1:
                self.record_action('set_gaussian_bound_1', f'Set Gaussian lower bound at λ={bound_value:.2f} Å')
            elif len(self.bounds) == 2:
                self.record_action('set_gaussian_bound_2', f'Set Gaussian upper bound at λ={bound_value:.2f} Å')
            self.fig.canvas.draw_idle()  # Update plot with the new bound line

            # If two bounds are selected, fit the Gaussian
            if self.gaussian_mode and len(self.bounds) == 2:
                self.gaussian_mode = False
                left_bound, right_bound = sorted(self.bounds)
                
                # Check for partial continuum overlap
                has_partial_overlap, overlap_msg = self.check_continuum_partial_overlap(left_bound, right_bound)
                if has_partial_overlap:
                    print(f"WARNING: {overlap_msg}")
                    print("Fit aborted to avoid ambiguous continuum handling.")
                    # Clear bounds
                    for line in self.bound_lines:
                        line.remove()
                    self.bound_lines.clear()
                    self.bounds.clear()
                    plt.draw()
                    return

                # Check for existing continuum
                existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
                comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                # Handle optional error spectrum
                if self.err is not None:
                    comp_err = self.err[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                else:
                    comp_err = None

                if existing_continuum is not None:
                    # Use the existing continuum
                    continuum_subtracted_y = comp_y - existing_continuum
                    continuum_for_plot = existing_continuum
                    print("Using existing continuum for Gaussian fit.")

                else:
                    # No continuum defined - fit directly to data without continuum subtraction
                    continuum_subtracted_y = comp_y
                    continuum_for_plot = np.zeros_like(comp_y)
                    print("No existing continuum found; fitting Gaussian directly to data.")

                # Fit Gaussian to the continuum-subtracted data
                if len(comp_x) > 0:
                    if np.mean(continuum_subtracted_y) > 0:
                        initial_guess = [max(continuum_subtracted_y) - min(continuum_subtracted_y), np.mean(comp_x), np.std(comp_x)]
                    else:
                        initial_guess = [min(continuum_subtracted_y) - max(continuum_subtracted_y), np.mean(comp_x), np.std(comp_x)]
                    
                    # --- START NEW GUESS METHOD --- #
                    # Find peak index
                    peak_index = np.argmax(np.abs(continuum_subtracted_y))
                    peak_x = comp_x[peak_index]
                    peak_y = continuum_subtracted_y[peak_index]

                    # Estimate amplitude and sign
                    amplitude_guess = peak_y

                    # Estimate stddev from FWHM (rough approximation)
                    half_max = amplitude_guess / 2.0
                    try:
                        # Get indices where the signal crosses half max
                        indices_above_half = np.where(np.abs(continuum_subtracted_y) > np.abs(half_max))[0]
                        if len(indices_above_half) >= 2:
                            fwhm_estimate = comp_x[indices_above_half[-1]] - comp_x[indices_above_half[0]]
                            stddev_guess = fwhm_estimate / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to stddev
                        else:
                            stddev_guess = np.std(comp_x)  # Fallback
                    except:
                        stddev_guess = np.std(comp_x)

                    # Use peak location for mean
                    mean_guess = peak_x

                    initial_guess = [amplitude_guess, mean_guess, stddev_guess]

                    # --- END NEW GUESS METHOD --- #
                    
                    # curve_fit with optional sigma (errors)
                    if comp_err is not None:
                        params, pcov = curve_fit(self.gaussian, comp_x, continuum_subtracted_y, sigma=comp_err, p0=initial_guess, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
                    else:
                        params, pcov = curve_fit(self.gaussian, comp_x, continuum_subtracted_y, p0=initial_guess, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]))
                    amp, mean, stddev = params
                    perr = np.sqrt(np.diag(pcov))
                    amp_err, mean_err, stddev_err = perr

                    # Plot the fit and store fit info
                    x_fit = comp_x
                    y_fit = self.gaussian(x_fit, amp, mean, stddev) + continuum_for_plot
                    residuals = comp_y - y_fit
                    # Calculate chi2 with optional errors
                    if comp_err is not None:
                        chi2 = np.sum((residuals ** 2) / comp_err) # Calculate chi2
                    else:
                        chi2 = np.sum(residuals ** 2) # Chi2 without errors
                    chi2_nu = chi2 / (len(x_fit) - len(params))# Calculate chi2 d.o.f.
                    interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                    y_plt = interpolator(x_plt)
                    fit_line, = self.ax.plot(x_plt, y_plt, color='red', linestyle='--')
                    # Store each component’s parameters
                    self.gaussian_fits.append({
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': chi2,
                    'chi2_nu': chi2_nu,
                    'component_id': self.component_id,
                    'amp': amp, 'amp_err': amp_err, 'mean': mean, 'mean_err': mean_err, 'stddev': stddev, 'stddev_err': stddev_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength  if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift
                    })
                    # Register with ItemTracker
                    position_str = f"λ: {mean:.2f} Å"
                    self.register_item('gaussian', f'Gaussian', fit_dict=self.gaussian_fits[-1], line_obj=fit_line,
                                     position=position_str, color='red')
                    
                    # Record action for undo/redo
                    self.record_action('fit_gaussian', f'Fit Gaussian at λ={mean:.2f} Å')
                    
                    print(f"  Fit ID: {self.fit_id}")
                    print(f"  Component ID: {self.component_id}")
                    print(f"  Velocity mode: {self.is_velocity_mode}")
                    print(f"  Line ID: {line_id}")
                    print(f"  Line Wavelength: {line_wavelength}")
                    print(f"  Amplitude: {amp}+-{amp_err}")
                    print(f"  Mean: {mean}+-{mean_err}")
                    print(f"  Std_dev: {stddev}+-{stddev_err}")
                    print(f"  Bounds: ({left_bound}, {right_bound})")
                    print(f"  Chi-squared: {chi2}")
                    print(f"  Chi-squared_nu: {chi2_nu}")
                    print(f"  Line Object: {fit_line}\n")

                    self.component_id += 1
                    # Force immediate redraw of the canvas
                    self.ax.figure.canvas.draw()
                    QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
                    print(f"Fitted parameters: Amplitude = {amp}+-{amp_err}, Mean = {mean}+-{mean_err}, Std Dev = {stddev}+-{stddev_err}")
                    # Update residual display if shown
                    if self.is_residual_shown:
                        self.calculate_and_plot_residuals()

                # Remove bound lines after fit
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()  # Clear the list of bound lines
                self.bounds = []
                self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
                self.fit_id += 1

        # Perform multi-Gaussian fit if Enter is pressed in multi-Gaussian mode
        elif self.multi_gaussian_mode_old and event.key == 'enter' and len(self.bounds) >= 4 and len(self.bounds) % 2 == 0:
            self.multi_gaussian_mode_old = False
            bound_pairs = [(self.bounds[i], self.bounds[i + 1]) for i in range(0, len(self.bounds), 2)]
            
            # Pre-flight check: Verify no partial overlaps with continuum before processing
            for left_bound, right_bound in bound_pairs:
                has_partial_overlap, overlap_msg = self.check_continuum_partial_overlap(left_bound, right_bound)
                if has_partial_overlap:
                    print(f"WARNING: {overlap_msg}")
                    print("Multi-Gaussian fit aborted to avoid ambiguous continuum handling.")
                    # Clear bounds
                    for line in self.bound_lines:
                        line.remove()
                    self.bound_lines.clear()
                    self.bounds.clear()
                    self.fig.canvas.draw_idle()
                    return
            
            comp_xs = []
            comp_ys = []
            comp_errs = []
            continuum_subtracted_ys = []
            continuum_ys = []
            line_id = None
            line_wavelength = None

            # Prepare data for fitting multiple Gaussians, applying continuum subtraction
            initial_guesses = []
            for left_bound, right_bound in bound_pairs:
                comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                # Handle optional error spectrum
                if self.err is not None:
                    comp_err = self.err[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                else:
                    comp_err = None
                comp_ys.append(comp_y)
                # self.ax.step(comp_x, comp_y, color='brown', linestyle='--') # [DEBUG]
                
                # Check for existing continuum within bounds
                existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
                if existing_continuum is not None:
                    # If an existing continuum is available, subtract it
                    continuum_subtracted_y = comp_y - existing_continuum
                    print(f"Using existing continuum for bounds {left_bound}-{right_bound}.")
                    continuum_y = np.array(existing_continuum)
                else:
                    # No continuum defined - fit directly to data
                    continuum_subtracted_y = comp_y
                    continuum_y = np.zeros_like(comp_y)
                    print(f"No existing continuum found; fitting directly to data for bounds {left_bound}-{right_bound}.")

                continuum_ys.append(continuum_y)
                comp_xs.extend(comp_x)
                # Add error to list only if available
                if comp_err is not None:
                    comp_errs.extend(comp_err)
                continuum_subtracted_ys.extend(continuum_subtracted_y)
                # Add initial guesses for Gaussian fitting
                initial_guesses.extend([max(continuum_subtracted_y) - min(continuum_subtracted_y), np.mean(comp_x), np.std(comp_x)]) # NOT SHARED SIGMA

            # Fit multiple Gaussians
            if len(comp_xs) > 0:
                comp_xs = np.array(comp_xs)
                continuum_subtracted_ys = np.array(continuum_subtracted_ys)
                # Use sigma if errors available, otherwise None
                sigma_param = np.array(comp_errs) if comp_errs else None
                params, pcov = curve_fit(self.multi_gaussian, comp_xs, continuum_subtracted_ys, sigma=sigma_param, p0=initial_guesses) # NOT SHARED SIGMA
                perr = np.sqrt(np.diag(pcov))
                for i in range(0, len(params), 3):
                    amp, mean, stddev = params[i:i+3]
                    amp_err, mean_err, stddev_err = perr[i:i+3]
                    x_fit = self.x_data[(self.x_data >= bound_pairs[i // 3][0]) & (self.x_data <= bound_pairs[i // 3][1])]
                    y_fit = self.gaussian(x_fit, amp, mean, stddev) + continuum_ys[i // 3]
                    continuum_sub_data = comp_ys[i // 3] - continuum_ys[i // 3]
                    residuals = continuum_sub_data - self.gaussian(x_fit, amp, mean, stddev)
                    # Calculate chi2 with optional errors
                    if comp_errs:
                        chi2 = np.sum((residuals ** 2) / comp_errs[i // 3]) # Calculate chi2
                    else:
                        chi2 = np.sum(residuals ** 2) # Chi2 without errors
                    chi2_nu = chi2 / (len(x_fit) - len(params))# Calculate chi2 d.o.f.
                    interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                    y_plt = interpolator(x_plt)
                    fit_line, = self.ax.plot(x_plt, y_plt, color='red', linestyle='--')
                    left_bound, right_bound = bound_pairs[i // 3]
                    gaussian_fit = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': chi2,
                    'chi2_nu': chi2_nu,
                    'component_id': self.component_id,
                    'amp': amp, 'amp_err': amp_err, 'mean': mean, 'mean_err': mean_err, 'stddev': stddev, 'stddev_err': stddev_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength  if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift
                    }
                    self.gaussian_fits.append(gaussian_fit)
                    # Register with ItemTracker
                    position_str = f"λ: {mean:.2f} Å"
                    self.register_item('gaussian', f'Gaussian', fit_dict=gaussian_fit, line_obj=fit_line,
                                     position=position_str, color='red')
                    
                    # Record action for undo/redo (only record once after all components)
                    if i == len(params) - 3:  # Last component
                        self.record_action('fit_multi_gaussian', f'Fit {len(bound_pairs)} Gaussians')
                    
                    print(f"  Fit ID: {self.fit_id}")
                    print(f"  Component ID: {self.component_id}")
                    print(f"  Velocity mode: {self.is_velocity_mode}")
                    print(f"  Line ID: {line_id}")
                    print(f"  Line Wavelength: {line_wavelength}")
                    print(f"  Amplitude: {amp}+-{amp_err}")
                    print(f"  Mean: {mean}+-{mean_err}")
                    print(f"  Std_dev: {stddev}+-{stddev_err}")
                    print(f"  Bounds: ({left_bound}, {right_bound})")
                    print(f"  Chi-squared: {chi2}")
                    print(f"  Chi-squared_nu: {chi2_nu}")
                    print(f"  Line Object: {fit_line}\n")

                    self.component_id += 1

                # Force immediate redraw of the canvas
                self.ax.figure.canvas.draw()
                QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
                self.fit_id += 1
                # Clear bound lines after fit
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()
                self.bounds = []
                self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
                for i in range(0, len(params), 3):
                    print(f"Simultaneous Gaussian fit parameters:\nGaussian {i//3 + 1}: Amplitude = {amp}+-{amp_err}, Mean = {mean}+-{mean_err}, Std Dev = {stddev}+-{stddev_err}")

        # Enter Voigt fit mode
        if event.key == 'n':
            if self.voigt_mode:
                self.voigt_mode = False
                print("Exiting Voigt fit mode.")
            else:
                self.voigt_mode = True
                self.bounds = []  # Reset bounds
                if self.bound_lines is not None:
                    for line in self.bound_lines:  # Remove any existing bound lines
                        line.remove()
                self.bound_lines.clear()  # Clear the list of bound lines
                print("Voigt fit mode: Press space to set left and right bounds.")
                plt.draw()  # Update the plot to remove old lines

        if event.key == ' ' and self.voigt_mode:  # Set bounds with space for Voigt fit
            line_id = None
            line_wavelength = None
            # If in velocity mode, convert velocity back to wavelength for processing
            bound_value = event.xdata
            if self.is_velocity_mode and self.rest_wavelength is not None:
                # Convert velocity back to wavelength
                bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
            
            self.bounds.append(bound_value)
            line = self.ax.axvline(event.xdata, color='#eca829', linestyle='--')  # Plot bound line for Voigt
            self.bound_lines.append(line)  # Store the line object
            print(f"Voigt bound set at x = {event.xdata}")
            # Record action for setting a Voigt bound
            if len(self.bounds) == 1:
                self.record_action('set_voigt_bound_1', f'Set Voigt lower bound at λ={bound_value:.2f} Å')
            elif len(self.bounds) == 2:
                self.record_action('set_voigt_bound_2', f'Set Voigt upper bound at λ={bound_value:.2f} Å')
            self.fig.canvas.draw_idle()  # Update plot with the new bound line

            # If two bounds are selected, implement Voigt fitting
            if len(self.bounds) == 2:
                left_bound, right_bound = sorted(self.bounds)
                
                # Check for partial continuum overlap
                has_partial_overlap, overlap_msg = self.check_continuum_partial_overlap(left_bound, right_bound)
                if has_partial_overlap:
                    print(f"WARNING: {overlap_msg}")
                    print("Fit aborted to avoid ambiguous continuum handling.")
                    # Clear bounds
                    for line in self.bound_lines:
                        line.remove()
                    self.bound_lines.clear()
                    self.bounds.clear()
                    plt.draw()
                    return

                # Check for existing continuum
                existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
                comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                # Handle optional error spectrum
                if self.err is not None:
                    comp_err = self.err[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                else:
                    comp_err = None

                # Subtract existing continuum if found, otherwise fit directly to data
                if existing_continuum is not None:
                    continuum_subtracted_y = comp_y - existing_continuum
                    continuum_for_plot = existing_continuum
                    print("Using existing continuum for Voigt fit.")
                else:
                    # No continuum defined - fit directly to data
                    continuum_subtracted_y = comp_y
                    continuum_for_plot = np.zeros_like(comp_y)
                    print("No existing continuum found; fitting Voigt directly to data.")

                left_bound, right_bound = sorted(self.bounds)

                # Initial parameters for the Voigt profile fitting
                initial_params = Parameters()
                if np.mean(continuum_subtracted_y) > 0:
                    initial_params.add('amp', value=max(continuum_subtracted_y) - min(continuum_subtracted_y))
                else:
                    initial_params.add('amp', value=min(continuum_subtracted_y) - max(continuum_subtracted_y))
                initial_params.add('center', value=np.mean(comp_x))
                initial_params.add('sigma', value=np.std(comp_x)/10, min=0)  # Constrain sigma to be >= 0
                initial_params.add('gamma', value=np.std(comp_x)/10, min=0)  # Constrain gamma to be >= 0

                # Create the Voigt model and perform the fit
                voigt_model = Model(self.voigt)
                test_output = voigt_model.eval(params=initial_params, x=comp_x)
                if np.isnan(test_output).any():
                    print("Warning: Model function generated NaN values with initial parameters.")
                # Use weights if errors available, otherwise None
                weights = 1/comp_err if comp_err is not None else None
                result = voigt_model.fit(continuum_subtracted_y, x=comp_x, params=initial_params, weights=weights)

                # Clear bounds
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()
                self.bounds = []
                self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
                # Visualize the fit
                x_fit = np.linspace(left_bound, right_bound, len(continuum_for_plot))
                y_fit = result.eval(x=x_fit) + continuum_for_plot
                residuals = comp_y - y_fit
                # Calculate chi2 with optional errors
                if comp_err is not None:
                    chi2 = np.sum((residuals ** 2) / comp_err) # Calculate combined chi2
                else:
                    chi2 = np.sum(residuals ** 2) # Chi2 without errors
                chi2_nu = chi2 / (len(x_fit) - len(result.params)) # Calculate chi2 d.o.f.
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                fit_line, = self.ax.plot(x_plt, y_plt, color='#eca829', linestyle='--')
                # TEMP - below I present one exploratory method for calculating column densities and other absorption line diagnostics 
                line_wavelength = 2795 # AA
                f = 0.5
                N = self.column_density(continuum_for_plot, comp_y, f, line_wavelength, comp_x)
                # Store the fit results
                fit_results = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': result.chisqr,
                    'chi2_nu': result.redchi,
                    'component_id': self.component_id,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'N': np.log10(N) if N else None
                }
                for name, param in result.params.items():
                    fit_results[name] = param.value
                    fit_results[f'{name}_err'] = param.stderr
                    if name == 'sigma':
                        b = np.sqrt(2) * param.value
                        m = 4.0359e-23 # g for Mg
                        print(f"NOTE: Using {m} g mass for all ionic species. You may want to change this.")
                        T_eff = self.T_eff(b, m)
                        fit_results['b'] = b
                        fit_results['logT_eff'] = np.log10(T_eff)
                self.voigt_fits.append(fit_results)
                # Update residual display if shown
                if self.is_residual_shown:
                    self.calculate_and_plot_residuals()
                # Register with ItemTracker
                position_str = f"λ: {fit_results.get('center', fit_results.get('mean', 0)):.2f} Å"
                self.register_item('voigt', f'Voigt', fit_dict=fit_results, line_obj=fit_results.get('line'),
                                 position=position_str, color='orange')
                
                # Record action for undo/redo
                self.record_action('fit_voigt', f'Fit Voigt at λ={fit_results.get("center", fit_results.get("mean", 0)):.2f} Å')
                
                print(f"  Fit ID: {self.fit_id}")
                print(f"  Component ID: {self.component_id}")
                print(f"  Velocity mode: {self.is_velocity_mode}")
                print(f"  Line ID: {line_id}")
                print(f"  Line Wavelength: {line_wavelength}")
                for name, param in result.params.items():
                    print(f"  {name}: {param.value}+-{param.stderr}")
                    if name == 'sigma':
                        print(f" b: {b}")
                        print(f" logT_eff: {np.log10(T_eff)}")
                print(f" logN: {np.log10(N)}")
                print(f"  Bounds: ({left_bound}, {right_bound})")
                print(f"  Chi-squared: {result.chisqr}")
                print(f"  Chi-squared_nu: {result.redchi}")
                print(f"  Line Object: {fit_line}\n")
                self.fit_id += 1
                self.component_id += 1
                self.ax.legend(loc='upper right')
                # Force immediate redraw of the canvas
                self.ax.figure.canvas.draw()
                QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw

                # Clear bounds for the next fitting operation
                self.bounds.clear()

                self.voigt_mode = False  # Disable Voigt mode

        # Enter multi Voigt fit mode
        if event.key == 'N':
            if self.multi_voigt_mode:
                self.multi_voigt_mode = False
            else:
                self.multi_voigt_mode = True
            self.bounds = []  # Reset bounds
            if self.bound_lines is not None:
                for line in self.bound_lines:  # Remove any existing bound lines
                    line.remove()
            self.bound_lines.clear()  # Clear the list of bound lines
            print("Multi Voigt fit mode: Press space to set left and right bounds.")
            plt.draw()  # Update the plot to remove old lines
            self.voigt_comps = []
            self.component_id = 0

        if event.key == ' ' and self.multi_voigt_mode:  # Set bounds with space for Voigt fit
            # If in velocity mode, convert velocity back to wavelength for processing
            bound_value = event.xdata
            if self.is_velocity_mode and self.rest_wavelength is not None:
                # Convert velocity back to wavelength
                bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
            
            self.bounds.append(bound_value)
            line = self.ax.axvline(event.xdata, color='#eca829', linestyle='--')  # Plot bound line for Voigt
            self.bound_lines.append(line)  # Store the line object
            print(f"Voigt bound set at x = {event.xdata}")
            plt.draw()  # Update plot with the new bound line

            # If two bounds are selected, prepare Voigt fitting
            if len(self.bounds) % 2 == 0:
                left_bound, right_bound = sorted(self.bounds[-2:])

                # Open the LineListWindow to select a line                
                available_line_lists = self.get_all_available_line_lists()
                self.line_list_window = LineListWindow(available_line_lists=available_line_lists)
                self.line_list_window.selected_line.connect(self.receive_voigt)  # Connect selection to assign function
                self.line_list_window.show()
                # Save bounds for later use in on_line_selected
                self.current_bounds = (left_bound, right_bound)

        elif event.key == 'enter' and self.multi_voigt_mode and len(self.bounds) >= 2 and len(self.bounds) % 2 == 0:
            # Extract bound pairs for each Voigt component
            bound_pairs = [(self.bounds[i], self.bounds[i + 1]) for i in range(0, len(self.bounds), 2)]
            combined_model = None
            params = Parameters()
            component_ratios = []

            # Unique identifiers for each component and fit
            num_profiles = len(bound_pairs)

            # Lists to accumulate plot data for each component
            initial_params = Parameters()
            combined_model = None
            # Define shared parameters for sigma and gamma
            comp_x = self.voigt_comps[0].get('comp_x') # Use first Voigt component to set initial guesses for sigma and gamma
            # Set up models for each Voigt profile within the bounds
            for idx, (left_bound, right_bound) in enumerate(bound_pairs):
                prefix = f"p{idx + 1}_"  # Unique prefix for each component
                
                # Prepare data for fitting within the current bound
                comp_x = self.voigt_comps[idx].get('comp_x')
                comp_y = self.voigt_comps[idx].get('comp_y')
                existing_continuum = self.voigt_comps[idx].get('existing_continuum')
                continuum_subtracted_y = self.voigt_comps[idx].get('continuum_subtracted_y')
                # Subtract existing continuum if found
                if existing_continuum is not None:
                    continuum_subtracted_y = comp_y - existing_continuum
                    cont_y = existing_continuum
                    print("Using existing continuum for Voigt fit.")
                else:
                    overall_continuum, continuum_params, _ = self.fit_continuum(self.x_data, self.spec, self.err)
                    continuum_subtracted_y = comp_y - overall_continuum
                    cont_y = overall_continuum
                    print("No existing continuum found; fitted new continuum.")

                # Define component ratio
                if idx == 0:
                    print(self.voigt_comps[idx].get('osc_strength'), self.voigt_comps[idx].get('line_wavelength'))
                    ratio_item0 = self.voigt_comps[idx].get('osc_strength') * self.voigt_comps[idx].get('line_wavelength') # W_1 / W_2 = (f_1 / f_2) * (lam_1 / lam_2)
                    component_ratios.append(1) # Leave the first component ratio item as just 1
                else:
                    ratio_item = (self.voigt_comps[idx].get('osc_strength') * self.voigt_comps[idx].get('line_wavelength'))/ratio_item0
                    component_ratios.append(ratio_item)# The remaining component ratio items are then defined in terms of the first component
                
                # Add parameters to the model with unique prefixes
                # Define parameters for each Voigt component
                if idx == 0:
                    # Set amplitude freely
                    if np.mean(continuum_subtracted_y) > 0:
                        initial_params.add(f'{prefix}amp', value=max(continuum_subtracted_y) - min(continuum_subtracted_y))
                    else:
                        initial_params.add(f'{prefix}amp', value=min(continuum_subtracted_y) - max(continuum_subtracted_y))
                else:
                    # Subsequent components: constrain amplitude to a ratio of the first component
                    initial_params.add(f'{prefix}ratio', value=component_ratios[idx], vary=False)  # Fixed distance parameter
                    initial_params.add(f'{prefix}amp', expr=f'p1_amp * {prefix}ratio')
                if idx == 0:
                    initial_params.add(f'{prefix}center', value=np.mean(comp_x)+np.std(comp_x), min=min(comp_x), max=max(comp_x)) # Introduce +np.std(comp_x) to guess to avoid getting stuck in local minimum during optimization
                    # Add redshift as a free parameter
                    initial_params.add(f'{prefix}line_wavelength', value=self.voigt_comps[idx].get('line_wavelength'), vary=False)
                    initial_params.add(f'{prefix}z_comp', expr=f"p1_center / p1_line_wavelength - 1")  # Express in terms of centroid of transition wavelength
                    # initial_params.add(f'{prefix}z_comp', value=self.redshift, min=0, max=self.redshift+2)  # Set reasonable bounds for redshift # Previous: calculated this redshift independently of centroid
                    if self.is_velocity_mode:
                        initial_params.add(f'{prefix}z_comp', value=(np.mean(comp_x)+0.2*np.std(comp_x))/self.voigt_comps[idx].get('line_wavelength'), min=(np.mean(comp_x)-10*np.std(comp_x))/self.voigt_comps[idx].get('line_wavelength'), max=(np.mean(comp_x)+10*np.std(comp_x))/self.voigt_comps[idx].get('line_wavelength'))  # Set reasonable bounds for redshift
                else:
                    # distance = self.voigt_comps[idx]['distance']  # Retrieve precomputed distance
                    # initial_params.add(f'{prefix}delta', value=distance, vary=False)  # Fixed distance parameter
                    delta_expr = f"({self.voigt_comps[idx]['line_wavelength']} - {self.voigt_comps[0]['line_wavelength']}) * (1 + p1_z_comp)"
                    initial_params.add(f'{prefix}delta', expr=delta_expr)  # Delta depends on redshift
                    initial_params.add(f'{prefix}center', expr=f"p1_center + {prefix}delta")
                    initial_params.add(f'{prefix}z_comp', expr=f"p1_z_comp")  # Express in terms of centroid of transition wavelength
                if idx == 0:
                    initial_params.add(f'{prefix}sigma', value=np.std(comp_x) / 10, min=0.0)
                    initial_params.add(f'{prefix}gamma', value=np.std(comp_x) / 10, min=0.0)
                else:
                    initial_params.add(f'{prefix}sigma', expr='p1_sigma')
                    initial_params.add(f'{prefix}gamma', expr='p1_gamma')

                # Create the Voigt model for this component
                model = Model(self.voigt, prefix=prefix)
                combined_model = model if combined_model is None else combined_model + model
            
            # Clear bounds
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            # Define self.fit_x to cover the entire range of all bound_pairs
            leftmost_bound = min(bound[0] for bound in bound_pairs)
            rightmost_bound = max(bound[1] for bound in bound_pairs)
            x_fit = self.x_data[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            y_fit = self.spec[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            err_fit = self.err[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            # Fit continuum over this entire range
            existing_continuum, _, _ = self.get_existing_continuum(leftmost_bound, rightmost_bound)
            continuum_subtracted_y = y_fit - existing_continuum
            result = combined_model.fit(continuum_subtracted_y, initial_params, x=x_fit, weights=1/err_fit)
            print(result.fit_report()) # [DEBUG]

            for idx, (left_bound, right_bound) in enumerate(bound_pairs):
                prefix = f'p{idx+1}_'
                # Extract fitted parameters for each Voigt component
                amp = result.params[f'{prefix}amp'].value
                center = result.params[f'{prefix}center'].value
                sigma = result.params[f'{prefix}sigma'].value
                gamma = result.params[f'{prefix}gamma'].value
                comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
                # Generate data for plotting
                x_fit = np.linspace(left_bound, right_bound, len(existing_continuum))
                y_fit = self.voigt(x_fit, amp, center, sigma, gamma) + existing_continuum
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                # Higher resolution for smooth plotting
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                fit_line, = self.ax.plot(x_plt, y_plt, color='#eca829', linestyle='--')
                # Get `line_id` and `line_wavelength` from `self.voigt_comps` for this component
                line_id = self.voigt_comps[idx].get('line_id') if idx < len(self.voigt_comps) else None
                line_wavelength = self.voigt_comps[idx].get('line_wavelength') if idx < len(self.voigt_comps) else None
                # TEMP - READ IN INFORMATION FROM MATCHING LINE ID IN EMLINES.TXT
                # line_wavelength = 2795 # AA
                f = self.voigt_comps[idx].get('osc_strength')
                mass = None
                N = self.column_density(existing_continuum, comp_y, f, line_wavelength, comp_x)
                # Store the fit results
                fit_results = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': result.chisqr,
                    'chi2_nu': result.redchi,
                    'component_id': self.component_id,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'N': np.log10(N) if N else None
                }
                for name, param in result.params.items():
                    match = re.search(r'p(\d+)_', name) # Search the prefix for the component number
                    if match:
                        comp_num = int(match.group(1))
                        if comp_num == idx+1: # Get results separately for each component
                            name = name[len(prefix):]  # Strip the prefix if not z_comp
                            fit_results[name] = param.value
                            fit_results[f'{name}_err'] = param.stderr
                            if name == 'sigma':
                                b = np.sqrt(2) * param.value
                                m = 4.0359e-23 # g for Mg
                                print(f"NOTE: Using {m} g mass for all ionic species. You may want to change this.")
                                T_eff = self.T_eff(b, m)
                                fit_results['b'] = b
                                fit_results['logT_eff'] = np.log10(T_eff)
                self.voigt_fits.append(fit_results)
                # Update residual display if shown
                if self.is_residual_shown:
                    self.calculate_and_plot_residuals()
                # Register with ItemTracker
                position_str = f"λ: {fit_results.get('center', fit_results.get('mean', 0)):.2f} Å"
                self.register_item('voigt', f'Voigt', fit_dict=fit_results, line_obj=fit_results.get('line'),
                                 position=position_str, color='orange')
                print("fit_results:",fit_results)
                print(f"  Fit ID: {self.fit_id}")
                print(f"  Component ID: {self.component_id}")
                print(f"  Velocity mode: {self.is_velocity_mode}")
                print(f"  Line ID: {line_id}")
                print(f"  Line Wavelength: {line_wavelength}")
                for name, param in result.params.items():
                    match = re.search(r'p(\d+)_', name) # Search the prefix for the component number
                    if match:
                        comp_num = int(match.group(1))
                    if comp_num == idx+1: # Get results separately for each component
                        name = name[len(prefix):]  # Strip the prefix
                        fit_results[name] = param.value
                        fit_results[f'{name}_err'] = param.stderr
                        print(f"  {name}: {param.value}+-{param.stderr}")
                        if name == 'sigma':
                            print(f"  b: {b}")
                            print(f"  logT_eff: {np.log10(T_eff)}")
                print(f"  logN: {np.log10(N)}")
                print(f"  Bounds: ({left_bound}, {right_bound})")
                print(f"  Chi-squared: {result.chisqr}")
                print(f"  Chi-squared_nu: {result.redchi}")
                print(f"  Line Object: {fit_line}\n")
                self.component_id += 1
            self.fit_id += 1
            self.continuum_subtracted_ys = []
            self.continuum_ys = []
            self.voigt_comps.clear()
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            self.ax.legend(loc='upper right')
            plt.draw()  # Update the plot with the fitted profiles

            # Clear multi-Voigt mode settings
            self.voigt_mode = False
            self.multi_voigt_mode = False

        # Enter multi Gaussian fit mode
        if event.key == 'D':
            if self.multi_gaussian_mode:
                self.multi_gaussian_mode = False
            else:
                self.multi_gaussian_mode = True
            print("self.bounds:", self.bounds)
            self.bounds = []  # Reset bounds
            print("self.bounds:", self.bounds)
            if self.bound_lines is not None:
                for line in self.bound_lines:  # Remove any existing bound lines
                    line.remove()
            self.bound_lines.clear()  # Clear the list of bound lines
            print("Multi Gaussian fit mode: Press space to set left and right bounds.")
            plt.draw()  # Update the plot to remove old lines
            self.gaussian_comps = []
            print("self.bounds:", self.bounds)

        if event.key == ' ' and self.multi_gaussian_mode:  # Set bounds with space for Gaussian fit
            print("self.bounds:", self.bounds)
            # If in velocity mode, convert velocity back to wavelength for processing
            bound_value = event.xdata
            if self.is_velocity_mode and self.rest_wavelength is not None:
                # Convert velocity back to wavelength
                bound_value = self.vel_to_wav(event.xdata, self.rest_wavelength, z=self.redshift)
            
            self.bounds.append(bound_value)
            print("self.bounds:", self.bounds)
            line = self.ax.axvline(event.xdata, color='red', linestyle='--')  # Plot bound line for Gaussian
            self.bound_lines.append(line)  # Store the line object
            print(f"Gaussian bound set at x = {event.xdata}")
            plt.draw()  # Update plot with the new bound line

            # If two bounds are selected, prepare Voigt fitting
            if len(self.bounds) % 2 == 0:
                print("self.bounds:", self.bounds)
                left_bound, right_bound = sorted(self.bounds[-2:])

                # Open the LineListWindow to select a line                
                available_line_lists = self.get_all_available_line_lists()
                self.line_list_window = LineListWindow(available_line_lists=available_line_lists)
                self.line_list_window.selected_line.connect(self.receive_gaussian)  # Connect selection to assign function
                self.line_list_window.show()
                # Save bounds for later use in on_line_selected
                self.current_bounds = (left_bound, right_bound)

        elif event.key == 'enter' and self.multi_gaussian_mode and len(self.bounds) >= 2 and len(self.bounds) % 2 == 0:
            # Extract bound pairs for each Voigt component
            bound_pairs = [(self.bounds[i], self.bounds[i + 1]) for i in range(0, len(self.bounds), 2)]
            combined_model = None
            params = Parameters()
            component_ratios = []

            # Unique identifiers for each component and fit
            num_profiles = len(bound_pairs)

            # Lists to accumulate plot data for each component
            initial_params = Parameters()
            combined_model = None
            # Define shared parameters for sigma and gamma
            comp_x = self.gaussian_comps[0].get('comp_x') # Use first Voigt component to set initial guesses for sigma and gamma
            # Set up models for each Voigt profile within the bounds
            for idx, (left_bound, right_bound) in enumerate(bound_pairs):
                prefix = f"p{idx + 1}_"  # Unique prefix for each component
                
                # Prepare data for fitting within the current bound
                comp_x = self.gaussian_comps[idx].get('comp_x')
                comp_y = self.gaussian_comps[idx].get('comp_y')
                existing_continuum = self.gaussian_comps[idx].get('existing_continuum')
                continuum_subtracted_y = self.gaussian_comps[idx].get('continuum_subtracted_y')
                # Subtract existing continuum if found
                if existing_continuum is not None:
                    continuum_subtracted_y = comp_y - existing_continuum
                    print("Using existing continuum for Gaussian fit.")
                else:
                    overall_continuum, continuum_params, _ = self.fit_continuum(self.x_data, self.spec, self.err)
                    continuum_subtracted_y = comp_y - overall_continuum
                    print("No existing continuum found; fitted new continuum.")

                # Define component ratio
                if idx == 0:
                    print(self.gaussian_comps[idx].get('osc_strength'), self.gaussian_comps[idx].get('line_wavelength'))
                    ratio_item0 = self.gaussian_comps[idx].get('osc_strength') * self.gaussian_comps[idx].get('line_wavelength') # W_1 / W_2 = (f_1 / f_2) * (lam_1 / lam_2)
                    component_ratios.append(1) # Leave the first component ratio item as just 1
                else:
                    ratio_item = (self.gaussian_comps[idx].get('osc_strength') * self.gaussian_comps[idx].get('line_wavelength'))/ratio_item0
                    print("ratio_item0:", ratio_item0)
                    print("ratio_item:", ratio_item)
                    component_ratios.append(ratio_item)# The remaining component ratio items are then defined in terms of the first component
                
                # Add parameters to the model with unique prefixes
                # Define parameters for each Voigt component
                if idx == 0:
                    # First component: set amplitude freely
                    if np.mean(continuum_subtracted_y) > 0:
                        initial_params.add(f'{prefix}amp', value=max(continuum_subtracted_y) - min(continuum_subtracted_y))
                    else:
                        initial_params.add(f'{prefix}amp', value=min(continuum_subtracted_y) - max(continuum_subtracted_y))
                else:
                    # Subsequent components: constrain amplitude to a ratio of the first component
                    print("component_ratios[idx]:",component_ratios[idx])
                    initial_params.add(f'{prefix}ratio', value=component_ratios[idx], vary=False)  # Fixed distance parameter
                    initial_params.add(f'{prefix}amp', expr=f'p1_amp * {prefix}ratio')
                if idx == 0:
                    initial_params.add(f'{prefix}mean', value=np.mean(comp_x)+np.std(comp_x), min=min(comp_x), max=max(comp_x)) # Introduce +np.std(comp_x) to guess to avoid getting stuck in local minimum during optimization
                else:
                    distance = self.gaussian_comps[idx]['distance']  # Retrieve precomputed distance
                    initial_params.add(f'{prefix}delta', value=distance, vary=False)  # Fixed distance parameter
                    initial_params.add(f'{prefix}mean', expr=f"p1_mean + {prefix}delta")
                if idx == 0:
                    initial_params.add(f'{prefix}stddev', value=np.std(comp_x) / 10, min=0)
                else:
                    initial_params.add(f'{prefix}stddev', expr='p1_stddev')

                # Create the Voigt model for this component
                model = Model(self.gaussian, prefix=prefix)
                combined_model = model if combined_model is None else combined_model + model
            
            # Clear bounds
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            # Define self.fit_x to cover the entire range of all bound_pairs
            leftmost_bound = min(bound[0] for bound in bound_pairs)
            rightmost_bound = max(bound[1] for bound in bound_pairs)
            x_fit = self.x_data[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            y_fit = self.spec[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            err_fit = self.err[(self.x_data >= leftmost_bound) & (self.x_data <= rightmost_bound)]
            # Fit continuum over this entire range
            existing_continuum, _, _ = self.get_existing_continuum(leftmost_bound, rightmost_bound)
            continuum_subtracted_y = y_fit - existing_continuum
            result = combined_model.fit(continuum_subtracted_y, initial_params, x=x_fit, weights=1/err_fit)
            print(result.fit_report()) # [DEBUG]

            for idx, (left_bound, right_bound) in enumerate(bound_pairs):
                prefix = f'p{idx+1}_'
                # Extract fitted parameters for each Voigt component
                amp = result.params[f'{prefix}amp'].value
                mean = result.params[f'{prefix}mean'].value
                stddev = result.params[f'{prefix}stddev'].value
                comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                existing_continuum, _, _ = self.get_existing_continuum(left_bound, right_bound)
                # Generate data for plotting
                x_fit = np.linspace(left_bound, right_bound, len(existing_continuum))
                y_fit = self.gaussian(x_fit, amp, mean, stddev) + existing_continuum
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                # Higher resolution for smooth plotting
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                fit_line, = self.ax.plot(x_plt, y_plt, color='red', linestyle='--')
                # Get `line_id` and `line_wavelength` from `self.gaussian_comps` for this component
                line_id = self.gaussian_comps[idx].get('line_id') if idx < len(self.gaussian_comps) else None
                line_wavelength = self.gaussian_comps[idx].get('line_wavelength') if idx < len(self.gaussian_comps) else None
                # Read oscillator strength of transition
                f = self.gaussian_comps[idx].get('osc_strength')
                mass = None # PLACEHOLDER FOR MASS
                N = self.column_density(existing_continuum, comp_y, f, line_wavelength, comp_x)
                # Store the fit results
                fit_results = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': result.chisqr,
                    'chi2_nu': result.redchi,
                    'component_id': self.component_id,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'N': np.log10(N) if N else None
                }
                for name, param in result.params.items():
                    match = re.search(r'p(\d+)_', name) # Search the prefix for the component number
                    if match:
                        comp_num = int(match.group(1))
                    if comp_num == idx+1: # Get results separately for each component
                        name = name[len(prefix):]  # Strip the prefix
                        fit_results[name] = param.value
                        fit_results[f'{name}_err'] = param.stderr if param.stderr is not None else 0.0
                        print(f"[DEBUG Listfit] Extracted {name}={param.value}, {name}_err={param.stderr}")
                        if name == 'stddev':
                            b = np.sqrt(2) * param.value
                            m = 4.0359e-23 # g for Mg
                            print(f"NOTE: Using {m} g mass for all ionic species. You may want to change this.")
                            T_eff = self.T_eff(b, m)
                            fit_results['b'] = b
                            fit_results['logT_eff'] = np.log10(T_eff)
                self.gaussian_fits.append(fit_results)
                print("fit_results:",fit_results)
                print(f"  Fit ID: {self.fit_id}")
                print(f"  Component ID: {self.component_id}")
                print(f"  Velocity mode: {self.is_velocity_mode}")
                print(f"  Line ID: {line_id}")
                print(f"  Line Wavelength: {line_wavelength}")
                for name, param in result.params.items():
                    match = re.search(r'p(\d+)_', name) # Search the prefix for the component number
                    if match:
                        comp_num = int(match.group(1))
                    if comp_num == idx+1: # Get results separately for each component
                        name = name[len(prefix):]  # Strip the prefix
                        fit_results[name] = param.value
                        fit_results[f'{name}_err'] = param.stderr
                        print(f"  {name}: {param.value}+-{param.stderr}")
                        if name == 'stddev':
                            print(f"  b: {b}")
                            print(f"  logT_eff: {np.log10(T_eff)}")
                print(f"  logN: {np.log10(N)}")
                print(f"  Bounds: ({left_bound}, {right_bound})")
                print(f"  Chi-squared: {result.chisqr}")
                print(f"  Chi-squared_nu: {result.redchi}")
                print(f"  Line Object: {fit_line}\n")
                self.component_id += 1
            self.fit_id += 1
            self.continuum_subtracted_ys = []
            self.continuum_ys = []
            self.gaussian_comps.clear()
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            self.ax.legend(loc='upper right')
            plt.draw()  # Update the plot with the fitted profiles

            # Clear multi-Voigt mode settings
            self.gaussian_mode = False
            self.multi_gaussian_mode = False


        # Check if the cursor is within the axes bounds
        if hasattr(event, 'xdata') and hasattr(event, 'ydata'):
            if event.xdata is not None and event.ydata is not None:
                if (self.x_lower_bound <= event.xdata <= self.x_upper_bound and self.y_lower_bound <= event.ydata <= self.y_upper_bound):
                    # Smoothing with keys '1'-'9'
                    if event.key in '123456789':
                        try:
                            print(f"Key pressed: {event.key}")
                            key_num = int(event.key)
                            # print(self.spectrum_line.get_xdata)
                            kernel_width = key_num  # Set the kernel width based on the key pressed
                            self.smooth_spectrum(kernel_width) # Saves the smoothed spectrum to the self.smoothed_spec variable
                            # self.spectrum_line.set_ydata(self.smoothed_spec)  # Update line with smoothed data
                            if self.spectrum_line:
                                self.step_spec.set_visible(False)  # Hide the old line instead of removing it
                                self.line_spec.set_visible(False)
                                self.spectrum_line.set_visible(False)
                                self.fig.canvas.draw_idle()
                                self.step_spec, = self.ax.step(self.x_data, self.smoothed_spec, color='black', where='mid')
                                self.line_spec, = self.ax.plot(self.x_data, self.smoothed_spec, color='black', visible=False)
                                # self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
                                # Make sure to hide the step/line plot
                                if self.is_step_plot:
                                    self.step_spec.set_visible(True)
                                    self.line_spec.set_visible(False)
                                    self.spectrum_line = self.step_spec
                                else:
                                    self.step_spec.set_visible(False)
                                    self.line_spec.set_visible(True)
                                    self.spectrum_line = self.line_spec
                                # self.spectrum_line.set_ydata(self.smoothed_spec)  # Update the data for the old line
                                # self.spectrum_line.set_visible(True)  # Make it visible again
                                self.fig.canvas.draw_idle()
                            # self.spectrum_line.remove()
                            # self.spectrum_line, = self.ax.step(self.x_data, self.smoothed_spec, color='black', where='mid') if self.is_step_plot else self.ax.plot(self.x_data, self.smoothed_spec, color='black')  # Update line with smoothed data
                            # print(self.spectrum_line.get_xdata)
                            # self.ax.step(self.x_data, self.smoothed_spec, where='mid')
                            self.fig.canvas.draw_idle()
                        except Exception as e:
                            print(f"Error: {e}")
                    # Reset to original spectrum with '0'
                    elif event.key == '0':
                        print("Key pressed:", event.key)
                        print("Going back to unsmoothed spectrum.")
                        # self.spectrum_line.set_ydata(self.original_spec)
                        if self.spectrum_line:
                            self.step_spec.set_visible(False)  # Hide the old line instead of removing it
                            self.line_spec.set_visible(False)
                            self.spectrum_line.set_visible(False)
                            self.fig.canvas.draw_idle()
                            self.step_spec, = self.ax.step(self.x_data, self.original_spec, color='black', where='mid')
                            self.line_spec, = self.ax.plot(self.x_data, self.original_spec, color='black', visible=False)
                            # self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
                            # Make sure to hide the step/line plot
                            if self.is_step_plot:
                                self.step_spec.set_visible(True)
                                self.line_spec.set_visible(False)
                                self.spectrum_line = self.step_spec
                            else:
                                self.step_spec.set_visible(False)
                                self.line_spec.set_visible(True)
                                self.spectrum_line = self.line_spec
                            # self.spectrum_line.set_ydata(self.smoothed_spec)  # Update the data for the old line
                            # self.spectrum_line.set_visible(True)  # Make it visible again
                            self.fig.canvas.draw_idle()
                        # self.spectrum_line, = self.ax.step(self.x_data, self.original_spec, color='black', where='mid') if self.is_step_plot else self.ax.plot(self.x_data, self.original_spec, color='black')  # Update line with original data
                        self.fig.canvas.draw_idle()

        if event.key == 'x':
            # Center x-bounds on cursor position
            center_x = event.xdata
            x_range = self.x_upper_bound - self.x_lower_bound
            half_range = x_range / 2
            self.x_lower_bound = center_x - half_range
            self.x_upper_bound = center_x + half_range
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'y':
            y_range = self.y_upper_bound - self.y_lower_bound
            self.y_lower_bound -= y_range * self.zoom_factor
            self.y_upper_bound += y_range * self.zoom_factor
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'Y':
            y_range = self.y_upper_bound - self.y_lower_bound
            self.y_lower_bound += y_range * self.zoom_factor
            self.y_upper_bound -= y_range * self.zoom_factor
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 't':
            x_range = self.x_upper_bound - self.x_lower_bound
            self.x_lower_bound -= x_range * self.zoom_factor
            self.x_upper_bound += x_range * self.zoom_factor
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'T':
            x_range = self.x_upper_bound - self.x_lower_bound
            self.x_lower_bound += x_range * self.zoom_factor
            self.x_upper_bound -= x_range * self.zoom_factor
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == '[':
            x_range = self.x_upper_bound - self.x_lower_bound
            self.x_lower_bound -= x_range
            self.x_upper_bound -= x_range
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == ']':
            x_range = self.x_upper_bound - self.x_lower_bound
            self.x_lower_bound += x_range
            self.x_upper_bound += x_range
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'O':
            self.y_upper_bound = event.ydata
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'P':
            self.y_lower_bound = event.ydata
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'i':
            self.x_upper_bound = event.xdata
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'u':
            self.x_lower_bound = event.xdata
            self.update_bounds()
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots or self.active_line_lists:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()
        elif event.key == 'e':
            # Toggle line list selector window
            self.show_line_list_selector()
        elif event.key in ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']:  # Corresponding to bands 0-5
            index = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')'].index(event.key)
            if index < len(self.band_ranges):
                self.toggle_instrument_bands(index)  # Call function to toggle between showing and hiding instrument bands
        elif event.key in ['-', '=', '_', '+']:
            index = ['-', '=', '_', '+'].index(event.key)
            self.toggle_filter_bands(index)
        elif event.key == '\\':  # Backslash functionality to reset bounds
            self.x_lower_bound = self.original_xlim[0]
            self.x_upper_bound = self.original_xlim[1]
            self.y_lower_bound = self.original_ylim[0]
            self.y_upper_bound = self.original_ylim[1]
            self.update_bounds()  # Update the plot bounds
            self.update_ticks(self.ax)
            if self.is_residual_shown:
                self.update_residual_ticks()
                self.update_residual_ybounds()
            if self.linelist_plots:
                self.display_linelist()
            if self.markers and self.labels:
                self.update_marker_and_label_positions()

        elif event.key == ';':
            # Toggle the total line for ALL fitted profiles (single, multi-gaussian, voigt, continuum, listfit)
            self.show_total_line = not self.show_total_line
            if self.show_total_line:
                # Ensure there is data to sum and plot
                if self.continuum_fits or self.voigt_fits or self.gaussian_fits or self.listfit_fits:
                    self.draw_total_line()
                    self.ax.figure.canvas.draw()
                else:
                    print("Warning: No fits available to sum for total line.")
                    self.show_total_line = not self.show_total_line
            
            # Clear the total line if toggled off
            else:
                total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total Line"]
                for line in total_lines:
                    line.remove()
                # Regenerate the legend to exclude removed lines
                handles, labels = self.ax.get_legend_handles_labels()
                filtered_handles_labels = [
                    (handle, label) for handle, label in zip(handles, labels) if label != "Total Line"
                ]
                if filtered_handles_labels:
                    filtered_handles, filtered_labels = zip(*filtered_handles_labels)
                else:
                    filtered_handles, filtered_labels = [], []

                self.ax.legend(filtered_handles, filtered_labels)
                self.ax.figure.canvas.draw()

        # Enter Gaussian fit mode
        elif event.key == 'd':
            if self.gaussian_mode:
                self.gaussian_mode = False
                print("Exiting Gaussian fit mode.")
            else:
                self.gaussian_mode = True
                self.multi_gaussian_mode = False
                self.multi_gaussian_mode_old = False
                self.bounds = []  # Reset bounds
                if self.bound_lines is not None:
                    for line in self.bound_lines:  # Remove any existing bound lines
                        line.remove()
                self.bound_lines.clear()  # Clear the list of bound lines
                print("Gaussian fit mode: Press space to set left and right bounds.")
                plt.draw()  # Update the plot to remove old lines

        # Enable multi-Gaussian fit mode with '|'
        elif event.key == '|':
            if self.gaussian_mode or self.multi_gaussian_mode_old:
                self.gaussian_mode = False
                self.multi_gaussian_mode_old = False
                print("Exiting multi-Gaussian fit mode.")
            else:
                self.multi_gaussian_mode_old = True
                self.gaussian_mode = False
                self.bounds = []
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()
                print("Multi-Gaussian fit mode: Press space to set multiple bounds, enter to fit.")
                plt.draw()

        # Delete Gaussian or Voigt fit with 'w' if cursor is within its bounds
        elif event.key == 'w':
            x_pos = event.xdata
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    try:
                        fit['line'].remove()
                    except (ValueError, AttributeError):
                        # Line may have already been removed or is invalid
                        pass
                    # Remove the fill area if it exists
                    if self.ew_fill:
                        try:
                            self.ew_fill.remove()
                        except (ValueError, AttributeError):
                            pass
                        self.ew_fill = None  # Reset the fill reference
                    # Find and unregister the item from Item Tracker
                    for item_id, item_info in list(self.item_id_map.items()):
                        if item_info.get('fit_dict') is fit:
                            self.unregister_item(item_id)
                            break
                    self.gaussian_fits.remove(fit)
                    print(f"Removed Gaussian fit within bounds ({left_bound}, {right_bound})")
                    self.fig.canvas.draw_idle()
                    
                    # Record action for undo/redo
                    self.record_action('delete_gaussian', f'Remove Gaussian fit')
                    
                    # Update residual display if shown
                    if self.is_residual_shown:
                        self.calculate_and_plot_residuals()
                    break
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    try:
                        fit['line'].remove()
                    except (ValueError, AttributeError):
                        # Line may have already been removed or is invalid
                        pass
                    # Remove the fill area if it exists
                    if self.ew_fill:
                        try:
                            self.ew_fill.remove()
                        except (ValueError, AttributeError):
                            pass
                        self.ew_fill = None  # Reset the fill reference
                    # Find and unregister the item from Item Tracker
                    for item_id, item_info in list(self.item_id_map.items()):
                        if item_info.get('fit_dict') is fit:
                            self.unregister_item(item_id)
                            break
                    self.voigt_fits.remove(fit)
                    print(f"Removed Voigt fit within bounds ({left_bound}, {right_bound})")
                    self.fig.canvas.draw_idle()
                    
                    # Record action for undo/redo
                    self.record_action('delete_voigt', f'Remove Voigt fit')
                    
                    # Update residual display if shown
                    if self.is_residual_shown:
                        self.calculate_and_plot_residuals()
                    break

        # Exit redshift mode with Escape
        if event.key == 'escape' and self.redshift_estimation_mode:
            self.redshift_estimation_mode = False
            print('Exiting redshift estimation mode.')

        # Enter redshift estimation mode
        if event.key == 'z':
            if self.redshift_estimation_mode:
                self.redshift_estimation_mode = False
                print('Exiting redshift estimation mode.')
            else:
                self.redshift_estimation_mode = True
                print('Redshift estimation mode: Select Gaussian to use for redshift estimation. Assign a line to it, and estimate the redshift.')
        # If in redshift estimation mode and spacebar is pressed
        elif self.redshift_estimation_mode and event.key == ' ':
            x_pos = event.xdata
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.plot_redshift_gaussian(fit)
                    self.center_profile, self.center_profile_err = fit['mean'], fit['mean_err']
                    print(f"Center of selected Gaussian: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
                    self.open_linelist_window()
                    break
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.plot_redshift_voigt(fit)
                    self.center_profile, self.center_profile_err = fit['center'], fit['center_err']
                    if self.center_profile_err is None:
                        raise ValueError(f"Error associated with the center of the Voigt profile is missing for x_pos = {self.center_profile:.6f}.")
                    print(f"Center of selected Voigt: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
                    self.open_linelist_window()
                    break
        elif event.key == 'z' and self.redshift_estimation_mode:
            self.redshift_estimation_mode = False
            print('Exiting redshift estimation mode.')

        if event.key == 'b':
            self.is_velocity_mode = not self.is_velocity_mode  # Toggle Velocity mode
            if self.is_velocity_mode:
                self.activate_velocity_mode()  # Enter velocity mode
                if self.is_residual_shown:
                    self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)")
                    self.update_residual_ticks()
                else:
                    self.ax.set_xlabel(r"Velocity (km s$^{-1}$)")
            else:
                # Exit velocity mode
                self.exit_velocity_mode()  
                self.rest_wavelength = None
                self.rest_id = None
                
                # Revert labels and limits to wavelength mode
                if self.is_residual_shown:
                    self.residual_ax.set_xlabel(self._get_wavelength_unit_label())
                    self.update_residual_ticks()
                else:
                    self.ax.set_xlabel(self._get_wavelength_unit_label())
                    
                # Update ticks and plot
                self.update_ticks(self.ax)
                if self.is_residual_shown:
                    self.update_residual_ticks()
                    self.update_residual_ybounds()
                    self.residual_ax.set_xlim(self.x_lower_bound, self.x_upper_bound)
                if self.markers and self.labels:
                    self.update_marker_and_label_positions()
                plt.draw()

        # Enter Bayes fit mode with ':' key
        if event.key == ':':
            if self.bayes_mode:
                self.bayes_mode = False
                print("Exiting Bayes fit mode.")
            else:
                self.bayes_mode = True
                self.bayes_bounds = []  # Reset bounds
                if self.bayes_bound_lines is not None:
                    for line in self.bayes_bound_lines:  # Remove any existing bound lines
                        line.remove()
                self.bayes_bound_lines.clear()  # Clear the list of bound lines
                print("Bayes fit mode: Press space to set left and right bounds.")
                plt.draw()

        # Select bounds to perform Bayes fit
        elif event.key == ' ' and self.bayes_mode:
            if len(self.bayes_bounds) < 2:
                self.bayes_bounds.append(event.xdata)
                line = self.ax.axvline(event.xdata, color='lightblue', linestyle='--')
                self.bayes_bound_lines.append(line)
                self.fig.canvas.draw_idle()
                if len(self.bayes_bounds) == 2:
                    self.bayes_bounds.sort()
                    print(f"Bayes fit bounds set: {self.bayes_bounds}")
                    self.prompt_bayes_fit()

        # Save all fits to a single consolidated file
        elif event.key == 'S':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_fits = []
            
            # Collect Gaussian fits
            for fit in self.gaussian_fits:
                fit_entry = {'type': 'gaussian'}
                fit_entry.update(fit)
                # Find and save the tracker name for this fit
                for item_id, item_info in self.item_id_map.items():
                    if item_info.get('fit_dict') is fit:
                        fit_entry['tracker_name'] = item_info.get('name', '')
                        break
                # Handle tuple bounds - convert to min/max
                if 'bounds' in fit_entry and isinstance(fit_entry['bounds'], (tuple, list)):
                    bounds = fit_entry.pop('bounds')
                    fit_entry['bounds_min'] = bounds[0]
                    fit_entry['bounds_max'] = bounds[1]
                all_fits.append(fit_entry)
            
            # Collect Voigt fits
            for fit in self.voigt_fits:
                fit_entry = {'type': 'voigt'}
                fit_entry.update(fit)
                # Find and save the tracker name for this fit
                for item_id, item_info in self.item_id_map.items():
                    if item_info.get('fit_dict') is fit:
                        fit_entry['tracker_name'] = item_info.get('name', '')
                        break
                # Handle tuple bounds - convert to min/max
                if 'bounds' in fit_entry and isinstance(fit_entry['bounds'], (tuple, list)):
                    bounds = fit_entry.pop('bounds')
                    fit_entry['bounds_min'] = bounds[0]
                    fit_entry['bounds_max'] = bounds[1]
                all_fits.append(fit_entry)
            
            # Collect Continuum fits
            for fit in self.continuum_fits:
                fit_entry = {'type': 'continuum'}
                fit_entry.update(fit)
                # Find and save the tracker name for this fit
                for item_id, item_info in self.item_id_map.items():
                    if item_info.get('fit_dict') is fit:
                        fit_entry['tracker_name'] = item_info.get('name', '')
                        break
                # Handle tuple bounds - convert to min/max
                if 'bounds' in fit_entry and isinstance(fit_entry['bounds'], (tuple, list)):
                    bounds = fit_entry.pop('bounds')
                    fit_entry['bounds_min'] = bounds[0]
                    fit_entry['bounds_max'] = bounds[1]
                all_fits.append(fit_entry)
            
            # Collect Listfit data
            for listfit in self.listfit_fits:
                bounds = listfit.get('bounds', (None, None))
                components = listfit.get('components', [])
                result = listfit.get('result')
                initial_guesses = listfit.get('initial_guesses', {})
                constraints = listfit.get('constraints', {})
                
                fit_entry = {
                    'type': 'listfit',
                    'bounds_min': bounds[0],
                    'bounds_max': bounds[1],
                    'components': str(components),  # Store as string representation
                    'initial_guesses': str(initial_guesses),
                    'constraints': str(constraints),
                    'chi_squared': result.chisqr if result else None,
                    'redchi': result.redchi if result else None,
                    'aic': result.aic if result else None,
                    'bic': result.bic if result else None,
                }
                
                # Find and save the tracker name for this listfit
                for item_id, item_info in self.item_id_map.items():
                    if item_info.get('fit_dict') is listfit:
                        fit_entry['tracker_name'] = item_info.get('name', '')
                        break
                
                # Add all result parameters - including polynomial coefficients
                if result:
                    for param_name, param in result.params.items():
                        fit_entry[f'param_{param_name}'] = param.value
                    # Debug output to verify parameters are being captured
                    param_names = [p for p in result.params.keys()]
                    if param_names:
                        print(f"[DEBUG] Listfit parameters saved: {', '.join(param_names[:5])}{'...' if len(param_names) > 5 else ''}")
                
                all_fits.append(fit_entry)
            
            if all_fits:
                # Create DataFrame and save
                df = pd.DataFrame(all_fits)
                # Drop non-serializable columns
                df.drop(columns=['line', 'patches'], inplace=True, errors='ignore')
                
                filename = f"qasap_fits_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {len(all_fits)} fits to {filename}")
                print(f"  - Gaussian: {sum(1 for f in all_fits if f['type'] == 'gaussian')}")
                print(f"  - Voigt: {sum(1 for f in all_fits if f['type'] == 'voigt')}")
                print(f"  - Polynomial: {sum(1 for f in all_fits if f['type'] == 'continuum')}")
                print(f"  - Listfit: {sum(1 for f in all_fits if f['type'] == 'listfit')}")
            else:
                print("No fits to save. Fit some profiles first (Gaussian, Voigt, Polynomial, or Listfit).")

        # Load fit file with 'a' key
        elif event.key == 'a':
            self.load_fit_file()

        elif event.key == 'K':
            file_path, _ = QFileDialog.getOpenFileName(
                parent=self,  # 'self' refers to the main widget of your application
                caption="Select a file to load",  # Window title
                directory='',  # Start in the current directory
                filter="CSV Files (*.csv);;All Files (*)"  # File filters
            )

            try:
                # Determine the file type based on the first few rows of the file
                df = pd.read_csv(file_path)
                # Convert 'bounds' column from string to tuple
                df['bounds'] = df['bounds'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                fit_type = None

                # Check column presence to infer fit type
                if {'amp', 'mean', 'stddev'}.issubset(df.columns):
                    fit_type = 'gaussian'
                elif {'a', 'b'}.issubset(df.columns):
                    fit_type = 'continuum'
                elif {'gamma', 'sigma'}.issubset(df.columns):
                    fit_type = 'voigt'
                else:
                    raise ValueError("Unrecognized file format. Ensure the file contains valid fit parameters.")

                # Process the loaded fits
                if fit_type == 'continuum':
                    self.continuum_fits = df.to_dict(orient='records')
                    for fit in self.continuum_fits:
                        fit['line'] = None  # Reinitialize the line object
                        # Reconstruct the continuum lines
                        x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                        y_plot = fit['a'] * x_plot + fit['b']
                        fit['line'], = self.ax.plot(x_plot, y_plot, color="magenta", linestyle='--')

                elif fit_type == 'gaussian':
                    print(df.to_dict(orient='records'))
                    self.gaussian_fits = df.to_dict(orient='records')
                    for fit in self.gaussian_fits:
                        fit['line'] = None  # Reinitialize the line object
                        # Reconstruct the plot for each fit
                        x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                        _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
                        existing_continuum = self.continuum_model(x_plot, a, b)
                        y_plot = self.gaussian(x_plot, fit['amp'], fit['mean'], fit['stddev']) + existing_continuum
                        fit['line'], = self.ax.plot(x_plot, y_plot, color="red", linestyle='--')

                elif fit_type == 'voigt':
                    self.voigt_fits = df.to_dict(orient='records')
                    for fit in self.voigt_fits:
                        fit['line'] = None  # Reinitialize the line object
                        # Reconstruct the plot for each Voigt profile
                        x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                        _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
                        existing_continuum = self.continuum_model(x_plot, a, b)
                        y_plot = self.voigt(x_plot, fit['amp'], fit['center'], fit['sigma'], fit['gamma']) + existing_continuum
                        fit['line'], = self.ax.plot(x_plot, y_plot, color="orange", linestyle='--')

                self.ax.legend(loc='upper right')
                print(f"Successfully loaded {fit_type.capitalize()} fits from {file_path}.")

            except Exception as e:
                print(f"Failed to load fits: {e}")

        # Save a pdf of the current plot
        if event.key == '`':
            self.save_plot_as_pdf()

        if event.key == ',':  # Use ',' key to assign line ID and wavelength
            x_pos = event.xdata  # Get x position of mouse click
            self.selected_gaussian = None
            self.selected_voigt = None

            # Find the Gaussian fit corresponding to the selected x position
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.selected_gaussian = fit
                    print(f"Gaussian with parameters amp: {self.selected_gaussian['amp']}, mean: {self.selected_gaussian['mean']}, stddev: {self.selected_gaussian['stddev']} selected.")
                    break  # Select only the first Gaussian fit found within the bounds

            # OR find the Voigt fit corresponding to the selected x position
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.selected_voigt = fit
                    print(f"Voigt with parameters amp: {self.selected_voigt['amp']}, center: {self.selected_voigt['center']}, sigma: {self.selected_voigt['sigma']}, gamma: {self.selected_voigt['gamma']}  selected.")
                    break  # Select only the first Voigt fit found within the bounds
            
            if self.selected_gaussian or self.selected_voigt:
                # Open the LineListWindow to select a line                
                available_line_lists = self.get_all_available_line_lists()
                self.line_list_window = LineListWindow(available_line_lists=available_line_lists)
                self.line_list_window.selected_line.connect(self.assign_line_to_fit)  # Connect selection to assign function
                self.line_list_window.setWindowFlags(self.line_list_window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
                self.line_list_window.show()
                self.line_list_window.raise_()
                self.line_list_window.activateWindow()
            
        elif event.key == '<':
            # Find the marker and label nearest to the cursor to remove
            x_pos = event.xdata  # Current cursor x-position
            for marker in self.markers:
                left_bound, right_bound = getattr(marker, 'bounds')
                if left_bound <= x_pos <= right_bound:
                    line_id = getattr(marker, 'line_id', 'Unknown')
                    
                    # Remove the marker
                    marker.remove()
                    self.markers.remove(marker)
                    
                    # Remove associated label(s)
                    labels_to_remove = []
                    for label in self.labels:
                        if getattr(label, 'marker', None) is marker:
                            label.remove()
                            labels_to_remove.append(label)
                    
                    for label in labels_to_remove:
                        self.labels.remove(label)
                    
                    # Remove from item tracker
                    marker_id = f"marker_{self.markers.index(marker) if marker in self.markers else 'removed'}_{line_id}"
                    # Find and remove from tracker by searching for marker with this line_id
                    for item_id, item_info in list(self.item_tracker.items.items()):
                        if item_info.get('type') == 'marker' and line_id in item_info.get('name', ''):
                            self.item_tracker.remove_item(item_id)
                            break
                    
                    # Record action
                    self.record_action('remove_marker', f'Remove Marker: {line_id}')
                    
                    print(f"Removed marker within bounds ({left_bound}, {right_bound})")
                    plt.draw()
                    break

    def show_listfit_window(self):
        """Display the listfit component selection window"""
        self.listfit_window = ListfitWindow(self.listfit_bounds)
        self.listfit_window.fit_requested.connect(self.perform_listfit)
        self.listfit_window.bounds_cleared.connect(self.clear_listfit_bounds)
        self.listfit_window.components_changed.connect(self._on_listfit_components_changed)
        self.listfit_window.show()
    
    def clear_listfit_bounds(self):
        """Clear listfit bounds and remove bound lines from plot"""
        for line in self.listfit_bound_lines:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.listfit_bound_lines.clear()
        self.listfit_bounds = []
        self.listfit_mode = False
        plt.draw()
    
    def _on_listfit_components_changed(self, components):
        """Handle listfit components being added/removed - update plot immediately"""
        try:
            self.listfit_components = components
            # Redraw the canvas to reflect component changes
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating listfit components: {e}")
            import traceback
            traceback.print_exc()


    def perform_listfit(self, components):
        """Perform multi-component fitting"""
        if not self.listfit_bounds or len(self.listfit_bounds) < 2:
            print("Error: Invalid bounds for listfit")
            return
        
        left_bound, right_bound = sorted(self.listfit_bounds)
        
        # Extract data within bounds
        mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
        x_fit = self.x_data[mask]
        y_fit = self.spec[mask]
        err_fit = self.err[mask] if self.err is not None else None
        
        if len(x_fit) == 0:
            print("Error: No data within bounds")
            return
        
        # Extract data masks to exclude pixels from fit
        data_masks = [comp for comp in components if comp['type'] == 'data_mask']
        
        # Apply data masks - exclude pixels in masked regions from fit
        fit_mask = np.ones(len(x_fit), dtype=bool)
        for data_mask in data_masks:
            min_lambda = data_mask.get('min_lambda')
            max_lambda = data_mask.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                # Exclude pixels within this mask region
                fit_mask &= ~((x_fit >= min_lambda) & (x_fit <= max_lambda))
        
        # Apply mask to data
        x_fit_masked = x_fit[fit_mask]
        y_fit_masked = y_fit[fit_mask]
        err_fit_masked = err_fit[fit_mask] if err_fit is not None else None
        
        if len(x_fit_masked) == 0:
            print("Error: All data excluded by data masks")
            return
        
        # Build the composite model
        composite_model = self.build_composite_model(components, x_fit_masked, y_fit_masked, err_fit_masked)
        
        if composite_model is None:
            print("Error: Could not build composite model")
            return
        
        # Perform the fit with optional weights
        try:
            if err_fit_masked is not None:
                result = composite_model.fit(y_fit_masked, x=x_fit_masked, weights=1.0/err_fit_masked)
            else:
                result = composite_model.fit(y_fit_masked, x=x_fit_masked)
        except RecursionError as e:
            error_msg = (
                "Fit failed due to circular or conflicting constraints!\n\n"
                "This usually happens when:\n"
                "  • Parameters have conflicting bounds (e.g., min > max)\n"
                "  • Constraints form circular dependencies\n"
                "  • Fixed values conflict with bounds or other constraints\n\n"
                "Please check your constraint settings:\n"
                "  1. Verify min < max for all bounds\n"
                "  2. Check for circular parameter linking\n"
                "  3. Ensure fixed values don't conflict with bounds"
            )
            print(f"Error: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Fit Failed - Invalid Constraints", error_msg)
            return
        except Exception as e:
            error_msg = f"Fit failed with error: {str(e)}\n\nPlease check your constraints and try again."
            print(f"Error: {error_msg}")
            QtWidgets.QMessageBox.critical(self, "Fit Failed", error_msg)
            return
        
        # Check if fit succeeded OR if it converged but just failed on error estimation
        fit_converged = result.success or (result.nfree > 0 and result.ndata > result.nfree)
        
        if not fit_converged:
            print(f"Fit failed: {result.message}")
            return
        
        # Print warning if fit converged but error bars couldn't be estimated
        if not result.success:
            print("Warning: Fit converged but error-bar estimation failed (tolerance too small).")
            print("This is often due to numerical precision limits. The fit parameters are still valid.")
        
        print("Listfit completed successfully!")
        
        # Print custom fit report if no error spectrum
        if err_fit is None:
            self._print_listfit_report_no_errors(result)
        else:
            print(result.fit_report())
        
        # Check fit quality and warn if poor
        self._check_listfit_quality(result, y_fit)
        
        # Plot the components (pass all components so masks can be visualized)
        self.plot_listfit_components(result, components, x_fit, y_fit, err_fit, left_bound, right_bound)
        
        # Update residual display if shown
        if self.is_residual_shown:
            self.calculate_and_plot_residuals()
        
        # Extract initial guesses and constraints for storage
        initial_guesses = self._extract_listfit_initial_guesses(result, components)
        constraints_info = self._extract_listfit_constraints(components)
        
        # Store the fit
        self.listfit_fits.append({
            'bounds': (left_bound, right_bound),
            'components': components,
            'result': result,
            'x_data': x_fit,
            'y_data': y_fit,
            'err_data': err_fit,
            'initial_guesses': initial_guesses,
            'constraints': constraints_info
        })
        
        # Record action for undo/redo
        self.record_action('perform_listfit', f'Perform Listfit ({len(self.listfit_fits)} total fits)')
        
        # Clear listfit mode
        self.listfit_mode = False
        for line in self.listfit_bound_lines:
            line.remove()
        self.listfit_bound_lines.clear()
        self.listfit_bounds = []
        
        self.fig.canvas.draw_idle()  # Redraw to show listfit results

    def build_composite_model(self, components, x_fit, y_fit, err_fit):
        """Build a composite lmfit Model from component list with improved initial guesses for blended profiles"""
        from scipy.signal import find_peaks
        
        model = None
        gauss_count = 0
        voigt_count = 0
        poly_count = 0
        
        # Create continuum mask for polynomial fitting (avoid line profiles)
        continuum_mask = self._identify_continuum_regions(x_fit, y_fit)
        
        # Extract polynomial guess masks and data masks
        polynomial_guess_masks = [comp for comp in components if comp['type'] == 'polynomial_guess_mask']
        data_masks = [comp for comp in components if comp['type'] == 'data_mask']
        
        # **NEW: Two-stage fitting for better continuum estimates**
        # Stage 1: Fit polynomial alone to get good initial guesses
        polynomial_fits = {}  # Cache fitted polynomial parameters
        poly_components = [comp for comp in components if comp['type'] == 'polynomial']
        
        if poly_components:
            print("[DEBUG] Stage 1: Pre-fitting polynomial continuum to get better initial guesses...")
            for comp in poly_components:
                order = comp.get('order', 1)
                prefix = f'p{poly_count}_'
                
                # Estimate and fit polynomial on continuum regions
                poly_coeffs = self._estimate_polynomial_coefficients(x_fit, y_fit, order, continuum_mask, polynomial_guess_masks)
                poly_coeffs_reversed = poly_coeffs[::-1]
                
                try:
                    # Create a temporary polynomial model and fit it
                    poly_model = PolynomialModel(degree=order, prefix=prefix, independent_vars=['x'])
                    for i in range(order + 1):
                        param_name = f'{prefix}c{i}'
                        poly_model.set_param_hint(param_name, value=poly_coeffs_reversed[i])
                    
                    # Fit polynomial only (to get better starting guesses)
                    result_poly = poly_model.fit(y_fit, x=x_fit)
                    
                    # Cache the fitted parameters
                    polynomial_fits[prefix] = result_poly.params
                    print(f"[DEBUG] Polynomial {prefix} pre-fit complete. Chi² reduced = {result_poly.redchi:.3e}")
                except Exception as e:
                    print(f"[DEBUG] Warning: Pre-fit of polynomial {prefix} failed: {e}")
                    polynomial_fits[prefix] = None
                
                poly_count += 1
            
            # Reset poly_count for actual model building
            poly_count = 0
        
        # Find all peaks upfront for multi-component Gaussian/Voigt fitting
        # For multiple Gaussians/Voigts, use region-based peak detection to handle blended profiles
        
        # Count how many Gaussians and Voigts we need to fit
        num_gaussians = len([c for c in components if c['type'] == 'gaussian'])
        num_voigts = len([c for c in components if c['type'] == 'voigt'])
        num_line_components = num_gaussians + num_voigts
        
        y_abs = np.abs(y_fit)
        signal_mean = np.mean(y_abs)
        signal_std = np.std(y_abs)
        
        # Strategy for peak detection based on number of components
        if num_line_components > 1:
            # Multiple components: divide wavelength range into sub-regions
            # and find the strongest peak in each region
            peaks = self._find_peaks_for_multiple_components(x_fit, y_fit, num_line_components)
        else:
            # Single component: use global peak detection
            height_threshold = max(signal_mean * 0.1, signal_std * 0.2)
            peaks, properties = find_peaks(y_abs, height=height_threshold, distance=2)
            
            # Sort by height if we found peaks
            if len(peaks) > 0:
                sorted_indices = np.argsort(-properties['peak_heights'])
                peaks = peaks[sorted_indices]
        
        peak_index_for_component = 0  # Track which peak to use next
        
        for comp in components:
            # Skip mask features - they're not fitted, only used for polynomial guess or data exclusion
            if comp['type'] in ['polynomial_guess_mask', 'data_mask']:
                continue
            
            if comp['type'] == 'gaussian':
                # Assign this Gaussian to the next available peak
                if peak_index_for_component < len(peaks):
                    amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(
                        x_fit, y_fit, peak_idx=peaks[peak_index_for_component]
                    )
                    peak_index_for_component += 1
                else:
                    # Fallback if we run out of detected peaks
                    amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(x_fit, y_fit)
                
                prefix = f'g{gauss_count}_'
                gauss_model = Model(self.gaussian, prefix=prefix, independent_vars=['x'])
                
                # Set initial guesses FIRST with the detected peak positions
                gauss_model.set_param_hint(f'{prefix}amp', value=amp_guess)
                gauss_model.set_param_hint(f'{prefix}mean', value=center_guess)
                gauss_model.set_param_hint(f'{prefix}stddev', value=sigma_guess, min=1e-6)
                
                # Apply constraints AFTER initial guesses
                # Constraints may override the initial guess (e.g., for fixed values or linked parameters)
                # but they won't restrict the initial value unnecessarily
                if 'constraints' in comp:
                    self._apply_gaussian_constraints(gauss_model, prefix, comp['constraints'])
                
                if model is None:
                    model = gauss_model
                else:
                    model = model + gauss_model
                gauss_count += 1
            
            elif comp['type'] == 'voigt':
                # Assign this Voigt to the next available peak
                if peak_index_for_component < len(peaks):
                    amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(
                        x_fit, y_fit, peak_idx=peaks[peak_index_for_component]
                    )
                    peak_index_for_component += 1
                else:
                    # Fallback if we run out of detected peaks
                    amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(x_fit, y_fit)
                
                gamma_guess = sigma_guess * 0.5  # Reasonable starting point for gamma
                
                prefix = f'v{voigt_count}_'
                voigt_model = Model(self.voigt, prefix=prefix, independent_vars=['x'])
                
                # Set initial guesses FIRST with the detected peak positions
                voigt_model.set_param_hint(f'{prefix}amp', value=amp_guess)
                voigt_model.set_param_hint(f'{prefix}center', value=center_guess)
                voigt_model.set_param_hint(f'{prefix}sigma', value=sigma_guess, min=1e-6)
                voigt_model.set_param_hint(f'{prefix}gamma', value=gamma_guess, min=1e-6)
                
                # Apply constraints AFTER initial guesses
                if 'constraints' in comp:
                    self._apply_voigt_constraints(voigt_model, prefix, comp['constraints'])
                
                if model is None:
                    model = voigt_model
                else:
                    model = model + voigt_model
                voigt_count += 1
            
            elif comp['type'] == 'polynomial':
                order = comp.get('order', 1)
                prefix = f'p{poly_count}_'
                # Create polynomial model using lmfit's built-in PolynomialModel
                poly_model = PolynomialModel(degree=order, prefix=prefix, independent_vars=['x'])
                
                # **Use pre-fitted polynomial parameters if available (Stage 1 results)**
                if prefix in polynomial_fits and polynomial_fits[prefix] is not None:
                    print(f"[DEBUG] Using pre-fitted parameters for polynomial {prefix}")
                    # Use the fitted parameters from Stage 1
                    for i in range(order + 1):
                        param_name = f'{prefix}c{i}'
                        if param_name in polynomial_fits[prefix]:
                            fitted_value = polynomial_fits[prefix][param_name].value
                            poly_model.set_param_hint(param_name, value=fitted_value)
                            print(f"[DEBUG]   {param_name} = {fitted_value}")
                else:
                    # Fallback to coefficient estimation if pre-fit failed
                    print(f"[DEBUG] Using estimated coefficients for polynomial {prefix} (no pre-fit available)")
                    poly_coeffs = self._estimate_polynomial_coefficients(x_fit, y_fit, order, continuum_mask, polynomial_guess_masks)
                    
                    # np.polyfit returns coefficients from highest to lowest degree (x^n, ..., x^0)
                    # but lmfit expects them from lowest to highest (c0=x^0, c1=x^1, ..., cn=x^n)
                    # So we need to reverse them
                    poly_coeffs_reversed = poly_coeffs[::-1]
                    
                    for i in range(order + 1):
                        param_name = f'{prefix}c{i}'
                        poly_model.set_param_hint(param_name, value=poly_coeffs_reversed[i])
                
                if model is None:
                    model = poly_model
                else:
                    model = model + poly_model
                poly_count += 1
        
        return model

    def _estimate_polynomial_coefficients(self, x_fit, y_fit, order, continuum_mask, polynomial_guess_masks=None):
        """Estimate polynomial coefficients using robust iterative sigma-clipping on median-filtered data
        
        Args:
            x_fit: x data
            y_fit: y data
            order: polynomial order
            continuum_mask: boolean mask of continuum regions
            polynomial_guess_masks: list of {'min_lambda': x1, 'max_lambda': x2} to exclude from guess
        """
        from scipy.ndimage import median_filter
        
        # Create mask for user-specified wavelength ranges to exclude
        exclude_mask = np.zeros(len(x_fit), dtype=bool)
        if polynomial_guess_masks:
            for mask_feat in polynomial_guess_masks:
                min_lambda = mask_feat.get('min_lambda')
                max_lambda = mask_feat.get('max_lambda')
                if min_lambda is not None and max_lambda is not None:
                    exclude_mask |= (x_fit >= min_lambda) & (x_fit <= max_lambda)
        
        # Use median filter to estimate the smooth continuum (removes line profiles)
        # Window size should be large enough to span a line profile but small enough to preserve continuum shape
        window_size = max(5, int(len(y_fit) * 0.08))  # ~8% of spectrum width for better continuum
        # Make window size odd (required for median_filter)
        if window_size % 2 == 0:
            window_size += 1
        
        median_filtered = median_filter(y_fit, size=window_size)
        
        # Combine continuum mask and user-defined exclusion mask
        combined_mask = continuum_mask & ~exclude_mask
        
        # Use iterative sigma-clipping on the median-filtered data to reject outliers
        # This helps identify the true continuum level more robustly
        continuum_estimate = self._robust_continuum_estimate(x_fit, median_filtered, order, combined_mask)
        if continuum_estimate is not None:
            return continuum_estimate
        
        # Fallback 1: Try fitting to combined mask regions on median-filtered data
        if combined_mask.sum() > order + 1:
            x_masked = x_fit[combined_mask]
            y_masked = median_filtered[combined_mask]
            try:
                poly_coeffs = np.polyfit(x_masked, y_masked, order)
                return poly_coeffs
            except:
                pass
        
        # Fallback 2: Fit to the full median-filtered data (exclude user-masked regions)
        if (~exclude_mask).sum() > order + 1:
            x_unmasked = x_fit[~exclude_mask]
            y_unmasked = median_filtered[~exclude_mask]
            try:
                poly_coeffs = np.polyfit(x_unmasked, y_unmasked, order)
                return poly_coeffs
            except:
                pass
        
        # Fallback 3: Fit lower order polynomial if current order fails
        for reduced_order in range(order - 1, -1, -1):
            try:
                if (~exclude_mask).sum() > reduced_order + 1:
                    x_unmasked = x_fit[~exclude_mask]
                    y_unmasked = median_filtered[~exclude_mask]
                    poly_coeffs = np.polyfit(x_unmasked, y_unmasked, reduced_order)
                else:
                    poly_coeffs = np.polyfit(x_fit, median_filtered, reduced_order)
                # Pad with zeros to match the requested order
                padding = np.zeros(order - reduced_order)
                poly_coeffs = np.concatenate([padding, poly_coeffs])
                return poly_coeffs
            except:
                continue
        
        # Last resort fallback: constant estimate (median of unmasked data)
        if (~exclude_mask).sum() > 0:
            constant_value = np.median(median_filtered[~exclude_mask])
        else:
            constant_value = np.median(median_filtered)
        poly_coeffs = np.zeros(order + 1)
        poly_coeffs[-1] = constant_value  # Set constant term
        return poly_coeffs

    def _robust_continuum_estimate(self, x_fit, y_filtered, order, combined_mask, n_iterations=3, sigma_clip=2.0):
        """Use iterative sigma-clipping to robustly estimate the continuum and fit polynomial"""
        try:
            # Start with the combined mask (continuum regions + user-specified safe regions)
            mask = combined_mask.copy()
            
            for iteration in range(n_iterations):
                if mask.sum() <= order + 1:
                    # Not enough points left, abort
                    return None
                
                # Fit polynomial to current set of points
                try:
                    poly_coeffs = np.polyfit(x_fit[mask], y_filtered[mask], order)
                    poly_fit = np.polyval(poly_coeffs, x_fit)
                except:
                    return None
                
                # Calculate residuals
                residuals = y_filtered - poly_fit
                std_residuals = np.std(residuals[mask])
                
                if std_residuals == 0:
                    # No variation, use this fit
                    return poly_coeffs
                
                # Sigma-clip: reject points that deviate too much from the fit
                # (These are likely line profiles, not continuum)
                new_mask = np.abs(residuals) < sigma_clip * std_residuals
                
                # If mask didn't change much, we've converged
                if np.sum(new_mask == mask) > len(mask) * 0.95:
                    return poly_coeffs
                
                mask = new_mask
            
            # Return the final fit
            if mask.sum() > order + 1:
                poly_coeffs = np.polyfit(x_fit[mask], y_filtered[mask], order)
                return poly_coeffs
            
            return None
        except:
            return None

    def _extract_listfit_initial_guesses(self, result, components):
        """Extract initial guesses from lmfit result and components"""
        guesses = {}
        for param_name, param in result.params.items():
            if param.init_value is not None:
                guesses[param_name] = param.init_value
        return guesses
    
    def _extract_listfit_constraints(self, components):
        """Extract constraint information from components"""
        constraints = {}
        for comp in components:
            if comp['type'] in ['gaussian', 'voigt']:
                comp_id = comp.get('id', '')
                if 'constraints' in comp:
                    constraints[f"{comp['type']}_{comp_id}"] = comp['constraints']
        return constraints
    
    def _print_listfit_report_no_errors(self, result):
        """Print custom fit report when no error spectrum is available.
        
        Without error bars, chi-squared is not properly defined.
        Instead, we report the Sum of Squared Residuals (SSR).
        """
        report = result.fit_report()
        
        # Replace chi-square references with SSR (Sum of Squared Residuals)
        report = report.replace('chi-square', 'sum of squared residuals (SSR)')
        report = report.replace('reduced chi-square', 'mean squared residual')
        
        # Add explanation
        explanation = (
            "\n## NOTE: No error spectrum provided ##\n"
            "The fit quality metrics are reported as:\n"
            "  • Sum of Squared Residuals (SSR) = Σ(data - model)²\n"
            "  • Mean Squared Residual = SSR / (N - n_params)\n"
            "These are NOT true chi-squared values (which require error bars).\n"
            "Lower values indicate better fit quality.\n"
        )
        
        print(report)
        print(explanation)

    def _check_listfit_quality(self, result, y_fit):
        """Check the quality of the Listfit and warn user if fit is poor"""
        warnings = []
        
        # Criterion 1: Reduced chi-square
        # For a good fit, reduced chi-square should be close to 1
        # Much > 1 indicates poor fit (underfitting or bad initial guesses)
        rchi = result.redchi
        if rchi is not None and rchi > 5.0:
            warnings.append(f"High reduced chi-square ({rchi:.2f} >> 1): Fit may be underfitting data. Consider adding more components or higher-order polynomial.")
        elif rchi is not None and rchi > 2.0:
            warnings.append(f"Moderate reduced chi-square ({rchi:.2f} > 1): Fit quality could be improved.")
        
        # Criterion 2: R-squared value
        # R-squared should be close to 1 for a good fit
        # Can extract from fit_report string or calculate from residuals
        try:
            # Calculate R-squared from best_fit and data
            ss_res = np.sum((y_fit - result.best_fit) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            if r_squared < 0.80:
                warnings.append(f"Low R-squared ({r_squared:.3f} < 0.80): Poor goodness-of-fit. Consider adjusting components.")
            elif r_squared < 0.90:
                warnings.append(f"Moderate R-squared ({r_squared:.3f} < 0.90): Consider refitting with better initial guesses.")
        except:
            pass
        
        # Criterion 3: Check for suspicious parameter values (very large)
        # These might indicate the optimizer went to extreme values
        extreme_value = False
        for param_name, param in result.params.items():
            if param.value is not None:
                # Check if parameter value is extremely large
                if abs(param.value) > 1e6:
                    extreme_value = True
                    break
        
        if extreme_value:
            warnings.append("Some parameters have extreme values: Fit may be unstable. Try refitting with adjusted component bounds.")
        
        # Print warnings
        if warnings:
            print("\n" + "="*70)
            print("WARNING: FIT QUALITY WARNINGS - CONSIDER RE-FITTING")
            print("="*70)
            for warning in warnings:
                print(f"  • {warning}")
            print("="*70)
            print("Tip: Try adjusting initial parameter guesses or component configuration.")
            print("="*70 + "\n")
    
    def _apply_gaussian_constraints(self, model, prefix, constraints):
        """Apply constraints to a Gaussian model"""
        if not constraints:
            return
        
        # Amplitude bounds or fixed value
        if constraints.get('amplitude_fixed'):
            fixed_val = constraints.get('amplitude_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}amp', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}amp', vary=False)
        else:
            amp_min, amp_max = constraints.get('amplitude_bounds', ('', ''))
            if amp_min:
                model.set_param_hint(f'{prefix}amp', min=float(amp_min))
            if amp_max:
                model.set_param_hint(f'{prefix}amp', max=float(amp_max))
        
        # Mean bounds or fixed value
        if constraints.get('mean_fixed'):
            fixed_val = constraints.get('mean_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}mean', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}mean', vary=False)
        else:
            mean_min, mean_max = constraints.get('mean_bounds', ('', ''))
            if mean_min:
                model.set_param_hint(f'{prefix}mean', min=float(mean_min))
            if mean_max:
                model.set_param_hint(f'{prefix}mean', max=float(mean_max))
        
        # Sigma bounds or fixed value
        if constraints.get('sigma_fixed'):
            fixed_val = constraints.get('sigma_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}stddev', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}stddev', vary=False)
        else:
            sigma_min, sigma_max = constraints.get('sigma_bounds', ('', ''))
            if sigma_min:
                model.set_param_hint(f'{prefix}stddev', min=float(sigma_min))
            if sigma_max:
                model.set_param_hint(f'{prefix}stddev', max=float(sigma_max))
        
        # Apply linked constraints (multiple parameter linking)
        linked_constraints = constraints.get('linked_constraints', [])
        for linked in linked_constraints:
            parameter = linked.get('parameter', 'mean')
            expression = linked.get('expression', '')
            if expression:
                # Map parameter name to parameter key
                param_map = {'mean': 'mean', 'stddev': 'stddev', 'amp': 'amp', 'sigma': 'sigma'}
                param_key = param_map.get(parameter, parameter)
                model.set_param_hint(f'{prefix}{param_key}', expr=expression)
    
    def _apply_voigt_constraints(self, model, prefix, constraints):
        """Apply constraints to a Voigt model"""
        if not constraints:
            return
        
        # Amplitude bounds or fixed value
        if constraints.get('amplitude_fixed'):
            fixed_val = constraints.get('amplitude_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}amp', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}amp', vary=False)
        else:
            amp_min, amp_max = constraints.get('amplitude_bounds', ('', ''))
            if amp_min:
                model.set_param_hint(f'{prefix}amp', min=float(amp_min))
            if amp_max:
                model.set_param_hint(f'{prefix}amp', max=float(amp_max))
        
        # Center bounds or fixed value
        if constraints.get('center_fixed'):
            fixed_val = constraints.get('center_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}center', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}center', vary=False)
        else:
            center_min, center_max = constraints.get('center_bounds', ('', ''))
            if center_min:
                model.set_param_hint(f'{prefix}center', min=float(center_min))
            if center_max:
                model.set_param_hint(f'{prefix}center', max=float(center_max))
        
        # Sigma bounds or fixed value
        if constraints.get('sigma_fixed'):
            fixed_val = constraints.get('sigma_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}sigma', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}sigma', vary=False)
        else:
            sigma_min, sigma_max = constraints.get('sigma_bounds', ('', ''))
            if sigma_min:
                model.set_param_hint(f'{prefix}sigma', min=float(sigma_min))
            if sigma_max:
                model.set_param_hint(f'{prefix}sigma', max=float(sigma_max))
        
        # Gamma bounds or fixed value
        if constraints.get('gamma_fixed'):
            fixed_val = constraints.get('gamma_fixed_value')
            if fixed_val:
                model.set_param_hint(f'{prefix}gamma', value=float(fixed_val), vary=False)
            else:
                model.set_param_hint(f'{prefix}gamma', vary=False)
        else:
            gamma_min, gamma_max = constraints.get('gamma_bounds', ('', ''))
            if gamma_min:
                model.set_param_hint(f'{prefix}gamma', min=float(gamma_min))
            if gamma_max:
                model.set_param_hint(f'{prefix}gamma', max=float(gamma_max))
        
        # Apply linked constraints (multiple parameter linking)
        linked_constraints = constraints.get('linked_constraints', [])
        for linked in linked_constraints:
            parameter = linked.get('parameter', 'center')
            expression = linked.get('expression', '')
            if expression:
                # Map parameter name to parameter key
                param_map = {'center': 'center', 'sigma': 'sigma', 'amp': 'amp', 'gamma': 'gamma'}
                param_key = param_map.get(parameter, parameter)
                model.set_param_hint(f'{prefix}{param_key}', expr=expression)

    def _clamp_to_bounds(self, value, bounds):
        """Clamp a value to be within specified bounds.
        
        Args:
            value: The value to clamp
            bounds: Tuple of (min_str, max_str) where strings are empty if not specified
                   (from constraints like amplitude_bounds, mean_bounds, etc.)
        
        Returns:
            Clamped value
        """
        min_str, max_str = bounds
        
        if not min_str and not max_str:
            return value  # No bounds, return as-is
        
        try:
            if min_str:
                min_val = float(min_str)
                value = max(value, min_val)
            if max_str:
                max_val = float(max_str)
                value = min(value, max_val)
        except (ValueError, TypeError):
            pass  # If conversion fails, just return value as-is
        
        return value

    def _find_peaks_for_multiple_components(self, x_fit, y_fit, num_components):
        """Find peaks for multiple blended components by dividing wavelength range.
        
        For heavily blended profiles, divide the x-range into sub-regions
        and find the strongest peak in each region. This ensures each component
        gets a different starting wavelength.
        
        Args:
            x_fit: wavelength array
            y_fit: flux array
            num_components: number of Gaussians/Voigts to fit
        
        Returns:
            Array of peak indices, one per component (or fewer if not enough found)
        """
        from scipy.signal import find_peaks
        
        y_abs = np.abs(y_fit)
        peaks = []
        
        if num_components <= 1:
            return np.array(peaks)
        
        # Divide the wavelength range into num_components regions
        region_size = len(x_fit) / num_components
        
        for region_idx in range(num_components):
            # Define this region's bounds
            region_start = int(region_idx * region_size)
            region_end = int((region_idx + 1) * region_size)
            if region_idx == num_components - 1:
                region_end = len(x_fit)  # Ensure last region goes to the end
            
            # Get data in this region
            region_y = y_abs[region_start:region_end]
            
            if len(region_y) > 0:
                # Find the strongest point in this region (could be a peak or part of blended profile)
                strongest_idx_in_region = np.argmax(region_y)
                peak_idx = region_start + strongest_idx_in_region
                peaks.append(peak_idx)
        
        return np.array(peaks)

    def _identify_continuum_regions(self, x_fit, y_fit):
        """Identify regions likely to be continuum (not dominated by line profiles)"""
        from scipy.signal import find_peaks
        
        # Use peak detection to identify line profile regions
        median_y = np.median(y_fit)
        deviation = np.abs(y_fit - median_y)
        
        # Find peaks in deviation from median
        peaks, _ = find_peaks(deviation, height=np.std(y_fit)*0.5)
        
        # Mark regions around peaks as NOT continuum
        continuum_mask = np.ones(len(x_fit), dtype=bool)
        peak_width = max(2, int(len(x_fit) * 0.05))  # ~5% of range around each peak
        
        for peak_idx in peaks:
            start = max(0, peak_idx - peak_width)
            end = min(len(x_fit), peak_idx + peak_width)
            continuum_mask[start:end] = False
        
        # Ensure edges are marked as continuum (usually safe regions)
        edge_width = max(2, int(len(x_fit) * 0.1))
        continuum_mask[:edge_width] = True
        continuum_mask[-edge_width:] = True
        
        # Need at least some continuum points
        if continuum_mask.sum() < 3:
            # Fallback: mark everything as continuum
            continuum_mask[:] = True
        
        return continuum_mask

    def _estimate_gaussian_params(self, x_fit, y_fit, peak_idx=None):
        """Estimate Gaussian parameters using peak detection with FWHM
        
        Args:
            x_fit: x data
            y_fit: y data
            peak_idx: Optional specific peak index to use. If None, uses the global maximum.
        """
        # Find the peak to use
        if peak_idx is not None:
            peak_index = peak_idx
        else:
            peak_index = np.argmax(np.abs(y_fit))
        
        peak_x = x_fit[peak_index]
        peak_y = y_fit[peak_index]
        
        # Amplitude is the peak value
        amp_guess = peak_y
        
        # Estimate sigma from FWHM (Full Width at Half Maximum) around this peak
        half_max = amp_guess / 2.0
        try:
            # Find indices where signal is above half maximum, but limit search to region around peak
            search_width = len(x_fit) // 3  # Search within 1/3 of spectrum on each side
            search_start = max(0, peak_index - search_width)
            search_end = min(len(x_fit), peak_index + search_width)
            
            # Find indices in the search region where signal is above half maximum
            if amp_guess > 0:
                indices_above_half = np.where(y_fit[search_start:search_end] > half_max)[0] + search_start
            else:
                indices_above_half = np.where(y_fit[search_start:search_end] < half_max)[0] + search_start
            
            if len(indices_above_half) >= 2:
                fwhm_estimate = x_fit[indices_above_half[-1]] - x_fit[indices_above_half[0]]
                # Convert FWHM to sigma: FWHM = 2.355 * sigma for Gaussian
                sigma_guess = fwhm_estimate / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                # Ensure sigma is positive and not too small
                sigma_guess = max(sigma_guess, (x_fit[-1] - x_fit[0]) / 100.0)
            else:
                # Fallback to a reasonable default
                sigma_guess = (x_fit[-1] - x_fit[0]) / 10.0
        except:
            # Fallback to a reasonable default
            sigma_guess = (x_fit[-1] - x_fit[0]) / 10.0
        
        # Center is at the peak location
        center_guess = peak_x
        
        return amp_guess, center_guess, sigma_guess

    def plot_listfit_components(self, result, components, x_fit, y_fit, err_fit, left_bound, right_bound):
        """Plot the fitted components with different colors"""
        x_smooth = np.linspace(x_fit.min(), x_fit.max(), len(x_fit) * 50)
        
        # Color mapping for components
        colors = {'gaussian': 'red', 'voigt': 'orange', 'polynomial': 'magenta'}
        
        # Plot mask regions as gray fill patches
        data_masks = [comp for comp in components if comp['type'] == 'data_mask']
        polynomial_guess_masks = [comp for comp in components if comp['type'] == 'polynomial_guess_mask']
        
        mask_count = 0
        for mask in data_masks:
            min_lambda = mask.get('min_lambda')
            max_lambda = mask.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                patch = self.ax.axvspan(min_lambda, max_lambda, alpha=0.2, color='gray', zorder=1)
                # Register with ItemTracker
                position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                self.register_item('data_mask', f'Data Mask {mask_count+1} ({min_lambda:.2f}-{max_lambda:.2f} Å)', 
                                 patch_obj=patch, position=position_str, color='gray')
                mask_count += 1
        
        poly_mask_count = 0
        for mask in polynomial_guess_masks:
            min_lambda = mask.get('min_lambda')
            max_lambda = mask.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                patch = self.ax.axvspan(min_lambda, max_lambda, alpha=0.1, color='lightgray', zorder=0.5, linestyle='--', edgecolor='gray', linewidth=1)
                # Register with ItemTracker
                position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                self.register_item('polynomial_guess_mask', f'Poly Guess Mask {poly_mask_count+1} ({min_lambda:.2f}-{max_lambda:.2f} Å)', 
                                 patch_obj=patch, position=position_str, color='lightgray')
                poly_mask_count += 1
        
        # Plot individual components
        gauss_count = 0
        voigt_count = 0
        poly_count = 0
        
        for comp in components:
            comp_type = comp['type']
            
            # Skip mask types - they're already plotted above
            if comp_type in ['polynomial_guess_mask', 'data_mask']:
                continue
            
            color = colors[comp_type]
            params = result.params
            
            if comp_type == 'gaussian':
                prefix = f'g{gauss_count}_'
                g_amp = params[f'{prefix}amp'].value
                g_mean = params[f'{prefix}mean'].value
                g_stddev = params[f'{prefix}stddev'].value
                
                # Extract errors from lmfit results
                g_amp_err = params[f'{prefix}amp'].stderr if params[f'{prefix}amp'].stderr is not None else 0.0
                g_mean_err = params[f'{prefix}mean'].stderr if params[f'{prefix}mean'].stderr is not None else 0.0
                g_stddev_err = params[f'{prefix}stddev'].stderr if params[f'{prefix}stddev'].stderr is not None else 0.0
                
                y_component = self.gaussian(x_smooth, g_amp, g_mean, g_stddev)
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle='--', linewidth=2)
                
                # Add to gaussian_fits for redshift mode
                gaussian_fit = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'component_id': self.component_id,
                    'amp': g_amp, 'amp_err': g_amp_err,
                    'mean': g_mean, 'mean_err': g_mean_err,
                    'stddev': g_stddev, 'stddev_err': g_stddev_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': None,
                    'line_wavelength': None,
                    'line': line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift
                }
                self.gaussian_fits.append(gaussian_fit)
                # Register with ItemTracker
                position_str = f"λ: {g_mean:.2f} Å"
                item_id = self.register_item('gaussian', f'Gaussian {gauss_count+1}', fit_dict=gaussian_fit, 
                                           line_obj=line, position=position_str, color=color)
                self.component_id += 1
                gauss_count += 1
            
            elif comp_type == 'voigt':
                prefix = f'v{voigt_count}_'
                v_amp = params[f'{prefix}amp'].value
                v_center = params[f'{prefix}center'].value
                v_sigma = params[f'{prefix}sigma'].value
                v_gamma = params[f'{prefix}gamma'].value
                
                # Extract errors from lmfit results
                v_amp_err = params[f'{prefix}amp'].stderr if params[f'{prefix}amp'].stderr is not None else 0.0
                v_center_err = params[f'{prefix}center'].stderr if params[f'{prefix}center'].stderr is not None else 0.0
                v_sigma_err = params[f'{prefix}sigma'].stderr if params[f'{prefix}sigma'].stderr is not None else 0.0
                v_gamma_err = params[f'{prefix}gamma'].stderr if params[f'{prefix}gamma'].stderr is not None else 0.0
                
                y_component = self.voigt(x_smooth, v_amp, v_center, v_sigma, v_gamma)
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle='--', linewidth=2)
                
                # Add to voigt_fits for redshift mode
                voigt_fit = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'component_id': self.component_id,
                    'amp': v_amp, 'amp_err': v_amp_err,
                    'center': v_center, 'center_err': v_center_err,
                    'sigma': v_sigma, 'sigma_err': v_sigma_err,
                    'gamma': v_gamma, 'gamma_err': v_gamma_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': None,
                    'line_wavelength': None,
                    'line': line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift
                }
                self.voigt_fits.append(voigt_fit)
                # Register with ItemTracker
                position_str = f"λ: {v_center:.2f} Å"
                item_id = self.register_item('voigt', f'Voigt {voigt_count+1}', fit_dict=voigt_fit,
                                           line_obj=line, position=position_str, color=color)
                self.component_id += 1
                voigt_count += 1
            
            elif comp_type == 'polynomial':
                prefix = f'p{poly_count}_'
                order = comp.get('order', 1)
                poly_coeffs = []
                for i in range(order + 1):
                    poly_coeffs.append(params[f'{prefix}c{i}'].value)
                # Reverse coefficients for np.polyval (expects highest order first)
                poly_coeffs = poly_coeffs[::-1]
                y_component = np.polyval(poly_coeffs, x_smooth)
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle='--', linewidth=2)
                # Register with ItemTracker - store metadata for deletion handling
                position_str = f"λ: {left_bound:.2f}-{right_bound:.2f} Å"
                poly_fit_dict = {
                    'listfit_bounds': (left_bound, right_bound),
                    'poly_index': poly_count,
                    'order': order
                }
                item_id = self.register_item('polynomial', f'Polynomial (order={order})', fit_dict=poly_fit_dict, line_obj=line,
                                           position=position_str, color=color)
                poly_count += 1
        
        # Plot the total fit result (for visualization only, NOT included in draw_total_line to avoid double-counting)
        y_total_fit = result.eval(x=x_smooth)
        line, = self.ax.plot(x_smooth, y_total_fit, label='Total Listfit', color='#003d7a', linestyle='-', linewidth=2)
        
        # Register Total Listfit with ItemTracker
        position_str = f"λ: {left_bound:.2f}-{right_bound:.2f} Å"
        total_fit_dict = {
            'listfit_bounds': (left_bound, right_bound),
            'is_total_fit': True
        }
        self.register_item('listfit_total', f'Total Listfit', fit_dict=total_fit_dict, line_obj=line,
                          position=position_str, color='#003d7a')
        
        self.ax.legend(loc='upper right')



