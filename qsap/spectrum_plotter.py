# qsap Spectrum Plotter --- v0.12
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
  where QSAP was launched from.
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

# Configure matplotlib to handle exceptions gracefully
import matplotlib as mpl
mpl.rcParams['figure.raise_window'] = False

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.cm as cm
from matplotlib import font_manager
from astropy.io import fits
import os

# Suppress matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib')

# Register Computer Modern Unicode fonts from local fonts/ directory (qsap/fonts/)
fonts_dir = os.path.join(os.path.dirname(__file__), '..', 'fonts')
fonts_dir = os.path.abspath(fonts_dir)
if os.path.isdir(fonts_dir):
    for font_file in os.listdir(fonts_dir):
        if font_file.endswith('.otf'):
            font_path = os.path.join(fonts_dir, font_file)
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception as e:
                pass  # Silently skip fonts that fail to load
    # Configure matplotlib to use Computer Modern Serif Unicode
    plt.rcParams['font.serif'] = ['CMU Serif', 'DejaVu Serif']
    plt.rcParams['font.sans-serif'] = ['CMU Sans Serif', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'serif'

# scipy and lmfit are imported lazily in functions that need them to speed up startup
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QIcon, QKeyEvent
from PyQt5.QtWidgets import QFileDialog
from datetime import datetime
import ast
import re
from qsap.spectrum_io import SpectrumIO

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
from .qsap_file_handler import QSAPFileHandler


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
        
        # Enable text selection and copying
        self.text_edit.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | 
            QtCore.Qt.TextSelectableByKeyboard | 
            QtCore.Qt.LinksAccessibleByMouse
        )
        
        # Set size policy to expand to fill available space
        self.text_edit.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.text_edit.setMinimumHeight(100)  # Set minimum height instead of maximum
        self.text_edit.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f5f5f5;
                color: #333333;
                font-family: 'Courier New', monospace;
                font-size: 12pt;
                border: 1px solid #cccccc;
            }
        """)
        
        # Enable context menu for copy/paste operations
        self.text_edit.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.text_edit.customContextMenuRequested.connect(self.show_context_menu)
        
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        # Set the OutputPanel itself to expand
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
    
    def show_context_menu(self, position):
        """Show context menu for copy/select all operations."""
        menu = QtWidgets.QMenu()
        
        # Copy action
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.text_edit.copy)
        
        # Select All action
        select_all_action = menu.addAction("Select All")
        select_all_action.triggered.connect(self.text_edit.selectAll)
        
        # Clear action
        menu.addSeparator()
        clear_action = menu.addAction("Clear")
        clear_action.triggered.connect(self.clear_output)
        
        menu.exec_(self.text_edit.mapToGlobal(position))
    
    def append_text(self, text):
        """Append text to the output panel."""
        self.text_edit.appendPlainText(text.rstrip('\n'))
        # Auto-scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_output(self):
        """Clear the output panel."""
        self.text_edit.clear()
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for copy/select all."""
        if event.key() == QtCore.Qt.Key_C and event.modifiers() == QtCore.Qt.ControlModifier:
            # Ctrl+C: Copy selected text
            self.text_edit.copy()
            event.accept()
        elif event.key() == QtCore.Qt.Key_A and event.modifiers() == QtCore.Qt.ControlModifier:
            # Ctrl+A: Select all text
            self.text_edit.selectAll()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def restore_streams(self):
        """Restore original stdout/stderr."""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr


class HelpWindow(QtWidgets.QDialog):
    """Help window displaying all keyboard shortcuts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        from qsap.ui_utils import get_qsap_icon
        self.setWindowTitle("QSAP - Keyboard Shortcuts Help")
        self.setWindowIcon(get_qsap_icon())
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
        return """# QSAP Keyboard Shortcuts

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
- **q** or **Q** - Quit QSAP

## Help
- **?** - Show this help window

## File Storage
All saved screenshots, redshifts, and profile info are by default stored in the directory where QSAP was launched from.
This can be changed in the settings menu.
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
        self.flux_scale_factor = 1.0  # Track the flux scaling applied to the spectrum
        self.file_flag = file_flag
        self.lsf = lsf
        self.lsf_kernel_x = None
        self.lsf_kernel_y = None

        # Process LSF
        self.process_lsf(lsf)

        # Set window properties first
        from qsap.ui_utils import get_qsap_icon
        self.setWindowTitle("QSAP - Spectrum Viewer")
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(100, 100, 1200, 700)

        # Initialize the control panel (will be docked later in plot_spectrum)
        # This must be done BEFORE create_menu_bar() since the menu bar references windows created here
        self.init_controlpanel()
        
        # Initialize settings panel
        self.save_directory = str(Path.cwd())  # Default to app launch directory
        self.load_directory = str(Path.cwd())  # Default to app launch directory
        self.qsap_handler = QSAPFileHandler(self.save_directory)  # Initialize QSAP file handler
        self.init_settings_panel()
        
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
        self.poly_order = 2  # Default polynomial order for continuum fitting

        self.listfit_mode = False
        self.listfit_bounds = []
        self.listfit_bound_lines = []
        self.listfit_components = []
        self.listfit_fits = []  # Stores completed listfit results
        self.listfit_component_lines = {}  # Store plotted component lines by component ID
        self.listfit_polynomials = {}  # Store polynomial data for listfit
        self.deleted_listfit_polynomials = set()  # Track deleted polynomial item_ids for residual calculation

        self.redshift_estimation_mode = False
        self.rest_wavelength = None  # Set to `None` if no initial rest wavelength
        self.rest_id = None
        self.wavelength_unit = "Å"  # Current wavelength unit (Å, nm, or µm)

        # Equivalent Width Calculation Mode
        self.calculate_ew_enabled = True  # Checkbox: "Calculate EW automatically" (ON by default)
        self.plot_mc_profiles_enabled = False  # Checkbox: "Plot MC Profiles"
        self.mc_profile_lines_current = []  # Store MC profile line objects for removal
        self.calculate_ew_selection_mode = False  # Mode for selecting profiles via spacebar/mouse click

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

        # Legend tracking - track which profile types have been added to legend
        self.legend_profile_types = set()  # Tracks which profile types have a legend entry

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
        # Connect visibility changes if signal is available
        if hasattr(self.action_history_window, 'visibilityChanged'):
            self.action_history_window.visibilityChanged.connect(self.on_window_visibility_changed)
        if hasattr(self.action_history_window, 'destroyed'):
            self.action_history_window.destroyed.connect(self.on_window_visibility_changed)
        
        # Track if this is the first spectrum load (for initial action recording)
        self.is_first_load = True
        self.initial_spectrum_file = fits_file
        
        # Active line lists tracking (resources_dir and line_list_selector already initialized earlier)
        self.active_line_lists = []  # {linelist: LineList, color: str}
        self.current_linelist_lines = []  # Store plotted linelist lines for removal
        # Line list annotation offsets (normalized 0-1 relative to plotting window)
        self.linelist_x_offset = 0.02  # Default 2% offset in x direction (positive = right)
        self.linelist_y_offset = 0.02  # Default 2% offset in y direction (positive = up)
        
        # Load color configuration
        self.colors = self._load_color_config()

    def _load_color_config(self):
        """Load color configuration from config_colors.json"""
        import json
        config_path = Path(__file__).parent / 'config_colors.json'
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to hardcoded defaults if config file not found
            return {
                "profiles": {
                    "gaussian": {"color": "red", "linestyle": "--", "linewidth": 1.5},
                    "voigt": {"color": "orange", "linestyle": "--", "linewidth": 1.5},
                    "continuum_line": {"color": "magenta", "linestyle": "--", "linewidth": 1.5},
                    "continuum_region": {"color": "magenta", "alpha": 0.3},
                    "total_line": {"color": "#003d7a", "linestyle": "-", "linewidth": 2}
                },
                "spectrum": {
                    "data": {"color": "black", "linestyle": "-"},
                    "error": {"color": "red", "linestyle": "--", "alpha": 0.4}
                },
                "residual": {"color": "royalblue", "linestyle": "-"},
                "preview": {"color": "lime", "linestyle": "-", "linewidth": 2},
                "reference_lines": {"color": "gray", "linestyle": "--", "linewidth": 1}
            }

    def create_menu_bar(self):
        """Create the menu bar with QSAP, File, Edit, and View menus"""
        menubar = self.menuBar()
        
        # Create QSAP menu (application menu on macOS)
        self.qsap_menu = menubar.addMenu("QSAP")
        
        # Add Quit action to QSAP menu
        quit_action = self.qsap_menu.addAction("Quit QSAP")
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
        
        # Create Help menu
        self.help_menu = menubar.addMenu("Help")
        help_action = self.help_menu.addAction("Show Help")
        help_action.setShortcut("?")
        help_action.triggered.connect(self.show_help_window)

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
        
        # Add Control Panel Dock Widget
        if hasattr(self, 'control_panel_dock') and self.control_panel_dock is not None:
            is_visible = self.control_panel_dock.isVisible()
            windows.append(("Control Panel", self.control_panel_dock, is_visible))
        
        # Add Right Dock Widget (Fitting Options)
        if hasattr(self, 'right_dock_widget') and self.right_dock_widget is not None:
            is_visible = self.right_dock_widget.isVisible()
            windows.append(("Fitting Menu", self.right_dock_widget, is_visible))
        
        # Add Bottom Dock Widget (Terminal/Output)
        if hasattr(self, 'bottom_dock_widget') and self.bottom_dock_widget is not None:
            is_visible = self.bottom_dock_widget.isVisible()
            windows.append(("Terminal", self.bottom_dock_widget, is_visible))
        
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
            from qsap.help_window import HelpWindow
            self.help_window = HelpWindow()
            # Connect visibility changes to update the View menu if signal available
            if hasattr(self.help_window, 'visibilityChanged'):
                self.help_window.visibilityChanged.connect(self.on_window_visibility_changed)
            if hasattr(self.help_window, 'destroyed'):
                self.help_window.destroyed.connect(self.on_window_visibility_changed)
        except Exception as e:
            pass  # Silently skip if help_window module not available
    
    def _create_line_list_selector(self):
        """Create the Line List Selector if it doesn't exist"""
        try:
            from qsap.line_list_selector import LineListSelector
            self.line_list_selector = LineListSelector(self.resources_dir)
            self.line_list_selector.line_lists_changed.connect(self.on_line_lists_changed)
            # Connect visibility changes to update the View menu if signal available
            if hasattr(self.line_list_selector, 'visibilityChanged'):
                self.line_list_selector.visibilityChanged.connect(self.on_window_visibility_changed)
            if hasattr(self.line_list_selector, 'destroyed'):
                self.line_list_selector.destroyed.connect(self.on_window_visibility_changed)
        except Exception as e:
            pass  # Silently skip if line_list_selector module not available
    
    def _create_listfit_window(self):
        """Create the Listfit window if it doesn't exist"""
        try:
            # Listfit window initialization - may need special setup
            # For now, we'll skip auto-creation as it may have dependencies
            pass
        except Exception as e:
            print(f"Could not create Listfit window: {e}")

    def toggle_window(self, window_obj):
        """Toggle the visibility of a window - resets to default position when shown"""
        if window_obj is None:
            return
        
        # Handle QDockWidget specially - use toggleViewAction() to restore as docked
        if isinstance(window_obj, QtWidgets.QDockWidget):
            # Use the dock widget's built-in toggle action to restore properly to default docked position
            window_obj.toggleViewAction().trigger()
        # Handle regular QWidget windows - reset position when showing
        elif hasattr(window_obj, 'isVisible'):
            if window_obj.isVisible():
                window_obj.hide()
            else:
                # Reset to default position when showing
                if window_obj == self.help_window and self.help_window is not None:
                    # Reset help window position
                    self.help_window.move(200, 200)
                elif window_obj == self.fit_information_window:
                    # Reset fit information window to default position
                    self.fit_information_window.setGeometry(100, 550, 1200, 350)
                elif window_obj == self.action_history_window:
                    # Reset action history window to default position
                    self.action_history_window.setGeometry(300, 300, 400, 300)
                elif window_obj == self.item_tracker:
                    # Reset item tracker to default position
                    self.item_tracker.setGeometry(100, 550, 600, 300)
                elif window_obj == self.line_list_selector and self.line_list_selector is not None:
                    # Reset line list selector to default position
                    self.line_list_selector.setGeometry(500, 400, 400, 300)
                
                window_obj.show()
                # Bring window to front and activate it
                if hasattr(window_obj, 'raise_'):
                    window_obj.raise_()
                if hasattr(window_obj, 'activateWindow'):
                    window_obj.activateWindow()
        
        # Refresh the View menu after toggling to update checkmarks
        self.update_view_menu()

    def refresh_view_menu(self):
        """Refresh the View menu - called after windows are created or shown/hidden"""
        self.update_view_menu()

    def on_window_visibility_changed(self):
        """Handle window visibility changes and update the View menu accordingly"""
        self.update_view_menu()

    def init_controlpanel(self):
        # Create a container widget for control panel (will be added to matplotlib window as dock later)
        self.control_panel_container = QtWidgets.QWidget()
        control_panel_layout = QVBoxLayout()
        control_panel_layout.setContentsMargins(10, 10, 10, 10)
        control_panel_layout.setSpacing(5)
        
        # ===== TOP ROW: Logo & File buttons (left) + Redshift section (right) =====
        top_row_layout = QHBoxLayout()
        top_row_layout.setSpacing(10)
        
        # LEFT SIDE: Logo and File buttons
        left_column_layout = QVBoxLayout()
        left_column_layout.setSpacing(5)
        left_column_layout.setAlignment(QtCore.Qt.AlignTop)
        
        # Logo (at top of left column)
        logo_label = QtWidgets.QLabel()
        logo_path = Path(__file__).parent.parent / 'logo' / 'qsap_logo.png'
        if logo_path.exists():
            pixmap = QtGui.QPixmap(str(logo_path))
            # Scale logo to reasonable size (max 80px wide)
            scaled_pixmap = pixmap.scaledToWidth(80, QtCore.Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            left_column_layout.addWidget(logo_label)
        
        # Version label under logo
        from qsap import __version__
        version_label = QtWidgets.QLabel(f"QSAP v{__version__}")
        version_label.setAlignment(QtCore.Qt.AlignCenter)
        version_label.setStyleSheet("font-size: 9px; color: #666666;")
        left_column_layout.addWidget(version_label)
        
        # File buttons (below version)
        self.open_button = QPushButton("Load Spectrum...")
        self.open_button.clicked.connect(self.open_spectrum_file)
        left_column_layout.addWidget(self.open_button)
        
        self.load_fit_button = QPushButton("Load Fit...")
        self.load_fit_button.clicked.connect(self.load_fit_file)
        self.load_fit_button.setEnabled(True)
        left_column_layout.addWidget(self.load_fit_button)
        
        left_column_layout.addStretch()
        
        # RIGHT SIDE: Redshift section
        redshift_group = QtWidgets.QGroupBox("Redshift")
        redshift_layout = QVBoxLayout()
        redshift_layout.setSpacing(2)
        
        # Redshift value input
        redshift_input_layout = QHBoxLayout()
        redshift_input_layout.setSpacing(5)
        self.label_redshift = QLabel("Value:")
        self.input_redshift = QLineEdit()
        self.input_redshift.setText(str(self.redshift))
        self.input_redshift.setMaximumWidth(80)
        redshift_input_layout.addWidget(self.label_redshift)
        redshift_input_layout.addWidget(self.input_redshift)
        redshift_input_layout.addStretch()
        redshift_layout.addLayout(redshift_input_layout)
        
        self.input_redshift.returnPressed.connect(self.apply_changes)
        
        # UP buttons row
        redshift_up_layout = QHBoxLayout()
        redshift_up_layout.setSpacing(2)
        
        self.button_redshift_increase_0001 = QPushButton("↑ 0.001")
        self.button_redshift_increase_0001.setMaximumWidth(80)
        self.button_redshift_increase_0001.clicked.connect(lambda: self.adjust_redshift(0.001))
        redshift_up_layout.addWidget(self.button_redshift_increase_0001)
        
        self.button_redshift_increase_001 = QPushButton("↑ 0.01")
        self.button_redshift_increase_001.setMaximumWidth(80)
        self.button_redshift_increase_001.clicked.connect(lambda: self.adjust_redshift(0.01))
        redshift_up_layout.addWidget(self.button_redshift_increase_001)
        
        self.button_redshift_increase_01 = QPushButton("↑ 0.1")
        self.button_redshift_increase_01.setMaximumWidth(80)
        self.button_redshift_increase_01.clicked.connect(lambda: self.adjust_redshift(0.1))
        redshift_up_layout.addWidget(self.button_redshift_increase_01)
        redshift_up_layout.addStretch()
        redshift_layout.addLayout(redshift_up_layout)
        
        # DOWN buttons row
        redshift_down_layout = QHBoxLayout()
        redshift_down_layout.setSpacing(2)
        
        self.button_redshift_decrease_0001 = QPushButton("↓ 0.001")
        self.button_redshift_decrease_0001.setMaximumWidth(80)
        self.button_redshift_decrease_0001.clicked.connect(lambda: self.adjust_redshift(-0.001))
        redshift_down_layout.addWidget(self.button_redshift_decrease_0001)
        
        self.button_redshift_decrease_001 = QPushButton("↓ 0.01")
        self.button_redshift_decrease_001.setMaximumWidth(80)
        self.button_redshift_decrease_001.clicked.connect(lambda: self.adjust_redshift(-0.01))
        redshift_down_layout.addWidget(self.button_redshift_decrease_001)
        
        self.button_redshift_decrease_01 = QPushButton("↓ 0.1")
        self.button_redshift_decrease_01.setMaximumWidth(80)
        self.button_redshift_decrease_01.clicked.connect(lambda: self.adjust_redshift(-0.1))
        redshift_down_layout.addWidget(self.button_redshift_decrease_01)
        redshift_down_layout.addStretch()
        redshift_layout.addLayout(redshift_down_layout)
        
        # Apply button under redshift section (left-aligned, not full width)
        apply_button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.setMaximumWidth(80)
        self.apply_button.clicked.connect(self.apply_changes)
        apply_button_layout.addWidget(self.apply_button)
        apply_button_layout.addStretch()
        redshift_layout.addLayout(apply_button_layout)
        
        redshift_group.setLayout(redshift_layout)
        
        # Add left and right to top row
        top_row_layout.addLayout(left_column_layout, 1)  # 50% width (logo + buttons)
        top_row_layout.addWidget(redshift_group, 1)  # 50% width
        
        control_panel_layout.addLayout(top_row_layout)
        
        # ===== MIDDLE ROW: Undo/Redo/Quit/Help buttons =====
        button_row_layout = QHBoxLayout()
        button_row_layout.setSpacing(5)
        
        self.undo_button = QPushButton("← Undo")
        self.undo_button.setMaximumWidth(80)
        self.undo_button.clicked.connect(self.on_undo)
        self.undo_button.setEnabled(False)
        button_row_layout.addWidget(self.undo_button)
        
        self.redo_button = QPushButton("Redo →")
        self.redo_button.setMaximumWidth(80)
        self.redo_button.clicked.connect(self.on_redo)
        self.redo_button.setEnabled(False)
        button_row_layout.addWidget(self.redo_button)
        
        button_row_layout.addSpacing(20)  # Add horizontal space
        
        self.help_button = QPushButton("Help (?)")
        self.help_button.setMaximumWidth(80)
        self.help_button.clicked.connect(self.show_help_window)
        button_row_layout.addWidget(self.help_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setMaximumWidth(80)
        self.quit_button.clicked.connect(self.quit_application)
        button_row_layout.addWidget(self.quit_button)
        
        button_row_layout.addStretch()
        
        control_panel_layout.addLayout(button_row_layout)
        control_panel_layout.addStretch()
        
        self.control_panel_container.setLayout(control_panel_layout)

    def init_settings_panel(self):
        """Initialize the Settings panel for save directory configuration"""
        self.settings_container = QtWidgets.QWidget()
        settings_layout = QVBoxLayout()
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)
        
        # Title
        settings_title = QtWidgets.QLabel("Settings")
        settings_title.setStyleSheet("font-weight: bold; font-size: 12px;")
        settings_layout.addWidget(settings_title)
        
        # Save Directory Section
        save_dir_layout = QVBoxLayout()
        save_dir_layout.setSpacing(5)
        
        save_dir_label = QtWidgets.QLabel("Save Directory:")
        save_dir_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        save_dir_layout.addWidget(save_dir_label)
        
        # Directory input field and browse button
        dir_input_layout = QHBoxLayout()
        dir_input_layout.setSpacing(5)
        
        self.save_directory_input = QLineEdit()
        self.save_directory_input.setText(self.save_directory)
        self.save_directory_input.editingFinished.connect(self.on_save_directory_changed)
        dir_input_layout.addWidget(self.save_directory_input)
        
        browse_button = QPushButton("Browse...")
        browse_button.setMaximumWidth(100)
        browse_button.clicked.connect(self.on_browse_save_directory)
        dir_input_layout.addWidget(browse_button)
        
        save_dir_layout.addLayout(dir_input_layout)
        
        settings_layout.addLayout(save_dir_layout)
        
        # Load Directory Section
        load_dir_layout = QVBoxLayout()
        load_dir_layout.setSpacing(5)
        
        load_dir_label = QtWidgets.QLabel("Load Directory:")
        load_dir_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        load_dir_layout.addWidget(load_dir_label)
        
        # Directory input field and browse button
        load_input_layout = QHBoxLayout()
        load_input_layout.setSpacing(5)
        
        self.load_directory_input = QLineEdit()
        self.load_directory_input.setText(self.load_directory)
        self.load_directory_input.editingFinished.connect(self.on_load_directory_changed)
        load_input_layout.addWidget(self.load_directory_input)
        
        load_browse_button = QPushButton("Browse...")
        load_browse_button.setMaximumWidth(100)
        load_browse_button.clicked.connect(self.on_browse_load_directory)
        load_input_layout.addWidget(load_browse_button)
        
        load_dir_layout.addLayout(load_input_layout)
        
        settings_layout.addLayout(load_dir_layout)
        settings_layout.addStretch()
        
        self.settings_container.setLayout(settings_layout)

    def on_browse_save_directory(self):
        """Open file dialog to select save directory"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Save Directory",
            self.save_directory,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if directory:
            self.save_directory = directory
            self.qsap_handler.save_directory = directory  # Update QSAP handler
            self.save_directory_input.setText(directory)

    def on_save_directory_changed(self):
        """Handle changes to the save directory input field"""
        new_directory = self.save_directory_input.text().strip()
        try:
            # Verify the directory exists
            Path(new_directory).resolve()
            self.save_directory = new_directory
            self.qsap_handler.save_directory = new_directory  # Update QSAP handler
        except Exception as e:
            # If invalid path, revert to previous value
            self.save_directory_input.setText(self.save_directory)

    def on_browse_load_directory(self):
        """Open file dialog to select load directory"""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Load Directory",
            self.load_directory,
            QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.DontResolveSymlinks
        )
        if directory:
            self.load_directory = directory
            self.load_directory_input.setText(directory)

    def on_load_directory_changed(self):
        """Handle changes to the load directory input field"""
        new_directory = self.load_directory_input.text().strip()
        try:
            # Verify the directory exists
            Path(new_directory).resolve()
            self.load_directory = new_directory
        except Exception as e:
            # If invalid path, revert to previous value
            self.load_directory_input.setText(self.load_directory)

    def show_help_window(self):
        """Show the help window."""
        if self.help_window is None:
            self.help_window = HelpWindow(self)
        self.help_window.show()
        self.help_window.raise_()
        self.help_window.activateWindow()

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
            # Update redshift from input field
            self.redshift = float(self.input_redshift.text())
            
            # Note: zoom_factor is no longer controlled via UI input field
            # It's only used internally for y-axis zoom operations (y and Y keys)
            
            # Format redshift without trailing zeros for display
            redshift_str = f"{self.redshift:.6f}".rstrip('0').rstrip('.')
            self.input_redshift.setText(redshift_str)
            
            # Update polynomial order from Options panel input field
            try:
                self.poly_order = int(self.options_poly_order_input.text())
                print(f"Polynomial order set to: {self.poly_order}")
            except ValueError:
                print("Invalid polynomial order")
            
            print(f"Applied Redshift: {self.redshift}")
            
            # Redisplay active line lists with new redshift
            if self.active_line_lists:
                self.display_linelist()
            
            # Also handle legacy linelist_plots for backwards compatibility
            if self.linelist_plots:
                self.clear_linelist()
                self.display_linelist()
            
            self.fig.canvas.draw_idle()  # Redraw the figure to update the display
        except ValueError:
            print("Invalid input for redshift or polynomial order. Please enter numerical values.")

    def open_spectrum_file(self):
        """Open a file dialog to select and load a new spectrum file."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        from qsap.format_picker_dialog import FormatPickerDialog
        
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
            try:
                self.clear_plot_and_reset()
                self.plot_spectrum()
                print(f"Loaded: {file_path}")
            except Exception as plot_error:
                print(f"Error during plot: {plot_error}")
                import traceback
                traceback.print_exc()
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            import traceback
            traceback.print_exc()

    def load_fit_file(self):
        """Load previously saved fits from .qsap or CSV files"""
        from PyQt5.QtWidgets import QFileDialog
        import pandas as pd
        
        dialog = QFileDialog(
            self,
            "Load Fit File(s)",
            self.load_directory,
            "QSAP Files (*.qsap);;CSV Files (*.csv);;All Files (*)"
        )
        dialog.setFileMode(QFileDialog.ExistingFiles)
        
        if dialog.exec_() != QFileDialog.Accepted:
            return
        
        file_paths = dialog.selectedFiles()
        if not file_paths:
            return
            
        self.load_directory = os.path.dirname(file_paths[0])
        
        # Load all selected files
        for file_path in file_paths:
            try:
                basename = os.path.basename(file_path)
                
                # Check if this is a .qsap file
                if file_path.endswith('.qsap'):
                    self._load_qsap_file(file_path)
                    print(f"Loaded fits from QSAP file: {basename}")
                else:
                    # Legacy CSV support
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
                
            except Exception as e:
                print(f"Error loading fit file {basename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        try:
            # Update plot and redraw all loaded fits
            # Only plot spectrum if not already loaded
            if len(self.wav) == 0 or len(self.spec) == 0:
                self.plot_spectrum()
            # Always redraw fits (will only add new ones that don't have lines yet)
            self._redraw_loaded_fits()
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error loading fit file: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_qsap_file(self, filepath):
        """Load fits from a .qsap file and populate relevant fit lists
        
        Args:
            filepath: Path to the .qsap file
        """
        qsap_data = self.qsap_handler.parse_qsap_file(filepath)
        metadata = qsap_data.get('metadata', {})
        components = qsap_data.get('components', [])
        
        fit_type = metadata.get('TYPE', 'Unknown')
        fit_mode = metadata.get('MODE', 'Unknown')
        
        print(f"Loading {fit_type} fit (mode: {fit_mode})")
        
        # Load scale factor from metadata if present
        if 'SCALE_FACTOR' in metadata:
            self.flux_scale_factor = metadata['SCALE_FACTOR']
            print(f"  Scale factor: {self.flux_scale_factor}")
        
        # Process components based on fit type
        if fit_type == 'Gaussian':
            for comp in components:
                fit_dict = self._parse_qsap_gaussian_component(comp)
                if fit_dict:
                    self.gaussian_fits.append(fit_dict)
                    # Register with ItemTracker
                    position_str = f"λ: {fit_dict.get('mean', 0):.2f} Å"
                    self.register_item('gaussian', f'Gaussian', fit_dict=fit_dict, position=position_str, 
                                     color=self.colors['profiles']['gaussian']['color'])
            print(f"  Loaded {len(components)} Gaussian components")
            
        elif fit_type == 'Voigt':
            for comp in components:
                fit_dict = self._parse_qsap_voigt_component(comp)
                if fit_dict:
                    self.voigt_fits.append(fit_dict)
                    # Register with ItemTracker
                    position_str = f"λ: {fit_dict.get('center', 0):.2f} Å"
                    self.register_item('voigt', f'Voigt', fit_dict=fit_dict, position=position_str,
                                     color='orange')
            print(f"  Loaded {len(components)} Voigt components")
            
        elif fit_type == 'Continuum':
            for comp in components:
                fit_dict = self._parse_qsap_continuum_component(comp)
                if fit_dict:
                    self.continuum_fits.append(fit_dict)
                    
                    # Register with ItemTracker
                    bounds = fit_dict.get('bounds', (0, 0))
                    bounds_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å"
                    self.register_item('continuum', f'Continuum (order {fit_dict.get("poly_order", 2)})',
                                     fit_dict=fit_dict, position=bounds_str, 
                                     color=self.colors['profiles']['continuum_line']['color'])
                    
                    # Recreate continuum regions/patches for visualization
                    # Check if individual_regions was stored, otherwise use the combined bounds
                    individual_regions = fit_dict.get('individual_regions', [bounds])
                    
                    for region_bounds in individual_regions:
                        if region_bounds[0] > 0 and region_bounds[1] > 0:  # Valid bounds
                            # Add to continuum_regions list
                            self.continuum_regions.append(region_bounds)
                            
                            # Create visual patch for the region
                            try:
                                continuum_region_cfg = self.colors['profiles']['continuum_region']
                                patch = self.ax.axvspan(region_bounds[0], region_bounds[1], 
                                                      color=continuum_region_cfg['color'], 
                                                      alpha=continuum_region_cfg['alpha'], 
                                                      hatch=continuum_region_cfg['hatch'])
                                self.continuum_patches.append({'patch': patch, 'bounds': region_bounds})
                                
                                # Register the region with ItemTracker
                                position_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
                                self.register_item('continuum_region', f'Continuum Region', patch_obj=patch,
                                                 position=position_str, color=continuum_region_cfg['color'], bounds=region_bounds)
                            except Exception as e:
                                print(f"Error creating continuum region patch: {e}")
            print(f"  Loaded {len(components)} Continuum fits")
            
        elif fit_type == 'Listfit':
            mask_count = 0
            poly_count = 0
            for comp in components:
                if comp.get('TYPE') == 'Gaussian':
                    fit_dict = self._parse_qsap_gaussian_component(comp)
                    if fit_dict:
                        self.gaussian_fits.append(fit_dict)
                        position_str = f"λ: {fit_dict.get('mean', 0):.2f} Å"
                        self.register_item('gaussian', f'Gaussian (listfit)', fit_dict=fit_dict, 
                                         position=position_str, color=self.colors['profiles']['gaussian']['color'])
                elif comp.get('TYPE') == 'Voigt':
                    fit_dict = self._parse_qsap_voigt_component(comp)
                    if fit_dict:
                        self.voigt_fits.append(fit_dict)
                        position_str = f"λ: {fit_dict.get('center', 0):.2f} Å"
                        self.register_item('voigt', f'Voigt (listfit)', fit_dict=fit_dict, 
                                         position=position_str, color='orange')
                elif comp.get('TYPE') == 'Polynomial':
                    fit_dict = self._parse_qsap_polynomial_component(comp)
                    if fit_dict:
                        # Use bounds from the parsed polynomial, or default to full spectrum
                        bounds = fit_dict.get('bounds', (self.x_data.min(), self.x_data.max()))
                        fit_dict['bounds'] = bounds
                        fit_dict['is_velocity_mode'] = False
                        self.continuum_fits.append(fit_dict)
                        position_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å"
                        self.register_item('polynomial', f'Polynomial (listfit, order={fit_dict.get("poly_order", 1)})', 
                                         fit_dict=fit_dict, position=position_str, 
                                         color=self.colors['profiles']['continuum_line']['color'])
                        poly_count += 1
                elif comp.get('TYPE') == 'PolynomialGuessMask':
                    mask_dict = self._parse_qsap_polynomial_guess_mask_component(comp)
                    if mask_dict:
                        min_lambda = mask_dict.get('min_lambda', 0)
                        max_lambda = mask_dict.get('max_lambda', 0)
                        position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                        self.register_item('polynomial_guess_mask', f'Polynomial Guess Mask (listfit)',
                                         fit_dict=mask_dict, position=position_str,
                                         color='lightblue', bounds=(min_lambda, max_lambda))
                        mask_count += 1
                elif comp.get('TYPE') == 'DataMask':
                    mask_dict = self._parse_qsap_data_mask_component(comp)
                    if mask_dict:
                        min_lambda = mask_dict.get('min_lambda', 0)
                        max_lambda = mask_dict.get('max_lambda', 0)
                        position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                        self.register_item('data_mask', f'Data Mask (listfit)',
                                         fit_dict=mask_dict, position=position_str,
                                         color='lightcoral', bounds=(min_lambda, max_lambda))
                        mask_count += 1
            print(f"  Loaded {len(components)} Listfit components (including {mask_count} masks, {poly_count} polynomials)")
            
        elif fit_type == 'Redshift':
            redshift_value = components[0].get('REDSHIFT') if components else None
            if isinstance(redshift_value, tuple):
                redshift_value = redshift_value[0]
            
            if redshift_value:
                print(f"  Loaded Redshift: {redshift_value}")
                print(f"  Line ID: {components[0].get('LINE_ID', 'Unknown')}")
                print(f"  Rest Wavelength: {components[0].get('LINE_WAVELENGTH_REST', 'Unknown')} Å")
                print(f"  Observed Wavelength: {components[0].get('LINE_WAVELENGTH_OBSERVED', 'Unknown')} Å")
    
    def _parse_qsap_gaussian_component(self, comp_dict):
        """Parse a Gaussian component from QSAP format"""
        fit_dict = {}
        
        # Core parameters
        if 'FIT_ID' in comp_dict:
            fit_dict['fit_id'] = comp_dict['FIT_ID']
        if 'COMPONENT_ID' in comp_dict:
            fit_dict['component_id'] = comp_dict['COMPONENT_ID']
        
        # Line information
        if 'LINE_ID' in comp_dict:
            fit_dict['line_id'] = comp_dict['LINE_ID']
        if 'LINE_WAVELENGTH' in comp_dict:
            fit_dict['line_wavelength'] = comp_dict['LINE_WAVELENGTH']
        if 'REST_WAVELENGTH' in comp_dict:
            fit_dict['rest_wavelength'] = comp_dict['REST_WAVELENGTH']
        
        # Parse amplitude with error
        if 'AMPLITUDE' in comp_dict:
            val, err = comp_dict['AMPLITUDE'] if isinstance(comp_dict['AMPLITUDE'], tuple) else (comp_dict['AMPLITUDE'], None)
            fit_dict['amp'] = val
            if err:
                fit_dict['amp_err'] = err
        
        # Parse mean with error
        if 'MEAN' in comp_dict:
            val, err = comp_dict['MEAN'] if isinstance(comp_dict['MEAN'], tuple) else (comp_dict['MEAN'], None)
            fit_dict['mean'] = val
            if err:
                fit_dict['mean_err'] = err
        
        # Parse std_dev with error
        if 'STD_DEV' in comp_dict:
            val, err = comp_dict['STD_DEV'] if isinstance(comp_dict['STD_DEV'], tuple) else (comp_dict['STD_DEV'], None)
            fit_dict['stddev'] = val
            if err:
                fit_dict['stddev_err'] = err
        
        # Bounds
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Quality metrics
        if 'CHI_SQUARED' in comp_dict:
            fit_dict['chi2'] = comp_dict['CHI_SQUARED']
        if 'CHI_SQUARED_NU' in comp_dict:
            fit_dict['chi2_nu'] = comp_dict['CHI_SQUARED_NU']
        
        # Mode information (default to False if not present)
        fit_dict['is_velocity_mode'] = comp_dict.get('VELOCITY_MODE', False)
        
        # System redshift
        if 'SYSTEM_REDSHIFT' in comp_dict:
            fit_dict['z_sys'] = comp_dict['SYSTEM_REDSHIFT']
        
        return fit_dict if fit_dict else None
    
    def _parse_qsap_voigt_component(self, comp_dict):
        """Parse a Voigt component from QSAP format"""
        fit_dict = {}
        
        # Core parameters
        if 'FIT_ID' in comp_dict:
            fit_dict['fit_id'] = comp_dict['FIT_ID']
        if 'COMPONENT_ID' in comp_dict:
            fit_dict['component_id'] = comp_dict['COMPONENT_ID']
        
        # Line information
        if 'LINE_ID' in comp_dict:
            fit_dict['line_id'] = comp_dict['LINE_ID']
        if 'LINE_WAVELENGTH' in comp_dict:
            fit_dict['line_wavelength'] = comp_dict['LINE_WAVELENGTH']
        if 'REST_WAVELENGTH' in comp_dict:
            fit_dict['rest_wavelength'] = comp_dict['REST_WAVELENGTH']
        
        # Parse amplitude with error
        if 'AMPLITUDE' in comp_dict:
            val, err = comp_dict['AMPLITUDE'] if isinstance(comp_dict['AMPLITUDE'], tuple) else (comp_dict['AMPLITUDE'], None)
            fit_dict['amplitude'] = val
            if err:
                fit_dict['amplitude_err'] = err
        
        # Parse mean/center with error
        if 'MEAN' in comp_dict:
            val, err = comp_dict['MEAN'] if isinstance(comp_dict['MEAN'], tuple) else (comp_dict['MEAN'], None)
            fit_dict['center'] = val
            fit_dict['mean'] = val
            if err:
                fit_dict['center_err'] = err
                fit_dict['mean_err'] = err
        
        # Parse sigma with error
        if 'SIGMA' in comp_dict:
            val, err = comp_dict['SIGMA'] if isinstance(comp_dict['SIGMA'], tuple) else (comp_dict['SIGMA'], None)
            fit_dict['sigma'] = val
            if err:
                fit_dict['sigma_err'] = err
        
        # Parse gamma with error
        if 'GAMMA' in comp_dict:
            val, err = comp_dict['GAMMA'] if isinstance(comp_dict['GAMMA'], tuple) else (comp_dict['GAMMA'], None)
            fit_dict['gamma'] = val
            if err:
                fit_dict['gamma_err'] = err
        
        # Doppler parameter
        if 'B_DOPPLER' in comp_dict:
            fit_dict['b'] = comp_dict['B_DOPPLER']
        if 'LOG_T_EFF' in comp_dict:
            fit_dict['logT_eff'] = comp_dict['LOG_T_EFF']
        
        # Bounds
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Quality metrics
        if 'CHI_SQUARED' in comp_dict:
            fit_dict['chi2'] = comp_dict['CHI_SQUARED']
        if 'CHI_SQUARED_NU' in comp_dict:
            fit_dict['chi2_nu'] = comp_dict['CHI_SQUARED_NU']
        
        # Mode information (default to False if not present)
        fit_dict['is_velocity_mode'] = comp_dict.get('VELOCITY_MODE', False)
        
        # System redshift
        if 'SYSTEM_REDSHIFT' in comp_dict:
            fit_dict['z_sys'] = comp_dict['SYSTEM_REDSHIFT']
        
        return fit_dict if fit_dict else None
    
    def _parse_qsap_continuum_component(self, comp_dict):
        """Parse a continuum component from QSAP format"""
        fit_dict = {}
        
        # Polynomial order
        if 'POLY_ORDER' in comp_dict:
            fit_dict['poly_order'] = comp_dict['POLY_ORDER']
        
        # Bounds (combined min-max for backward compatibility)
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Individual regions (new format)
        if 'NUM_REGIONS' in comp_dict:
            num_regions = comp_dict['NUM_REGIONS']
            individual_regions = []
            for idx in range(num_regions):
                lower_key = f'REGION_{idx}_LOWER'
                upper_key = f'REGION_{idx}_UPPER'
                if lower_key in comp_dict and upper_key in comp_dict:
                    region = (comp_dict[lower_key], comp_dict[upper_key])
                    individual_regions.append(region)
            if individual_regions:
                fit_dict['individual_regions'] = individual_regions
        
        # Polynomial coefficients
        coeffs = []
        coeffs_err = []
        coeff_idx = 0
        while f'COEFF_{coeff_idx}' in comp_dict:
            val = comp_dict[f'COEFF_{coeff_idx}']
            if isinstance(val, tuple):
                coeffs.append(val[0])
                coeffs_err.append(val[1])
            else:
                coeffs.append(val)
                coeffs_err.append(None)
            coeff_idx += 1
        
        if coeffs:
            fit_dict['coeffs'] = np.array(coeffs)
            fit_dict['coeffs_err'] = np.array(coeffs_err)
            
            # Reconstruct covariance matrix from coefficient errors (diagonal approximation)
            # When loading from .qsap file, we only have coefficient errors, not full covariance
            # Create diagonal covariance matrix from coefficient errors squared
            if all(err is not None for err in coeffs_err):
                pcov = np.diag(np.array(coeffs_err) ** 2)
                fit_dict['covariance'] = pcov
        
        # Mode information
        if 'VELOCITY_MODE' in comp_dict:
            fit_dict['is_velocity_mode'] = comp_dict['VELOCITY_MODE']
        
        return fit_dict if fit_dict else None
    
    def _parse_qsap_polynomial_guess_mask_component(self, comp_dict):
        """Parse a polynomial guess mask component from QSAP format"""
        mask_dict = {
            'type': 'polynomial_guess_mask'
        }
        
        if 'MIN_LAMBDA' in comp_dict:
            mask_dict['min_lambda'] = comp_dict['MIN_LAMBDA']
        if 'MAX_LAMBDA' in comp_dict:
            mask_dict['max_lambda'] = comp_dict['MAX_LAMBDA']
        
        return mask_dict if mask_dict else None
    
    def _parse_qsap_data_mask_component(self, comp_dict):
        """Parse a data mask component from QSAP format"""
        mask_dict = {
            'type': 'data_mask'
        }
        
        if 'MIN_LAMBDA' in comp_dict:
            mask_dict['min_lambda'] = comp_dict['MIN_LAMBDA']
        if 'MAX_LAMBDA' in comp_dict:
            mask_dict['max_lambda'] = comp_dict['MAX_LAMBDA']
        
        return mask_dict if mask_dict else None
    
    def _parse_qsap_polynomial_component(self, comp_dict):
        """Parse a polynomial component from QSAP format (for listfit)"""
        fit_dict = {}
        
        # Polynomial order
        if 'POLY_ORDER' in comp_dict:
            fit_dict['poly_order'] = comp_dict['POLY_ORDER']
        
        # Bounds
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Polynomial coefficients
        coeffs = []
        coeffs_err = []
        coeff_idx = 0
        while f'COEFF_{coeff_idx}' in comp_dict:
            val = comp_dict[f'COEFF_{coeff_idx}']
            if isinstance(val, tuple):
                coeffs.append(val[0])
                coeffs_err.append(val[1])
            else:
                coeffs.append(val)
                coeffs_err.append(None)
            coeff_idx += 1
        
        if coeffs:
            fit_dict['coeffs'] = np.array(coeffs)
            fit_dict['coeffs_err'] = np.array(coeffs_err)
        
        return fit_dict if fit_dict else None
    
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
                    gaussian_color = self.colors['profiles']['gaussian']
                    
                    # Add label only for the first gaussian
                    label = 'Gaussian' if 'gaussian' not in self.legend_profile_types else None
                    fit['line'], = self.ax.plot(x_plot, y_plot, color=gaussian_color['color'], linestyle=gaussian_color['linestyle'], label=label)
                    if label:
                        self.legend_profile_types.add('gaussian')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Gaussian (μ={fit['mean']:.1f}, σ={fit['stddev']:.1f})"
                    self.register_item('gaussian', name, fit_dict=fit, line_obj=fit['line'], 
                                     color=gaussian_color['color'], bounds=fit['bounds'])
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
                    voigt_color = self.colors['profiles']['voigt']
                    
                    # Add label only for the first voigt
                    label = 'Voigt' if 'voigt' not in self.legend_profile_types else None
                    fit['line'], = self.ax.plot(x_plot, y_plot, color=voigt_color['color'], linestyle=voigt_color['linestyle'], label=label)
                    if label:
                        self.legend_profile_types.add('voigt')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Voigt (c={fit['center']:.1f}, σ={fit['sigma']:.1f}, γ={fit['gamma']:.1f})"
                    self.register_item('voigt', name, fit_dict=fit, line_obj=fit['line'],
                                     color=voigt_color['color'], bounds=fit['bounds'])
                except Exception as e:
                    print(f"Error redrawing Voigt fit: {e}")
        
        # Redraw Continuum fits
        for idx, fit in enumerate(self.continuum_fits):
            if fit.get('line') is None:
                try:
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 100)
                    
                    # Handle both old format (a, b) and new format (coeffs)
                    if 'coeffs' in fit:
                        y_plot = np.polyval(fit['coeffs'], x_plot)
                    else:
                        # Old format fallback for backward compatibility
                        y_plot = fit['a'] * x_plot + fit['b']
                    
                    continuum_color = self.colors['profiles']['continuum_line']
                    
                    # Add label only for the first continuum
                    label = 'Continuum' if 'continuum' not in self.legend_profile_types else None
                    fit['line'], = self.ax.plot(x_plot, y_plot, color=continuum_color['color'], linestyle=continuum_color['linestyle'], label=label)
                    if label:
                        self.legend_profile_types.add('continuum')
                    
                    # Register with item tracker
                    poly_order = fit.get('poly_order', 1)
                    name = f"Continuum (order {poly_order})"
                    # Create position string safely
                    bounds = fit.get('bounds', (0, 0))
                    if bounds and bounds[0] is not None and bounds[1] is not None:
                        position_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å"
                    else:
                        position_str = "Continuum"
                    self.register_item('continuum', name, fit_dict=fit, line_obj=fit['line'],
                                     position=position_str, color=continuum_color['color'], bounds=bounds)
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
                        # Use the standard listfit color from config
                        total_color = self.colors['profiles']['total_line']
                        listfit_line, = self.ax.plot(x_plot, y_plot, label='Total Listfit', color=total_color['color'], linestyle=total_color['linestyle'], linewidth=total_color['linewidth'])
                        listfit['line'] = listfit_line
                        
                        # Register with item tracker - use saved name if available
                        n_components = len(listfit.get('components', []))
                        chi2 = listfit.get('quality_metrics', {}).get('chisqr', 0)
                        name = listfit.get('_tracker_name') or f"Total Listfit ({n_components} components, χ²={chi2:.2f})"
                        self.register_item('listfit_total', name, fit_dict=listfit, line_obj=listfit_line,
                                         color=total_color['color'], bounds=bounds)
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
                self.record_action('open_qsap', 'Open qsap')
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
        
        # Extract and store flux scale factor from metadata
        if meta and 'scale_factor' in meta:
            self.flux_scale_factor = meta['scale_factor']
        else:
            self.flux_scale_factor = 1.0  # Default to no scaling
        
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
        """Quit the QSAP application gracefully"""
        try:
            # Show goodbye message in terminal
            message = "Quitting QSAP. Bye!"
            box_width = len(message)
            print("\n" + "╔" + "═" * (box_width) + "╗")
            print("║" + message + "║")
            print("╚" + "═" * (box_width) + "╝\n")
            
            # Close all child windows
            if hasattr(self, 'help_window') and self.help_window is not None:
                try:
                    self.help_window.close()
                except:
                    pass
            if hasattr(self, 'item_tracker') and self.item_tracker is not None:
                try:
                    self.item_tracker.close()
                except:
                    pass
            if hasattr(self, 'fit_information_window') and self.fit_information_window is not None:
                try:
                    self.fit_information_window.close()
                except:
                    pass
            if hasattr(self, 'linelist_selector') and self.linelist_selector is not None:
                try:
                    self.linelist_selector.close()
                except:
                    pass
            
            # Close main window
            self.close()
            
            # Quit the application
            QtWidgets.QApplication.quit()
        except Exception as e:
            print(f"Error during quit: {e}")
            import sys
            sys.exit(0)
    
    def closeEvent(self, event):
        """Handle window close event gracefully"""
        try:
            # Close all child windows
            if hasattr(self, 'help_window') and self.help_window is not None:
                try:
                    self.help_window.close()
                except:
                    pass
            if hasattr(self, 'item_tracker') and self.item_tracker is not None:
                try:
                    self.item_tracker.close()
                except:
                    pass
            if hasattr(self, 'fit_information_window') and self.fit_information_window is not None:
                try:
                    self.fit_information_window.close()
                except:
                    pass
            if hasattr(self, 'linelist_selector') and self.linelist_selector is not None:
                try:
                    self.linelist_selector.close()
                except:
                    pass
            
            # Accept the close event
            event.accept()
        except Exception as e:
            print(f"Error during window close: {e}")
            event.accept()
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
            title = "QSAP - Load a Spectrum to Begin"

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
            # Create new figure directly (not via plt)
            self.fig = Figure(figsize=(10, 6))
            self.ax = self.fig.add_subplot(111)
            
            # Create canvas and set as central widget
            self.canvas = FigureCanvas(self.fig)
            
            # Create wrapper widget with toolbar for spectrum plotter
            wrapper_widget = QtWidgets.QWidget()
            wrapper_layout = QVBoxLayout()
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(0)
            
            # Add matplotlib toolbar
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            wrapper_layout.addWidget(self.toolbar)
            
            # Add canvas
            wrapper_layout.addWidget(self.canvas)
            wrapper_widget.setLayout(wrapper_layout)
            wrapper_widget.setMinimumHeight(350)  # Spectrum plotter minimum size
            
            # Create vertical splitter for spectrum only (central widget)
            center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            center_splitter.setContentsMargins(0, 0, 0, 0)
            
            # Add spectrum plotter (only section)
            center_splitter.addWidget(wrapper_widget)
            
            # Don't add terminal to splitter - it will be a dockable widget
            
            self.center_splitter = center_splitter
            self.setCentralWidget(center_splitter)
            
            # Create tab widget for dockable Control Panel + Item Tracker + Settings
            top_tab_widget = QtWidgets.QTabWidget()
            top_tab_widget.addTab(self.control_panel_container, "Control Panel")
            top_tab_widget.addTab(self.item_tracker, "Item Tracker")
            top_tab_widget.addTab(self.settings_container, "Settings")
            
            # Create dock widget for the tab widget (Control Panel - top/left)
            self.control_panel_dock = QtWidgets.QDockWidget("Control Panel", self)
            self.control_panel_dock.setWidget(top_tab_widget)
            self.control_panel_dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea)
            self.control_panel_dock.setMinimumWidth(200)
            self.control_panel_dock.setMaximumHeight(350)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.control_panel_dock)
            
            # Connect visibility changes if available
            if hasattr(self.control_panel_dock, 'visibilityChanged'):
                self.control_panel_dock.visibilityChanged.connect(self.on_window_visibility_changed)
            if hasattr(self.control_panel_dock, 'destroyed'):
                self.control_panel_dock.destroyed.connect(self.on_window_visibility_changed)
            
            # Create output panel (terminal at bottom) as a dockable widget
            self.output_panel = OutputPanel()
            self.output_panel.setMinimumHeight(150)
            terminal_dock = QtWidgets.QDockWidget("Terminal", self)
            terminal_dock.setWidget(self.output_panel)
            terminal_dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.TopDockWidgetArea | QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, terminal_dock)
            self.bottom_dock_widget = terminal_dock
            
            # Connect visibility changes if available
            if hasattr(terminal_dock, 'visibilityChanged'):
                terminal_dock.visibilityChanged.connect(self.on_window_visibility_changed)
            if hasattr(terminal_dock, 'destroyed'):
                terminal_dock.destroyed.connect(self.on_window_visibility_changed)
            
            # Connect matplotlib events to SpectrumPlotter handlers
            self.fig.canvas.mpl_connect('key_press_event', self.on_key)
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
            self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)
            
            # Ensure canvas can receive focus for keyboard events
            self.canvas.setFocus()
        else:
            # Reuse existing figure - clear axes
            self.ax.clear()
        
        # Plot the spectrum
        data_cfg = self.colors['spectrum']['data']
        error_cfg = self.colors['spectrum']['error']
        self.step_spec, = self.ax.step(self.x_data, self.spec, label='Data', color=data_cfg['color'], where='mid', zorder=0)
        self.line_spec, = self.ax.plot(self.x_data, self.spec, color=data_cfg['color'], visible=False, zorder=0)
        
        # Only plot error if errors exist
        if self.err is not None:
            self.step_error, = self.ax.step(self.x_data, self.err, color=error_cfg['color'], linestyle=error_cfg['linestyle'], alpha=error_cfg['alpha'], label='Error', where='mid', zorder=0)
            self.line_error, = self.ax.plot(self.x_data, self.err, color=error_cfg['color'], linestyle=error_cfg['linestyle'], alpha=error_cfg['alpha'], visible=False, zorder=0)
            self.error_line = self.step_error if self.is_step_plot else self.line_error
        else:
            self.step_error = None
            self.line_error = None
            self.error_line = None
        
        self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
        ref_cfg = self.colors['reference_lines']
        self.ax.plot(self.x_data, [0] * len(self.x_data), color=ref_cfg['color'], linestyle=ref_cfg['linestyle'], linewidth=ref_cfg['linewidth']) # Add horizontal line at y=0
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
        from qsap import __version__
        # Update title with version (no matplotlib manager anymore)
        self.setWindowTitle(f"QSAP - Quick Spectrum Analysis Program (v{__version__})")

        self.update_legend()

        # Show the Item Tracker window (in background)
        self.item_tracker.item_deleted.connect(self.on_item_deleted_from_tracker)
        self.item_tracker.item_selected.connect(self.on_item_selected_from_tracker)
        self.item_tracker.item_individually_deselected.connect(self.on_item_individually_deselected_from_tracker)
        self.item_tracker.item_deselected.connect(self.on_item_deselected_from_tracker)
        self.item_tracker.estimate_redshift.connect(self.on_estimate_redshift_from_tracker)
        self.item_tracker.calculate_ew.connect(self.on_calculate_ew_from_tracker)
        # Connect visibility changes to update the View menu if signal available
        if hasattr(self.item_tracker, 'visibilityChanged'):
            self.item_tracker.visibilityChanged.connect(self.on_window_visibility_changed)
        if hasattr(self.item_tracker, 'destroyed'):
            self.item_tracker.destroyed.connect(self.on_window_visibility_changed)

        # Connect Fit Information window signals
        self.fit_information_window.item_selected.connect(self.on_fit_info_item_selected)
        self.fit_information_window.item_deselected.connect(self.on_fit_info_item_deselected)
        self.fit_information_window.setGeometry(100, 550, 1200, 350)  # Position below main window
        # Connect visibility changes to update the View menu if signal available
        if hasattr(self.fit_information_window, 'visibilityChanged'):
            self.fit_information_window.visibilityChanged.connect(self.on_window_visibility_changed)
        if hasattr(self.fit_information_window, 'destroyed'):
            self.fit_information_window.destroyed.connect(self.on_window_visibility_changed)

        # Connect keyboard and mouse events to the canvas
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        # Connect x-bounds update to the axes
        self.ax.callbacks.connect('xlim_changed', self.update_residual_xbounds)

        # Update the View menu now that the spectrum plotter figure has been created
        self.update_view_menu()

        # On first plot creation, set up dock widgets
        if is_first_plot:
            # Add right dock widget for controls
            self.setup_right_dock(self)
            
            # Show the window
            self.show()
            
            # Refresh the View menu to include the new dock widget
            self.update_view_menu()

    def setup_right_dock(self, mpl_window):
        """Create and setup the right dock widget for fitting options"""
        # Create dock widget
        dock_widget = QtWidgets.QDockWidget("Options", mpl_window)
        dock_widget.setAllowedAreas(QtCore.Qt.RightDockWidgetArea)
        
        # Create main content widget
        dock_content = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        
        # ===== FITTING SECTION TITLE =====
        fitting_title = QtWidgets.QLabel("Fit")
        fitting_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #0078d4;")
        main_layout.addWidget(fitting_title)
        
        fitting_layout = QtWidgets.QVBoxLayout()
        fitting_layout.setContentsMargins(15, 5, 10, 10)
        fitting_layout.setSpacing(6)
        
        # --- CONTINUUM SUBSECTION ---
        continuum_label = QtWidgets.QLabel("Continuum")
        continuum_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        fitting_layout.addWidget(continuum_label)
        
        continuum_sub_layout = QtWidgets.QVBoxLayout()
        continuum_sub_layout.setContentsMargins(10, 5, 10, 5)
        continuum_sub_layout.setSpacing(4)
        
        # Continuum mode dropdown
        cont_dropdown_layout = QtWidgets.QHBoxLayout()
        self.continuum_mode_dropdown = QtWidgets.QComboBox()
        self.continuum_mode_dropdown.addItem("")  # Empty (inactive)
        self.continuum_mode_dropdown.addItem("Continuum Region(s)     [m]")
        self.continuum_mode_dropdown.currentTextChanged.connect(self.on_continuum_mode_changed)
        cont_dropdown_layout.addWidget(self.continuum_mode_dropdown)
        cont_dropdown_layout.addStretch()
        continuum_sub_layout.addLayout(cont_dropdown_layout)
        
        # Polynomial order label
        poly_order_label = QtWidgets.QLabel("Polynomial order")
        poly_order_label.setStyleSheet("font-size: 10px;")
        continuum_sub_layout.addWidget(poly_order_label)
        
        # Polynomial order with +/- buttons
        poly_order_layout = QtWidgets.QHBoxLayout()
        poly_order_layout.setSpacing(4)
        poly_minus_btn = QtWidgets.QPushButton("-")
        poly_minus_btn.setMaximumWidth(30)
        poly_minus_btn.clicked.connect(self.on_poly_order_minus)
        self.options_poly_order_input = QtWidgets.QLineEdit()
        self.options_poly_order_input.setText("1")
        self.options_poly_order_input.setMaximumWidth(50)
        poly_validator = QIntValidator(0, 10, self)
        self.options_poly_order_input.setValidator(poly_validator)
        self.options_poly_order_input.editingFinished.connect(self.on_poly_order_changed)
        poly_plus_btn = QtWidgets.QPushButton("+")
        poly_plus_btn.setMaximumWidth(30)
        poly_plus_btn.clicked.connect(self.on_poly_order_plus)
        poly_order_layout.addWidget(poly_minus_btn)
        poly_order_layout.addWidget(self.options_poly_order_input)
        poly_order_layout.addWidget(poly_plus_btn)
        poly_order_layout.addStretch()
        continuum_sub_layout.addLayout(poly_order_layout)
        
        # Continuum Enter button
        continuum_enter_layout = QtWidgets.QHBoxLayout()
        self.continuum_enter_button = QtWidgets.QPushButton("Enter")
        self.continuum_enter_button.setEnabled(False)
        self.continuum_enter_button.clicked.connect(self.on_continuum_enter_clicked)
        continuum_enter_layout.addWidget(self.continuum_enter_button)
        continuum_enter_layout.addStretch()
        continuum_sub_layout.addLayout(continuum_enter_layout)
        
        fitting_layout.addLayout(continuum_sub_layout)
        
        # --- LINE PROFILES SUBSECTION ---
        line_profiles_label = QtWidgets.QLabel("Line Profiles")
        line_profiles_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        fitting_layout.addWidget(line_profiles_label)
        
        line_profiles_sub_layout = QtWidgets.QVBoxLayout()
        line_profiles_sub_layout.setContentsMargins(10, 5, 10, 5)
        line_profiles_sub_layout.setSpacing(4)
        
        # Dropdown for Gaussian mode selection
        gaussian_dropdown_layout = QtWidgets.QHBoxLayout()
        self.gaussian_mode_dropdown = QtWidgets.QComboBox()
        self.gaussian_mode_dropdown.addItem("")  # Empty (inactive)
        self.gaussian_mode_dropdown.addItem("Single Gaussian     [g]")
        self.gaussian_mode_dropdown.addItem("Multi Gaussian      [|]")
        self.gaussian_mode_dropdown.currentTextChanged.connect(self.on_gaussian_mode_changed)
        gaussian_dropdown_layout.addWidget(self.gaussian_mode_dropdown)
        gaussian_dropdown_layout.addStretch()
        line_profiles_sub_layout.addLayout(gaussian_dropdown_layout)
        
        # Enter button for Multi Gaussian mode
        gaussian_enter_layout = QtWidgets.QHBoxLayout()
        self.gaussian_enter_button = QtWidgets.QPushButton("Enter")
        self.gaussian_enter_button.setEnabled(False)
        self.gaussian_enter_button.clicked.connect(self.on_gaussian_enter_clicked)
        gaussian_enter_layout.addWidget(self.gaussian_enter_button)
        gaussian_enter_layout.addStretch()
        line_profiles_sub_layout.addLayout(gaussian_enter_layout)
        
        fitting_layout.addLayout(line_profiles_sub_layout)
        
        # --- ADVANCED SUBSECTION ---
        advanced_label = QtWidgets.QLabel("Advanced")
        advanced_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        fitting_layout.addWidget(advanced_label)
        
        advanced_sub_layout = QtWidgets.QVBoxLayout()
        advanced_sub_layout.setContentsMargins(10, 5, 10, 5)
        advanced_sub_layout.setSpacing(4)
        
        # Dropdown for Advanced mode selection
        advanced_dropdown_layout = QtWidgets.QHBoxLayout()
        self.advanced_mode_dropdown = QtWidgets.QComboBox()
        self.advanced_mode_dropdown.addItem("")  # Empty (inactive)
        self.advanced_mode_dropdown.addItem("Listfit            [H]")
        self.advanced_mode_dropdown.addItem("Bayes Fit          [:]")
        self.advanced_mode_dropdown.currentTextChanged.connect(self.on_advanced_mode_changed)
        advanced_dropdown_layout.addWidget(self.advanced_mode_dropdown)
        advanced_dropdown_layout.addStretch()
        advanced_sub_layout.addLayout(advanced_dropdown_layout)
        
        fitting_layout.addLayout(advanced_sub_layout)
        
        # Add fitting layout to main
        main_layout.addLayout(fitting_layout)
        
        # ===== CALCULATE SECTION TITLE =====
        calculate_title = QtWidgets.QLabel("Calculate")
        calculate_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #0078d4;")
        main_layout.addWidget(calculate_title)
        
        calculate_layout = QtWidgets.QVBoxLayout()
        calculate_layout.setContentsMargins(15, 5, 10, 10)
        calculate_layout.setSpacing(4)
        
        # Dropdown for Calculate mode selection
        calculate_dropdown_layout = QtWidgets.QHBoxLayout()
        self.calculate_mode_dropdown = QtWidgets.QComboBox()
        self.calculate_mode_dropdown.addItem("")  # Empty (inactive)
        self.calculate_mode_dropdown.addItem("Estimate Redshift     [z]")
        self.calculate_mode_dropdown.addItem("Velocity x-axis      [b]")
        self.calculate_mode_dropdown.currentTextChanged.connect(self.on_calculate_mode_changed)
        calculate_dropdown_layout.addWidget(self.calculate_mode_dropdown)
        calculate_dropdown_layout.addStretch()
        calculate_layout.addLayout(calculate_dropdown_layout)
        
        # Plot Residual button
        plot_residual_layout = QtWidgets.QHBoxLayout()
        self.plot_residual_button = QtWidgets.QPushButton("Plot Residual     [r]")
        self.plot_residual_button.clicked.connect(self.toggle_residual_panel)
        plot_residual_layout.addWidget(self.plot_residual_button)
        plot_residual_layout.addStretch()
        calculate_layout.addLayout(plot_residual_layout)
        
        # Plot Total Line button
        plot_total_line_layout = QtWidgets.QHBoxLayout()
        self.plot_total_line_button = QtWidgets.QPushButton("Plot Total Line    [;]")
        self.plot_total_line_button.clicked.connect(self.toggle_total_line)
        plot_total_line_layout.addWidget(self.plot_total_line_button)
        plot_total_line_layout.addStretch()
        calculate_layout.addLayout(plot_total_line_layout)
        
        # --- EQUIVALENT WIDTH SUBSECTION ---
        calculate_layout.addSpacing(8)
        ew_label = QtWidgets.QLabel("Equivalent Width")
        ew_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        calculate_layout.addWidget(ew_label)
        
        ew_sub_layout = QtWidgets.QVBoxLayout()
        ew_sub_layout.setContentsMargins(10, 5, 10, 5)
        ew_sub_layout.setSpacing(4)
        
        # Dropdown for Calculate EW mode selection
        calculate_ew_dropdown_layout = QtWidgets.QHBoxLayout()
        self.calculate_ew_mode_dropdown = QtWidgets.QComboBox()
        self.calculate_ew_mode_dropdown.addItem("")  # Empty (inactive)
        self.calculate_ew_mode_dropdown.addItem("Calculate Equivalent Width     [v]")
        self.calculate_ew_mode_dropdown.currentTextChanged.connect(self.on_calculate_ew_mode_changed)
        calculate_ew_dropdown_layout.addWidget(self.calculate_ew_mode_dropdown)
        calculate_ew_dropdown_layout.addStretch()
        ew_sub_layout.addLayout(calculate_ew_dropdown_layout)
        
        # Checkbox: Calculate EW automatically (on by default)
        self.calculate_ew_auto_checkbox = QtWidgets.QCheckBox("Calculate EW automatically")
        self.calculate_ew_auto_checkbox.setChecked(True)
        self.calculate_ew_auto_checkbox.stateChanged.connect(self.on_calculate_ew_auto_toggled)
        ew_sub_layout.addWidget(self.calculate_ew_auto_checkbox)
        
        # Checkbox: Plot MC Profiles Automatically
        self.plot_mc_profiles_checkbox = QtWidgets.QCheckBox("Plot MC Profiles Automatically")
        self.plot_mc_profiles_checkbox.setChecked(False)
        self.plot_mc_profiles_checkbox.stateChanged.connect(self.on_plot_mc_profiles_toggled)
        ew_sub_layout.addWidget(self.plot_mc_profiles_checkbox)
        
        # Button: Delete All MC Profiles
        delete_mc_button_layout = QtWidgets.QHBoxLayout()
        self.delete_all_mc_profiles_button = QtWidgets.QPushButton("Delete All MC Profiles")
        self.delete_all_mc_profiles_button.clicked.connect(self.delete_all_mc_profiles)
        delete_mc_button_layout.addWidget(self.delete_all_mc_profiles_button)
        delete_mc_button_layout.addStretch()
        ew_sub_layout.addLayout(delete_mc_button_layout)
        
        calculate_layout.addLayout(ew_sub_layout)
        
        main_layout.addLayout(calculate_layout)
        
        # --- DEACTIVATE ALL BUTTON ---
        main_layout.addSpacing(5)
        deactivate_button_layout = QtWidgets.QVBoxLayout()
        self.deactivate_all_button = QtWidgets.QPushButton("Deactivate All [Esc]")
        self.deactivate_all_button.clicked.connect(self.on_deactivate_all)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.deactivate_all_button)
        button_layout.addStretch()
        deactivate_button_layout.addLayout(button_layout)
        # Add helper text
        deactivate_help_label = QtWidgets.QLabel("Click this to deactivate all active processes from this Options tab.")
        deactivate_help_label.setStyleSheet("font-size: 10px; color: gray;")
        deactivate_button_layout.addWidget(deactivate_help_label)
        main_layout.addLayout(deactivate_button_layout)
        
        # Add stretch to push content to top
        main_layout.addStretch()
        
        # Wrap main_layout in a scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(main_layout)
        scroll_area.setWidget(scroll_widget)
        
        # Apply styling to indicate scrollability
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #f5f5f5;
                border-left: 3px solid #e0e0e0;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #b0b0b0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #808080;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # Set the scroll area as the dock widget's content
        dock_content.setLayout(QtWidgets.QVBoxLayout())
        dock_content.layout().addWidget(scroll_area)
        dock_widget.setWidget(dock_content)
        
        # Set minimum width for the dock
        dock_widget.setMinimumWidth(220)
        
        # Add dock widget to the right side
        mpl_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock_widget)
        
        # Store reference
        self.right_dock_widget = dock_widget

    def on_gaussian_mode_changed(self, mode_text):
        """Handle change in Gaussian fitting mode from dropdown"""
        # Strip keystroke indicator for comparison - try different space counts
        mode_clean = mode_text.split("     [")  # 5 spaces for Single Gaussian
        if len(mode_clean) > 1:
            mode_text = mode_clean[0]
        else:
            mode_clean = mode_text.split("      [")  # 6 spaces for Multi Gaussian
            if len(mode_clean) > 1:
                mode_text = mode_clean[0]
            else:
                mode_text = mode_text  # Keep original if no bracket found
        
        mode_text = mode_text.strip()  # Remove any trailing whitespace
        
        if mode_text == "":
            # Deactivated
            self.gaussian_mode = False
            self.multi_gaussian_mode_old = False
            self.gaussian_enter_button.setEnabled(False)
        elif mode_text == "Single Gaussian":
            # Activate single Gaussian mode (same as pressing 'd')
            self.gaussian_mode = True
            self.multi_gaussian_mode_old = False
            self.bounds = []
            if hasattr(self, 'bound_lines'):
                self.bound_lines.clear()
            print("Single Gaussian mode activated from Options panel")
            self.gaussian_enter_button.setEnabled(False)
            self.record_action('activate_single_gaussian', 'Activated Single Gaussian Mode')
        elif mode_text == "Multi Gaussian":
            # Activate multi Gaussian mode (same as pressing 'D')
            self.multi_gaussian_mode_old = True
            self.gaussian_mode = False
            self.bounds = []
            if hasattr(self, 'bound_lines'):
                self.bound_lines.clear()
            print("Multi Gaussian mode activated from Options panel")
            self.update_gaussian_enter_button()
            self.record_action('activate_multi_gaussian', 'Activated Multi Gaussian Mode')

    def on_continuum_mode_changed(self, mode_text):
        """Handle change in Continuum fitting mode from dropdown"""
        # Strip keystroke indicator for comparison
        mode_clean = mode_text.split("     [")
        if len(mode_clean) > 1:
            mode_text = mode_clean[0]
        else:
            mode_text = mode_clean[0]
        if mode_text == "":
            # Deactivated - exit continuum mode
            self.continuum_mode = False
            self.continuum_regions = []
            for patch_info in self.continuum_patches:
                if 'patch' in patch_info:
                    patch = patch_info['patch']
                    if patch in self.ax.patches:
                        patch.remove()
            self.continuum_patches.clear()
            self.continuum_enter_button.setEnabled(False)
            self.fig.canvas.draw_idle()
            print("Continuum mode deactivated from Options panel")
        elif mode_text == "Continuum Region(s)":
            # Activate continuum mode (same as pressing 'm')
            self.continuum_mode = True
            self.continuum_regions = []
            self.continuum_patches = []
            self.continuum_enter_button.setEnabled(True)
            print("Continuum fitting mode activated from Options panel")
            print("Use the spacebar to define the left bound and then the right bound of a continuum region.")
            print("You can define multiple continuum regions by selecting multiple pairs of bounds.")
            print("When the regions are set, hit Enter or Return to fit a polynomial (the order of the polynomial is configurable in Options).")
            self.record_action('activate_continuum_mode', 'Activated Continuum Mode')

    def on_advanced_mode_changed(self, mode_text):
        """Handle change in Advanced fitting mode (Listfit or Bayes) from dropdown"""
        # Strip keystroke indicator for comparison
        mode_clean = mode_text.split("            [")
        if len(mode_clean) > 1:
            mode_text = mode_clean[0]
        else:
            mode_text = mode_clean[0]
        
        if mode_text == "":
            # Deactivated - exit both listfit and bayes modes
            if self.listfit_mode:
                for line in self.listfit_bound_lines:
                    try:
                        line.remove()
                    except (ValueError, NotImplementedError):
                        pass
                self.listfit_bound_lines.clear()
                self.listfit_bounds = []
                self.listfit_mode = False
            if self.bayes_mode:
                for line in self.bayes_bound_lines:
                    try:
                        line.remove()
                    except (ValueError, NotImplementedError):
                        pass
                self.bayes_bound_lines.clear()
                self.bayes_bounds = []
                self.bayes_mode = False
            if self.ax is not None:
                self.ax.figure.canvas.draw_idle()
            print("Advanced modes deactivated from Options panel")
        elif mode_text == "Listfit":
            # Activate listfit mode (same as pressing 'H')
            self.listfit_mode = True
            self.listfit_bounds = []
            self.listfit_bound_lines = []
            self.listfit_components = []
            print("Listfit mode: Use the spacebar to define left and right boundaries.")
            self.record_action('activate_listfit_mode', 'Activated Listfit Mode')
        elif mode_text == "Bayes Fit":
            # Activate bayes mode (same as pressing ':')
            if self.bayes_mode:
                self.bayes_mode = False
                print("Exiting Bayes fit mode.")
            else:
                self.bayes_mode = True
                self.bayes_bounds = []
                if self.bayes_bound_lines is not None:
                    for line in self.bayes_bound_lines:
                        if line in self.ax.lines:
                            line.remove()
                self.bayes_bound_lines = []
                print("Bayes fit mode: Use the spacebar to define left and right boundaries.")
                self.record_action('activate_bayes_mode', 'Activated Bayes Fit Mode')

    def on_calculate_mode_changed(self, mode_text):
        """Handle change in Calculate mode (Redshift Estimation or Velocity) from dropdown"""
        # Strip keystroke indicator for comparison
        mode_clean = mode_text.split("     [")
        if len(mode_clean) > 1:
            mode_text = mode_clean[0]
        else:
            mode_clean = mode_text.split("      [")
            if len(mode_clean) > 1:
                mode_text = mode_clean[0]
            else:
                mode_text = mode_clean[0]
        
        if mode_text == "":
            # Deactivated - exit both redshift and velocity modes
            if self.redshift_estimation_mode:
                self.redshift_estimation_mode = False
                print('Redshift estimation mode deactivated from Options panel')
            if self.is_velocity_mode:
                self.exit_velocity_mode()
                self.rest_wavelength = None
                self.rest_id = None
                print('Velocity mode deactivated from Options panel')
            if self.ax is not None:
                self.ax.figure.canvas.draw_idle()
        elif mode_text == "Estimate Redshift":
            # Activate redshift estimation mode (same as pressing 'z')
            if self.redshift_estimation_mode:
                self.redshift_estimation_mode = False
                print('Exiting redshift estimation mode.')
            else:
                self.redshift_estimation_mode = True
                print('Redshift estimation mode: Select Gaussian to use for redshift estimation. Assign a line to it, and estimate the redshift.')
                self.record_action('activate_redshift_estimation', 'Activated Redshift Estimation Mode')
        elif mode_text == "Velocity x-axis":
            # Activate velocity mode (same as pressing 'b')
            self.is_velocity_mode = not self.is_velocity_mode  # Toggle Velocity mode
            if self.is_velocity_mode:
                self.activate_velocity_mode()  # Enter velocity mode
                if self.is_residual_shown:
                    self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)")
                    self.update_residual_ticks()
                else:
                    self.ax.set_xlabel(r"Velocity (km s$^{-1}$)")
                print("Velocity mode activated from Calculate menu")
                self.record_action('activate_velocity_mode', 'Activated Velocity Mode')
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
                print("Velocity mode deactivated from Calculate menu")
                self.record_action('deactivate_velocity_mode', 'Deactivated Velocity Mode')
                plt.draw()

    def on_calculate_ew_mode_changed(self, mode_text):
        """Handle change in Calculate Equivalent Width mode from dropdown"""
        # Strip keystroke indicator for comparison
        mode_clean = mode_text.split("     [")
        if len(mode_clean) > 1:
            mode_text = mode_clean[0]
        else:
            mode_text = mode_clean[0]
        
        if mode_text == "":
            # Deactivated - exit EW selection mode
            if self.calculate_ew_selection_mode:
                self.calculate_ew_selection_mode = False
                print('Calculate Equivalent Width mode deactivated from Options panel')
        elif mode_text == "Calculate Equivalent Width":
            # Activate EW selection mode (same as pressing 'v')
            if self.calculate_ew_selection_mode:
                self.calculate_ew_selection_mode = False
                print('Exiting Calculate Equivalent Width mode.')
                # Update dropdown back to empty
                self.calculate_ew_mode_dropdown.blockSignals(True)
                self.calculate_ew_mode_dropdown.setCurrentIndex(0)
                self.calculate_ew_mode_dropdown.blockSignals(False)
            else:
                self.calculate_ew_selection_mode = True
                print('Calculate Equivalent Width mode: Use spacebar to select a profile for EW calculation, or use Item Tracker context menu.')
                self.record_action('activate_calculate_ew_mode', 'Activated Calculate Equivalent Width Mode')

    def on_poly_order_minus(self):
        """Handle minus button for polynomial order"""
        try:
            current = int(self.options_poly_order_input.text())
            if current > 0:
                current -= 1
                self.options_poly_order_input.setText(str(current))
                self.poly_order = current
                print(f"Polynomial order set to: {current}")
        except ValueError:
            pass

    def on_poly_order_plus(self):
        """Handle plus button for polynomial order"""
        try:
            current = int(self.options_poly_order_input.text())
            if current < 10:
                current += 1
                self.options_poly_order_input.setText(str(current))
                self.poly_order = current
                print(f"Polynomial order set to: {current}")
        except ValueError:
            pass

    def on_poly_order_changed(self):
        """Handle direct edit of polynomial order field"""
        try:
            value = int(self.options_poly_order_input.text())
            if 0 <= value <= 10:
                self.poly_order = value
                print(f"Polynomial order set to: {value}")
        except ValueError:
            self.options_poly_order_input.setText(str(self.poly_order))

    def on_continuum_enter_clicked(self):
        """Handle Enter button click to perform continuum fit"""
        if not self.continuum_mode or len(self.continuum_regions) == 0:
            print("No continuum regions defined. Define regions with spacebar first.")
            return
        
        # Trigger the continuum fit logic (same as pressing 'enter' in continuum mode)
        # Combine all defined regions into a single dataset for fitting
        combined_wav = []
        combined_spec = []
        combined_err = []

        for region in self.continuum_regions:
            start, end = region
            mask = (self.x_data >= start) & (self.x_data <= end)
            if np.any(mask):
                combined_wav.extend(self.x_data[mask])
                combined_spec.extend(self.spec[mask])
                if self.err is not None:
                    combined_err.extend(self.err[mask])

        # Store the individual regions (not the combined bound) in the fit
        individual_regions = list(self.continuum_regions)
        region_bounds = (min(region[0] for region in self.continuum_regions), 
                        max(region[1] for region in self.continuum_regions))

        combined_wav = np.array(combined_wav)
        combined_spec = np.array(combined_spec)
        combined_err = np.array(combined_err) if combined_err else None

        # Fit the continuum
        continuum, coeffs, perr, pcov = self.fit_continuum(combined_wav, combined_spec, combined_err, 
                                                            poly_order=self.poly_order)

        # Plot the fitted continuum
        x_plot = np.linspace(region_bounds[0], region_bounds[1], 500)
        continuum_full = np.polyval(coeffs, x_plot)
        continuum_cfg = self.colors['profiles']['continuum_line']
        continuum_line, = self.ax.plot(x_plot, continuum_full, color=continuum_cfg['color'], 
                                      linestyle=continuum_cfg['linestyle'], alpha=0.8)
        if self.is_residual_shown:
            self.calculate_and_plot_residuals()
        self.update_legend()
        self.ax.figure.canvas.draw()
        QtWidgets.QApplication.processEvents()

        # Store continuum fit
        continuum_fit = {
            'bounds': region_bounds,
            'individual_regions': individual_regions,  # Store each region separately
            'coeffs': coeffs,
            'coeffs_err': perr,
            'covariance': pcov,  # Store full covariance matrix
            'poly_order': self.poly_order,
            'patches': self.continuum_patches,
            'line': continuum_line,
            'is_velocity_mode': self.is_velocity_mode
        }
        self.continuum_fits.append(continuum_fit)
        
        # Register with ItemTracker
        bounds_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
        self.register_item('continuum', f'Continuum (order {self.poly_order})', fit_dict=continuum_fit,
                         line_obj=continuum_line, position=bounds_str, color=continuum_cfg['color'])
        
        self.record_action('fit_continuum', f'Fit Continuum (order {self.poly_order})')
        
        # Save fit to .qsap file and print
        self.save_and_print_qsap_fit(continuum_fit, 'Continuum', 'Single')
        
        # Clean up and deactivate
        self.continuum_regions = []
        self.continuum_patches = []
        self.continuum_mode = False
        
        # Reset dropdown to blank
        self.continuum_mode_dropdown.blockSignals(True)
        self.continuum_mode_dropdown.setCurrentIndex(0)
        self.continuum_mode_dropdown.blockSignals(False)
        self.continuum_enter_button.setEnabled(False)

    def reset_advanced_dropdown(self):
        """Reset Advanced dropdown to blank and deactivate related modes"""
        self.listfit_mode = False
        self.bayes_mode = False
        self.bayes_bounds = []
        for line in self.bayes_bound_lines:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.bayes_bound_lines.clear()
        if hasattr(self, 'advanced_mode_dropdown'):
            self.advanced_mode_dropdown.blockSignals(True)
            self.advanced_mode_dropdown.setCurrentIndex(0)
            self.advanced_mode_dropdown.blockSignals(False)
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
    
    def reset_calculate_dropdown(self):
        """Reset Calculate dropdown to blank and deactivate related modes"""
        self.redshift_estimation_mode = False
        self.is_velocity_mode = False
        if hasattr(self, 'calculate_mode_dropdown'):
            self.calculate_mode_dropdown.blockSignals(True)
            self.calculate_mode_dropdown.setCurrentIndex(0)
            self.calculate_mode_dropdown.blockSignals(False)
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
    
    def _cleanup_redshift_highlighting(self):
        """Remove neon green highlighting from redshift selected line"""
        if hasattr(self, 'redshift_selected_line') and self.redshift_selected_line:
            # Restore the line to its original color based on its fit type
            if hasattr(self, 'gaussian_fits'):
                for fit in self.gaussian_fits:
                    if 'line' in fit and fit['line'] is self.redshift_selected_line:
                        gaussian_cfg = self.colors['profiles']['gaussian']
                        fit['line'].set_color(gaussian_cfg['color'])
                        fit['line'].set_linewidth(gaussian_cfg['linewidth'])
                        fit['line'].set_zorder(2)  # Reset z-order to bring back above data
                        break
            if hasattr(self, 'voigt_fits'):
                for fit in self.voigt_fits:
                    if 'line' in fit and fit['line'] is self.redshift_selected_line:
                        voigt_cfg = self.colors['profiles']['voigt']
                        fit['line'].set_color(voigt_cfg['color'])
                        fit['line'].set_linewidth(voigt_cfg['linewidth'])
                        fit['line'].set_zorder(2)  # Reset z-order to bring back above data
                        break
            self.redshift_selected_line = None
        
        # Remove any preview plots
        if hasattr(self, 'current_gaussian_plot') and self.current_gaussian_plot:
            try:
                self.current_gaussian_plot.remove()
            except (ValueError, RuntimeError):
                pass
            self.current_gaussian_plot = None
        if hasattr(self, 'current_voigt_plot') and self.current_voigt_plot:
            try:
                self.current_voigt_plot.remove()
            except (ValueError, RuntimeError):
                pass
            self.current_voigt_plot = None
        
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
    
    def on_deactivate_all(self):
        """Deactivate all active fitting modes"""
        modes_deactivated = []
        
        # Deactivate continuum mode if active
        if self.continuum_mode:
            self.continuum_mode = False
            self.continuum_regions = []
            # Remove patches
            for patch_info in self.continuum_patches:
                if 'patch' in patch_info:
                    patch = patch_info['patch']
                    if patch in self.ax.patches:
                        patch.remove()
            self.continuum_patches.clear()
            # Remove bound lines if any
            for line in self.bound_lines:
                line.remove()
            self.continuum_mode_dropdown.blockSignals(True)
            self.continuum_mode_dropdown.setCurrentIndex(0)
            self.continuum_mode_dropdown.blockSignals(False)
            self.continuum_enter_button.setEnabled(False)
            modes_deactivated.append("Continuum")
        
        # Deactivate Gaussian mode if active
        if self.gaussian_mode or self.multi_gaussian_mode_old:
            self.gaussian_mode = False
            self.multi_gaussian_mode_old = False
            # Remove bounds
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            self.bounds.clear()
            self.gaussian_mode_dropdown.blockSignals(True)
            self.gaussian_mode_dropdown.setCurrentIndex(0)
            self.gaussian_mode_dropdown.blockSignals(False)
            self.gaussian_enter_button.setEnabled(False)
            modes_deactivated.append("Gaussian")
        
        # Deactivate Advanced modes (Listfit, Bayes)
        if self.listfit_mode:
            for line in self.listfit_bound_lines:
                try:
                    line.remove()
                except (ValueError, NotImplementedError):
                    pass
            self.listfit_bound_lines.clear()
            self.listfit_bounds = []
            self.listfit_mode = False
            modes_deactivated.append("Listfit")
        if self.bayes_mode:
            for line in self.bayes_bound_lines:
                try:
                    line.remove()
                except (ValueError, NotImplementedError):
                    pass
            self.bayes_bound_lines.clear()
            self.bayes_bounds = []
            self.bayes_mode = False
            modes_deactivated.append("Bayes Fit")
        if hasattr(self, 'advanced_mode_dropdown'):
            self.advanced_mode_dropdown.blockSignals(True)
            self.advanced_mode_dropdown.setCurrentIndex(0)
            self.advanced_mode_dropdown.blockSignals(False)
        
        # Deactivate Calculate modes (Redshift estimation, Velocity)
        if self.redshift_estimation_mode:
            self.redshift_estimation_mode = False
            modes_deactivated.append("Redshift Estimation")
        if self.is_velocity_mode:
            self.is_velocity_mode = False
            modes_deactivated.append("Velocity Mode")
        if hasattr(self, 'calculate_mode_dropdown'):
            self.calculate_mode_dropdown.blockSignals(True)
            self.calculate_mode_dropdown.setCurrentIndex(0)
            self.calculate_mode_dropdown.blockSignals(False)
        
        # Redraw canvas
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
        
        if modes_deactivated:
            print(f"Deactivated: {', '.join(modes_deactivated)}")
        else:
            print("No active fitting modes to deactivate.")

    def update_gaussian_enter_button(self):
        """Update the enabled state of the Enter button based on number of bounds"""
        if self.multi_gaussian_mode_old:
            # For Multi Gaussian: enable if even number of bounds >= 4
            if len(self.bounds) >= 4 and len(self.bounds) % 2 == 0:
                self.gaussian_enter_button.setEnabled(True)
            else:
                self.gaussian_enter_button.setEnabled(False)
        else:
            self.gaussian_enter_button.setEnabled(False)

    def on_gaussian_enter_clicked(self):
        """Handle Enter button click to perform Multi Gaussian fit"""
        if not self.multi_gaussian_mode_old or len(self.bounds) < 4:
            print("Invalid Multi Gaussian configuration: need at least 4 bounds (2 profiles minimum)")
            return
        
        # Perform the fit with the current bounds
        self.perform_multi_gaussian_fit()

    def perform_multi_gaussian_fit(self):
        """Perform multi-gaussian fitting with current bounds"""
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
                # Update button state
                self.update_gaussian_enter_button()
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
        sigma_maxes = []  # Store sigma_max for each component
        for left_bound, right_bound in bound_pairs:
            comp_x = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
            comp_y = self.spec[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
            # Handle optional error spectrum
            if self.err is not None:
                comp_err = self.err[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
            else:
                comp_err = None
            comp_ys.append(comp_y)
            
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
            mean_guess = np.mean(comp_x)
            # Calculate max sigma for this component FIRST
            sigma_max = self._calculate_max_sigma(left_bound, right_bound, mean_guess, epsilon=0.05)
            sigma_maxes.append(sigma_max)
            # Cap initial sigma guess to stay within bounds (use 50% of max for safety)
            sigma_guess = min(np.std(comp_x), sigma_max * 0.5)
            initial_guesses.extend([max(continuum_subtracted_y) - min(continuum_subtracted_y), mean_guess, sigma_guess])

        # Fit multiple Gaussians
        if len(comp_xs) > 0:
            comp_xs = np.array(comp_xs)
            continuum_subtracted_ys = np.array(continuum_subtracted_ys)
            # Use sigma if errors available, otherwise None
            sigma_param = np.array(comp_errs) if comp_errs else None
            
            # Build bounds with sigma constraints for each component
            num_components = len(bound_pairs)
            lower_bounds = [-np.inf] * (num_components * 3)
            upper_bounds = [np.inf] * (num_components * 3)
            for i, sigma_max in enumerate(sigma_maxes):
                lower_bounds[i * 3 + 2] = 0  # sigma >= 0
                upper_bounds[i * 3 + 2] = sigma_max  # sigma <= sigma_max
            
            # Lazy import of scipy for curve fitting
            from scipy.optimize import curve_fit
            params, pcov = curve_fit(self.multi_gaussian, comp_xs, continuum_subtracted_ys, sigma=sigma_param, p0=initial_guesses, bounds=(lower_bounds, upper_bounds))
            perr = np.sqrt(np.diag(pcov))
            for i in range(0, len(params), 3):
                amp, mean, stddev = params[i:i+3]
                amp_err, mean_err, stddev_err = perr[i:i+3]
                x_fit = self.x_data[(self.x_data >= bound_pairs[i // 3][0]) & (self.x_data <= bound_pairs[i // 3][1])]
                y_fit = self.gaussian(x_fit, amp, mean, stddev) + continuum_ys[i // 3]
                continuum_sub_data = comp_ys[i // 3] - continuum_ys[i // 3]
                residuals = continuum_sub_data - self.gaussian(x_fit, amp, mean, stddev)
                # Calculate chi2 (simpler without errors since they're concatenated)
                chi2 = np.sum(residuals ** 2)  # Chi2 without decomposed errors
                chi2_nu = chi2 / (len(x_fit) - 3)  # 3 params per component
                
                # Extract this component's covariance from the full covariance matrix
                comp_cov_indices = [i, i+1, i+2]
                comp_cov = pcov[np.ix_(comp_cov_indices, comp_cov_indices)]
                
                # DEBUG: Verify covariance structure
                print(f"[DEBUG] Multi-Gaussian component {self.component_id}: comp_cov shape = {comp_cov.shape}, has_data = {comp_cov is not None}")
                
                # Lazy import of scipy for interpolation
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                gaussian_cfg = self.colors['profiles']['gaussian']
                fit_line, = self.ax.plot(x_plt, y_plt, color=gaussian_cfg['color'], linestyle=gaussian_cfg['linestyle'])
                left_bound, right_bound = bound_pairs[i // 3]
                gaussian_fit = {
                'fit_id': self.fit_id,
                'is_velocity_mode': self.is_velocity_mode,
                'chi2': chi2,
                'chi2_nu': chi2_nu,
                'has_errors': sigma_param is not None,  # Track if errors were available
                'component_id': self.component_id,
                'amp': amp, 'amp_err': amp_err, 'mean': mean, 'mean_err': mean_err, 'stddev': stddev, 'stddev_err': stddev_err,
                'bounds': (left_bound, right_bound),
                'line_id': line_id if line_id else None,
                'line_wavelength': line_wavelength  if line_wavelength else None,
                'line': fit_line,
                'rest_wavelength': self.rest_wavelength,
                'rest_id': self.rest_id,
                'z_sys': self.redshift,
                'covariance': comp_cov.tolist()  # Store covariance matrix as list
                }
                # DEBUG: Verify gaussian_fit has covariance
                print(f"[DEBUG] gaussian_fit created with covariance={('covariance' in gaussian_fit)}, bounds={gaussian_fit.get('bounds')}")
                self.gaussian_fits.append(gaussian_fit)
                # Register with ItemTracker
                position_str = f"λ: {mean:.2f} Å"
                gaussian_cfg = self.colors['profiles']['gaussian']
                self.register_item('gaussian', f'Gaussian', fit_dict=gaussian_fit, line_obj=fit_line,
                                 position=position_str, color=gaussian_cfg['color'])
                
                # Record action for undo/redo (only record once after all components)
                if i == len(params) - 3:  # Last component
                    self.record_action('fit_multi_gaussian', f'Fit {len(bound_pairs)} Gaussians')
                
                self.component_id += 1

            # Force immediate redraw of the canvas
            self.ax.figure.canvas.draw()
            QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
            
            # Save all components to single .qsap file with Multi-Gaussian MODE
            # Get all gaussians for this fit_id
            multi_gaussian_components = [g for g in self.gaussian_fits if g.get('fit_id') == self.fit_id]
            if multi_gaussian_components:
                # DEBUG: Verify structure before saving
                for idx, comp in enumerate(multi_gaussian_components):
                    has_bounds = 'bounds' in comp
                    has_covariance = 'covariance' in comp
                    print(f"[DEBUG] Multi-Gaussian component {idx}: has_bounds={has_bounds}, has_covariance={has_covariance}")
                self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian')
            
            self.fit_id += 1
            # Clear bound lines after fit
            for line in self.bound_lines:
                line.remove()
            self.bound_lines.clear()
            self.bounds = []
            self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
        
        # Update button state
        self.update_gaussian_enter_button()
        # Update dropdown to blank and deactivate mode
        self.gaussian_mode_dropdown.blockSignals(True)
        self.gaussian_mode_dropdown.setCurrentIndex(0)
        self.gaussian_mode_dropdown.blockSignals(False)
        print('Exiting Multi Gaussian mode.')

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
            # print(f"Using Gaussian LSF with FWHM {lsf_width} km/s.")
        
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
            # Lazy import for interpolation
            from scipy.interpolate import interp1d
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

    def _calculate_max_sigma(self, left_bound, right_bound, mean, epsilon=0.05):
        """
        Calculate maximum sigma for Gaussian to prevent unphysically broad (flat-top) profiles.
        
        Ensures the Gaussian value reaches < epsilon * amplitude at the boundaries.
        
        Parameters
        ----------
        left_bound : float
            Left boundary of fitting region
        right_bound : float
            Right boundary of fitting region
        mean : float
            Center (mean) of the Gaussian
        epsilon : float
            Threshold fraction (default 0.05 = 5%)
            At boundary: G(x) < epsilon * |A|
            
        Returns
        -------
        sigma_max : float
            Maximum allowed standard deviation
            
        Notes
        -----
        Derived from: sigma_max = d / sqrt(-2*ln(epsilon))
        where d = min distance from mean to boundary
        This constraint is scale-invariant and works across all flux magnitudes.
        """
        # Distance to nearest boundary
        dist_to_boundary = min(abs(mean - left_bound), abs(mean - right_bound))
        
        # Maximum sigma that keeps Gaussian at epsilon*A at boundary
        # sigma_max = d / sqrt(-2 * ln(epsilon))
        sigma_max = dist_to_boundary / np.sqrt(-2 * np.log(epsilon))
        
        return sigma_max

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
        residual_cfg = self.colors['residual']
        ref_cfg = self.colors['reference_lines']
        self.residual_line, = self.residual_ax.step(self.x_data, self.residuals, color=residual_cfg['color'], where='mid')
        self.residual_ax.plot(self.x_data, [0] * len(self.x_data), color=ref_cfg['color'], linestyle=ref_cfg['linestyle'], linewidth=ref_cfg['linewidth']) # Add horizontal line at y=0
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

    def toggle_total_line(self):
        """Toggle the total line for ALL fitted profiles (single, multi-gaussian, voigt, continuum, listfit)"""
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
            total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total"]
            for line in total_lines:
                line.remove()
            # Regenerate the legend to exclude removed lines
            self.update_legend()
            self.ax.figure.canvas.draw()

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

        # Add Listfit polynomial components to residuals (all active components)
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
                
                # Add polynomial components from the listfit (already filtered - deleted ones removed)
                poly_count = 0
                for comp in components:
                    if comp['type'] == 'polynomial':
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
        ref_cfg = self.colors['reference_lines']
        self.ax.plot(self.x_data, [0] * len(self.x_data), color=ref_cfg['color'], linestyle=ref_cfg['linestyle'], linewidth=ref_cfg['linewidth'])

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
                continuum_region_cfg = self.colors['profiles']['continuum_region']
                new_patch = self.ax.axvspan(vel_start, vel_end, color=continuum_region_cfg['color'], alpha=continuum_region_cfg['alpha'])
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
        ref_cfg = self.colors['reference_lines']
        self.ax.plot(self.x_data, [0] * len(self.x_data), color=ref_cfg['color'], linestyle=ref_cfg['linestyle'], linewidth=ref_cfg['linewidth'])

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
                continuum_region_cfg = self.colors['profiles']['continuum_region']
                new_patch = self.ax.axvspan(wav_start, wav_end, color=continuum_region_cfg['color'], alpha=continuum_region_cfg['alpha'])
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
        from scipy.special import wofz  # Lazy import for Voigt profile
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
                pcov = None

            # Return the final continuum and parameters
            return continuum, coeffs, perr, pcov
        except RuntimeError as e:
            print(f"Error in fitting continuum: {e}")
            return None, None, None, None

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
        preview_cfg = self.colors['preview']
        self.current_gaussian_plot = self.ax.plot(x, plot_data, color=preview_cfg['color'], linestyle=preview_cfg['linestyle'], linewidth=preview_cfg['linewidth'])[0]  # Store the first element (line object)
        
        # Highlight the original fit line in neon green for redshift mode
        if 'line' in fit and fit['line']:
            neon_green = '#39FF14'  # Neon green color
            fit['line'].set_color(neon_green)
            fit['line'].set_linewidth(3.0)  # Make it thicker for visibility
            fit['line'].set_zorder(10)  # Bring to front
            self.redshift_selected_line = fit['line']
            print(f"Highlighted Gaussian fit line in neon green")
        
        self.ax.figure.canvas.draw_idle()  # Refresh the plot

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
        preview_cfg = self.colors['preview']
        self.current_voigt_plot = self.ax.plot(x, plot_data, color=preview_cfg['color'], linestyle=preview_cfg['linestyle'], linewidth=preview_cfg['linewidth'])[0]
        
        # Highlight the original fit line in neon green for redshift mode
        if 'line' in fit and fit['line']:
            neon_green = '#39FF14'  # Neon green color
            fit['line'].set_color(neon_green)
            fit['line'].set_linewidth(3.0)  # Make it thicker for visibility
            fit['line'].set_zorder(10)  # Bring to front
            self.redshift_selected_line = fit['line']
            print(f"Highlighted Voigt fit line in neon green")
        
        self.ax.figure.canvas.draw_idle()  # Refresh the plot to show the updated Voigt profile

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
                    
                    # Create a mixed coordinate annotation:
                    # x: data coordinates (wavelength), y: normalized axes coordinates (0-1)
                    # This way the annotation stays at the same relative position even when panning/zooming
                    from matplotlib.transforms import blended_transform_factory
                    trans = blended_transform_factory(self.ax.transData, self.ax.transAxes)
                    
                    # Apply offset to the normalized y position
                    y_position = self.linelist_y_offset
                    
                    label = self.ax.text(shifted_wl_display + (self.linelist_x_offset * (xlim[1] - xlim[0])), 
                                        y_position, line.name,
                                        rotation=90, verticalalignment='bottom', 
                                        color=color, fontsize=8,
                                        transform=trans)
                    self.current_linelist_lines.append((vline, label))
        
        # Use canvas.draw_idle() instead of plt.draw() to ensure proper update
        if self.ax and self.ax.figure:
            self.ax.figure.canvas.draw_idle()

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
        
        # Update line list annotation offsets from the selector
        if self.line_list_selector:
            self.linelist_x_offset = self.line_list_selector.linelist_x_offset
            self.linelist_y_offset = self.line_list_selector.linelist_y_offset
        
        # Redisplay line lists
        if self.active_line_lists:
            self.display_linelist()
        else:
            self.clear_linelist()
            if self.ax and self.ax.figure:
                self.ax.figure.canvas.draw_idle()

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
    
    def save_and_print_qsap_fit(self, fit_data, fit_type, fit_mode='Single'):
        """Save fit to .qsap file and print contents to terminal
        
        Args:
            fit_data: Single dict or list of dicts with fit parameters
            fit_type: 'Gaussian', 'Voigt', 'Continuum', or 'Listfit'
            fit_mode: 'Single', 'Multi-Gaussian', 'Listfit', etc.
            
        Returns:
            Tuple of (filepath, file_content)
        """
        # Calculate equivalent width using Monte Carlo error propagation
        # Only if "Calculate EW automatically" checkbox is enabled
        if self.calculate_ew_enabled and fit_type in ['Gaussian', 'Voigt', 'Listfit']:
            # Get the continuum fit to use for EW calculation
            continuum_fit_dict = None
            if self.continuum_fits:
                # Use the most recent continuum fit
                continuum_fit_dict = self.continuum_fits[-1]
            
            if isinstance(fit_data, list):
                for fit in fit_data:
                    # For Listfit, extract the actual profile type from the component
                    if fit_type == 'Listfit':
                        component_type = fit.get('type', '').lower()
                        # Skip non-profile components (polynomial, masks, diagnostics)
                        if component_type not in ['gaussian', 'voigt']:
                            continue
                        component_fit_type = component_type
                    else:
                        # For multi-Gaussian or multi-Voigt, use the provided fit_type
                        component_fit_type = fit_type.lower()
                    
                    # DEBUG: Check structure of fit dict
                    component_id = fit.get('component_id', '?')
                    has_bounds = 'bounds' in fit
                    has_covariance = 'covariance' in fit
                    print(f"[DEBUG] Component {component_id}: has_bounds={has_bounds}, has_covariance={has_covariance}")
                    
                    try:
                        ew_result = self._calculate_equivalent_width_monte_carlo(
                            fit, continuum_fit_dict, component_fit_type
                        )
                        if ew_result:
                            # Store best, median, mean
                            fit['ew_best'] = ew_result.get('ew_best')
                            fit['ew_median'] = ew_result.get('ew_median')
                            fit['ew_mean'] = ew_result.get('ew_mean')
                            # Store for file output (backward compatibility)
                            fit['equivalent_width'] = ew_result['ew']
                            fit['equivalent_width_1sigma_lower'] = ew_result.get('ew_1sigma_lower')
                            fit['equivalent_width_1sigma_upper'] = ew_result.get('ew_1sigma_upper')
                            fit['equivalent_width_2sigma_lower'] = ew_result.get('ew_2sigma_lower')
                            fit['equivalent_width_2sigma_upper'] = ew_result.get('ew_2sigma_upper')
                            fit['equivalent_width_3sigma_lower'] = ew_result.get('ew_3sigma_lower')
                            fit['equivalent_width_3sigma_upper'] = ew_result.get('ew_3sigma_upper')
                            print(f"[EW] Calculated for component {component_id}")
                            
                            # Plot MC profiles if enabled
                            if self.plot_mc_profiles_enabled:
                                self.plot_mc_profiles(fit, ew_result, component_fit_type)
                        else:
                            print(f"[EW] No result for component {component_id}")
                    except Exception as e:
                        print(f"[EW] Error calculating EW for component {component_id}: {e}")
            else:
                ew_result = self._calculate_equivalent_width_monte_carlo(
                    fit_data, continuum_fit_dict, fit_type.lower()
                )
                if ew_result:
                    # Store best, median, mean
                    fit_data['ew_best'] = ew_result.get('ew_best')
                    fit_data['ew_median'] = ew_result.get('ew_median')
                    fit_data['ew_mean'] = ew_result.get('ew_mean')
                    # Store for file output (backward compatibility)
                    fit_data['equivalent_width'] = ew_result['ew']
                    fit_data['equivalent_width_1sigma_lower'] = ew_result.get('ew_1sigma_lower')
                    fit_data['equivalent_width_1sigma_upper'] = ew_result.get('ew_1sigma_upper')
                    fit_data['equivalent_width_2sigma_lower'] = ew_result.get('ew_2sigma_lower')
                    fit_data['equivalent_width_2sigma_upper'] = ew_result.get('ew_2sigma_upper')
                    fit_data['equivalent_width_3sigma_lower'] = ew_result.get('ew_3sigma_lower')
                    fit_data['equivalent_width_3sigma_upper'] = ew_result.get('ew_3sigma_upper')
                    
                    # Plot MC profiles if enabled
                    if self.plot_mc_profiles_enabled:
                        self.plot_mc_profiles(fit_data, ew_result, fit_type.lower())
        
        # Build spectrum info dict
        spectrum_info = {
            'wavelength_unit': self.wavelength_unit,
            'velocity_mode': self.is_velocity_mode,
            'scale_factor': self.flux_scale_factor,
        }
        if self.x_data is not None and len(self.x_data) > 0:
            spectrum_info['wavelength_range'] = (self.x_data[0], self.x_data[-1])
        if self.rest_wavelength:
            spectrum_info['rest_wavelength'] = self.rest_wavelength
        
        # Create .qsap file
        if fit_type == 'Gaussian':
            filepath, content = self.qsap_handler.create_gaussian_qsap(
                fit_data, self.fits_file, fit_mode, spectrum_info
            )
        elif fit_type == 'Voigt':
            filepath, content = self.qsap_handler.create_voigt_qsap(
                fit_data, self.fits_file, fit_mode, spectrum_info
            )
        elif fit_type == 'Continuum':
            filepath, content = self.qsap_handler.create_continuum_qsap(
                fit_data, self.fits_file, spectrum_info
            )
        elif fit_type == 'Listfit':
            filepath, content = self.qsap_handler.create_listfit_qsap(
                fit_data, self.fits_file, spectrum_info
            )
        else:
            raise ValueError(f"Unknown fit type: {fit_type}")
        
        # Print the file contents to terminal
        print("\n" + "="*70)
        print(f"FIT SAVED TO: {os.path.basename(filepath)}")
        print("="*70)
        print(content)
        print("="*70 + "\n")
        
        return filepath, content
    
    def _format_param_value(self, value, error=None):
        """Format parameter value with error for redshift data"""
        if error is None or error != error:  # Check for NaN
            return f"{value}"
        return f"{value}±{error}"
    
    def _calculate_equivalent_width(self, fit_dict, fit_type='gaussian'):
        """Calculate equivalent width for a fitted profile
        
        Equivalent width is the width of an imaginary perfectly black line that contains 
        the same integrated area as the observed profile.
        EW = ∫(1 - f_obs/f_continuum) dλ = ∫(f_continuum - f_obs)/f_continuum dλ
        
        Args:
            fit_dict: Dictionary containing fit parameters
            fit_type: 'gaussian' or 'voigt'
            
        Returns:
            Dictionary with 'ew' and 'ew_err' keys, or None if calculation fails
            
        Note: Place between triple-hash marks (###) to easily comment out this development feature
        """
        ### EQUIVALENT WIDTH CALCULATION - IN DEVELOPMENT
        try:
            bounds = fit_dict.get('bounds', (None, None))
            if bounds[0] is None or bounds[1] is None:
                return None
            
            # Get continuum level - try to get it from stored continuum or calculate from data
            continuum_level = None
            
            # Try to get continuum from existing continuum fits
            if self.continuum_fits:
                for cont_fit in self.continuum_fits:
                    cont_bounds = cont_fit.get('bounds', (None, None))
                    if cont_bounds[0] is not None and cont_bounds[1] is not None:
                        if cont_bounds[0] <= bounds[0] and bounds[1] <= cont_bounds[1]:
                            # This continuum covers our profile region
                            x_center = (bounds[0] + bounds[1]) / 2.0
                            if 'coeffs' in cont_fit:
                                continuum_level = np.polyval(cont_fit['coeffs'], x_center)
                            break
            
            # If no continuum found, estimate from data endpoints
            if continuum_level is None:
                # Find data points at the edges of the fit region
                left_mask = (self.x_data >= bounds[0] - (bounds[1] - bounds[0]) * 0.1) & \
                            (self.x_data <= bounds[0])
                right_mask = (self.x_data >= bounds[1]) & \
                             (self.x_data <= bounds[1] + (bounds[1] - bounds[0]) * 0.1)
                
                left_cont = np.nanmedian(self.spec[left_mask]) if np.any(left_mask) else np.nan
                right_cont = np.nanmedian(self.spec[right_mask]) if np.any(right_mask) else np.nan
                
                if not np.isnan(left_cont) and not np.isnan(right_cont):
                    continuum_level = (left_cont + right_cont) / 2.0
                elif not np.isnan(left_cont):
                    continuum_level = left_cont
                elif not np.isnan(right_cont):
                    continuum_level = right_cont
            
            if continuum_level is None or continuum_level <= 0:
                return None
            
            # Create high-resolution wavelength grid for integration
            x_int = np.linspace(bounds[0], bounds[1], 200)
            
            # Evaluate profile at high resolution
            if fit_type.lower() == 'gaussian':
                amp = fit_dict.get('amp', fit_dict.get('amplitude'))
                mean = fit_dict.get('mean')
                stddev = fit_dict.get('stddev', fit_dict.get('std_dev'))
                if any(v is None for v in [amp, mean, stddev]):
                    return None
                y_profile = self.gaussian(x_int, amp, mean, stddev)
                amp_err = fit_dict.get('amp_err')
                stddev_err = fit_dict.get('stddev_err', fit_dict.get('std_dev_err'))
                
            elif fit_type.lower() == 'voigt':
                amp = fit_dict.get('amplitude')
                mean = fit_dict.get('mean', fit_dict.get('center'))
                sigma = fit_dict.get('sigma')
                gamma = fit_dict.get('gamma')
                if any(v is None for v in [amp, mean, sigma, gamma]):
                    return None
                y_profile = self.voigt(x_int, amp, mean, sigma, gamma)
                amp_err = fit_dict.get('amplitude_err')
                sigma_err = fit_dict.get('sigma_err')
                
            else:
                return None
            
            # Calculate equivalent width: EW = ∫ (continuum - profile) / continuum dλ
            # For absorption lines (profile < continuum): EW is positive
            # For emission lines (profile > continuum): EW is negative
            normalized_diff = (continuum_level - y_profile) / continuum_level
            
            # Integrate using trapezoidal rule
            ew = np.trapz(normalized_diff, x_int)
            
            # Apply sign convention: 
            # - Emission lines (positive amplitude): EW should be negative
            # - Absorption lines (negative amplitude): EW should be positive
            amp = fit_dict.get('amp', fit_dict.get('amplitude'))
            if amp is not None and amp > 0:
                # Emission line - make EW negative
                ew = -abs(ew)
            else:
                # Absorption line - keep EW positive
                ew = abs(ew)
            
            # Estimate uncertainty - approximation using wave resolution and flux errors
            # EW_err ~ (∂EW/∂amp) * amp_err for conservative estimate
            if fit_type.lower() == 'gaussian' and amp_err is not None and stddev_err is not None:
                # Approximate derivative: ∂EW/∂amp ≈ Δλ / continuum  
                delta_lambda = bounds[1] - bounds[0]
                ew_err_amp = (delta_lambda / continuum_level) * amp_err
                ew_err_width = (amp / continuum_level) * stddev_err * 2.355  # FWHM = 2.355*sigma
                ew_err = np.sqrt(ew_err_amp**2 + ew_err_width**2)
            elif fit_type.lower() == 'voigt' and amp_err is not None and sigma_err is not None:
                delta_lambda = bounds[1] - bounds[0]
                ew_err_amp = (delta_lambda / continuum_level) * amp_err
                ew_err_width = (amp / continuum_level) * sigma_err
                ew_err = np.sqrt(ew_err_amp**2 + ew_err_width**2)
            else:
                ew_err = None
            
            return {'ew': ew, 'ew_err': ew_err}
        except Exception as e:
            # Development mode - print error but don't crash
            print(f"[DEV] EW calculation error: {e}")
            return None
        ### END EQUIVALENT WIDTH CALCULATION
    
    def _get_profile_params_from_dict(self, fit_dict, fit_type):
        """Extract profile parameters from fit dictionary based on fit type
        
        Returns tuple of (param_list, param_names)
        """
        fit_type = fit_type.lower()
        
        if fit_type == 'gaussian':
            params = [fit_dict.get('amp'), fit_dict.get('mean'), fit_dict.get('stddev')]
            names = ['amp', 'mean', 'stddev']
        elif fit_type == 'voigt':
            params = [fit_dict.get('amplitude'), fit_dict.get('center', fit_dict.get('mean')), 
                     fit_dict.get('sigma'), fit_dict.get('gamma')]
            names = ['amplitude', 'center', 'sigma', 'gamma']
        else:
            return None, None
        
        return params, names
    
    def _evaluate_profile(self, x, fit_type, params):
        """Evaluate a profile at wavelength points x
        
        Args:
            x: wavelength array
            fit_type: 'gaussian', 'voigt', etc.
            params: list of profile parameters in correct order
        
        Returns:
            Profile values at x
        """
        fit_type = fit_type.lower()
        
        if fit_type == 'gaussian':
            amp, mean, stddev = params
            return self.gaussian(x, amp, mean, stddev)
        elif fit_type == 'voigt':
            amp, center, sigma, gamma = params
            return self.voigt(x, amp, center, sigma, gamma)
        else:
            raise ValueError(f"Unknown fit type: {fit_type}")
    
    def _calculate_equivalent_width_monte_carlo(self, fit_dict, continuum_fit_dict, fit_type='gaussian', n_samples=1000):
        """Calculate equivalent width using Monte Carlo error propagation
        
        This method samples from the joint posterior distribution of:
        - Gaussian/Voigt profile parameters and their covariance
        - Continuum polynomial and its covariance
        
        Then calculates EW, sigma, and 3-sigma credible intervals from the resulting distribution.
        
        Args:
            fit_dict: Dictionary with fitted profile parameters and covariance
            continuum_fit_dict: Dictionary with continuum polynomial coefficients and covariance
            fit_type: 'gaussian', 'voigt', or profile function identifier
            n_samples: Number of Monte Carlo samples (default 1000)
            
        Returns:
            Dictionary with:
            - 'ew': median EW
            - 'ew_1sigma_lower', 'ew_1sigma_upper': 1-sigma credible interval bounds
            - 'ew_2sigma_lower', 'ew_2sigma_upper': 2-sigma credible interval bounds
            - 'ew_3sigma_lower', 'ew_3sigma_upper': 3-sigma credible interval bounds
            - 'ew_samples': full array of samples (optional, for diagnostics)
        """
        try:
            # Require continuum fit to proceed
            if continuum_fit_dict is None or 'coeffs' not in continuum_fit_dict:
                print("[MC] ERROR: Continuum fit required for MC EW calculation")
                return None
            
            bounds = fit_dict.get('bounds')
            if bounds[0] is None or bounds[1] is None:
                return None
            
            # Get covariance matrices
            profile_cov = fit_dict.get('covariance')
            if profile_cov is None:
                return None
            if isinstance(profile_cov, list):
                profile_cov = np.array(profile_cov)
            
            cont_cov = continuum_fit_dict.get('covariance') if continuum_fit_dict else None
            
            # Get continuum polynomial if available
            cont_coeffs = None
            if continuum_fit_dict and 'coeffs' in continuum_fit_dict:
                cont_coeffs = np.array(continuum_fit_dict['coeffs'])
                if isinstance(cont_cov, list):
                    cont_cov = np.array(cont_cov)
                
                # Validate we have proper covariance matrix
                if cont_cov is None:
                    print("[MC] ERROR: Continuum covariance matrix is missing")
                    return None
            
            # Integration grid
            x_int = np.linspace(bounds[0], bounds[1], 200)
            
            ew_samples = []
            profile_samples = []  # Store all realized profiles
            continuum_samples = []  # Store all realized continua
            
            # Monte Carlo loop
            for sample_idx in range(n_samples):
                # Extract profile parameters based on fit_type
                profile_params, param_names = self._get_profile_params_from_dict(fit_dict, fit_type)
                if profile_params is None:
                    return None
                
                # Sample profile parameters from multivariate normal
                profile_sample = np.random.multivariate_normal(profile_params, profile_cov)
                
                # Evaluate profile at high resolution
                profile = self._evaluate_profile(x_int, fit_type, profile_sample)
                
                # Sample continuum polynomial if available
                if cont_coeffs is not None and cont_cov is not None:
                    cont_sample = np.random.multivariate_normal(cont_coeffs, cont_cov)
                    continuum = np.polyval(cont_sample, x_int)
                else:
                    # Cannot proceed without continuum if one was used in the fit
                    print("[MC] ERROR: Cannot compute MC EW without continuum covariance matrix")
                    return None
                
                # Ensure non-zero continuum to avoid division issues
                continuum = np.maximum(continuum, 1e-10)
                
                # Store samples for later plotting
                profile_samples.append(profile)
                continuum_samples.append(continuum)
                
                # Calculate EW for this sample
                # Note: profile is the residual (flux - continuum) from the fit
                # EW = ∫(continuum - flux)/continuum dλ = -∫residual/continuum dλ
                normalized = -profile / continuum
                ew = np.trapz(normalized, x_int)
                
                ew_samples.append(ew)
            
            ew_samples = np.array(ew_samples)
            profile_samples_array = np.array(profile_samples)
            continuum_samples_array = np.array(continuum_samples)
            
            # Filter out NaN and inf values (numerical artifacts only)
            valid_mask = np.isfinite(ew_samples)
            valid_ew_samples = ew_samples[valid_mask]
            valid_profile_samples = profile_samples_array[valid_mask]
            valid_continuum_samples = continuum_samples_array[valid_mask]
            
            n_invalid = np.sum(~valid_mask)
            if n_invalid > 0:
                print(f"[MC] WARNING: {n_invalid} samples produced NaN/inf (numerical artifacts)")
                if n_invalid > len(ew_samples) * 0.1:
                    print("[MC]   This may indicate issues with the fit covariance matrix")
                    print("[MC]   Consider: checking continuum fit quality, adjusting polynomial order")
            
            if len(valid_ew_samples) == 0:
                print("[MC] ERROR: All samples are NaN/inf - cannot compute EW statistics")
                return None
            
            # Calculate statistics from the distribution
            median_ew = np.percentile(valid_ew_samples, 50)
            mean_ew = np.mean(valid_ew_samples)
            
            # Calculate "best" EW from the fitted parameters (no MC)
            profile_params, _ = self._get_profile_params_from_dict(fit_dict, fit_type)
            best_profile = self._evaluate_profile(x_int, fit_type, profile_params)
            if cont_coeffs is not None:
                best_continuum = np.polyval(cont_coeffs, x_int)
            else:
                continuum_level = self._get_continuum_level_estimate(bounds)
                if continuum_level is None or continuum_level <= 0:
                    continuum_level = 1.0
                best_continuum = np.ones_like(x_int) * continuum_level
            best_continuum = np.maximum(best_continuum, 1e-10)
            best_normalized = -best_profile / best_continuum
            best_ew = np.trapz(best_normalized, x_int)
            # Note: profile and best_profile are residuals (flux - continuum) from the fit
            
            # 1-sigma (16th-84th percentile)
            p16_1 = np.percentile(valid_ew_samples, 16)
            p84_1 = np.percentile(valid_ew_samples, 84)
            
            # 2-sigma (2.28th-97.72th percentile)
            p228_2 = np.percentile(valid_ew_samples, 2.28)
            p9772_2 = np.percentile(valid_ew_samples, 97.72)
            
            # 3-sigma (0.135th-99.865th percentile)
            p0135_3 = np.percentile(valid_ew_samples, 0.135)
            p99865_3 = np.percentile(valid_ew_samples, 99.865)
            
            return {
                'ew_best': best_ew,
                'ew_median': median_ew,
                'ew_mean': mean_ew,
                'ew': median_ew,  # Keep for backward compatibility
                'ew_1sigma_lower': median_ew - p16_1,
                'ew_1sigma_upper': p84_1 - median_ew,
                'ew_2sigma_lower': median_ew - p228_2,
                'ew_2sigma_upper': p9772_2 - median_ew,
                'ew_3sigma_lower': median_ew - p0135_3,
                'ew_3sigma_upper': p99865_3 - median_ew,
                'ew_samples': valid_ew_samples,  # Keep only valid samples
                'x_grid': x_int,  # Wavelength grid for plotting
                'profile_samples': valid_profile_samples,  # Only valid realized profiles
                'continuum_samples': valid_continuum_samples  # Only valid realized continua
            }
        except Exception as e:
            print(f"[MC-EW] Monte Carlo EW calculation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_continuum_level_estimate(self, bounds):
        """Estimate continuum level from data at boundaries"""
        left_mask = (self.x_data >= bounds[0] - (bounds[1] - bounds[0]) * 0.1) & \
                    (self.x_data <= bounds[0])
        right_mask = (self.x_data >= bounds[1]) & \
                     (self.x_data <= bounds[1] + (bounds[1] - bounds[0]) * 0.1)
        
        left_cont = np.nanmedian(self.spec[left_mask]) if np.any(left_mask) else np.nan
        right_cont = np.nanmedian(self.spec[right_mask]) if np.any(right_mask) else np.nan
        
        if not np.isnan(left_cont) and not np.isnan(right_cont):
            return (left_cont + right_cont) / 2.0
        elif not np.isnan(left_cont):
            return left_cont
        elif not np.isnan(right_cont):
            return right_cont
        return None
    
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
        
        # Handle polynomial deletion - remove from listfit components list
        if item_type == 'polynomial':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            poly_index = fit_dict.get('poly_index')
            
            # Find the listfit this polynomial belongs to and remove the component
            if listfit_bounds is not None and poly_index is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        # Remove the component from the listfit's components list
                        components = listfit.get('components', [])
                        # Remove components in reverse order by index to avoid index shifting
                        for i, comp in enumerate(components):
                            if (comp.get('type') == 'polynomial' and 
                                comp.get('index') == poly_index):
                                components.pop(i)
                                break
                        break
        
        # Handle Gaussian deletion - remove from listfit components list
        elif item_type == 'gaussian':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            gauss_index = fit_dict.get('gauss_index')
            
            if listfit_bounds is not None and gauss_index is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i, comp in enumerate(components):
                            if comp.get('type') == 'gaussian' and comp.get('index') == gauss_index:
                                components.pop(i)
                                break
                        break
        
        # Handle Voigt deletion - remove from listfit components list
        elif item_type == 'voigt':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            voigt_index = fit_dict.get('voigt_index')
            
            if listfit_bounds is not None and voigt_index is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i, comp in enumerate(components):
                            if comp.get('type') == 'voigt' and comp.get('index') == voigt_index:
                                components.pop(i)
                                break
                        break
        
        # Handle Polynomial Guess Mask deletion - remove from listfit
        elif item_type == 'polynomial_guess_mask':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            min_lambda = fit_dict.get('min_lambda')
            max_lambda = fit_dict.get('max_lambda')
            
            if listfit_bounds is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i, comp in enumerate(components):
                            if (comp.get('type') == 'polynomial_guess_mask' and 
                                comp.get('min_lambda') == min_lambda and
                                comp.get('max_lambda') == max_lambda):
                                components.pop(i)
                                break
                        break
        
        # Handle Data Mask deletion - remove from listfit
        elif item_type == 'data_mask':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            min_lambda = fit_dict.get('min_lambda')
            max_lambda = fit_dict.get('max_lambda')
            
            if listfit_bounds is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i, comp in enumerate(components):
                            if (comp.get('type') == 'data_mask' and 
                                comp.get('min_lambda') == min_lambda and
                                comp.get('max_lambda') == max_lambda):
                                components.pop(i)
                                break
                        break
        
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
        
        # Update legend - remove profile types that no longer exist
        self._update_legend_profile_types()
        self.update_legend()
        
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
        
    def on_calculate_ew_from_tracker(self, item_id):
        """Handle calculate equivalent width action from ItemTracker context menu"""
        if item_id not in self.item_id_map:
            return
        
        item_info = self.item_id_map[item_id]
        item_type = item_info.get('type')
        fit_dict = item_info.get('fit_dict')
        
        if not fit_dict:
            print("[EW] No fit dictionary found for this item")
            return
        
        # Get the corresponding fit type
        if item_type == 'gaussian':
            fit_type = 'gaussian'
        elif item_type == 'voigt':
            fit_type = 'voigt'
        else:
            print(f"[EW] Cannot calculate EW for item type: {item_type}")
            return
        
        # Get the continuum fit
        continuum_fit_dict = self.continuum_fits[-1] if self.continuum_fits else None
        
        # Calculate EW
        ew_result = self._calculate_equivalent_width_monte_carlo(fit_dict, continuum_fit_dict, fit_type)
        
        if ew_result:
            print(f"\n[EW Calculation Results for {item_type.capitalize()}]")
            print(f"  Best:    {ew_result['ew_best']:.6f}")
            print(f"  Median:  {ew_result['ew_median']:.6f}")
            print(f"  Mean:    {ew_result['ew_mean']:.6f}")
            print(f"  1-sigma: -{ew_result['ew_1sigma_lower']:.6f}/+{ew_result['ew_1sigma_upper']:.6f}")
            print(f"  2-sigma: -{ew_result['ew_2sigma_lower']:.6f}/+{ew_result['ew_2sigma_upper']:.6f}")
            print(f"  3-sigma: -{ew_result['ew_3sigma_lower']:.6f}/+{ew_result['ew_3sigma_upper']:.6f}")
            
            # Store results in fit dict for later use
            fit_dict['ew_best'] = ew_result['ew_best']
            fit_dict['ew_median'] = ew_result['ew_median']
            fit_dict['ew_mean'] = ew_result['ew_mean']
            fit_dict['ew_1sigma_lower'] = ew_result['ew_1sigma_lower']
            fit_dict['ew_1sigma_upper'] = ew_result['ew_1sigma_upper']
            fit_dict['ew_2sigma_lower'] = ew_result['ew_2sigma_lower']
            fit_dict['ew_2sigma_upper'] = ew_result['ew_2sigma_upper']
            fit_dict['ew_3sigma_lower'] = ew_result['ew_3sigma_lower']
            fit_dict['ew_3sigma_upper'] = ew_result['ew_3sigma_upper']
            
            # Save EW results to dedicated .qsap file
            self.save_ew_qsap_file(ew_result, fit_dict, fit_type)
            
            # Plot MC profiles if enabled
            if self.plot_mc_profiles_enabled:
                self.plot_mc_profiles(fit_dict, ew_result, fit_type)
        else:
            print("[EW] Failed to calculate equivalent width")
        
    def on_calculate_ew_auto_toggled(self, state):
        """Handle toggle of 'Calculate EW automatically' checkbox"""
        self.calculate_ew_enabled = (state == QtCore.Qt.Checked)
        status = "enabled" if self.calculate_ew_enabled else "disabled"
        print(f"Automatic EW calculation {status}")
        
    def on_plot_mc_profiles_toggled(self, state):
        """Handle toggle of 'Plot MC Profiles' checkbox"""
        self.plot_mc_profiles_enabled = (state == QtCore.Qt.Checked)
        status = "enabled" if self.plot_mc_profiles_enabled else "disabled"
        print(f"MC Profile plotting {status}")
    
    def save_ew_qsap_file(self, ew_result, fit_dict, fit_type):
        """Save Equivalent Width results to a dedicated .qsap file
        
        Args:
            ew_result: Result dict from _calculate_equivalent_width_monte_carlo
            fit_dict: Dictionary with fitted profile parameters
            fit_type: 'gaussian' or 'voigt'
        """
        try:
            # Build spectrum info dict
            spectrum_info = {
                'wavelength_unit': self.wavelength_unit,
                'velocity_mode': self.is_velocity_mode,
            }
            if self.x_data is not None and len(self.x_data) > 0:
                spectrum_info['wavelength_range'] = (self.x_data[0], self.x_data[-1])
            
            # Create .qsap file for EW results
            filepath, content = self.qsap_handler.create_equivalent_width_qsap(
                ew_result, fit_dict, fit_type.capitalize(), self.fits_file, spectrum_info
            )
            
            # Print the file contents to output
            print("\n" + "="*70)
            print(f"EQUIVALENT WIDTH RESULTS SAVED TO: {os.path.basename(filepath)}")
            print("="*70)
            print(content)
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"[EW] Error saving EW results to file: {e}")
            import traceback
            traceback.print_exc()
    
    def delete_all_mc_profiles(self):
        """Delete all currently displayed MC profile lines"""
        for line in self.mc_profile_lines_current:
            try:
                line.remove()
            except (ValueError, AttributeError):
                pass
        self.mc_profile_lines_current.clear()
        
        # Redraw
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
        
        print("[MC] Deleted all MC profile visualizations")
    
    def plot_mc_profiles(self, fit_dict, ew_result, fit_type, n_samples=1000):
        """Plot Monte Carlo profile realizations as flux (residual profile + continuum)
        
        Args:
            fit_dict: Dictionary with fitted profile parameters and covariance
            ew_result: Result dict from _calculate_equivalent_width_monte_carlo (contains samples)
            fit_type: 'gaussian', 'voigt', etc.
            n_samples: Number of samples to plot (default 1000, limited by available samples)
        """
        try:
            # Get pre-computed samples from ew_result
            if 'profile_samples' not in ew_result or 'continuum_samples' not in ew_result:
                print("[MC] No sample data available for plotting")
                return
            
            x_grid = ew_result.get('x_grid')
            profile_samples = ew_result.get('profile_samples')  # residual profile
            continuum_samples = ew_result.get('continuum_samples')
            
            if x_grid is None or len(profile_samples) == 0:
                print("[MC] Invalid sample data")
                return
            
            # Limit to available samples
            n_available = len(profile_samples)
            n_plot = min(n_samples, n_available)
            
            # Get profile color from item_id_map (or use default)
            profile_color = 'blue'
            for item_id, item_info in self.item_id_map.items():
                if item_info.get('fit_dict') == fit_dict:
                    profile_color = item_info.get('color', 'blue')
                    break
            
            # Get continuum color from colors config
            continuum_color_dict = self.colors.get('profiles', {}).get('continuum_line', {})
            continuum_color = continuum_color_dict.get('color', 'black')
            
            # First, plot all continuum realizations (in continuum color, underneath)
            for sample_idx in range(n_plot):
                continuum = continuum_samples[sample_idx]
                line_cont, = self.ax.plot(x_grid, continuum, color=continuum_color, alpha=0.03, 
                                         linewidth=0.5, linestyle='-', zorder=1)
                self.mc_profile_lines_current.append(line_cont)
            
            # Then, plot all flux realizations on top (flux = residual + continuum)
            for sample_idx in range(n_plot):
                residual = profile_samples[sample_idx]
                continuum = continuum_samples[sample_idx]
                
                # Compute flux for this realization: flux = continuum + residual
                flux = continuum + residual
                
                # Plot flux (the actual realized spectrum) with low alpha
                line_flux, = self.ax.plot(x_grid, flux, color=profile_color, alpha=0.01, 
                                         linewidth=0.5, linestyle='-', zorder=2)
                self.mc_profile_lines_current.append(line_flux)
            
            # Redraw
            if self.ax is not None:
                self.ax.figure.canvas.draw_idle()
            
            print(f"[MC] Plotted {n_plot} Monte Carlo profile realizations")
            
            
            
        except Exception as e:
            print(f"[MC] Error plotting MC profiles: {e}")
            import traceback
            traceback.print_exc()
        
    def estimate_redshift(self, selected_id, selected_wavelength):
        self.selected_id, self.selected_rest_wavelength = selected_id, selected_wavelength  # Store the selected wavelength
        if self.selected_id is not None and self.selected_rest_wavelength is not None:
            # Capture the fit_dict early so we can use it throughout this method
            selected_fit_dict = None
            
            # Extract center from selected fit if not already set
            if not hasattr(self, 'center_profile') or self.center_profile is None:
                # Try to get from selected Gaussian
                if self.selected_gaussian:
                    selected_fit_dict = self.selected_gaussian  # Capture it here
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
                    selected_fit_dict = self.selected_voigt  # Capture it here
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
            
            # Also capture if we already have center_profile set (from previous call)
            if selected_fit_dict is None:
                if self.selected_gaussian:
                    selected_fit_dict = self.selected_gaussian
                elif self.selected_voigt:
                    selected_fit_dict = self.selected_voigt
            
            # Print center info like in redshift mode
            print(f"Center of selected Gaussian: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
            
            est_redshift = (self.center_profile - self.selected_rest_wavelength) / self.selected_rest_wavelength
            est_redshift_err = self.center_profile_err / self.selected_rest_wavelength
            print(f"Estimated Redshift: {est_redshift:.6f}+-{est_redshift_err:.6f}")
            
            # Calculate MC redshift estimation using the captured fit_dict
            mc_redshift_result = self._estimate_redshift_monte_carlo(
                selected_fit_dict,
                self.selected_rest_wavelength
            )
            
            # Debug: Print MC result
            print(f"[DEBUG] MC Redshift Result: {mc_redshift_result}")
            if mc_redshift_result:
                print(f"[DEBUG] MC z_best: {mc_redshift_result.get('z_best')}")
                print(f"[DEBUG] MC z_median: {mc_redshift_result.get('z_median')}")
            
            # Prepare redshift data for .qsap file
            redshift_data = {
                'REDSHIFT': self._format_param_value(est_redshift, est_redshift_err),
                'LINE_ID': self.selected_id or 'Unknown',
                'LINE_WAVELENGTH_REST': self.selected_rest_wavelength,
                'LINE_WAVELENGTH_OBSERVED': self.center_profile,
                'LINE_WAVELENGTH_OBSERVED_ERR': self.center_profile_err,
            }
            
            # Add MC results if available
            if mc_redshift_result:
                redshift_data['REDSHIFT_BEST'] = mc_redshift_result['z_best']
                redshift_data['REDSHIFT_MEDIAN'] = mc_redshift_result['z_median']
                redshift_data['REDSHIFT_MEAN'] = mc_redshift_result['z_mean']
                redshift_data['REDSHIFT_1SIGMA'] = f"-{mc_redshift_result['z_1sigma_lower']:.6f},+{mc_redshift_result['z_1sigma_upper']:.6f}"
                redshift_data['REDSHIFT_2SIGMA'] = f"-{mc_redshift_result['z_2sigma_lower']:.6f},+{mc_redshift_result['z_2sigma_upper']:.6f}"
                redshift_data['REDSHIFT_3SIGMA'] = f"-{mc_redshift_result['z_3sigma_lower']:.6f},+{mc_redshift_result['z_3sigma_upper']:.6f}"
                print("[DEBUG] MC redshift parameters added to redshift_data")
            else:
                print("[DEBUG] MC redshift result is None - MC parameters NOT added")
            
            # Get the parent gaussian/voigt fit ID if available
            parent_fit_id = None
            if selected_fit_dict:
                parent_fit_id = selected_fit_dict.get('fit_id')
            
            # Save to .qsap file and print contents
            try:
                filepath, content = self.qsap_handler.create_redshift_qsap(
                    redshift_data, 
                    self.fits_file,
                    parent_fit_id=parent_fit_id,
                    parent_component_id=self.component_id
                )
                
                # Print the file contents to terminal
                print("\n" + "="*70)
                print(f"REDSHIFT SAVED TO: {os.path.basename(filepath)}")
                print("="*70)
                print(content)
                print("="*70 + "\n")
            except Exception as e:
                print(f"[ERROR] Failed to save redshift file: {e}")
                import traceback
                traceback.print_exc()
                return
            
            # Cleanup highlighting and reset Calculate dropdown
            self._cleanup_redshift_highlighting()
            self.reset_calculate_dropdown()
            print("Exiting redshift estimation mode.")
        else:
            print("No line selected.")
    
    def _estimate_redshift_monte_carlo(self, fit_dict, rest_wavelength, n_samples=1000):
        """Estimate redshift using Monte Carlo error propagation
        
        Args:
            fit_dict: Dictionary with fitted profile parameters and covariance
            rest_wavelength: Rest wavelength of the emission/absorption line
            n_samples: Number of MC samples (default 1000)
            
        Returns:
            Dictionary with redshift statistics, or None if calculation fails
        """
        try:
            print("\n" + "="*70)
            print("[MC-REDSHIFT] *** ENTERING MC REDSHIFT CALCULATION ***")
            print("="*70)
            print(f"[MC-REDSHIFT] fit_dict is None: {fit_dict is None}")
            print(f"[MC-REDSHIFT] fit_dict type: {type(fit_dict)}")
            
            if fit_dict:
                keys = list(fit_dict.keys())
                print(f"[MC-REDSHIFT] fit_dict.keys(): {keys}")
                print(f"[MC-REDSHIFT] 'covariance' in fit_dict: {'covariance' in fit_dict}")
                print(f"[MC-REDSHIFT] 'mean' in fit_dict: {'mean' in fit_dict}")
                print(f"[MC-REDSHIFT] 'center' in fit_dict: {'center' in fit_dict}")
            
            if fit_dict is None or 'covariance' not in fit_dict:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: fit_dict is None or no covariance ***")
                print("="*70 + "\n")
                return None
            
            # Get the center parameter and its index in the covariance matrix
            # For Gaussian: [amplitude, mean, stddev]
            # For Voigt: [amplitude, center, sigma, gamma]
            if 'mean' in fit_dict:  # Gaussian
                center = fit_dict.get('mean')
                center_idx = 1  # mean is index 1
                param_names = ['amplitude', 'mean', 'stddev']
                print(f"[MC-REDSHIFT] Detected Gaussian fit, center={center}")
            elif 'center' in fit_dict:  # Voigt
                center = fit_dict.get('center')
                center_idx = 1  # center is index 1
                param_names = ['amplitude', 'center', 'sigma', 'gamma']
                print(f"[MC-REDSHIFT] Detected Voigt fit, center={center}")
            else:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: No 'mean' or 'center' found ***")
                print("="*70 + "\n")
                return None
            
            if center is None:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: center is None ***")
                print("="*70 + "\n")
                return None
            
            # Get the covariance matrix
            cov = fit_dict.get('covariance')
            print(f"[MC-REDSHIFT] covariance retrieved: {cov is not None}")
            if cov is None:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: covariance is None ***")
                print("="*70 + "\n")
                return None
            if isinstance(cov, list):
                cov = np.array(cov)
            
            # We only need the uncertainty in the center parameter
            # Extract the center variance from the covariance matrix
            center_var = cov[center_idx, center_idx]
            print(f"[MC-REDSHIFT] center_var: {center_var}, center_idx: {center_idx}")
            if center_var <= 0:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: center_var <= 0 ***")
                print("="*70 + "\n")
                return None
            
            # Sample centers from the posterior distribution
            z_samples = []
            for sample_idx in range(n_samples):
                # Sample center from univariate normal
                center_sample = np.random.normal(center, np.sqrt(center_var))
                
                # Calculate redshift for this sample
                z = (center_sample - rest_wavelength) / rest_wavelength
                z_samples.append(z)
            
            z_samples = np.array(z_samples)
            print(f"[MC-REDSHIFT] Generated {len(z_samples)} samples")
            
            # Filter out NaN/inf values (numerical artifacts only)
            valid_mask = np.isfinite(z_samples)
            valid_z_samples = z_samples[valid_mask]
            print(f"[MC-REDSHIFT] Valid samples after filtering: {len(valid_z_samples)}")
            
            if len(valid_z_samples) == 0:
                print(f"[MC-REDSHIFT] *** RETURNING NONE: no valid samples ***")
                print("="*70 + "\n")
                return None
            
            # Calculate statistics
            z_best = (center - rest_wavelength) / rest_wavelength
            z_median = np.percentile(valid_z_samples, 50)
            z_mean = np.mean(valid_z_samples)
            
            # 1-sigma (16th-84th percentile)
            p16_1 = np.percentile(valid_z_samples, 16)
            p84_1 = np.percentile(valid_z_samples, 84)
            
            # 2-sigma (2.28th-97.72th percentile)
            p228_2 = np.percentile(valid_z_samples, 2.28)
            p9772_2 = np.percentile(valid_z_samples, 97.72)
            
            # 3-sigma (0.135th-99.865th percentile)
            p0135_3 = np.percentile(valid_z_samples, 0.135)
            p99865_3 = np.percentile(valid_z_samples, 99.865)
            
            result = {
                'z_best': z_best,
                'z_median': z_median,
                'z_mean': z_mean,
                'z_1sigma_lower': z_median - p16_1,
                'z_1sigma_upper': p84_1 - z_median,
                'z_2sigma_lower': z_median - p228_2,
                'z_2sigma_upper': p9772_2 - z_median,
                'z_3sigma_lower': z_median - p0135_3,
                'z_3sigma_upper': p99865_3 - z_median,
            }
            print(f"[MC-REDSHIFT] *** SUCCESS! z_best={result['z_best']:.6f} ***")
            print("="*70 + "\n")
            return result
        except Exception as e:
            print(f"[MC-REDSHIFT] *** ERROR: {e} ***")
            import traceback
            traceback.print_exc()
            print("="*70 + "\n")
            return None
    
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
        marker_cfg = self.colors['markers']['voigt'] if profile_type == 'Voigt' else self.colors['markers']['gaussian']
        marker_color = marker_cfg['color']
        marker_lw = marker_cfg.get('linewidth', 2)
        
        # Draw a vertical line as a marker at the specified position
        marker, = self.ax.plot(
            [center_or_mean, center_or_mean],
            [y_pos, y_pos + 0.05 * (y_max - y_min)],
            color=marker_color,
            lw=marker_lw
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
    
    def create_standalone_marker_from_linelist(self, x_position):
        """Create a standalone marker at x_position using line list selection"""
        available_line_lists = self.get_all_available_line_lists()
        self.marker_linelist_window = LineListWindow(available_line_lists=available_line_lists)
        self.marker_linelist_window.x_position = x_position  # Store x position
        self.marker_linelist_window.selected_line.connect(self.on_standalone_marker_line_selected)
        self.marker_linelist_window.setWindowFlags(self.marker_linelist_window.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
        self.marker_linelist_window.show()
        self.marker_linelist_window.raise_()
        self.marker_linelist_window.activateWindow()
    
    def on_standalone_marker_line_selected(self, line_id, wavelength):
        """Handle line selection for standalone marker creation"""
        if hasattr(self, 'marker_linelist_window'):
            x_position = self.marker_linelist_window.x_position
            # Create a standalone marker with a marker color (use gaussian color by default)
            marker_color = self.colors['markers']['gaussian']['color']
            # Create marker with default bounds (just at the position)
            bounds = (x_position - 1, x_position + 1)
            
            # Get current y-axis limits and calculate the y-position
            x_min, x_max = self.ax.get_xlim()
            y_min, y_max = self.ax.get_ylim()
            x_pos_add = (x_max - x_min) * 0.02
            y_pos = y_min + 0.925 * (y_max - y_min)
            
            # Draw marker at x_position
            marker, = self.ax.plot(
                [x_position, x_position],
                [y_pos, y_pos + 0.05 * (y_max - y_min)],
                color=marker_color,
                lw=2
            )
            # Attach attributes to marker
            setattr(marker, 'bounds', bounds)
            setattr(marker, 'center', x_position)
            setattr(marker, 'line_id', line_id)
            self.markers.append(marker)
            
            # Add label
            label = self.ax.text(
                x_position + x_pos_add, y_pos - 0.10 * (y_max - y_min),
                line_id,
                color=marker_color,
                verticalalignment='center',
                horizontalalignment='center',
                rotation='vertical'
            )
            setattr(label, 'bounds', bounds)
            setattr(label, 'center', x_position)
            setattr(label, 'marker', marker)
            self.labels.append(label)
            
            # Register in item tracker (do NOT connect to any fit)
            marker_id = f"marker_standalone_{len(self.markers)-1}_{line_id}"
            self.item_tracker.add_item(marker_id, 'marker', f'Marker: {line_id}', position=f'{x_position:.2f} Å', color=marker_color, line_obj=marker)
            
            # Register in item_id_map for removal
            self.item_id_map[marker_id] = {
                'type': 'marker',
                'fit_dict': None,
                'line_obj': marker,
                'name': f'Marker: {line_id}',
                'position': f'{x_position:.2f} Å',
                'color': marker_color
            }
            
            print(f"Created standalone marker: {line_id} at λ={x_position:.2f} Å")
            self.record_action('create_standalone_marker', f'Create Marker: {line_id}')
            self.ax.figure.canvas.draw_idle()
    
    def create_standalone_marker_from_text(self, x_position):
        """Create a standalone marker at x_position with custom text input"""
        # Open text input dialog
        text, ok = QtWidgets.QInputDialog.getText(
            self, 'Marker Label', 'Enter text for marker label:'
        )
        
        if ok and text:
            self.on_standalone_marker_text_entered(x_position, text)
    
    def on_standalone_marker_text_entered(self, x_position, label_text):
        """Handle marker creation with text input"""
        marker_color = self.colors['markers']['gaussian']['color']
        bounds = (x_position - 1, x_position + 1)
        
        # Get current y-axis limits and calculate the y-position
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_pos_add = (x_max - x_min) * 0.02
        y_pos = y_min + 0.925 * (y_max - y_min)
        
        # Draw marker at x_position
        marker, = self.ax.plot(
            [x_position, x_position],
            [y_pos, y_pos + 0.05 * (y_max - y_min)],
            color=marker_color,
            lw=2
        )
        # Attach attributes to marker
        setattr(marker, 'bounds', bounds)
        setattr(marker, 'center', x_position)
        setattr(marker, 'line_id', label_text)
        self.markers.append(marker)
        
        # Add label with custom text
        label = self.ax.text(
            x_position + x_pos_add, y_pos - 0.10 * (y_max - y_min),
            label_text,
            color=marker_color,
            verticalalignment='center',
            horizontalalignment='center',
            rotation='vertical'
        )
        setattr(label, 'bounds', bounds)
        setattr(label, 'center', x_position)
        setattr(label, 'marker', marker)
        self.labels.append(label)
        
        # Register in item tracker
        marker_id = f"marker_text_{len(self.markers)-1}_{label_text}"
        self.item_tracker.add_item(marker_id, 'marker', f'Marker: {label_text}', position=f'{x_position:.2f} Å', color=marker_color, line_obj=marker)
        
        # Register in item_id_map for removal
        self.item_id_map[marker_id] = {
            'type': 'marker',
            'fit_dict': None,
            'line_obj': marker,
            'name': f'Marker: {label_text}',
            'position': f'{x_position:.2f} Å',
            'color': marker_color
        }
        
        print(f"Created text marker: {label_text} at λ={x_position:.2f} Å")
        self.record_action('create_text_marker', f'Create Text Marker: {label_text}')
        self.ax.figure.canvas.draw_idle()

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
        import lmfit  # Lazy import for voigt fitting
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

    def on_canvas_click(self, event):
        """Handle canvas click events to refocus on the plot area"""
        # Set focus to the canvas when clicked to deselect text fields
        if event.inaxes or event.xdata is not None:
            self.canvas.setFocus()

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
            'listfit_polynomials': deepcopy(self.listfit_polynomials),
            'deleted_listfit_polynomials': deepcopy(self.deleted_listfit_polynomials),
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
        self.listfit_polynomials = deepcopy(state.get('listfit_polynomials', {}))
        self.redshift = state.get('redshift', 0.0)
        self.fit_id = state.get('fit_id', 0)
        self.component_id = state.get('component_id', 0)
        
        # Clear tracking sets and restore deleted polynomials
        self.deleted_listfit_polynomials = deepcopy(state.get('deleted_listfit_polynomials', set()))
        
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
                    # Recreate the axvspan patch with continuum_region color and hatching (same as original)
                    continuum_region_cfg = self.colors['profiles']['continuum_region']
                    patch = self.ax.axvspan(bounds[0], bounds[1], color=continuum_region_cfg['color'], alpha=continuum_region_cfg['alpha'], hatch=continuum_region_cfg['hatch'])
                    # Store the patch object back in the patch_info dict
                    patch_info['patch'] = patch
                    # Register the region patch with ItemTracker
                    position_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å"
                    self.register_item('continuum_region', f'Continuum Region', patch_obj=patch, 
                                     position=position_str, color=continuum_region_cfg['color'], bounds=bounds)
            
            # Re-plot all continuum fits
            for fit in self.continuum_fits:
                if 'line' in fit and fit['line'] is not None:
                    continuum_cfg = self.colors['profiles']['continuum_line']
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color=continuum_cfg['color'], linestyle=continuum_cfg['linestyle'], linewidth=1.5, label='Continuum')
                    # Re-register with tracker
                    bounds = fit.get('bounds')
                    bounds_str = f"λ: {bounds[0]:.2f}-{bounds[1]:.2f} Å" if bounds else "Continuum"
                    self.register_item('continuum', f'Continuum (order {fit.get("poly_order", 1)})', 
                                     fit_dict=fit, line_obj=fit['line'], 
                                     position=bounds_str, color=continuum_cfg['color'])
            
            # Re-plot all Gaussian fits
            for fit in self.gaussian_fits:
                if 'line' in fit and fit['line'] is not None:
                    gaussian_cfg = self.colors['profiles']['gaussian']
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color=gaussian_cfg['color'], linestyle=gaussian_cfg['linestyle'], linewidth=gaussian_cfg['linewidth'], label='Gaussian')
                    # Re-register with tracker
                    mean = fit.get('mean', 0)
                    position_str = f"λ: {mean:.2f} Å"
                    self.register_item('gaussian', 'Gaussian', fit_dict=fit, 
                                     line_obj=fit['line'], position=position_str, color=gaussian_cfg['color'])
            
            # Re-plot all Voigt fits
            for fit in self.voigt_fits:
                if 'line' in fit and fit['line'] is not None:
                    voigt_cfg = self.colors['profiles']['voigt']
                    self.ax.plot(fit['line'].get_xdata(), fit['line'].get_ydata(), 
                               color=voigt_cfg['color'], linestyle=voigt_cfg['linestyle'], linewidth=voigt_cfg['linewidth'], label='Voigt')
                    # Re-register with tracker
                    center = fit.get('center', fit.get('mean', 0))
                    position_str = f"λ: {center:.2f} Å"
                    self.register_item('voigt', 'Voigt', fit_dict=fit, 
                                     line_obj=fit['line'], position=position_str, color=voigt_cfg['color'])
            
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
        """Handle key press events - forward to canvas so matplotlib can handle them"""
        if isinstance(event, QKeyEvent):
            # Check for Ctrl+Z (undo) or Cmd+Z (undo) on macOS
            if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier or event.modifiers() & Qt.MetaModifier):
                # Check if Shift is also pressed (for Redo)
                if event.modifiers() & Qt.ShiftModifier:
                    self.on_redo()
                else:
                    self.on_undo()
                event.accept()
                return
            
            # Check for 'q' or 'Q' to quit
            if event.key() == Qt.Key_Q:
                self.quit_application()
                event.accept()
                return
            
            # For all other keys, ensure canvas has focus and forward to it
            if hasattr(self, 'canvas'):
                self.canvas.setFocus()
                # Let the canvas handle the key - matplotlib will convert it to a matplotlib event
                # and call our on_key handler through mpl_connect
                self.canvas.keyPressEvent(event)
                event.accept()
                return
        
        # Fallback to parent class
        super().keyPressEvent(event)

    def update_total_line_if_shown(self):
        """Redraw the total line if it's currently displayed. Called when fits change."""
        if self.show_total_line and (self.continuum_fits or self.voigt_fits or self.gaussian_fits or self.listfit_fits):
            # Remove existing total line
            total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total"]
            for line in total_lines:
                line.remove()
            
            # Redraw the total line
            self.draw_total_line()
            self.ax.figure.canvas.draw()
    
    def draw_total_line(self):
        """Draw the total line from all fitted profiles."""
        # Lazy imports for drawing total line
        from scipy.interpolate import interp1d
        
        # Remove any existing total lines first to avoid duplicates
        existing_total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total"]
        for line in existing_total_lines:
            line.remove()
        
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
            # Handle both loaded fits (no 'line' object) and normal fits (with 'line' object)
            if 'line' in fit and fit['line'] is not None:
                fit_x = fit['line'].get_xdata()
                fit_y = fit['line'].get_ydata()
                left_bound, right_bound = min(fit_x), max(fit_x)
            else:
                # Reconstruct from Gaussian parameters for loaded fits
                left_bound, right_bound = fit.get('bounds', (0, 0))
                if left_bound == 0 and right_bound == 0:
                    continue  # Skip if no bounds
                fit_x = np.linspace(left_bound, right_bound, 500)
                # Reconstruct Gaussian: A * exp(-(x - mean)^2 / (2 * stddev^2))
                amplitude = fit.get('amp', 1.0)
                mean = fit.get('mean', (left_bound + right_bound) / 2)
                stddev = fit.get('stddev', 1.0)
                fit_y = amplitude * np.exp(-((fit_x - mean) ** 2) / (2 * stddev ** 2))
            
            # Get continuum coefficients and interpolate to fit line's x values
            existing_continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
            if existing_continuum_vals is not None:
                # Interpolate continuum to fit line's x values
                continuum_interp = interp1d(self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)], 
                                           existing_continuum_vals, bounds_error=False, fill_value='extrapolate')
                existing_continuum = continuum_interp(fit_x)
            else:
                existing_continuum = np.zeros_like(fit_y)
            
            fit_is_velocity = fit.get('is_velocity_mode', False)
            if fit_is_velocity and self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit_is_velocity and not self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit_is_velocity and self.is_velocity_mode:
                profile_interp = interp1d(self.wav_to_vel(fit_x, fit.get('rest_wavelength'), z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif fit_is_velocity and not self.is_velocity_mode:
                profile_interp = interp1d(self.vel_to_wav(fit_x, fit.get('rest_wavelength'), z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            total_gaussian += profile_interp(x_plot)

        # Combine Voigt fits
        total_voigt = np.zeros_like(x_plot)
        for fit in self.voigt_fits:
            # Handle both loaded fits (no 'line' object) and normal fits (with 'line' object)
            if 'line' in fit and fit['line'] is not None:
                fit_x = fit['line'].get_xdata()
                fit_y = fit['line'].get_ydata()
                left_bound, right_bound = min(fit_x), max(fit_x)
            else:
                # Reconstruct from Voigt parameters for loaded fits
                left_bound, right_bound = fit.get('bounds', (0, 0))
                if left_bound == 0 and right_bound == 0:
                    continue  # Skip if no bounds
                fit_x = np.linspace(left_bound, right_bound, 500)
                # Reconstruct Voigt profile
                from scipy.special import wofz
                amplitude = fit.get('amplitude', 1.0)
                center = fit.get('center', (left_bound + right_bound) / 2)
                sigma = fit.get('sigma', 1.0)
                gamma = fit.get('gamma', 1.0)
                z = ((fit_x - center) + 1j * gamma) / (sigma * np.sqrt(2.0))
                fit_y = amplitude * wofz(z).real / (sigma * np.sqrt(2.0 * np.pi))
            
            # Get continuum coefficients and interpolate to fit line's x values
            existing_continuum_vals, a, b = self.get_existing_continuum(left_bound, right_bound)
            if existing_continuum_vals is not None:
                # Interpolate continuum to fit line's x values
                continuum_interp = interp1d(self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)], 
                                           existing_continuum_vals, bounds_error=False, fill_value='extrapolate')
                existing_continuum = continuum_interp(fit_x)
            else:
                existing_continuum = np.zeros_like(fit_y)
            
            fit_is_velocity = fit.get('is_velocity_mode', False)
            if fit_is_velocity and self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit_is_velocity and not self.is_velocity_mode:
                profile_interp = interp1d(fit_x, fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif not fit_is_velocity and self.is_velocity_mode:
                profile_interp = interp1d(self.wav_to_vel(fit_x, fit.get('rest_wavelength'), z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            elif fit_is_velocity and not self.is_velocity_mode:
                profile_interp = interp1d(self.vel_to_wav(fit_x, fit.get('rest_wavelength'), z=self.redshift), fit_y - existing_continuum, bounds_error=False, fill_value=0)
            total_voigt += profile_interp(x_plot)

        # Combine Listfit fits - only extract polynomials (gaussians/voigts already in their respective lists)
        total_listfit = np.zeros_like(x_plot)
        for fit in self.listfit_fits:
            left_bound, right_bound = fit.get('bounds', (0, 0))
            mask = (x_plot >= left_bound) & (x_plot <= right_bound)
            if mask.any():
                components = fit.get('components', [])
                result = fit.get('result')
                
                # Calculate polynomial components from the listfit only
                # (Gaussians and Voigts are already counted via gaussian_fits and voigt_fits lists)
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

        # Plot the total line using the configured color
        total_line_cfg = self.colors['profiles']['total_line']
        self.ax.plot(x_plot, total_y, label="Total", color=total_line_cfg['color'], 
                     linestyle=total_line_cfg['linestyle'], linewidth=total_line_cfg['linewidth'])
        self.update_legend()

    def update_legend(self):
        """Update the legend to show only profile lines (one per type) and special lines like Total."""
        handles, labels = self.ax.get_legend_handles_labels()
        
        if not handles:
            # No legend entries
            self.ax.legend(loc='upper right')
            return
        
        # Create a filtered list keeping only the first occurrence of each label
        seen_labels = set()
        filtered_handles_labels = []
        
        for handle, label in zip(handles, labels):
            # Keep lines with labels (profile types) only if we haven't seen them before
            # Also keep special lines like "Total", "Total Listfit", etc.
            if label and (label not in seen_labels or label.startswith('Total')):
                filtered_handles_labels.append((handle, label))
                if not label.startswith('Total'):  # Total appears multiple times, don't track it
                    seen_labels.add(label)
        
        if filtered_handles_labels:
            filtered_handles, filtered_labels = zip(*filtered_handles_labels)
        else:
            filtered_handles, filtered_labels = [], []
        
        self.ax.legend(filtered_handles, filtered_labels, loc='upper right')

    def _update_legend_profile_types(self):
        """Update the legend_profile_types set based on which profile types still have fits."""
        # Check which profile types still have instances
        has_gaussian = len(self.gaussian_fits) > 0
        has_voigt = len(self.voigt_fits) > 0
        has_continuum = len(self.continuum_fits) > 0
        
        # Update the legend_profile_types set to reflect current state
        if has_gaussian:
            self.legend_profile_types.add('gaussian')
        else:
            self.legend_profile_types.discard('gaussian')
        
        if has_voigt:
            self.legend_profile_types.add('voigt')
        else:
            self.legend_profile_types.discard('voigt')
        
        if has_continuum:
            self.legend_profile_types.add('continuum')
        else:
            self.legend_profile_types.discard('continuum')

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
            
            # Deactivate all modes with 'escape' key
            if event.key == 'escape':
                self.on_deactivate_all()
                return

        # Guard: ensure bounds are initialized before processing other key events
        if self.x_lower_bound is None or self.x_upper_bound is None or self.y_lower_bound is None or self.y_upper_bound is None:
            return  # Bounds not yet initialized, skip key processing

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

        # COMMENTED OUT: Old 'v' key EW calculation functionality
        # Replaced with dropdown menu in Options panel and spacebar/right-click selection
        # If users select a profile via spacebar or right-click context menu,
        # they can calculate EW for that specific profile.
        # See: on_calculate_ew_mode_changed() and on_calculate_ew_from_tracker()
        #
        # if event.key == 'v':  # Use 'v' key to calculate equivalent width
        #     if event.xdata is None:
        #         print("Please click inside the plot area to use this function.")
        #         return
        #     x_pos = event.xdata  # Get x position of mouse click
        #
        #     # Find the Gaussian fit corresponding to the selected x position
        #     selected_gaussian = None
        #     for fit in self.gaussian_fits:
        #         left_bound, right_bound = fit['bounds']
        #         if left_bound <= x_pos <= right_bound:
        #             selected_gaussian = fit
        #             print(f"Gaussian with parameters amp:{selected_gaussian['amp']}, mean: {selected_gaussian['mean']}, stddev: {selected_gaussian['stddev']} selected.")
        #             break  # Select only the first Gaussian fit found within the bounds
        #
        #     # OR find the Voigt fit corresponding to the selected x position
        #     selected_voigt = None
        #     for fit in self.voigt_fits:
        #         left_bound, right_bound = fit['bounds']
        #         if left_bound <= x_pos <= right_bound:
        #             selected_voigt = fit
        #             print(f"Voigt with parameters amp: {selected_voigt['amp']}, center: {selected_voigt['center']}, sigma: {selected_voigt['sigma']}, gamma: {selected_voigt['gamma']}  selected.")
        #             break  # Select only the first Gaussian fit found within the bounds
        #     
        #     if selected_gaussian or selected_voigt:
        #         if selected_gaussian:
        #             # Get Gaussian parameters for the selected fit
        #             bounds = selected_gaussian['bounds']
        #             amp = selected_gaussian['amp']
        #             mean = selected_gaussian['mean']
        #             stddev = selected_gaussian['stddev']
        #             
        #             # Define the Gaussian function based on selected parameters
        #             gaussian_function = lambda x: self.gaussian(x, amp, mean, stddev)
        #         if selected_voigt:
        #             # Get Voigt parameters for the selected fit
        #             bounds = selected_voigt['bounds']
        #             amp = selected_voigt['amp']
        #             center = selected_voigt['center']
        #             gamma = selected_voigt['gamma']
        #             sigma = selected_voigt['sigma']
        #             
        #             # Define the Voigt function based on selected parameters
        #             voigt_function = lambda x: self.voigt(x, amp, center, sigma, gamma)
        #
        #         # Retrieve the fitted continuum over the Gaussian's bounds
        #         continuum_within_bounds = self.get_existing_continuum(bounds[0], bounds[1])
        #
        #         if continuum_within_bounds is not None:
        #             # Extract continuum values and parameters from returned data
        #             _, a, b = continuum_within_bounds
        #
        #             # Calculate Equivalent Width
        #             if selected_gaussian:
        #                 ew = self.calculate_equivalent_width(gaussian_function, (a, b), bounds)
        #                 # Plot the filled area between Gaussian and continuum
        #                 x_fill = np.linspace(bounds[0], bounds[1], 100)
        #                 y_gaussian = gaussian_function(x_fill)
        #                 y_continuum = self.continuum_model(x_fill, a, b)
        #             if selected_voigt:
        #                 ew = self.calculate_equivalent_width(voigt_function, (a, b), bounds)
        #                 # Plot the filled area between Gaussian and continuum
        #                 x_fill = np.linspace(bounds[0], bounds[1], 100)
        #                 y_gaussian = voigt_function(x_fill)
        #                 y_continuum = self.continuum_model(x_fill, a, b)
        #             
        #             # Remove the previous fill region if it exists
        #             if self.ew_fill:
        #                 self.ew_fill.remove()
        #             
        #             # Create a new fill region and store its reference
        #             self.ew_fill = self.ax.fill_between(x_fill, y_gaussian + y_continuum, y_continuum, color='cyan', alpha=0.7)
        #             plt.draw()  # Redraw plot to show the filled area
        #         else:
        #             print("No continuum specified. Unable to calculate EW.")
        
        # NEW: 'v' key activates Calculate Equivalent Width selection mode
        if event.key == 'v':
            if self.calculate_ew_selection_mode:
                self.calculate_ew_selection_mode = False
                print('Exiting Calculate Equivalent Width mode.')
            else:
                self.calculate_ew_selection_mode = True
                print('Calculate Equivalent Width mode: Use spacebar to select a profile, or right-click in Item Tracker to calculate EW.')
        # If in EW selection mode and spacebar is pressed
        elif self.calculate_ew_selection_mode and event.key == ' ':
            if event.xdata is None:
                print("Please click inside the plot area to select a profile.")
                return
            x_pos = event.xdata
            # Try to find Gaussian at this position
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    # Find the corresponding item_id in item_tracker
                    for item_id, item_info in self.item_id_map.items():
                        if item_info.get('fit_dict') == fit and item_info.get('type') == 'gaussian':
                            self.on_calculate_ew_from_tracker(item_id)
                            return
            # Try to find Voigt at this position
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    # Find the corresponding item_id in item_tracker
                    for item_id, item_info in self.item_id_map.items():
                        if item_info.get('fit_dict') == fit and item_info.get('type') == 'voigt':
                            self.on_calculate_ew_from_tracker(item_id)
                            return

        if event.key == 'm':
            if self.continuum_mode:
                # Already in continuum mode, so pressing 'm' again exits it
                self.continuum_mode = False
                self.continuum_regions = []  # Clear any defined regions
                # Update dropdown to blank
                self.continuum_mode_dropdown.blockSignals(True)
                self.continuum_mode_dropdown.setCurrentIndex(0)
                self.continuum_mode_dropdown.blockSignals(False)
                print('Exiting continuum mode.')
            else:
                # Enter continuum mode
                self.continuum_mode = True
                # Update dropdown to show active mode
                self.continuum_mode_dropdown.blockSignals(True)
                self.continuum_mode_dropdown.setCurrentText("Continuum Region(s)")
                self.continuum_mode_dropdown.blockSignals(False)
                self.continuum_enter_button.setEnabled(True)
                print("Continuum fitting mode: Use the spacebar to define the left bound and then the right bound of a continuum region.")
                print("You can define multiple continuum regions by selecting multiple pairs of bounds.")
                print("When the regions are set, hit Enter or Return to fit a polynomial (the order of the polynomial is configurable in Options).")
                print(f"Current polynomial order: {self.poly_order}")

        if event.key == 'enter' and self.continuum_mode:
            # Update polynomial order from Options panel input field
            try:
                self.poly_order = int(self.options_poly_order_input.text())
                print(f"Using polynomial order: {self.poly_order}")
            except ValueError:
                print("Invalid polynomial order, using default order 1")
                self.poly_order = 1
            
            # Check if we have valid continuum regions
            if not self.continuum_regions:
                self.continuum_mode = False
                # Reset dropdown to blank
                self.continuum_mode_dropdown.blockSignals(True)
                self.continuum_mode_dropdown.setCurrentIndex(0)
                self.continuum_mode_dropdown.blockSignals(False)
                self.continuum_enter_button.setEnabled(False)
                print('Exiting continuum mode.')
                return
            
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
            try:
                region_bounds = (min(region[0] for region in self.continuum_regions), max(region[1] for region in self.continuum_regions))
            except (ValueError, TypeError) as e:
                self.continuum_mode = False
                # Reset dropdown to blank
                self.continuum_mode_dropdown.blockSignals(True)
                self.continuum_mode_dropdown.setCurrentIndex(0)
                self.continuum_mode_dropdown.blockSignals(False)
                self.continuum_enter_button.setEnabled(False)
                print('Exiting continuum mode due to error.')
                return

            # Convert to numpy arrays
            combined_wav = np.array(combined_wav)
            combined_spec = np.array(combined_spec)
            combined_err = np.array(combined_err) if combined_err else None

            # Fit the continuum using the combined data
            continuum, coeffs, perr, pcov = self.fit_continuum(combined_wav, combined_spec, combined_err, poly_order=self.poly_order)
            
            # Print fit parameters
            poly_str = f"Continuum fit (order {self.poly_order}):"
            for i, (coeff, err) in enumerate(zip(coeffs, perr)):
                poly_str += f" c{i}={coeff:.6e}±{err:.6e}"
            print(poly_str)

            # Plot the fitted continuum only between the minimum and maximum continuum region wavelengths
            # Create wavelength array between region bounds for plotting
            x_plot = np.linspace(region_bounds[0], region_bounds[1], 500)
            continuum_full = np.polyval(coeffs, x_plot)
            continuum_cfg = self.colors['profiles']['continuum_line']
            continuum_line, = self.ax.plot(x_plot, continuum_full, color=continuum_cfg['color'], linestyle=continuum_cfg['linestyle'], alpha=0.8)
            if self.is_residual_shown:
                self.calculate_and_plot_residuals()
            self.update_legend()
            # Force immediate redraw of the canvas
            self.ax.figure.canvas.draw()
            QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
            print("Fitting completed for defined regions.")
            # Add continuum fit
            continuum_fit = {
                'bounds': region_bounds,  # Tuple (left_bound, right_bound)
                'individual_regions': list(self.continuum_regions),  # Store each region separately
                'coeffs': coeffs,               # Polynomial coefficients
                'coeffs_err': perr,             # Coefficient errors
                'covariance': pcov,             # Full covariance matrix for MC sampling
                'poly_order': self.poly_order,  # Polynomial order
                'patches': self.continuum_patches,    # The patches object is for plotting
                'line': continuum_line,            
                'is_velocity_mode': self.is_velocity_mode
            }
            self.continuum_fits.append(continuum_fit)
            # Register with ItemTracker
            bounds_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
            continuum_cfg = self.colors['profiles']['continuum_line']
            self.register_item('continuum', f'Continuum (order {self.poly_order})', fit_dict=continuum_fit,
                             line_obj=continuum_line, position=bounds_str, color=continuum_cfg['color'])
            
            # Record action for undo/redo
            self.record_action('fit_continuum', f'Fit Continuum (order {self.poly_order})')
            
            # Save fit to .qsap file and print
            self.save_and_print_qsap_fit(continuum_fit, 'Continuum', 'Single')
            
            self.continuum_regions = [] # Clear continuum_regions
            self.continuum_mode = False # Exit continuum mode
            # Reset dropdown to blank
            self.continuum_mode_dropdown.blockSignals(True)
            self.continuum_mode_dropdown.setCurrentIndex(0)
            self.continuum_mode_dropdown.blockSignals(False)
            self.continuum_enter_button.setEnabled(False)
            print('Exiting continuum mode.')
        elif event.key == ' ' and self.continuum_mode:
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
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
                continuum_region_cfg = self.colors['profiles']['continuum_region']
                patch = self.ax.axvspan(self.continuum_regions[-1][0], event.xdata, color=continuum_region_cfg['color'], alpha=continuum_region_cfg['alpha'], hatch=continuum_region_cfg['hatch'])
                region_bounds = (self.continuum_regions[-1][0], event.xdata)
                self.continuum_patches.append({'patch': patch, 'bounds': region_bounds}) # Store the patch
                # Register the region patch with ItemTracker
                position_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
                self.register_item('continuum_region', f'Continuum Region', patch_obj=patch, 
                                 position=position_str, color=continuum_region_cfg['color'], bounds=region_bounds)
                # Record action for defining a continuum region
                self.record_action('define_continuum_region', f'Define Continuum Region λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å')
                # self.continuum_patches.append(patch) # Store the patch
                self.fig.canvas.draw_idle()  # Update plot with the new region
        # Remove continuum region
        if event.key == 'M':
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
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
            gaussian_cfg = self.colors['profiles']['gaussian']
            line = self.ax.axvline(event.xdata, color=gaussian_cfg['color'], linestyle='--')  # Plot bound line using displayed coords
            self.bound_lines.append(line)  # Store the line object
            print(f"Bound set at x = {event.xdata}")
            # Record action for setting a bound
            if len(self.bounds) == 1:
                self.record_action('set_gaussian_bound_1', f'Set Gaussian lower bound at λ={bound_value:.2f} Å')
            elif len(self.bounds) == 2:
                self.record_action('set_gaussian_bound_2', f'Set Gaussian upper bound at λ={bound_value:.2f} Å')
            self.fig.canvas.draw_idle()  # Update plot with the new bound line
            
            # Update button state for Multi Gaussian mode
            self.update_gaussian_enter_button()

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
                    
                    # Calculate maximum sigma to prevent unphysical broad Gaussians
                    sigma_max = self._calculate_max_sigma(left_bound, right_bound, mean_guess, epsilon=0.05)
                    
                    # Cap the initial stddev guess to stay within bounds (use 50% of max for safety)
                    initial_guess[2] = min(stddev_guess, sigma_max * 0.5)
                    
                    # curve_fit with optional sigma (errors) and constrained sigma bounds
                    # Lazy import of scipy for curve fitting
                    from scipy.optimize import curve_fit
                    if comp_err is not None:
                        params, pcov = curve_fit(self.gaussian, comp_x, continuum_subtracted_y, sigma=comp_err, p0=initial_guess, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, sigma_max]))
                    else:
                        params, pcov = curve_fit(self.gaussian, comp_x, continuum_subtracted_y, p0=initial_guess, bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, sigma_max]))
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
                    
                    # Lazy import of scipy for interpolation
                    from scipy.interpolate import interp1d
                    interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                    y_plt = interpolator(x_plt)
                    gaussian_cfg = self.colors['profiles']['gaussian']
                    fit_line, = self.ax.plot(x_plt, y_plt, color=gaussian_cfg['color'], linestyle=gaussian_cfg['linestyle'])
                    # Store each component’s parameters
                    self.gaussian_fits.append({
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': chi2,
                    'chi2_nu': chi2_nu,
                    'has_errors': comp_err is not None,
                    'component_id': self.component_id,
                    'amp': amp, 'amp_err': amp_err, 'mean': mean, 'mean_err': mean_err, 'stddev': stddev, 'stddev_err': stddev_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength  if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'covariance': pcov.tolist()  # Store covariance matrix as list
                    })
                    # Register with ItemTracker
                    position_str = f"λ: {mean:.2f} Å"
                    gaussian_cfg = self.colors['profiles']['gaussian']
                    self.register_item('gaussian', f'Gaussian', fit_dict=self.gaussian_fits[-1], line_obj=fit_line,
                                     position=position_str, color=gaussian_cfg['color'])
                    
                    # Record action for undo/redo
                    self.record_action('fit_gaussian', f'Fit Gaussian at λ={mean:.2f} Å')
                    
                    # Save fit to .qsap file and print
                    self.save_and_print_qsap_fit(self.gaussian_fits[-1], 'Gaussian', 'Single')

                    self.component_id += 1
                    # Force immediate redraw of the canvas
                    self.ax.figure.canvas.draw()
                    QtWidgets.QApplication.processEvents()  # Process Qt events to ensure redraw
                    # Update residual display if shown
                    if self.is_residual_shown:
                        self.calculate_and_plot_residuals()

                # Remove bound lines after fit
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()  # Clear the list of bound lines
                self.bounds = []
                self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
                # Deactivate Gaussian mode and reset dropdown
                self.gaussian_mode = False
                self.gaussian_mode_dropdown.blockSignals(True)
                self.gaussian_mode_dropdown.setCurrentIndex(0)
                self.gaussian_mode_dropdown.blockSignals(False)
                self.gaussian_enter_button.setEnabled(False)
                print('Exiting Gaussian mode.')
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
            sigma_maxes = []  # Store sigma_max for each component
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
                mean_guess = np.mean(comp_x)
                # Calculate max sigma for this component FIRST
                sigma_max = self._calculate_max_sigma(left_bound, right_bound, mean_guess, epsilon=0.05)
                sigma_maxes.append(sigma_max)
                # Cap initial sigma guess to stay within bounds (use 50% of max for safety)
                sigma_guess = min(np.std(comp_x), sigma_max * 0.5)
                initial_guesses.extend([max(continuum_subtracted_y) - min(continuum_subtracted_y), mean_guess, sigma_guess]) # NOT SHARED SIGMA

            # Fit multiple Gaussians
            if len(comp_xs) > 0:
                comp_xs = np.array(comp_xs)
                continuum_subtracted_ys = np.array(continuum_subtracted_ys)
                # Use sigma if errors available, otherwise None
                sigma_param = np.array(comp_errs) if comp_errs else None
                
                # Build bounds with sigma constraints for each component
                num_components = len(bound_pairs)
                lower_bounds = [-np.inf] * (num_components * 3)
                upper_bounds = [np.inf] * (num_components * 3)
                for i, sigma_max in enumerate(sigma_maxes):
                    lower_bounds[i * 3 + 2] = 0  # sigma >= 0
                    upper_bounds[i * 3 + 2] = sigma_max  # sigma <= sigma_max
                
                # Lazy import of scipy for curve fitting
                from scipy.optimize import curve_fit
                from scipy.interpolate import interp1d
                params, pcov = curve_fit(self.multi_gaussian, comp_xs, continuum_subtracted_ys, sigma=sigma_param, p0=initial_guesses, bounds=(lower_bounds, upper_bounds)) # NOT SHARED SIGMA
                perr = np.sqrt(np.diag(pcov))
                
                # Track component boundaries in concatenated arrays for later chi2 calculation
                component_boundaries = []
                cumulative_idx = 0
                for comp_x in [self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)] for left_bound, right_bound in bound_pairs]:
                    component_boundaries.append((cumulative_idx, cumulative_idx + len(comp_x)))
                    cumulative_idx += len(comp_x)
                for i in range(0, len(params), 3):
                    amp, mean, stddev = params[i:i+3]
                    amp_err, mean_err, stddev_err = perr[i:i+3]
                    x_fit = self.x_data[(self.x_data >= bound_pairs[i // 3][0]) & (self.x_data <= bound_pairs[i // 3][1])]
                    y_fit = self.gaussian(x_fit, amp, mean, stddev) + continuum_ys[i // 3]
                    continuum_sub_data = comp_ys[i // 3] - continuum_ys[i // 3]
                    residuals = continuum_sub_data - self.gaussian(x_fit, amp, mean, stddev)
                    
                    # Extract this component's covariance from the full covariance matrix
                    comp_cov_indices = [i, i+1, i+2]
                    comp_cov = pcov[np.ix_(comp_cov_indices, comp_cov_indices)]
                    
                    # Calculate chi2 or SSR depending on whether errors are available
                    if sigma_param is not None and i // 3 < len(component_boundaries):
                        start_idx, end_idx = component_boundaries[i // 3]
                        component_errors = sigma_param[start_idx:end_idx]
                        chi2 = np.sum((residuals / component_errors) ** 2)  # Proper chi-squared
                    else:
                        chi2 = np.sum(residuals ** 2)  # SSR without errors
                    chi2_nu = chi2 / (len(x_fit) - 3)  # 3 params per component
                    interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                    x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                    y_plt = interpolator(x_plt)
                    gaussian_cfg = self.colors['profiles']['gaussian']
                    fit_line, = self.ax.plot(x_plt, y_plt, color=gaussian_cfg['color'], linestyle=gaussian_cfg['linestyle'])
                    left_bound, right_bound = bound_pairs[i // 3]
                    gaussian_fit = {
                    'fit_id': self.fit_id,
                    'is_velocity_mode': self.is_velocity_mode,
                    'chi2': chi2,
                    'chi2_nu': chi2_nu,
                    'has_errors': sigma_param is not None,  # Track if errors were available
                    'component_id': self.component_id,
                    'amp': amp, 'amp_err': amp_err, 'mean': mean, 'mean_err': mean_err, 'stddev': stddev, 'stddev_err': stddev_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': line_id if line_id else None,
                    'line_wavelength': line_wavelength  if line_wavelength else None,
                    'line': fit_line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'covariance': comp_cov.tolist()  # Store covariance matrix as list
                    }
                    self.gaussian_fits.append(gaussian_fit)
                    # Register with ItemTracker
                    position_str = f"λ: {mean:.2f} Å"
                    gaussian_cfg = self.colors['profiles']['gaussian']
                    self.register_item('gaussian', f'Gaussian', fit_dict=gaussian_fit, line_obj=fit_line,
                                     position=position_str, color=gaussian_cfg['color'])
                    
                    # Record action for undo/redo (only record once after all components)
                    if i == len(params) - 3:  # Last component
                        self.record_action('fit_multi_gaussian', f'Fit {len(bound_pairs)} Gaussians')
                    
                    # Component details now printed in .qsap file output below
                    # print(f"  Fit ID: {self.fit_id}")
                    # print(f"  Component ID: {self.component_id}")
                    # print(f"  Velocity mode: {self.is_velocity_mode}")
                    # print(f"  Line ID: {line_id}")
                    # print(f"  Line Wavelength: {line_wavelength}")
                    # print(f"  Amplitude: {amp}+-{amp_err}")
                    # print(f"  Mean: {mean}+-{mean_err}")
                    # print(f"  Std_dev: {stddev}+-{stddev_err}")
                    # print(f"  Bounds: ({left_bound}, {right_bound})")
                    # print(f"  Chi-squared: {chi2}")
                    # print(f"  Chi-squared_nu: {chi2_nu}")
                    # print(f"  Line Object: {fit_line}\n")

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

                # Save multi-gaussian fit to .qsap file
                multi_gaussian_components = [g for g in self.gaussian_fits if g.get('fit_id') == self.fit_id - 1]
                if multi_gaussian_components:
                    self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian')

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
                print("When a pair of bounds are set, a single Voigt profile will be fitted automatically within those bounds.")
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
            voigt_cfg = self.colors['profiles']['voigt']
            line = self.ax.axvline(event.xdata, color=voigt_cfg['color'], linestyle='--')  # Plot bound line for Voigt
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
                # Calculate max sigma to prevent unphysical broad profiles
                center_guess = np.mean(comp_x)
                sigma_max = self._calculate_max_sigma(left_bound, right_bound, center_guess, epsilon=0.05)
                
                initial_params = Parameters()
                if np.mean(continuum_subtracted_y) > 0:
                    initial_params.add('amp', value=max(continuum_subtracted_y) - min(continuum_subtracted_y))
                else:
                    initial_params.add('amp', value=min(continuum_subtracted_y) - max(continuum_subtracted_y))
                initial_params.add('center', value=center_guess)
                initial_params.add('sigma', value=np.std(comp_x)/10, min=0, max=sigma_max)  # Constrain sigma with upper bound
                initial_params.add('gamma', value=np.std(comp_x)/10, min=0, max=sigma_max)  # Constrain gamma similarly

                # Lazy imports for lmfit model fitting
                from lmfit import Model, Parameters
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
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                voigt_cfg = self.colors['profiles']['voigt']
                fit_line, = self.ax.plot(x_plt, y_plt, color=voigt_cfg['color'], linestyle=voigt_cfg['linestyle'])
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
                    'has_errors': comp_err is not None,  # Track if errors were available
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
                
                # Save fit to .qsap file and print
                self.save_and_print_qsap_fit(fit_results, 'Voigt', 'Single')
                
                # Reset Advanced dropdown if Bayes mode was active
                if self.bayes_mode:
                    self.reset_advanced_dropdown()
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
                self.update_legend()
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
            print("When a pair of bounds are set, a single Voigt profile will be fitted automatically within those bounds.")
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
            voigt_cfg = self.colors['profiles']['voigt']
            line = self.ax.axvline(event.xdata, color=voigt_cfg['color'], linestyle='--')  # Plot bound line for Voigt
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
            # Lazy imports for lmfit model fitting
            from lmfit import Model, Parameters
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
                    overall_continuum, continuum_params, _, _ = self.fit_continuum(self.x_data, self.spec, self.err)
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
                    initial_params.add(f'{prefix}sigma', value=np.std(comp_x) / 10, min=0.0, max=self._calculate_max_sigma(left_bound, right_bound, np.mean(comp_x), epsilon=0.05))
                    initial_params.add(f'{prefix}gamma', value=np.std(comp_x) / 10, min=0.0, max=self._calculate_max_sigma(left_bound, right_bound, np.mean(comp_x), epsilon=0.05))
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
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                # Higher resolution for smooth plotting
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                voigt_cfg = self.colors['profiles']['voigt']
                fit_line, = self.ax.plot(x_plt, y_plt, color=voigt_cfg['color'], linestyle=voigt_cfg['linestyle'])
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
            self.update_legend()
            plt.draw()  # Update the plot with the fitted profiles

            # Clear multi-Voigt mode settings
            self.voigt_mode = False
            self.multi_voigt_mode = False

        # Enter multi Gaussian fit mode
        if event.key == 'D':
            if self.multi_gaussian_mode or self.multi_gaussian_mode_old:
                self.multi_gaussian_mode = False
                self.multi_gaussian_mode_old = False
                # Update dropdown to blank
                self.gaussian_mode_dropdown.blockSignals(True)
                self.gaussian_mode_dropdown.setCurrentIndex(0)
                self.gaussian_mode_dropdown.blockSignals(False)
            else:
                self.multi_gaussian_mode_old = True
                self.multi_gaussian_mode = False
                self.gaussian_mode = False
                # Update dropdown to show active mode
                self.gaussian_mode_dropdown.blockSignals(True)
                self.gaussian_mode_dropdown.setCurrentText("Multi Gaussian")
                self.gaussian_mode_dropdown.blockSignals(False)
            print("self.bounds:", self.bounds)
            self.bounds = []  # Reset bounds
            print("self.bounds:", self.bounds)
            if self.bound_lines is not None:
                for line in self.bound_lines:  # Remove any existing bound lines
                    line.remove()
            self.bound_lines.clear()  # Clear the list of bound lines
            print("Multi Gaussian fit mode: Press space to set left and right bounds.")
            print("When a pair of bounds are set, a single Gaussian will be fitted automatically within those bounds.")
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
            gaussian_cfg = self.colors['profiles']['gaussian']
            line = self.ax.axvline(event.xdata, color=gaussian_cfg['color'], linestyle='--')  # Plot bound line for Gaussian
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
            # Lazy imports for lmfit model fitting
            from lmfit import Model, Parameters
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
                    overall_continuum, continuum_params, _, _ = self.fit_continuum(self.x_data, self.spec, self.err)
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
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit, kind='cubic', bounds_error=False, fill_value='extrapolate')
                # Higher resolution for smooth plotting
                x_plt = np.linspace(x_fit.min(), x_fit.max(), 10 * len(x_fit))
                y_plt = interpolator(x_plt)
                gaussian_cfg = self.colors['profiles']['gaussian']
                fit_line, = self.ax.plot(x_plt, y_plt, color=gaussian_cfg['color'], linestyle=gaussian_cfg['linestyle'])
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
            self.update_legend()
            plt.draw()  # Update the plot with the fitted profiles

            # Clear multi-Gaussian mode settings
            self.gaussian_mode = False
            self.multi_gaussian_mode = False
            # Reset dropdown to blank
            self.gaussian_mode_dropdown.blockSignals(True)
            self.gaussian_mode_dropdown.setCurrentIndex(0)
            self.gaussian_mode_dropdown.blockSignals(False)


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
                                self.step_spec, = self.ax.step(self.x_data, self.smoothed_spec, color='black', where='mid', zorder=0)
                                self.line_spec, = self.ax.plot(self.x_data, self.smoothed_spec, color='black', visible=False, zorder=0)
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
                            self.step_spec, = self.ax.step(self.x_data, self.original_spec, color='black', where='mid', zorder=0)
                            self.line_spec, = self.ax.plot(self.x_data, self.original_spec, color='black', visible=False, zorder=0)
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
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
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
            # Toggle the total line for ALL fitted profiles
            self.toggle_total_line()

        # Enter Gaussian fit mode
        elif event.key == 'd':
            if self.gaussian_mode or self.multi_gaussian_mode_old:
                self.gaussian_mode = False
                self.multi_gaussian_mode_old = False
                # Update dropdown to blank
                self.gaussian_mode_dropdown.blockSignals(True)
                self.gaussian_mode_dropdown.setCurrentIndex(0)
                self.gaussian_mode_dropdown.blockSignals(False)
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
                # Update dropdown to show active mode
                self.gaussian_mode_dropdown.blockSignals(True)
                self.gaussian_mode_dropdown.setCurrentText("Single Gaussian     [g]")
                self.gaussian_mode_dropdown.blockSignals(False)
                print("Gaussian fit mode: Press space to set left and right bounds.")
                print("When a pair of bounds are set, a single Gaussian will be fitted automatically within those bounds.")
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
                print("Multi-Gaussian fit mode: Press space to set multiple bounds, enter to fit multiple Gaussians simultaneously.")
                print("The bounds are set in pairs: (left, right) for each profile. A minimum of two pairs of bounds are required to fit.")
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
            self._cleanup_redshift_highlighting()
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
            # Try to find Gaussian at this position
            for fit in self.gaussian_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.selected_gaussian = fit  # Capture the fit dict
                    self.selected_voigt = None
                    self.plot_redshift_gaussian(fit)
                    self.center_profile, self.center_profile_err = fit['mean'], fit['mean_err']
                    print(f"Center of selected Gaussian: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
                    self.open_linelist_window()
                    return
            # Try to find Voigt at this position
            for fit in self.voigt_fits:
                left_bound, right_bound = fit['bounds']
                if left_bound <= x_pos <= right_bound:
                    self.selected_voigt = fit  # Capture the fit dict
                    self.selected_gaussian = None
                    self.plot_redshift_voigt(fit)
                    self.center_profile, self.center_profile_err = fit['center'], fit['center_err']
                    if self.center_profile_err is None:
                        raise ValueError(f"Error associated with the center of the Voigt profile is missing for x_pos = {self.center_profile:.6f}.")
                    print(f"Center of selected Voigt: {self.center_profile:.6f}+-{self.center_profile_err:.6f}")
                    self.open_linelist_window()
                    return
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
                
                # Reset Calculate dropdown when exiting velocity mode
                self.reset_calculate_dropdown()

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



        # Save a pdf of the current plot
        if event.key == '`':
            self.save_plot_as_pdf()

        if event.key == ',':  # Use ',' key to assign line ID and wavelength
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
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
                plt.draw()
            
        elif event.key == '<':
            # Find the marker and label nearest to the cursor to remove
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
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
        
        elif event.key == '.':
            # Create a standalone marker from line list at cursor position
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
            x_pos = event.xdata
            if x_pos is not None:
                self.create_standalone_marker_from_linelist(x_pos)
        
        elif event.key == '>':
            # Create a standalone marker from custom text input at cursor position
            if event.xdata is None:
                print("Please click inside the plot area to use this function.")
                return
            x_pos = event.xdata
            if x_pos is not None:
                self.create_standalone_marker_from_text(x_pos)

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
        try:
            composite_model = self.build_composite_model(components, x_fit_masked, y_fit_masked, err_fit_masked)
        except Exception as e:
            print(f"Error building composite model: {e}")
            import traceback
            traceback.print_exc()
            return
        
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
        self._check_listfit_quality(result, y_fit, err_fit)
        
        # Plot the components (pass all components so masks can be visualized)
        print("[DEBUG] About to plot listfit components...")
        try:
            self.plot_listfit_components(result, components, x_fit, y_fit, err_fit, left_bound, right_bound)
            print("[DEBUG] Successfully plotted listfit components")
        except Exception as e:
            print(f"[DEBUG] Error plotting listfit components: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Update residual display if shown
        if self.is_residual_shown:
            print("[DEBUG] Updating residual display...")
            try:
                self.calculate_and_plot_residuals()
                print("[DEBUG] Successfully updated residual display")
            except Exception as e:
                print(f"[DEBUG] Error updating residual display: {e}")
                import traceback
                traceback.print_exc()
        
        # Extract initial guesses and constraints for storage
        print("[DEBUG] Extracting fit information...")
        try:
            initial_guesses = self._extract_listfit_initial_guesses(result, components)
            constraints_info = self._extract_listfit_constraints(components)
            print("[DEBUG] Successfully extracted fit information")
        except Exception as e:
            print(f"[DEBUG] Error extracting fit information: {e}")
            import traceback
            traceback.print_exc()
            initial_guesses = {}
            constraints_info = {}
        
        # Store the fit
        print("[DEBUG] Storing fit results...")
        try:
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
            print("[DEBUG] Successfully stored fit results")
        except Exception as e:
            print(f"[DEBUG] Error storing fit results: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Record action for undo/redo
        self.record_action('perform_listfit', f'Perform Listfit ({len(self.listfit_fits)} total fits)')
        
        # Extract fit data for .qsap file (store components separately)
        listfit_fit_data = []
        
        # Track component counters to match what build_composite_model uses
        gauss_count = 0
        voigt_count = 0
        poly_count = 0
        
        for comp in components:
            comp_dict = {'type': comp['type']}
            
            if comp['type'] == 'gaussian':
                # Use counter-based naming to match build_composite_model
                prefix = f'g{gauss_count}_'
                amp_name = f'{prefix}amp'
                mean_name = f'{prefix}mean'
                sigma_name = f'{prefix}stddev'
                
                if all(name in result.params for name in [amp_name, mean_name, sigma_name]):
                    # Store initial guess values
                    comp_dict['amp_initial'] = result.params[amp_name].init_value
                    comp_dict['mean_initial'] = result.params[mean_name].init_value
                    comp_dict['stddev_initial'] = result.params[sigma_name].init_value
                    
                    # Store best fit parameters with errors
                    comp_dict['amp'] = result.params[amp_name].value
                    comp_dict['amp_err'] = result.params[amp_name].stderr
                    comp_dict['mean'] = result.params[mean_name].value
                    comp_dict['mean_err'] = result.params[mean_name].stderr
                    comp_dict['stddev'] = result.params[sigma_name].value
                    comp_dict['stddev_err'] = result.params[sigma_name].stderr
                    comp_dict['bounds'] = (left_bound, right_bound)
                    comp_dict['is_velocity_mode'] = self.is_velocity_mode
                
                gauss_count += 1
            
            elif comp['type'] == 'voigt':
                # Use counter-based naming to match build_composite_model
                prefix = f'v{voigt_count}_'
                amp_name = f'{prefix}amp'
                mean_name = f'{prefix}center'
                sigma_name = f'{prefix}sigma'
                gamma_name = f'{prefix}gamma'
                
                if all(name in result.params for name in [amp_name, mean_name, sigma_name, gamma_name]):
                    # Store initial guess values
                    comp_dict['amplitude_initial'] = result.params[amp_name].init_value
                    comp_dict['mean_initial'] = result.params[mean_name].init_value
                    comp_dict['sigma_initial'] = result.params[sigma_name].init_value
                    comp_dict['gamma_initial'] = result.params[gamma_name].init_value
                    
                    # Store best fit parameters with errors
                    comp_dict['amplitude'] = result.params[amp_name].value
                    comp_dict['amplitude_err'] = result.params[amp_name].stderr
                    comp_dict['mean'] = result.params[mean_name].value
                    comp_dict['mean_err'] = result.params[mean_name].stderr
                    comp_dict['sigma'] = result.params[sigma_name].value
                    comp_dict['sigma_err'] = result.params[sigma_name].stderr
                    comp_dict['gamma'] = result.params[gamma_name].value
                    comp_dict['gamma_err'] = result.params[gamma_name].stderr
                    comp_dict['bounds'] = (left_bound, right_bound)
                    comp_dict['is_velocity_mode'] = self.is_velocity_mode
                
                voigt_count += 1
            
            elif comp['type'] == 'polynomial':
                # Use counter-based naming to match build_composite_model
                prefix = f'p{poly_count}_'
                coeffs_initial = []
                coeffs = []
                coeffs_err = []
                order = comp.get('order', 1)
                for i in range(order + 1):
                    coeff_name = f'{prefix}c{i}'
                    if coeff_name in result.params:
                        coeffs_initial.append(result.params[coeff_name].init_value)
                        coeffs.append(result.params[coeff_name].value)
                        coeffs_err.append(result.params[coeff_name].stderr)
                
                if coeffs:
                    comp_dict['poly_order'] = order
                    comp_dict['coeffs_initial'] = coeffs_initial
                    comp_dict['coeffs'] = coeffs
                    comp_dict['coeffs_err'] = coeffs_err
                    comp_dict['bounds'] = (left_bound, right_bound)
                
                poly_count += 1
            
            elif comp['type'] == 'polynomial_guess_mask':
                # Store mask parameters as-is (no fit results needed)
                comp_dict['min_lambda'] = comp.get('min_lambda')
                comp_dict['max_lambda'] = comp.get('max_lambda')
            
            elif comp['type'] == 'data_mask':
                # Store mask parameters as-is (no fit results needed)
                comp_dict['min_lambda'] = comp.get('min_lambda')
                comp_dict['max_lambda'] = comp.get('max_lambda')
            
            # Only add non-empty component dictionaries
            if len(comp_dict) > 1:  # More than just 'type'
                listfit_fit_data.append(comp_dict)
        
        # Add fit diagnostics as a special metadata entry
        diagnostics_dict = {
            'type': 'fit_diagnostics',
            'ssr': float(np.sum(result.residual ** 2)) if result.residual is not None else None,
            'ssr_nu': result.redchi if result.redchi is not None else None,
            'chi2': float(result.chisqr) if hasattr(result, 'chisqr') and result.chisqr is not None else None,
            'chi2_reduced': result.redchi if result.redchi is not None else None,
            'akaike_info_criterion': result.aic if hasattr(result, 'aic') and result.aic is not None else None,
            'bayesian_info_criterion': result.bic if hasattr(result, 'bic') and result.bic is not None else None,
            'r_squared': None,  # Calculate if we have data
            'n_data_points': result.ndata if hasattr(result, 'ndata') else None,
            'n_parameters': result.nvarys if hasattr(result, 'nvarys') else None,
            'n_degrees_freedom': result.nfree if hasattr(result, 'nfree') else None,
            'fit_success': result.success if hasattr(result, 'success') else None
        }
        
        # Calculate R-squared if possible
        try:
            ss_res = np.sum(result.residual ** 2) if result.residual is not None else np.sum((y_fit - result.best_fit) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            diagnostics_dict['r_squared'] = float(r_squared)
        except:
            pass
        
        # Add diagnostics to fit data
        listfit_fit_data.append(diagnostics_dict)
        
        # Save fit to .qsap file
        if listfit_fit_data:
            self.save_and_print_qsap_fit(listfit_fit_data, 'Listfit', 'Listfit')
        
        # Clear listfit mode
        self.listfit_mode = False
        for line in self.listfit_bound_lines:
            line.remove()
        self.listfit_bound_lines.clear()
        self.listfit_bounds = []
        
        # Reset Advanced dropdown after successful fit
        self.reset_advanced_dropdown()
        
        self.fig.canvas.draw_idle()  # Redraw to show listfit results

    def build_composite_model(self, components, x_fit, y_fit, err_fit):
        """Build a composite lmfit Model from component list with improved initial guesses for blended profiles"""
        from scipy.signal import find_peaks
        from lmfit import Model, Parameters
        
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
                    from lmfit.models import PolynomialModel
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
        
        print(f"[DEBUG] Peak detection starting... num_gaussians={num_gaussians}, num_voigts={num_voigts}")
        
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
        
        print(f"[DEBUG] Peak detection completed. Found {len(peaks)} peaks")
        
        peak_index_for_component = 0  # Track which peak to use next
        
        print(f"[DEBUG] Processing {len(components)} components...")
        
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
                from lmfit.models import PolynomialModel
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

    def _check_listfit_quality(self, result, y_fit, err_fit=None):
        """Check the quality of the Listfit and warn user if fit is poor
        
        Args:
            result: lmfit fit result object
            y_fit: the fitted data
            err_fit: error spectrum (None if not available)
        """
        warnings = []
        
        # Check if error spectrum is actually available (not None and not empty)
        has_error_spectrum = err_fit is not None and len(err_fit) > 0
        
        # If there's NO error spectrum, show a different warning
        if not has_error_spectrum:
            print("\n" + "="*70)
            print("WARNING: NO ERROR SPECTRUM USED FOR FIT")
            print("="*70)
            # Calculate SSR from residuals
            ssr = np.sum(result.residual ** 2) if result.residual is not None else np.sum((y_fit - result.best_fit) ** 2)
            print(f"  • No error spectrum provided for fit. Consider refitting with error spectrum.")
            print(f"  • Sum of Squared Residuals (SSR) = {ssr:.3e}")
            print("  • SSR alone is difficult to interpret without knowing typical noise levels.")
            print("="*70 + "\n")
            return
        
        # If there IS an error spectrum, use chi-squared based warnings
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
        # Get colors from config
        gaussian_cfg = self.colors['profiles']['gaussian']
        voigt_cfg = self.colors['profiles']['voigt']
        continuum_cfg = self.colors['profiles']['continuum_line']
        colors = {'gaussian': gaussian_cfg['color'], 'voigt': voigt_cfg['color'], 'polynomial': continuum_cfg['color']}
        
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
                
                # Check if this component's parameters are in the fit result
                if f'{prefix}amp' not in params:
                    print(f"[DEBUG] Warning: Gaussian component {gauss_count} not found in fit result. Skipping...")
                    gauss_count += 1
                    continue
                
                g_amp = params[f'{prefix}amp'].value
                g_mean = params[f'{prefix}mean'].value
                g_stddev = params[f'{prefix}stddev'].value
                
                # Extract errors from lmfit results
                g_amp_err = params[f'{prefix}amp'].stderr if params[f'{prefix}amp'].stderr is not None else 0.0
                g_mean_err = params[f'{prefix}mean'].stderr if params[f'{prefix}mean'].stderr is not None else 0.0
                g_stddev_err = params[f'{prefix}stddev'].stderr if params[f'{prefix}stddev'].stderr is not None else 0.0
                
                y_component = self.gaussian(x_smooth, g_amp, g_mean, g_stddev)
                # Add label only for the first listfit gaussian
                label = 'Gaussian' if 'gaussian' not in self.legend_profile_types else None
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle=gaussian_cfg['linestyle'], linewidth=gaussian_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('gaussian')
                
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
                
                # Check if this component's parameters are in the fit result
                if f'{prefix}amp' not in params:
                    print(f"[DEBUG] Warning: Voigt component {voigt_count} not found in fit result. Skipping...")
                    voigt_count += 1
                    continue
                
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
                # Add label only for the first listfit voigt
                label = 'Voigt' if 'voigt' not in self.legend_profile_types else None
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle=voigt_cfg['linestyle'], linewidth=voigt_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('voigt')
                
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
                
                # Check if this component's parameters are in the fit result
                if f'{prefix}c0' not in params:
                    print(f"[DEBUG] Warning: Polynomial component {poly_count} not found in fit result. Skipping...")
                    poly_count += 1
                    continue
                
                poly_coeffs = []
                for i in range(order + 1):
                    poly_coeffs.append(params[f'{prefix}c{i}'].value)
                # Reverse coefficients for np.polyval (expects highest order first)
                poly_coeffs = poly_coeffs[::-1]
                y_component = np.polyval(poly_coeffs, x_smooth)
                # Add label only for the first listfit polynomial
                label = 'Continuum' if 'continuum' not in self.legend_profile_types else None
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle=continuum_cfg['linestyle'], linewidth=continuum_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('continuum')
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



