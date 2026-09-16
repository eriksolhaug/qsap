# qsap Spectrum Plotter --- v0.12
r"""
Main spectrum plotter widget for interactive spectral analysis

KEYBOARD SHORTCUTS:

Navigation Controls:
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

Spectrum Display:
  1-9                  - Apply Gaussian smoothing with different kernel sizes
  0                    - Remove smoothing (restore original spectrum)
  ~ (tilde)            - Toggle between step plot and line plot
  ` (backtick)         - Save screenshot of plot

Fitting Modes:
  m                    - Enter continuum fitting mode (define regions with SPACE)
  M                    - Remove a continuum region
  ENTER (in continuum mode) - Fit polynomial continuum to defined regions
  d                    - Enter Single Mode Gaussian fit (click to fit, SPACE to select bounds)
  | (pipe)             - Enter Multi-Gaussian fit mode (fit multiple Gaussians simultaneously)
  n                    - Single mode Voigt profile fitting
  e                    - Open line list selector window (new line list system)
  H                    - Open Listfit window for composite fitting (RECOMMENDED FOR COMPLEX FITS)
  : (colon)            - Perform Bayesian fitting (MCMC) - follow along prompts in terminal - this works well for simple profile fits but has not been tested on cases more complex than a line+Gaussian/Voigt
  
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
mpl.rcParams['axes.unicode_minus'] = False  # Use ASCII minus signs instead of Unicode for better compatibility

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
from .fit_diagnostics_panel import FitDiagnosticsPanel
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


# Helper function for numpy compatibility (trapz vs trapezoid in numpy 2.0+)
def trapz_compat(y, x=None):
    """Compatible trapezoid integration for numpy < 2.0 and >= 2.0
    
    In numpy >= 2.0, np.trapz was renamed to np.trapezoid.
    This function tries np.trapz first, then falls back to np.trapezoid.
    """
    if hasattr(np, 'trapz'):
        return np.trapz(y, x)
    else:
        return np.trapezoid(y, x)


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


class SmoothingJoystick(QtWidgets.QWidget):
    """Custom joystick widget for controlling smoothing parameters"""
    
    # Signals for joystick movement (0-1 range, centered at 0.5)
    x_moved = QtCore.pyqtSignal(float)  # Left-right (Median control)
    y_moved = QtCore.pyqtSignal(float)  # Up-down (Gaussian control)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)
        self.setStyleSheet("background-color: #f0f0f0; border: 2px solid #333;")
        
        # Joystick position (0-1 range)
        self.x_pos = 0.5  # Center
        self.y_pos = 0.5  # Center
        
        # Low sensitivity parameters
        self.sensitivity = 0.02  # Small movement threshold
        self.deadzone = 0.05  # Neutral zone around center
        
        self.setFocusPolicy(Qt.StrongFocus)
        
    def mouseMoveEvent(self, event):
        """Track mouse movement to control joystick"""
        if event.buttons() & Qt.LeftButton:
            # Normalize mouse position to 0-1 range
            rect = self.rect()
            x = event.x() / rect.width()
            y = 1.0 - (event.y() / rect.height())  # Invert Y (up is positive)
            
            # Clamp to 0-1 range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            # Apply deadzone
            if abs(x - 0.5) < self.deadzone:
                x = 0.5
            if abs(y - 0.5) < self.deadzone:
                y = 0.5
            
            self.x_pos = x
            self.y_pos = y
            
            # Emit signals with low sensitivity
            self.x_moved.emit(x)
            self.y_moved.emit(y)
            
            self.update()
    
    def mousePressEvent(self, event):
        """Start joystick control"""
        if event.button() == Qt.LeftButton:
            self.mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Joystick stays in place when released (no auto-return to center)"""
        # Joystick position is retained, not reset
        pass
    
    def paintEvent(self, event):
        """Draw the joystick"""
        painter = QtGui.QPainter(self)
        rect = self.rect()
        
        # Draw background circle
        painter.setBrush(QtGui.QColor("#e8e8e8"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#333"), 2))
        painter.drawEllipse(rect)
        
        # Draw crosshairs (neutral position)
        center_x = rect.width() / 2
        center_y = rect.height() / 2
        painter.setPen(QtGui.QPen(QtGui.QColor("#999"), 1, Qt.DashLine))
        painter.drawLine(int(center_x - 20), int(center_y), int(center_x + 20), int(center_y))
        painter.drawLine(int(center_x), int(center_y - 20), int(center_x), int(center_y + 20))
        
        # Draw current position indicator (joystick knob)
        knob_x = self.x_pos * rect.width()
        knob_y = (1.0 - self.y_pos) * rect.height()
        
        # Check if in deadzone
        if abs(self.x_pos - 0.5) < self.deadzone and abs(self.y_pos - 0.5) < self.deadzone:
            painter.setBrush(QtGui.QColor("#cccccc"))
        else:
            painter.setBrush(QtGui.QColor("#2196F3"))
        
        painter.setPen(QtGui.QPen(QtGui.QColor("#333"), 2))
        painter.drawEllipse(int(knob_x - 8), int(knob_y - 8), 16, 16)
        
        # Draw labels
        painter.setPen(QtGui.QColor("#666"))
        painter.setFont(QtGui.QFont("Arial", 7))
        painter.drawText(rect.adjusted(5, 5, -5, -5), Qt.AlignTop | Qt.AlignLeft, "↕ Gaussian σ")
        painter.drawText(rect.adjusted(5, 5, -5, -5), Qt.AlignBottom | Qt.AlignLeft, "↔ Median")
        
        painter.end()


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
        
        # Initialize smoothing panel
        self.init_smoothing_panel()
        
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
        self.smoothed_spec = []  # For smoothed spectrum data
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
        self.current_continuum_fit_id = None  # Track fit_id for all items in current continuum session
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
        
        # Guess drawing mode for listfit components
        self.guess_drawing_mode = False
        self.current_component_for_guess = None  # Component dict being edited
        self.current_component_row = None  # Row index in listfit window
        self.guess_preview_line = None  # Temporary line showing preview Gaussian/Voigt
        self.guess_center = None  # Wavelength center of the guess
        self.guess_sigma = None  # Width/sigma of the guess
        self.guess_amp = None  # Amplitude being dragged
        self.guess_lines = {}  # Store persistent guess lines by component_id
        # Direct drag tracking for simplified guess drawing (click-drag to draw)
        self.guess_mouse_down = False  # Whether mouse button is currently pressed
        self.guess_drag_start_x = None  # Wavelength where drag started
        self.guess_drag_start_y = None  # Flux where drag started
        self.guess_drag_end_x = None  # Current wavelength during drag
        self.guess_drag_end_y = None  # Current flux during drag
        self.guess_polynomial_line = None  # Temporary line for polynomial guess
        self.guess_polynomial_clicked_points = None  # Markers for clicked points during polynomial drawing
        self.guess_polynomial_baseline_line = None  # Temporary line showing polynomial baseline during Gaussian/Voigt drag
        self._polynomial_guess_confirmed = False  # Flag to track if polynomial guess was confirmed
        
        # Polynomial multi-point click mode for listfit
        self.polynomial_points = []  # List of (x, y) tuples for polynomial guess
        self.polynomial_order = 1  # Order of polynomial (will be set from component)
        self.polynomial_preview_points = []  # Line objects for showing clicked points
        
        # Mask drawing mode
        self.mask_drawing_mode = False  # True when drawing mask regions
        self.mask_type = None  # 'data_mask' or 'polynomial_mask'
        self.mask_regions = []  # List of (x_start, x_end) tuples for drawn regions
        self.mask_preview_rects = []  # Rectangle patches for visual feedback
        self.mask_drag_start_x = None  # X coordinate where mask drag started
        
        # Constraint bounds setting mode
        self.constraint_bounds_mode = False  # True when setting bounds for a constraint
        self.constraint_parameter = None  # Parameter being set (amp, mu, sigma, etc.)
        self.constraint_bounds_drag_start_x = None  # X coordinate where drag started
        self.constraint_bounds_drag_end_x = None  # X coordinate where drag ended
        self.constraint_bounds_preview_line = None  # Temporary vertical line showing bounds
        self.current_constraint_editor = None  # Reference to ConstraintEditor widget for updating bounds
        self.current_constraint_editor_dialog = None  # Reference to dialog containing the editor

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

        # Error Spectrum and Residual Display Options
        self.error_spectrum_mode = "Default"  # "Default" (red dashed line) or "Shaded" (light gray band)
        self.residual_display_mode = "None"  # "None", "Sigma" (residual/error), or "Shaded"
        self.step_error = None  # Step plot version of error line
        self.line_error = None  # Line plot version of error line
        self.error_band_fill = None  # Fill_between object for shaded error spectrum

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
        
        # Fit Diagnostics tracking (session-only, rebuilt from .qsap files on load)
        # Color palette: 12 distinct, colorblind-safe colors
        self.FIT_COLORS = [
            '#E41A1C',  # Red
            '#377EB8',  # Blue
            '#4DAF4A',  # Green
            '#FF7F00',  # Orange
            '#984EA3',  # Purple
            '#A65628',  # Brown
            '#F781BF',  # Pink
            '#999999',  # Grey
            '#E7298A',  # Magenta
            '#66C2A5',  # Teal
            '#FC8D62',  # Coral
            '#8DA0CB',  # Slate
        ]
        self.fit_counter = 0  # Auto-incrementing fit ID
        self.fit_colors = {}  # Map fit_id → color hex string
        self.fit_metadata = {}  # Map fit_id → {type, R², χ²_red, AIC, BIC, n_params, condition_num, flag, timestamp}
        self.fit_items = {}  # Map fit_id → set of item_ids belonging to this fit

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

        # Smoothing controls
        self.median_kernel_size = 1  # Median filter kernel size (default 1 = no smoothing)
        self.gaussian_sigma = 0.0  # Gaussian filter sigma (default 0 = no smoothing)
        self.smoothing_interactive_mode = False  # Is interactive smoothing enabled?
        self.smoothing_drag_start_x = None  # Track drag start position for smoothing
        self.smoothing_drag_start_y = None
        self.smoothing_prev_median = 1  # Previous applied median kernel (for reset)
        self.smoothing_prev_gaussian = 0.0  # Previous applied Gaussian sigma (for reset)
        self.last_applied_median = 1  # Last successfully applied median value
        self.last_applied_gaussian = 0.0  # Last successfully applied Gaussian value

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
        
        # Fit Diagnostics Panel
        self.fit_diagnostics_panel = FitDiagnosticsPanel()
        
        self.fit_information_window = FitInformationWindow()
        self.item_id_counter = 0
        self.item_id_map = {}  # Maps item_id to {'type': 'gaussian', 'fit': fit_dict, ...}
        self.lmfit_results = {}  # Maps fit_id to lmfit result objects (stored separately to avoid cleanup issues)
        self.highlighted_item_ids = set()  # Track all currently highlighted items
        self.redshift_selected_line = None  # Track the line object selected for redshift
        
        # Connect items_changed signal to update total line if displayed
        self.item_tracker.items_changed.connect(self.update_total_line_if_shown)
        
        # Connect display toggle signal to handle visibility changes
        self.item_tracker.item_display_toggled.connect(self.on_item_display_toggled)
        
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
        
        # Add separator
        settings_layout.addSpacing(15)
        
        # Remove All Plotted Features Section
        remove_features_layout = QVBoxLayout()
        remove_features_layout.setSpacing(5)
        
        remove_title = QtWidgets.QLabel("Data Management")
        remove_title.setStyleSheet("font-weight: bold; font-size: 11px;")
        remove_features_layout.addWidget(remove_title)
        
        self.remove_all_features_button = QtWidgets.QPushButton("Remove All Plotted Features")
        self.remove_all_features_button.setStyleSheet("background-color: #fff3cd; color: #856404; font-weight: bold;")
        self.remove_all_features_button.clicked.connect(self.on_remove_all_plotted_features)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.remove_all_features_button)
        button_layout.addStretch()
        remove_features_layout.addLayout(button_layout)
        
        # Add helper text
        remove_help_label = QtWidgets.QLabel("Remove all plotted features (guesses, fits, etc.) except the spectrum.")
        remove_help_label.setStyleSheet("font-size: 9px; color: #666666;")
        remove_features_layout.addWidget(remove_help_label)
        
        settings_layout.addLayout(remove_features_layout)
        
        settings_layout.addStretch()
        
        self.settings_container.setLayout(settings_layout)

    def init_smoothing_panel(self):
        """Initialize the Smoothing widget panel with controls for interactive smoothing"""
        # Create scroll area for smoothing controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; }
            QScrollBar:vertical { width: 10px; }
            QScrollBar::handle:vertical { background: #888888; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #555555; }
        """)
        
        # Create the actual content widget
        content_widget = QtWidgets.QWidget()
        smoothing_layout = QVBoxLayout()
        smoothing_layout.setContentsMargins(10, 10, 10, 10)
        smoothing_layout.setSpacing(8)
        
        # Title
        smoothing_title = QtWidgets.QLabel("Smoothing Controls")
        title_font = smoothing_title.font()
        title_font.setBold(True)
        title_font.setPointSize(11)
        smoothing_title.setFont(title_font)
        smoothing_layout.addWidget(smoothing_title)
        
        # Median Kernel Input
        median_layout = QHBoxLayout()
        median_layout.setSpacing(5)
        median_label = QLabel("Median Kernel (pixels):")
        self.smoothing_median_input = QLineEdit()
        self.smoothing_median_input.setText("1")
        self.smoothing_median_input.setMaximumWidth(80)
        self.smoothing_median_input.setToolTip("Median filter kernel size in pixels (odd integer, 1=off)")
        self.smoothing_median_input.returnPressed.connect(self.on_smoothing_apply)
        median_layout.addWidget(median_label)
        median_layout.addWidget(self.smoothing_median_input)
        median_layout.addStretch()
        smoothing_layout.addLayout(median_layout)
        
        # Gaussian Sigma Input
        gaussian_layout = QHBoxLayout()
        gaussian_layout.setSpacing(5)
        gaussian_label = QLabel("Gaussian Sigma:")
        self.smoothing_gaussian_input = QLineEdit()
        self.smoothing_gaussian_input.setText("0.0")
        self.smoothing_gaussian_input.setMaximumWidth(80)
        self.smoothing_gaussian_input.setToolTip("Gaussian filter sigma (0=off)")
        self.smoothing_gaussian_input.returnPressed.connect(self.on_smoothing_apply)
        gaussian_layout.addWidget(gaussian_label)
        gaussian_layout.addWidget(self.smoothing_gaussian_input)
        gaussian_layout.addStretch()
        smoothing_layout.addLayout(gaussian_layout)
        
        # Apply Button
        apply_button = QPushButton("Apply Smoothing")
        apply_button.clicked.connect(self.on_smoothing_apply)
        apply_button.setToolTip("Apply median and/or Gaussian smoothing to spectrum")
        apply_button.setMaximumWidth(180)
        
        # Reset Button
        reset_button = QPushButton("Reset to Previous")
        reset_button.clicked.connect(self.on_smoothing_reset)
        reset_button.setToolTip("Restore previous smoothing settings")
        reset_button.setMaximumWidth(180)
        
        # Original Data Button
        original_button = QPushButton("Reset to Original")
        original_button.clicked.connect(self.on_smoothing_original)
        original_button.setToolTip("Remove all smoothing, show original spectrum")
        original_button.setMaximumWidth(180)
        
        # Toggle Step/Line Plot Button
        toggle_plot_button = QPushButton("Toggle Step/Line Plot")
        toggle_plot_button.clicked.connect(self.on_toggle_step_line)
        toggle_plot_button.setToolTip("Toggle between step and line plot (~ key)")
        toggle_plot_button.setMaximumWidth(180)
        
        # Button Grid Layout (2x2)
        button_grid = QtWidgets.QGridLayout()
        button_grid.setSpacing(5)
        button_grid.addWidget(apply_button, 0, 0)
        button_grid.addWidget(reset_button, 0, 1)
        button_grid.addWidget(original_button, 1, 0)
        button_grid.addWidget(toggle_plot_button, 1, 1)
        smoothing_layout.addLayout(button_grid)
        
        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        smoothing_layout.addWidget(separator)
        
        # Joystick Label
        joystick_label = QtWidgets.QLabel("Smoothing Joystick")
        joystick_font = joystick_label.font()
        joystick_font.setBold(True)
        joystick_font.setPointSize(10)
        joystick_label.setFont(joystick_font)
        smoothing_layout.addWidget(joystick_label)
        
        # Joystick Widget
        self.smoothing_joystick = SmoothingJoystick()
        self.smoothing_joystick.x_moved.connect(self.on_joystick_x_moved)
        self.smoothing_joystick.y_moved.connect(self.on_joystick_y_moved)
        smoothing_layout.addWidget(self.smoothing_joystick, alignment=Qt.AlignCenter)
        
        # Joystick Info
        joystick_info = QtWidgets.QLabel(
            "<small><b>Joystick Control (Symmetric):</b><br>"
            "Left/Right: Adjust Median Kernel (center = no smoothing)<br>"
            "Up/Down: Adjust Gaussian Sigma (center = no smoothing)<br>"
            "Joystick stays in place when released<br>"
            "(Auto-applies smoothing)</small>"
        )
        joystick_info.setWordWrap(True)
        joystick_info.setStyleSheet("color: #666666; font-size: 8pt;")
        smoothing_layout.addWidget(joystick_info)
        
        # Separator
        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.HLine)
        separator2.setFrameShadow(QtWidgets.QFrame.Sunken)
        smoothing_layout.addWidget(separator2)
        
        # Interactive Mode Checkbox
        self.smoothing_interactive_checkbox = QtWidgets.QCheckBox("Enable Smoothing with Click-and-Drag in Plotter")
        self.smoothing_interactive_checkbox.setChecked(False)
        self.smoothing_interactive_checkbox.stateChanged.connect(self.on_smoothing_interactive_mode_changed)
        self.smoothing_interactive_checkbox.setToolTip("When checked: Up-down drag adjusts Gaussian, Left-right adjusts Median")
        smoothing_layout.addWidget(self.smoothing_interactive_checkbox)
        
        # Info label
        info_label = QtWidgets.QLabel(
            "<small><b>Click-and-Drag Smoothing (inside the plotter):</b><br>"
            "Up-down drag: adjust Gaussian<br>"
            "Left-right drag: adjust Median<br>"
            "Release to apply</small>"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666666; font-size: 8pt;")
        smoothing_layout.addWidget(info_label)
        
        smoothing_layout.addStretch()
        content_widget.setLayout(smoothing_layout)
        
        # Set content widget into scroll area
        scroll_area.setWidget(content_widget)
        
        # Store scroll area as smoothing_container for tab integration
        self.smoothing_container = scroll_area

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
            self.qsap_handler.save_directory = directory
            self.save_directory_input.setText(directory)

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
                # Don't print polynomial order changes triggered by Apply button
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
        # Use native macOS file picker (spurious Finder window issue was fixed)
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
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
        # Use native macOS file picker for modern appearance
        dialog.setOption(QFileDialog.DontUseNativeDialog, False)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        
        if dialog.exec_() != QFileDialog.Accepted:
            return
        
        file_paths = dialog.selectedFiles()
        if not file_paths:
            return
            
        self.load_directory = os.path.dirname(file_paths[0])
        file_covariance_data = None  # Initialize before loop so it's accessible after
        
        # Load all selected files
        for file_path in file_paths:
            try:
                basename = os.path.basename(file_path)
                
                # Check if this is a .qsap file
                if file_path.endswith('.qsap'):
                    file_covariance_data = self._load_qsap_file(file_path)
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
            
            # Clear 'line' objects from all loaded fits so they get redrawn properly
            for fit in self.gaussian_fits:
                if 'line' in fit:
                    fit.pop('line', None)
            for fit in self.voigt_fits:
                if 'line' in fit:
                    fit.pop('line', None)
            for fit in self.continuum_fits:
                if 'line' in fit:
                    fit.pop('line', None)
            print(f"[DEBUG] Cleared line objects from all loaded fits")
            
            # Always redraw fits (will only add new ones that don't have lines yet)
            self._redraw_loaded_fits()
            self.fig.canvas.draw_idle()
            
            # Distribute covariance matrices to loaded fits if available
            # (covariance_data from last loaded QSAP file, if applicable)
            if file_covariance_data:
                self._distribute_covariance_to_fits(file_covariance_data)
                
                # After covariance distribution, update the panel with covariance data
                # This allows correlation matrix plotting for loaded fits
                self._update_panel_with_covariance(file_covariance_data)
            
        except Exception as e:
            print(f"Error loading fit file: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_qsap_file(self, filepath):
        """Load fits from a .qsap file and populate relevant fit lists
        
        Args:
            filepath: Path to the .qsap file
            
        Returns:
            covariance_data: Full covariance matrix info from QSAP
        """
        qsap_data = self.qsap_handler.parse_qsap_file(filepath)
        metadata = qsap_data.get('metadata', {})
        components = qsap_data.get('components', [])
        
        # Load full covariance matrix and component registry (for MC sampling)
        covariance_data = self.qsap_handler.load_covariance_from_qsap(filepath)
        
        fit_type = metadata.get('TYPE', 'Unknown')
        fit_mode = metadata.get('MODE', 'Unknown')
        
        print(f"Loading {fit_type} fit (mode: {fit_mode})")
        print(f"[DEBUG] Metadata keys: {list(metadata.keys())}")
        print(f"[DEBUG] Components count: {len(components)}, first few component sections: {[c.get('TYPE') or list(c.keys())[0] if c else 'empty' for c in components[:5]]}")
        
        # Load scale factor from metadata if present
        if 'SCALE_FACTOR' in metadata:
            self.flux_scale_factor = metadata['SCALE_FACTOR']
            print(f"  Scale factor: {self.flux_scale_factor}")
        
        # Extract FIT_DIAGNOSTICS section (may be in components list since parser treats all sections as components)
        fit_diagnostics = None
        remaining_components = []
        for comp in components:
            # Check if this is the FIT_DIAGNOSTICS section (has CHI2, R_SQUARED, etc.)
            if 'CHI2' in comp or 'R_SQUARED' in comp or 'AKAIKE_INFO_CRITERION' in comp:
                fit_diagnostics = comp
                print(f"[DEBUG] Found FIT_DIAGNOSTICS: {list(comp.keys())}")
            elif comp.get('TYPE'):  # This is an actual component with TYPE field
                remaining_components.append(comp)
            else:
                # Skip sections without TYPE and without diagnostic keys
                pass
        
        components = remaining_components  # Use only actual component sections
        
        # Extract quality metrics from FIT_DIAGNOSTICS section
        quality_metrics = {}
        if fit_diagnostics:
            quality_metrics = {
                'chi2': fit_diagnostics.get('CHI2') or fit_diagnostics.get('SSR'),
                'chi2_reduced': fit_diagnostics.get('CHI2_REDUCED') or fit_diagnostics.get('SSR_NU'),
                'r_squared': fit_diagnostics.get('R_SQUARED'),
                'akaike': fit_diagnostics.get('AKAIKE_INFO_CRITERION'),
                'bayesian': fit_diagnostics.get('BAYESIAN_INFO_CRITERION'),
                'n_data': fit_diagnostics.get('N_DATA_POINTS'),
                'n_params': fit_diagnostics.get('N_PARAMETERS'),
            }
            print(f"[DEBUG] Quality metrics from FIT_DIAGNOSTICS: {quality_metrics}")
        else:
            print(f"[DEBUG] No FIT_DIAGNOSTICS section found")
            quality_metrics = {
                'chi2': None,
                'chi2_reduced': None,
                'r_squared': None,
                'akaike': None,
                'bayesian': None,
                'n_data': None,
                'n_params': None,
            }
        
        # Process components based on fit type
        # Assign one fit_id for all components of this loaded fit
        loaded_fit_id = self.next_fit_id()
        self.assign_fit_color(loaded_fit_id)
        
        # Use n_params from diagnostics if available, otherwise count components
        n_params = quality_metrics.get('n_params') or 0
        n_data = quality_metrics.get('n_data')
        
        if fit_type == 'Gaussian':
            for comp in components:
                fit_dict = self._parse_qsap_gaussian_component(comp)
                if fit_dict:
                    fit_dict['_fit_id'] = loaded_fit_id  # Store fit_id for color grouping during redraw
                    self.gaussian_fits.append(fit_dict)
                    # Only count if n_params not already from diagnostics
                    if not quality_metrics.get('n_params'):
                        n_params += 3  # Gaussian has 3 parameters (amplitude, mean, stddev)
            print(f"  Loaded {len(components)} Gaussian components")
            
            # Register metadata for Fit Diagnostics
            self._register_loaded_fit_metadata(loaded_fit_id, 'Single Gaussian' if len(components) == 1 else 'Multi-Gaussian',
                                             quality_metrics)
            
        elif fit_type == 'Voigt':
            for comp in components:
                fit_dict = self._parse_qsap_voigt_component(comp)
                if fit_dict:
                    fit_dict['_fit_id'] = loaded_fit_id  # Store fit_id for color grouping during redraw
                    self.voigt_fits.append(fit_dict)
                    # Only count if n_params not already from diagnostics
                    if not quality_metrics.get('n_params'):
                        n_params += 4  # Voigt has 4 parameters (amplitude, center, sigma, gamma)
            print(f"  Loaded {len(components)} Voigt components")
            
            # Register metadata for Fit Diagnostics
            self._register_loaded_fit_metadata(loaded_fit_id, 'Single Voigt' if len(components) == 1 else 'Multi-Voigt',
                                             quality_metrics)
            
        elif fit_type == 'Continuum':
            for comp in components:
                fit_dict = self._parse_qsap_continuum_component(comp)
                if fit_dict:
                    fit_dict['_fit_id'] = loaded_fit_id  # Store fit_id for color grouping during redraw
                    self.continuum_fits.append(fit_dict)
                    
                    # Count parameters from polynomial order
                    poly_order = fit_dict.get('poly_order', 1)
                    n_params += (poly_order + 1)
                    
                    # Recreate continuum regions/patches for visualization
                    # Check if individual_regions was stored, otherwise use the combined bounds
                    individual_regions = fit_dict.get('individual_regions')
                    if not individual_regions and 'bounds' in fit_dict:
                        individual_regions = [fit_dict['bounds']]
                    
                    if individual_regions:
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
                                                     position=position_str, color=continuum_region_cfg['color'], bounds=region_bounds,
                                                     fit_id=loaded_fit_id)
                                except Exception as e:
                                    print(f"Error creating continuum region patch: {e}")
            print(f"  Loaded {len(components)} Continuum fits")
            
            # Register metadata for Fit Diagnostics
            self._register_loaded_fit_metadata(loaded_fit_id, 'Continuum', quality_metrics)
            
        elif fit_type == 'Listfit':
            # Assign single fit_id for all components of this listfit
            listfit_fit_id = self.next_fit_id()
            self.assign_fit_color(listfit_fit_id)
            
            mask_count = 0
            poly_count = 0
            listfit_n_params = 0
            
            # Extract tie expressions and covariance info for inline extraction
            tie_expressions = {}
            if covariance_data and 'tie_expressions' in covariance_data:
                tie_expressions = covariance_data['tie_expressions']
                print(f"[DEBUG_LOAD_TIES] Loaded tie_expressions from covariance_data: {tie_expressions}")
            else:
                print(f"[DEBUG_LOAD_TIES] WARNING: No tie_expressions in covariance_data (keys: {list(covariance_data.keys()) if covariance_data else 'None'})")
            
            print(f"[DEBUG_LOAD_TIES] covariance_data keys: {list(covariance_data.keys()) if covariance_data else 'None'}")
            
            full_cov = None
            free_param_names = []
            free_param_values = []  # Store best-fit values for all free parameters
            if covariance_data and 'free_parameter_names' in covariance_data and 'free_parameter_covariance' in covariance_data:
                full_cov = covariance_data['free_parameter_covariance']
                free_param_names = covariance_data['free_parameter_names']
                print(f"[DEBUG_LOAD_TIES] Extracted free_param_names: {free_param_names}")
                print(f"[DEBUG_LOAD_TIES] Extracted full_cov shape: {full_cov.shape if hasattr(full_cov, 'shape') else type(full_cov)}")
                # Build a list of free parameter values in the same order
                if 'free_parameter_values' in covariance_data:
                    free_param_values = covariance_data['free_parameter_values']
                else:
                    # If not directly available, try to build from _free_param_dict (fallback)
                    if '_free_param_dict' in covariance_data:
                        free_param_dict = covariance_data['_free_param_dict']
                        free_param_values = [free_param_dict.get(pname, 0.0) for pname in free_param_names]
            
            # Track component indices for covariance extraction
            gauss_idx = 0
            voigt_idx = 0
            poly_idx = 0
            
            for comp in components:
                if comp.get('TYPE') == 'Gaussian':
                    fit_dict = self._parse_qsap_gaussian_component(comp)
                    if fit_dict:
                        fit_dict['component_id'] = f'gaussian_{gauss_idx}'  # Set component ID for parameter matching
                        fit_dict['_fit_id'] = listfit_fit_id  # Store fit_id for color grouping during redraw
                        fit_dict['is_listfit_component'] = True  # Mark as Listfit component - don't add continuum
                        # Store tie expressions from QSAP in the component dict (for MC EW calculation)
                        if tie_expressions:
                            fit_dict['tie_expressions'] = tie_expressions
                        
                        # Store global free parameter values for MC access (e.g., z1, other global params)
                        if free_param_names and free_param_values:
                            fit_dict['free_param_names_all'] = free_param_names
                            fit_dict['free_param_values_all'] = free_param_values

                        
                        # Extract covariance for this Gaussian's free parameters
                        # For tied Gaussians, only some parameters may be free (e.g., just amplitude)
                        if full_cov is not None and free_param_names:
                            # Find all free parameters that belong to this Gaussian
                            param_indices = []
                            param_names_extracted = []
                            
                            # Check for parameters specific to this Gaussian index
                            for pname in free_param_names:
                                if pname.startswith(f'g{gauss_idx}_'):
                                    param_indices.append(free_param_names.index(pname))
                                    param_names_extracted.append(pname)
                            
                            if param_indices:
                                try:
                                    # Extract covariance submatrix for these parameters
                                    if len(param_indices) == 1:
                                        # Single parameter - covariance is just the variance
                                        submatrix = np.array([[full_cov[param_indices[0], param_indices[0]]]])
                                    else:
                                        submatrix = full_cov[np.ix_(param_indices, param_indices)]
                                    
                                    fit_dict['covariance'] = submatrix
                                    fit_dict['free_params_for_cov'] = param_names_extracted
                                    fit_dict['all_free_param_names'] = free_param_names  # Store full param list for tied param resolution
                                    fit_dict['full_covariance'] = full_cov  # Store full matrix for tied parameter access
                                    print(f"[DEBUG] Extracted {len(param_indices)} free parameter(s) for Gaussian {gauss_idx}: {param_names_extracted}")
                                except Exception as e:
                                    print(f"[DEBUG] Could not extract Gaussian {gauss_idx} covariance: {e}")
                            else:
                                print(f"[DEBUG] WARNING: Gaussian {gauss_idx} has no free parameters in covariance matrix (fully tied?)")
                        
                        self.gaussian_fits.append(fit_dict)
                        bounds = fit_dict.get('bounds', 'MISSING')
                        print(f"[DEBUG] Loaded Gaussian: mean={fit_dict.get('mean'):.2f}, bounds={bounds}")
                        listfit_n_params += 3
                    gauss_idx += 1
                    
                elif comp.get('TYPE') == 'Voigt':
                    fit_dict = self._parse_qsap_voigt_component(comp)
                    if fit_dict:
                        fit_dict['component_id'] = f'voigt_{voigt_idx}'  # Set component ID for parameter matching
                        fit_dict['_fit_id'] = listfit_fit_id  # Store fit_id for color grouping during redraw
                        fit_dict['is_listfit_component'] = True  # Mark as Listfit component - don't add continuum
                        # Store tie expressions from QSAP in the component dict (for MC EW calculation)
                        if tie_expressions:
                            fit_dict['tie_expressions'] = tie_expressions
                        
                        # Store global free parameter values for MC access (e.g., z1, other global params)
                        if free_param_names and free_param_values:
                            fit_dict['free_param_names_all'] = free_param_names
                            fit_dict['free_param_values_all'] = free_param_values

                        
                        # Extract covariance for this Voigt's free parameters
                        if full_cov is not None and free_param_names:
                            param_indices = []
                            param_names_extracted = []
                            
                            for pname in free_param_names:
                                if pname.startswith(f'v{voigt_idx}_'):
                                    param_indices.append(free_param_names.index(pname))
                                    param_names_extracted.append(pname)
                            
                            if param_indices:
                                try:
                                    if len(param_indices) == 1:
                                        submatrix = np.array([[full_cov[param_indices[0], param_indices[0]]]])
                                    else:
                                        submatrix = full_cov[np.ix_(param_indices, param_indices)]
                                    
                                    fit_dict['covariance'] = submatrix
                                    fit_dict['free_params_for_cov'] = param_names_extracted
                                    fit_dict['all_free_param_names'] = free_param_names
                                    fit_dict['full_covariance'] = full_cov
                                    print(f"[DEBUG] Extracted {len(param_indices)} free parameter(s) for Voigt {voigt_idx}: {param_names_extracted}")
                                except Exception as e:
                                    print(f"[DEBUG] Could not extract Voigt {voigt_idx} covariance: {e}")
                            else:
                                print(f"[DEBUG] WARNING: Voigt {voigt_idx} has no free parameters in covariance matrix (fully tied?)")
                        
                        self.voigt_fits.append(fit_dict)
                        bounds = fit_dict.get('bounds', 'MISSING')
                        print(f"[DEBUG] Loaded Voigt: center={fit_dict.get('center'):.2f}, bounds={bounds}")
                        listfit_n_params += 4
                    voigt_idx += 1
                    
                elif comp.get('TYPE') == 'Polynomial':
                    fit_dict = self._parse_qsap_polynomial_component(comp)
                    if fit_dict:
                        # Use bounds from the parsed polynomial (should be from BOUNDS_LOWER/BOUNDS_UPPER), or default to full spectrum
                        bounds = fit_dict.get('bounds', (self.x_data.min(), self.x_data.max()))
                        if 'bounds' not in fit_dict:
                            print(f"[DEBUG] Polynomial bounds missing; using full spectrum: {bounds}")
                        else:
                            print(f"[DEBUG] Loaded Polynomial: poly_order={fit_dict.get('poly_order')}, bounds={bounds}")
                        fit_dict['bounds'] = bounds
                        fit_dict['is_velocity_mode'] = False
                        fit_dict['component_id'] = f'polynomial_{poly_idx}'  # Set component ID for parameter matching
                        fit_dict['_fit_id'] = listfit_fit_id  # Store fit_id for color grouping during redraw
                        fit_dict['is_listfit_component'] = True  # Mark as Listfit component
                        # Store tie expressions from QSAP in the component dict (for MC EW calculation)
                        if tie_expressions:
                            fit_dict['tie_expressions'] = tie_expressions
                        
                        # Store global free parameter values for MC access (e.g., z1, other global params)
                        if free_param_names and free_param_values:
                            fit_dict['free_param_names_all'] = free_param_names
                            fit_dict['free_param_values_all'] = free_param_values

                        
                        # Extract polynomial covariance submatrix
                        if full_cov is not None and free_param_names:
                            param_indices = []
                            param_names_extracted = []
                            
                            for pname in free_param_names:
                                if pname.startswith(f'p{poly_idx}_'):
                                    param_indices.append(free_param_names.index(pname))
                                    param_names_extracted.append(pname)
                            
                            if param_indices:
                                try:
                                    if len(param_indices) == 1:
                                        submatrix = np.array([[full_cov[param_indices[0], param_indices[0]]]])
                                    else:
                                        submatrix = full_cov[np.ix_(param_indices, param_indices)]
                                    
                                    fit_dict['covariance'] = submatrix
                                    fit_dict['free_params_for_cov'] = param_names_extracted
                                    fit_dict['all_free_param_names'] = free_param_names
                                    fit_dict['full_covariance'] = full_cov
                                    print(f"[DEBUG] Extracted {len(param_indices)} free parameter(s) for Polynomial {poly_idx}: {param_names_extracted}")
                                except Exception as e:
                                    print(f"[DEBUG] Could not extract Polynomial {poly_idx} covariance: {e}")
                            else:
                                print(f"[DEBUG] WARNING: Polynomial {poly_idx} has no free parameters in covariance matrix (all tied?)")
                        
                        self.continuum_fits.append(fit_dict)
                        print(f"[DEBUG] Added polynomial to continuum_fits (total={len(self.continuum_fits)}), has 'coeffs'={('coeffs' in fit_dict)}")
                        poly_count += 1
                        
                        # Count polynomial parameters
                        poly_order = fit_dict.get('poly_order', 1)
                        listfit_n_params += (poly_order + 1)
                    poly_idx += 1
                    
                elif comp.get('TYPE') == 'Chebyshev':
                    fit_dict = self._parse_qsap_chebyshev_component(comp)
                    if fit_dict:
                        # Set component ID for Chebyshev (using separate counter, prefix 'ch')
                        cheb_count = len([f for f in self.continuum_fits if f.get('type') == 'chebyshev'])
                        fit_dict['component_id'] = f'chebyshev_{cheb_count}'
                        fit_dict['_fit_id'] = listfit_fit_id
                        fit_dict['is_listfit_component'] = True
                        fit_dict['type'] = 'chebyshev'  # Mark type explicitly
                        fit_dict['listfit_source'] = True  # CRITICAL: Mark as from listfit so _get_listfit_continuum finds it
                        
                        # Store tie expressions and free parameters
                        if tie_expressions:
                            fit_dict['tie_expressions'] = tie_expressions
                        if free_param_names and free_param_values:
                            fit_dict['free_param_names_all'] = free_param_names
                            fit_dict['free_param_values_all'] = free_param_values
                        
                        # Extract Chebyshev covariance submatrix
                        # Chebyshev coefficients are named c{i}_c0, c{i}_c1, etc.
                        cheb_idx = len([f for f in self.continuum_fits if f.get('type') == 'chebyshev'])
                        if full_cov is not None and free_param_names:
                            param_indices = []
                            param_names_extracted = []
                            
                            for pname in free_param_names:
                                if pname.startswith(f'c{cheb_idx}_'):
                                    param_indices.append(free_param_names.index(pname))
                                    param_names_extracted.append(pname)
                            
                            if param_indices:
                                try:
                                    if len(param_indices) == 1:
                                        submatrix = np.array([[full_cov[param_indices[0], param_indices[0]]]])
                                    else:
                                        submatrix = full_cov[np.ix_(param_indices, param_indices)]
                                    
                                    fit_dict['covariance'] = submatrix
                                    fit_dict['free_params_for_cov'] = param_names_extracted
                                    fit_dict['all_free_param_names'] = free_param_names
                                    fit_dict['full_covariance'] = full_cov
                                    print(f"[DEBUG] Extracted {len(param_indices)} free parameter(s) for Chebyshev: {param_names_extracted}")
                                except Exception as e:
                                    print(f"[DEBUG] Could not extract Chebyshev covariance: {e}")
                            else:
                                print(f"[DEBUG] WARNING: Chebyshev has no free parameters in covariance matrix (all tied?)")
                        
                        self.continuum_fits.append(fit_dict)
                        print(f"[DEBUG] Added Chebyshev to continuum_fits (total={len(self.continuum_fits)})")
                        
                        # Count Chebyshev parameters
                        cheb_degree = fit_dict.get('degree', 1)
                        listfit_n_params += (cheb_degree + 1)
                
                elif comp.get('TYPE') == 'PolynomialGuessMask':
                    mask_dict = self._parse_qsap_polynomial_guess_mask_component(comp)
                    if mask_dict:
                        min_lambda = mask_dict.get('min_lambda', 0)
                        max_lambda = mask_dict.get('max_lambda', 0)
                        position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                        self.register_item('polynomial_guess_mask', f'Polynomial Guess Mask (listfit)',
                                         fit_dict=mask_dict, position=position_str,
                                         color='lightblue', bounds=(min_lambda, max_lambda),
                                         fit_id=listfit_fit_id)
                        mask_count += 1
                elif comp.get('TYPE') == 'DataMask':
                    mask_dict = self._parse_qsap_data_mask_component(comp)
                    if mask_dict:
                        min_lambda = mask_dict.get('min_lambda', 0)
                        max_lambda = mask_dict.get('max_lambda', 0)
                        position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                        self.register_item('data_mask', f'Data Mask (listfit)',
                                         fit_dict=mask_dict, position=position_str,
                                         color='lightcoral', bounds=(min_lambda, max_lambda),
                                         fit_id=listfit_fit_id)
                        mask_count += 1
            print(f"  Loaded {len(components)} Listfit components (including {mask_count} masks, {poly_count} polynomials)")
            
            # Update n_params in quality_metrics if we calculated it and it wasn't in diagnostics
            if not quality_metrics.get('n_params') and listfit_n_params > 0:
                quality_metrics['n_params'] = listfit_n_params
            
            # Register metadata for Fit Diagnostics
            self._register_loaded_fit_metadata(listfit_fit_id, 'Listfit', quality_metrics)
            
        elif fit_type == 'Redshift':
            redshift_value = components[0].get('REDSHIFT') if components else None
            if isinstance(redshift_value, tuple):
                redshift_value = redshift_value[0]
            
            if redshift_value:
                print(f"  Loaded Redshift: {redshift_value}")
                print(f"  Line ID: {components[0].get('LINE_ID', 'Unknown')}")
                print(f"  Rest Wavelength: {components[0].get('LINE_WAVELENGTH_REST', 'Unknown')} Å")
                print(f"  Observed Wavelength: {components[0].get('LINE_WAVELENGTH_OBSERVED', 'Unknown')} Å")
        
        # Return covariance data for this file so caller can process it
        return covariance_data
    
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
        elif 'MU' in comp_dict:
            # Handle alternative field name (older QSAP format)
            val, err = comp_dict['MU'] if isinstance(comp_dict['MU'], tuple) else (comp_dict['MU'], None)
            fit_dict['mean'] = val
            if err:
                fit_dict['mean_err'] = err
        
        # Parse std_dev with error
        if 'STD_DEV' in comp_dict:
            val, err = comp_dict['STD_DEV'] if isinstance(comp_dict['STD_DEV'], tuple) else (comp_dict['STD_DEV'], None)
            fit_dict['stddev'] = val
            if err:
                fit_dict['stddev_err'] = err
        elif 'SIGMA' in comp_dict:
            # Handle alternative field name (older QSAP format)
            val, err = comp_dict['SIGMA'] if isinstance(comp_dict['SIGMA'], tuple) else (comp_dict['SIGMA'], None)
            fit_dict['stddev'] = val
            if err:
                fit_dict['stddev_err'] = err
        
        # Bounds
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Quality metrics
        if 'CHI_SQUARED' in comp_dict:
            fit_dict['chi2'] = comp_dict['CHI_SQUARED']
        elif 'SSR' in comp_dict:
            # Handle alternative field name (Sum of Squared Residuals)
            fit_dict['chi2'] = comp_dict['SSR']
        
        if 'CHI_SQUARED_NU' in comp_dict:
            fit_dict['chi2_nu'] = comp_dict['CHI_SQUARED_NU']
        elif 'SSR_NU' in comp_dict:
            # Handle alternative field name (SSR per degree of freedom)
            fit_dict['chi2_nu'] = comp_dict['SSR_NU']
        
        # Mode information (default to False if not present)
        fit_dict['is_velocity_mode'] = comp_dict.get('VELOCITY_MODE', False)
        
        # System redshift
        if 'SYSTEM_REDSHIFT' in comp_dict:
            fit_dict['z_sys'] = comp_dict['SYSTEM_REDSHIFT']
        
        # Covariance matrix (3x3 for Gaussian: amp, mean, stddev)
        cov_matrix = []
        for i in range(3):
            row = []
            for j in range(3):
                cov_key = f'COV_{i}_{j}'
                if cov_key in comp_dict:
                    row.append(comp_dict[cov_key])
            if row:
                cov_matrix.append(row)
        if cov_matrix and len(cov_matrix) == 3 and all(len(row) == 3 for row in cov_matrix):
            fit_dict['covariance'] = np.array(cov_matrix)
        
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
        elif 'MU' in comp_dict:
            # Handle alternative field name (older QSAP format)
            val, err = comp_dict['MU'] if isinstance(comp_dict['MU'], tuple) else (comp_dict['MU'], None)
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
        elif 'SSR' in comp_dict:
            # Handle alternative field name (Sum of Squared Residuals)
            fit_dict['chi2'] = comp_dict['SSR']
        
        if 'CHI_SQUARED_NU' in comp_dict:
            fit_dict['chi2_nu'] = comp_dict['CHI_SQUARED_NU']
        elif 'SSR_NU' in comp_dict:
            # Handle alternative field name (SSR per degree of freedom)
            fit_dict['chi2_nu'] = comp_dict['SSR_NU']
        
        # Mode information (default to False if not present)
        fit_dict['is_velocity_mode'] = comp_dict.get('VELOCITY_MODE', False)
        
        # System redshift
        if 'SYSTEM_REDSHIFT' in comp_dict:
            fit_dict['z_sys'] = comp_dict['SYSTEM_REDSHIFT']
        
        # Covariance matrix (4x4 for Voigt: amplitude, center, sigma, gamma)
        cov_matrix = []
        for i in range(4):
            row = []
            for j in range(4):
                cov_key = f'COV_{i}_{j}'
                if cov_key in comp_dict:
                    row.append(comp_dict[cov_key])
            if row:
                cov_matrix.append(row)
        if cov_matrix and len(cov_matrix) == 4 and all(len(row) == 4 for row in cov_matrix):
            fit_dict['covariance'] = np.array(cov_matrix)
        
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
            print(f"[DEBUG_POLY_LOAD] Loaded polynomial: order={fit_dict.get('poly_order')}, coeffs={coeffs}, bounds={fit_dict.get('bounds')}")
        
        return fit_dict if fit_dict else None
    
    def _parse_qsap_chebyshev_component(self, comp_dict):
        """Parse a Chebyshev polynomial component from QSAP format (for listfit)
        
        Chebyshev coefficients are stored in the rescaled [-1, 1] frame.
        Domain bounds (lam_min, lam_max) are essential for proper evaluation.
        """
        fit_dict = {}
        
        # Chebyshev degree
        if 'DEGREE' in comp_dict:
            fit_dict['degree'] = comp_dict['DEGREE']
        
        # Domain bounds (CRITICAL)
        if 'DOMAIN_MIN' in comp_dict and 'DOMAIN_MAX' in comp_dict:
            fit_dict['lam_min'] = comp_dict['DOMAIN_MIN']
            fit_dict['lam_max'] = comp_dict['DOMAIN_MAX']
        else:
            print(f"[WARNING] Chebyshev component missing domain bounds (DOMAIN_MIN/MAX) - cannot evaluate!")
            return None
        
        # Fit bounds (wavelength range)
        if 'BOUNDS_LOWER' in comp_dict and 'BOUNDS_UPPER' in comp_dict:
            fit_dict['bounds'] = (comp_dict['BOUNDS_LOWER'], comp_dict['BOUNDS_UPPER'])
        
        # Chebyshev coefficients (in [-1, 1] rescaled frame)
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
            print(f"[DEBUG_CHEB_LOAD] Loaded Chebyshev: degree={fit_dict.get('degree')}, domain=[{fit_dict['lam_min']:.2f}, {fit_dict['lam_max']:.2f}], coeffs={coeffs}, bounds={fit_dict.get('bounds')}")
        
        return fit_dict if fit_dict else None
    
    def _redraw_loaded_fits(self):
        """Redraw all loaded fit lines on the plot and register with tracker"""
        # Redraw Gaussian fits
        for idx, fit in enumerate(self.gaussian_fits):
            if fit.get('line') is None:  # Only if line hasn't been created yet
                try:
                    # For Single/Multi Gaussian mode: bounds come from user-selected fit range
                    # For Listfit mode: bounds come from the full listfit fitting range (should be in BOUNDS_LOWER/BOUNDS_UPPER)
                    # If bounds are missing (e.g., older .qsap files), calculate from profile parameters
                    if 'bounds' not in fit or fit['bounds'] is None:
                        mean = fit.get('mean', 0)
                        stddev = fit.get('stddev', 10)
                        fit['bounds'] = (mean - 5*stddev, mean + 5*stddev)
                        print(f"[WARNING] Gaussian bounds missing; calculated from mean/stddev: {fit['bounds']}")
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 2000)
                    
                    # Plot profile alone from y=0 (unified with Listfit convention)
                    # The continuum is displayed separately as its own line
                    y_plot = self.gaussian(x_plot, fit['amp'], fit['mean'], fit['stddev'])
                    gaussian_color = self.colors['profiles']['gaussian']
                    
                    # Add label only for the first gaussian
                    label = 'Gaussian' if 'gaussian' not in self.legend_profile_types else None
                    fit['line'], = self.ax.plot(x_plot, y_plot, color=gaussian_color['color'], linestyle=gaussian_color['linestyle'], label=label)
                    if label:
                        self.legend_profile_types.add('gaussian')
                    
                    # Register with item tracker - use saved name if available
                    # If this fit already has a fit_id from initial loading, use it; otherwise assign new one
                    name = fit.get('_tracker_name') or f"Gaussian (mu={fit['mean']:.1f}, sigma={fit['stddev']:.1f})"
                    existing_fit_id = fit.get('_fit_id') if '_fit_id' in fit else None
                    self.register_item('gaussian', name, fit_dict=fit, line_obj=fit['line'], 
                                     color=gaussian_color['color'], bounds=fit['bounds'], fit_id=existing_fit_id)
                except Exception as e:
                    print(f"Error redrawing Gaussian fit: {e}")
        
        # Redraw Voigt fits
        for idx, fit in enumerate(self.voigt_fits):
            if fit.get('line') is None:
                try:
                    # For Single/Multi Voigt mode: bounds come from user-selected fit range
                    # For Listfit mode: bounds come from the full listfit fitting range (should be in BOUNDS_LOWER/BOUNDS_UPPER)
                    # If bounds are missing (e.g., older .qsap files), calculate from profile parameters
                    if 'bounds' not in fit or fit['bounds'] is None:
                        center = fit.get('center', 0)
                        sigma = fit.get('sigma', 10)
                        fit['bounds'] = (center - 5*sigma, center + 5*sigma)
                        print(f"[WARNING] Voigt bounds missing; calculated from center/sigma: {fit['bounds']}")
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 2000)
                    
                    # Plot profile alone from y=0 (unified with Listfit convention)
                    # The continuum is displayed separately as its own line
                    y_plot = self.voigt(x_plot, fit['amp'], fit['center'], fit['sigma'], fit['gamma'])
                    voigt_color = self.colors['profiles']['voigt']
                    
                    # Add label only for the first voigt
                    label = 'Voigt' if 'voigt' not in self.legend_profile_types else None
                    fit['line'], = self.ax.plot(x_plot, y_plot, color=voigt_color['color'], linestyle=voigt_color['linestyle'], label=label)
                    if label:
                        self.legend_profile_types.add('voigt')
                    
                    # Register with item tracker - use saved name if available
                    name = fit.get('_tracker_name') or f"Voigt (c={fit['center']:.1f}, sigma={fit['sigma']:.1f}, gamma={fit['gamma']:.1f})"
                    existing_fit_id = fit.get('_fit_id') if '_fit_id' in fit else None
                    self.register_item('voigt', name, fit_dict=fit, line_obj=fit['line'],
                                     color=voigt_color['color'], bounds=fit['bounds'], fit_id=existing_fit_id)
                except Exception as e:
                    print(f"Error redrawing Voigt fit: {e}")
        
        # Redraw Continuum fits
        for idx, fit in enumerate(self.continuum_fits):
            if fit.get('line') is None:
                try:
                    # For Single/Multi Continuum mode: bounds come from user-selected fit range  
                    # For Listfit mode: bounds come from the full listfit fitting range (should be in BOUNDS_LOWER/BOUNDS_UPPER)
                    # If bounds are missing (e.g., older .qsap files), use full spectrum range
                    if 'bounds' not in fit or fit['bounds'] is None:
                        fit['bounds'] = (self.x_data.min(), self.x_data.max())
                        print(f"[WARNING] Continuum bounds missing; using full spectrum range: {fit['bounds']}")
                    x_plot = np.linspace(fit['bounds'][0], fit['bounds'][1], 2000)
                    
                    # Check Chebyshev FIRST (before checking for 'coeffs' in general)
                    if fit.get('type') == 'chebyshev' and 'coeffs' in fit:
                        # Chebyshev continuum with domain rescaling
                        print(f"[DEBUG_CONTINUUM] Redrawing Chebyshev continuum (idx={idx}):")
                        print(f"  domain=[{fit['lam_min']:.2f}, {fit['lam_max']:.2f}], bounds={fit['bounds']}")
                        print(f"  coeffs={fit['coeffs']}, len={len(fit['coeffs'])}")
                        
                        # Rescale wavelengths to [-1, 1] frame (CRITICAL for correct evaluation)
                        x_rescaled = 2 * (x_plot - fit['lam_min']) / (fit['lam_max'] - fit['lam_min']) - 1
                        # Evaluate Chebyshev polynomial at rescaled points
                        y_plot = np.polynomial.chebyshev.chebval(x_rescaled, fit['coeffs'])
                        print(f"  y_plot range: {y_plot.min():.6f} to {y_plot.max():.6f}")
                    elif 'coeffs' in fit:
                        # Polynomial continuum (default case)
                        print(f"[DEBUG_CONTINUUM] Redrawing continuum (idx={idx}):")
                        print(f"  poly_order={fit.get('poly_order')}, bounds={fit['bounds']}")
                        print(f"  coeffs={fit['coeffs']}, len={len(fit['coeffs'])}")
                        print(f"  x_plot range: {x_plot.min():.2f} to {x_plot.max():.2f}")
                        # Coefficients stored as [c_N, c_{N-1}, ..., c_1, c_0] (high-to-low order, as np.polyval expects)
                        y_plot = np.polyval(fit['coeffs'], x_plot)
                        print(f"  y_plot range: {y_plot.min():.6f} to {y_plot.max():.6f}")
                    else:
                        # Old format fallback for backward compatibility
                        print(f"[DEBUG_CONTINUUM] Redrawing continuum (idx={idx}): using old format (a,b)")
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
                    existing_fit_id = fit.get('_fit_id') if '_fit_id' in fit else None
                    self.register_item('continuum', name, fit_dict=fit, line_obj=fit['line'],
                                     position=position_str, color=continuum_color['color'], bounds=bounds,
                                     fit_id=existing_fit_id)
                except Exception as e:
                    print(f"Error redrawing continuum fit (idx={idx}): {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"[DEBUG] Skipping continuum (idx={idx}): line already exists")
        
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
                    x_plot = np.linspace(bounds[0], bounds[1], 2000)
                    
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
                        existing_fit_id = listfit.get('_fit_id') if '_fit_id' in listfit else None
                        self.register_item('listfit_total', name, fit_dict=listfit, line_obj=listfit_line,
                                         color=total_color['color'], bounds=bounds, fit_id=existing_fit_id)
            except Exception as e:
                print(f"Error redrawing listfit composite: {e}")
                import traceback
                traceback.print_exc()
    
    def _distribute_covariance_to_fits(self, covariance_data):
        """Distribute covariance submatrices from global covariance to individual fit components
        
        Args:
            covariance_data: Dict from load_covariance_from_qsap with:
                - free_parameter_names: list of all free parameter names
                - free_parameter_covariance: full NxN covariance matrix
                - component_registry: maps component IDs to parameter lists
        """
        if not covariance_data or 'free_parameter_covariance' not in covariance_data:
            return
        
        full_covariance = covariance_data['free_parameter_covariance']
        free_param_names = covariance_data['free_parameter_names']
        component_registry = covariance_data.get('component_registry', {})
        
        # Build mapping from parameter names to indices
        param_to_idx = {name: idx for idx, name in enumerate(free_param_names)}
        
        # Distribute covariance to each component type
        for comp_id, comp_info in component_registry.items():
            comp_type = comp_info['type']
            param_names = comp_info['params']
            
            # Get indices of this component's parameters in the full covariance matrix
            param_indices = []
            for pname in param_names:
                if pname in param_to_idx:
                    param_indices.append(param_to_idx[pname])
            
            if not param_indices:
                continue  # Skip if parameters not found
            
            # Extract covariance submatrix for this component
            if len(param_indices) > 0:
                cov_submatrix = full_covariance[np.ix_(param_indices, param_indices)]
                
                # Match component_id (like "g0", "v1", "p0") to the fit dictionaries
                # by looking for matching component_id stored during parsing
                found = False
                
                if comp_type == 'gaussian':
                    for fit_dict in self.gaussian_fits:
                        # Check if this fit's component_id matches (stored as 'component_id')
                        # or match by order if component_id not stored
                        if fit_dict.get('component_id') == comp_id or (
                            not found and comp_id in fit_dict.get('_component_id', '')
                        ):
                            fit_dict['covariance'] = cov_submatrix
                            print(f"[DEBUG] Distributed covariance to Gaussian {comp_id}: shape {cov_submatrix.shape}")
                            found = True
                            break
                
                elif comp_type == 'voigt':
                    for fit_dict in self.voigt_fits:
                        if fit_dict.get('component_id') == comp_id or (
                            not found and comp_id in fit_dict.get('_component_id', '')
                        ):
                            fit_dict['covariance'] = cov_submatrix
                            print(f"[DEBUG] Distributed covariance to Voigt {comp_id}: shape {cov_submatrix.shape}")
                            found = True
                            break
                
                elif comp_type == 'polynomial':
                    for fit_dict in self.continuum_fits:
                        if fit_dict.get('component_id') == comp_id or (
                            not found and comp_id in fit_dict.get('_component_id', '')
                        ):
                            fit_dict['covariance'] = cov_submatrix
                            print(f"[DEBUG] Distributed covariance to Continuum {comp_id}: shape {cov_submatrix.shape}")
                            found = True
                            break
                
                if not found and len(param_indices) > 0:
                    print(f"[WARNING] Could not match covariance for component {comp_id} ({comp_type})")
    
    def _update_panel_with_covariance(self, covariance_data=None):
        """Update Fit Diagnostics panel with covariance data after distribution
        
        After _distribute_covariance_to_fits() adds covariance to fit dicts,
        this method updates the panel to include the covariance for correlation matrix plotting.
        
        For Listfits, stores the full free-parameter covariance matrix.
        For single components, stores component-specific submatrices.
        
        Args:
            covariance_data: Full covariance data from QSAP (needed for Listfit full matrix)
        """
        if not hasattr(self, 'fit_diagnostics_panel') or not self.fit_diagnostics_panel:
            print("[DEBUG] Fit Diagnostics panel not available for covariance update")
            return
        
        print(f"[DEBUG] _update_panel_with_covariance called with covariance_data={covariance_data is not None}")
        print(f"[DEBUG] fit_diagnostics_panel exists: {self.fit_diagnostics_panel}")
        print(f"[DEBUG] fit_diagnostics_panel.fits_data exists: {hasattr(self.fit_diagnostics_panel, 'fits_data')}")
        if hasattr(self.fit_diagnostics_panel, 'fits_data'):
            print(f"[DEBUG] fit_diagnostics_panel.fits_data keys: {list(self.fit_diagnostics_panel.fits_data.keys())}")
        
        # Identify which fit_ids are Listfits (have components in multiple lists)
        listfit_fit_ids = set()
        for fit in self.gaussian_fits:
            if fit.get('is_listfit_component'):
                listfit_fit_ids.add(fit.get('_fit_id'))
                print(f"[DEBUG] Found Gaussian with is_listfit_component, fit_id={fit.get('_fit_id')}")
        for fit in self.voigt_fits:
            if fit.get('is_listfit_component'):
                listfit_fit_ids.add(fit.get('_fit_id'))
                print(f"[DEBUG] Found Voigt with is_listfit_component, fit_id={fit.get('_fit_id')}")
        for fit in self.continuum_fits:
            if fit.get('is_listfit_component'):
                listfit_fit_ids.add(fit.get('_fit_id'))
                print(f"[DEBUG] Found Continuum with is_listfit_component, fit_id={fit.get('_fit_id')}")
        
        print(f"[DEBUG] Identified Listfit fit_ids: {listfit_fit_ids}")
        
        # For Listfits, store the full free-parameter covariance
        if listfit_fit_ids and covariance_data and 'free_parameter_covariance' in covariance_data:
            full_covariance = covariance_data['free_parameter_covariance']
            free_param_names = covariance_data['free_parameter_names']
            
            print(f"[DEBUG] Have covariance_data with full_covariance shape {full_covariance.shape}, {len(free_param_names)} param names")
            
            for listfit_id in listfit_fit_ids:
                print(f"[DEBUG] Checking if listfit_id {listfit_id} in fits_data...")
                if listfit_id in self.fit_diagnostics_panel.fits_data:
                    print(f"[DEBUG] Storing full covariance for Listfit {listfit_id}: shape {full_covariance.shape}, params: {free_param_names}")
                    self.fit_diagnostics_panel.fits_data[listfit_id]['covariance'] = full_covariance
                    self.fit_diagnostics_panel.fits_data[listfit_id]['parameter_names'] = free_param_names
                else:
                    print(f"[DEBUG] WARNING: Listfit {listfit_id} NOT in fits_data. Available keys: {list(self.fit_diagnostics_panel.fits_data.keys())}")
        else:
            print(f"[DEBUG] Cannot store Listfit covariance: listfit_fit_ids={listfit_fit_ids}, has_covariance_data={covariance_data is not None}, has_matrix={covariance_data.get('free_parameter_covariance') if covariance_data else None}")
        
        # Update Gaussian fits with covariance (non-Listfit or component-level)
        for fit in self.gaussian_fits:
            fit_id = fit.get('_fit_id')
            # Skip if this is a Listfit component (already handled above)
            if fit.get('is_listfit_component'):
                continue
            
            if fit_id and fit.get('covariance') is not None:
                print(f"[DEBUG] Updating Gaussian fit {fit_id} with covariance shape {fit['covariance'].shape}")
                param_names = ['Amplitude', 'Mean', 'StdDev']
                self.fit_diagnostics_panel.fits_data[fit_id]['covariance'] = fit['covariance']
                self.fit_diagnostics_panel.fits_data[fit_id]['parameter_names'] = param_names
        
        # Update Voigt fits with covariance (non-Listfit or component-level)
        for fit in self.voigt_fits:
            fit_id = fit.get('_fit_id')
            # Skip if this is a Listfit component
            if fit.get('is_listfit_component'):
                continue
            
            if fit_id and fit.get('covariance') is not None:
                print(f"[DEBUG] Updating Voigt fit {fit_id} with covariance shape {fit['covariance'].shape}")
                param_names = ['Amplitude', 'Center', 'Sigma', 'Gamma']
                self.fit_diagnostics_panel.fits_data[fit_id]['covariance'] = fit['covariance']
                self.fit_diagnostics_panel.fits_data[fit_id]['parameter_names'] = param_names
        
        # Update Continuum fits with covariance (non-Listfit or component-level)
        for fit in self.continuum_fits:
            fit_id = fit.get('_fit_id')
            # Skip if this is a Listfit component
            if fit.get('is_listfit_component'):
                continue
            
            if fit_id and fit.get('covariance') is not None:
                print(f"[DEBUG] Updating Continuum fit {fit_id} with covariance shape {fit['covariance'].shape}")
                poly_order = fit.get('poly_order', 1)
                param_names = [f'c{i}' for i in range(poly_order + 1)]
                self.fit_diagnostics_panel.fits_data[fit_id]['covariance'] = fit['covariance']
                self.fit_diagnostics_panel.fits_data[fit_id]['parameter_names'] = param_names
        
        print(f"[DEBUG] _update_panel_with_covariance COMPLETE")
    
    def _update_diagnostics_for_loaded_fits(self):
        """Update Fit Diagnostics panel for all loaded fits"""
        if not hasattr(self, 'fit_diagnostics_panel') or not self.fit_diagnostics_panel:
            print("[DEBUG] Fit Diagnostics panel not available")
            return
        
        print(f"[DEBUG] _update_diagnostics_for_loaded_fits: gaussian_fits={len(self.gaussian_fits)}, voigt_fits={len(self.voigt_fits)}, continuum_fits={len(self.continuum_fits)}, listfit_fits={len(self.listfit_fits)}")
        
        # Update diagnostics for Gaussian fits
        for i, fit in enumerate(self.gaussian_fits):
            fit_id = fit.get('_fit_id')
            print(f"[DEBUG] Gaussian fit {i}: _fit_id={fit_id}, keys={list(fit.keys())}")
            if fit_id is None:
                print(f"[DEBUG] Skipping Gaussian fit {i} - no _fit_id")
                continue
            
            # Build diagnostics dict from loaded fit data
            diagnostics = {}
            if 'chi2_nu' in fit:
                diagnostics['chi2_reduced'] = fit['chi2_nu']
            elif 'chi2' in fit:
                diagnostics['chi2_reduced'] = fit['chi2']
            
            # Use covariance if available
            covariance = fit.get('covariance')
            parameter_names = None
            
            # Call update to add row to diagnostics table
            color = self.fit_colors.get(fit_id, '#000000')
            print(f"[DEBUG] Calling update_fit_diagnostics for Gaussian {fit_id} with diagnostics={diagnostics}")
            self.fit_diagnostics_panel.update_fit_diagnostics(
                fit_id, 'gaussian', diagnostics, color,
                covariance=covariance, parameter_names=parameter_names
            )
        
        # Update diagnostics for Voigt fits
        for i, fit in enumerate(self.voigt_fits):
            fit_id = fit.get('_fit_id')
            print(f"[DEBUG] Voigt fit {i}: _fit_id={fit_id}, keys={list(fit.keys())}")
            if fit_id is None:
                print(f"[DEBUG] Skipping Voigt fit {i} - no _fit_id")
                continue
            
            diagnostics = {}
            if 'chi2_nu' in fit:
                diagnostics['chi2_reduced'] = fit['chi2_nu']
            elif 'chi2' in fit:
                diagnostics['chi2_reduced'] = fit['chi2']
            
            covariance = fit.get('covariance')
            color = self.fit_colors.get(fit_id, '#000000')
            print(f"[DEBUG] Calling update_fit_diagnostics for Voigt {fit_id} with diagnostics={diagnostics}")
            self.fit_diagnostics_panel.update_fit_diagnostics(
                fit_id, 'voigt', diagnostics, color,
                covariance=covariance, parameter_names=None
            )
        
        # Update diagnostics for Continuum fits
        for i, fit in enumerate(self.continuum_fits):
            fit_id = fit.get('_fit_id')
            print(f"[DEBUG] Continuum fit {i}: _fit_id={fit_id}, keys={list(fit.keys())}")
            if fit_id is None:
                print(f"[DEBUG] Skipping Continuum fit {i} - no _fit_id")
                continue
            
            diagnostics = {}
            if 'chi2_nu' in fit:
                diagnostics['chi2_reduced'] = fit['chi2_nu']
            elif 'chi2' in fit:
                diagnostics['chi2_reduced'] = fit['chi2']
            
            covariance = fit.get('covariance')
            color = self.fit_colors.get(fit_id, '#000000')
            print(f"[DEBUG] Calling update_fit_diagnostics for Continuum {fit_id} with diagnostics={diagnostics}")
            self.fit_diagnostics_panel.update_fit_diagnostics(
                fit_id, 'continuum', diagnostics, color,
                covariance=covariance, parameter_names=None
            )
        
        # Update diagnostics for Listfit fits (stored quality metrics)
        for i, fit in enumerate(self.listfit_fits):
            fit_id = fit.get('_fit_id')
            print(f"[DEBUG] Listfit fit {i}: _fit_id={fit_id}")
            if fit_id is None:
                print(f"[DEBUG] Skipping Listfit fit {i} - no _fit_id")
                continue
            
            # Extract quality metrics from listfit
            quality_metrics = fit.get('quality_metrics', {})
            diagnostics = {}
            
            if 'chisqr_red' in quality_metrics:
                diagnostics['chi2_reduced'] = quality_metrics['chisqr_red']
            elif 'chisqr' in quality_metrics:
                diagnostics['chi2_reduced'] = quality_metrics['chisqr']
            
            if 'r_squared' in quality_metrics:
                diagnostics['r_squared'] = quality_metrics['r_squared']
            
            if 'akaike' in quality_metrics:
                diagnostics['akaike'] = quality_metrics['akaike']
            
            if 'bayesian' in quality_metrics:
                diagnostics['bayesian'] = quality_metrics['bayesian']
            
            color = self.fit_colors.get(fit_id, '#000000')
            self.fit_diagnostics_panel.update_fit_diagnostics(
                fit_id, 'listfit', diagnostics, color,
                covariance=None, parameter_names=None
            )
    
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
        # **Important**: Clear Item Tracker FIRST so its deletion signals trigger handlers properly.
        # The handlers will remove items from internal lists (gaussian_fits, voigt_fits, etc.)
        # Do NOT manually clear fit lists before calling item_tracker.clear_all()
        # because the handlers need access to item_id_map to find item information
        self.item_tracker.clear_all()
        
        # Now that all Item Tracker items have been deleted via signals,
        # clear the internal fit lists and related data structures
        self.gaussian_fits.clear()
        self.voigt_fits.clear()
        self.continuum_fits.clear()
        self.listfit_fits.clear()
        self.continuum_patches.clear()
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
        import sys
        try:
            # Show goodbye message in terminal
            message = "Quitting QSAP. Bye!"
            box_width = len(message)
            print("\n" + "╔" + "═" * (box_width) + "╗")
            print("║" + message + "║")
            print("╚" + "═" * (box_width) + "╝\n")
            sys.stdout.flush()
            
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
            
            # Exit cleanly without triggering crash handlers
            sys.exit(0)
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
            # Make toolbar icons half as big (default is 24x24, set to 12x12)
            from PyQt5.QtCore import QSize
            self.toolbar.setIconSize(QSize(16, 16))
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
            
            # Create tab widget for dockable Control Panel + Item Tracker + Settings + Smoothing + Fit Diagnostics
            top_tab_widget = QtWidgets.QTabWidget()
            top_tab_widget.addTab(self.control_panel_container, "Control Panel")
            top_tab_widget.addTab(self.item_tracker, "Item Tracker")
            top_tab_widget.addTab(self.fit_diagnostics_panel, "Fit Diagnostics")
            top_tab_widget.addTab(self.settings_container, "Settings")
            top_tab_widget.addTab(self.smoothing_container, "Smoothing")
            
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
            self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
            self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
            
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
        
        # Set initial plot style to match the visible plot (step plot is displayed first)
        self.is_step_plot = True
        
        # Plot error spectrum using the new display mode system
        if self.err is not None:
            self.redraw_error_spectrum()
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
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
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
        
        # ===== DISPLAY SECTION TITLE =====
        display_title = QtWidgets.QLabel("Display")
        display_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #0078d4;")
        main_layout.addWidget(display_title)
        
        display_layout = QtWidgets.QVBoxLayout()
        display_layout.setContentsMargins(15, 5, 10, 10)
        display_layout.setSpacing(8)
        
        # --- ERROR SPECTRUM SUBSECTION ---
        error_spectrum_label = QtWidgets.QLabel("Error Spectrum")
        error_spectrum_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        display_layout.addWidget(error_spectrum_label)
        
        error_spectrum_group = QtWidgets.QGroupBox()
        error_spectrum_group.setStyleSheet("QGroupBox { border: none; margin: 0px; padding: 0px; }")
        error_spectrum_group_layout = QtWidgets.QVBoxLayout()
        error_spectrum_group_layout.setContentsMargins(10, 5, 10, 5)
        error_spectrum_group_layout.setSpacing(4)
        
        # Radio button group for error spectrum display
        self.error_spectrum_default_radio = QtWidgets.QRadioButton("Default (red dashed line)")
        self.error_spectrum_default_radio.setChecked(True)
        self.error_spectrum_default_radio.toggled.connect(self.on_error_spectrum_mode_changed)
        error_spectrum_group_layout.addWidget(self.error_spectrum_default_radio)
        
        self.error_spectrum_shaded_radio = QtWidgets.QRadioButton("Shaded (gray band ±error)")
        self.error_spectrum_shaded_radio.toggled.connect(self.on_error_spectrum_mode_changed)
        error_spectrum_group_layout.addWidget(self.error_spectrum_shaded_radio)
        
        error_spectrum_group.setLayout(error_spectrum_group_layout)
        display_layout.addWidget(error_spectrum_group)
        
        # --- RESIDUAL SUBSECTION ---
        residual_label = QtWidgets.QLabel("Residual")
        residual_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        display_layout.addWidget(residual_label)
        
        residual_group = QtWidgets.QGroupBox()
        residual_group.setStyleSheet("QGroupBox { border: none; margin: 0px; padding: 0px; }")
        residual_group_layout = QtWidgets.QVBoxLayout()
        residual_group_layout.setContentsMargins(10, 5, 10, 5)
        residual_group_layout.setSpacing(4)
        
        # Radio button group for residual display
        self.residual_none_radio = QtWidgets.QRadioButton("None (default behavior)")
        self.residual_none_radio.setChecked(True)
        self.residual_none_radio.toggled.connect(self.on_residual_display_mode_changed)
        residual_group_layout.addWidget(self.residual_none_radio)
        
        self.residual_sigma_radio = QtWidgets.QRadioButton("Sigma (residual / error)")
        self.residual_sigma_radio.toggled.connect(self.on_residual_display_mode_changed)
        residual_group_layout.addWidget(self.residual_sigma_radio)
        
        self.residual_shaded_radio = QtWidgets.QRadioButton("Shaded")
        self.residual_shaded_radio.toggled.connect(self.on_residual_display_mode_changed)
        residual_group_layout.addWidget(self.residual_shaded_radio)
        
        residual_group.setLayout(residual_group_layout)
        display_layout.addWidget(residual_group)
        
        main_layout.addLayout(display_layout)
        
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
            self.current_continuum_fit_id = None  # Clear fit tracking
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
            self.current_continuum_fit_id = self.next_fit_id()  # Assign new fit_id for this session
            self.assign_fit_color(self.current_continuum_fit_id)
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
            print("            Or press ENTER/RETURN to use the full spectral range.")
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
        # Convert curve_fit covariance to lmfit-like format
        pcov_from_dict = continuum_fit.get('covariance')
        if pcov_from_dict is not None:
            param_names = [f'p0_c{i}' for i in range(len(coeffs))]
            param_values = list(coeffs)
            mock_result = self._convert_curve_fit_to_lmfit_like(param_names, param_values, pcov_from_dict)
            self.save_and_print_qsap_fit(continuum_fit, 'Continuum', 'Single', lmfit_result=mock_result)
        else:
            self.save_and_print_qsap_fit(continuum_fit, 'Continuum', 'Single')
        
        # Clean up and deactivate
        self.continuum_regions = []
        self.continuum_patches = []
        self.continuum_mode = False
        self.current_continuum_fit_id = None  # Clear fit tracking
        
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
            self.current_continuum_fit_id = None  # Clear fit tracking
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
        
        # Deactivate guess drawing mode
        if self.guess_drawing_mode:
            self._cancel_guess()
            modes_deactivated.append("Guess Drawing")
        
        # Deactivate constraint bounds setting mode
        if self.constraint_bounds_mode:
            self.constraint_bounds_mode = False
            self.constraint_parameter = None
            self.constraint_bounds_drag_start_x = None
            self.constraint_bounds_drag_end_x = None
            if self.constraint_bounds_preview_line is not None:
                try:
                    self.constraint_bounds_preview_line.remove()
                except (ValueError, RuntimeError):
                    pass
            self.constraint_bounds_preview_line = None
            self.current_constraint_editor = None
            self.current_constraint_editor_dialog = None
            modes_deactivated.append("Constraint Bounds Setting")
        
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
            
            # Store full pcov for later use when saving to .qsap
            self._multi_gaussian_full_pcov = pcov
            self._multi_gaussian_param_names = []
            
            for i in range(0, len(params), 3):
                amp, mean, stddev = params[i:i+3]
                amp_err, mean_err, stddev_err = perr[i:i+3]
                x_fit = self.x_data[(self.x_data >= bound_pairs[i // 3][0]) & (self.x_data <= bound_pairs[i // 3][1])]
                y_fit_plot = self.gaussian(x_fit, amp, mean, stddev)  # For display (without continuum offset)
                continuum_sub_data = comp_ys[i // 3] - continuum_ys[i // 3]
                residuals = continuum_sub_data - self.gaussian(x_fit, amp, mean, stddev)
                # Calculate chi2 (simpler without errors since they're concatenated)
                chi2 = np.sum(residuals ** 2)  # Chi2 without decomposed errors
                chi2_nu = chi2 / (len(x_fit) - 3)  # 3 params per component
                
                # Extract this component's covariance from the full covariance matrix
                comp_cov_indices = [i, i+1, i+2]
                comp_cov = pcov[np.ix_(comp_cov_indices, comp_cov_indices)]
                
                # Track parameter names for covariance storage
                g_idx = i // 3
                self._multi_gaussian_param_names.extend([f'g{g_idx}_amp', f'g{g_idx}_mu', f'g{g_idx}_sigma'])
                
                # DEBUG: Verify covariance structure
                print(f"[DEBUG] Multi-Gaussian component {self.component_id}: comp_cov shape = {comp_cov.shape}, has_data = {comp_cov is not None}")
                
                # Lazy import of scipy for interpolation
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                
                # Convert full curve_fit covariance to lmfit-like format
                if hasattr(self, '_multi_gaussian_full_pcov') and self._multi_gaussian_full_pcov is not None:
                    param_names = self._multi_gaussian_param_names if hasattr(self, '_multi_gaussian_param_names') else []
                    param_values = list(params)  # All parameter values from curve_fit
                    if param_names and len(param_names) == len(param_values):
                        mock_result = self._convert_curve_fit_to_lmfit_like(param_names, param_values, self._multi_gaussian_full_pcov)
                        self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian', lmfit_result=mock_result)
                    else:
                        self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian')
                else:
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
            # Redraw residual panel to recalculate y-limits for new x-range
            self.redraw_residual_panel()
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

    # Smooth the spectrum with optional median and/or Gaussian filtering
    def smooth_spectrum(self, median_kernel=1, gaussian_sigma=0.0):
        """Apply median and/or Gaussian smoothing to spectrum.
        
        Args:
            median_kernel: Median filter kernel size (odd integer, 1=no smoothing)
            gaussian_sigma: Gaussian filter sigma (0=no smoothing)
        """
        from scipy.ndimage import gaussian_filter1d, median_filter
        
        if self.original_spec is None:
            print("Error: original_spec is not defined.")
            return False
        
        # Start with original spectrum
        result_spec = np.copy(self.original_spec)
        
        # Apply median filter if kernel > 1
        if median_kernel > 1:
            # Ensure kernel is odd
            if median_kernel % 2 == 0:
                median_kernel += 1
            result_spec = median_filter(result_spec, size=median_kernel)
        
        # Apply Gaussian filter if sigma > 0
        if gaussian_sigma > 0:
            result_spec = gaussian_filter1d(result_spec, sigma=gaussian_sigma)
        
        if median_kernel > 1 or gaussian_sigma > 0:
            self.smoothed_spec = result_spec
            return True
        else:
            self.smoothed_spec = result_spec
            return False

    def _update_spectrum_display(self, spec_data):
        """Update the plotted spectrum with new data (smoothed or original)
        
        NOTE: This method preserves the current plot style (step vs line).
        The plot style is ONLY changed by:
        - Pressing '~' key
        - Clicking 'Toggle Step/Line Plot' button
        - It is NOT changed by smoothing operations
        """
        if self.spectrum_line is None:
            return
        
        # Preserve current plot style
        saved_plot_style = self.is_step_plot
        
        # Hide old lines
        self.step_spec.set_visible(False)
        self.line_spec.set_visible(False)
        self.spectrum_line.set_visible(False)
        self.fig.canvas.draw_idle()
        
        # Create new step and line plots
        self.step_spec, = self.ax.step(self.x_data, spec_data, color='black', where='mid', zorder=0)
        self.line_spec, = self.ax.plot(self.x_data, spec_data, color='black', visible=False, zorder=0)
        
        # Restore the saved plot style (do NOT change the style during display update)
        self.is_step_plot = saved_plot_style
        
        # Update current spectrum_line reference based on preserved plot style
        if self.is_step_plot:
            self.step_spec.set_visible(True)
            self.line_spec.set_visible(False)
            self.spectrum_line = self.step_spec
        else:
            self.step_spec.set_visible(False)
            self.line_spec.set_visible(True)
            self.spectrum_line = self.line_spec
        
        self.fig.canvas.draw_idle()

    def on_smoothing_apply(self):
        """Apply smoothing based on current input values"""
        try:
            median_kernel = int(self.smoothing_median_input.text())
            gaussian_sigma = float(self.smoothing_gaussian_input.text())
        except ValueError:
            print("Invalid smoothing input values. Please enter integers for Median and floats for Gaussian.")
            return
        
        # Validate ranges
        if median_kernel < 1:
            median_kernel = 1
            self.smoothing_median_input.setText("1")
        if gaussian_sigma < 0:
            gaussian_sigma = 0.0
            self.smoothing_gaussian_input.setText("0.0")
        
        # Store previous values before applying new ones
        self.smoothing_prev_median = self.last_applied_median
        self.smoothing_prev_gaussian = self.last_applied_gaussian
        
        # Apply smoothing
        if self.smooth_spectrum(median_kernel, gaussian_sigma):
            self._update_spectrum_display(self.smoothed_spec)
            self.last_applied_median = median_kernel
            self.last_applied_gaussian = gaussian_sigma

    def on_smoothing_reset(self):
        """Reset to previous smoothing values"""
        self.smoothing_median_input.setText(str(self.smoothing_prev_median))
        self.smoothing_gaussian_input.setText(str(self.smoothing_prev_gaussian))
        self.on_smoothing_apply()

    def on_smoothing_original(self):
        """Reset to original unsmoothed spectrum and return joystick to center"""
        self.smoothing_median_input.setText("1")
        self.smoothing_gaussian_input.setText("0.0")
        self.smoothing_prev_median = self.last_applied_median
        self.smoothing_prev_gaussian = self.last_applied_gaussian
        self._update_spectrum_display(self.original_spec)
        self.last_applied_median = 1
        self.last_applied_gaussian = 0.0
        # Reset joystick to center
        self.smoothing_joystick.x_pos = 0.5
        self.smoothing_joystick.y_pos = 0.5
        self.smoothing_joystick.update()
        print("Restored original spectrum")

    def on_toggle_step_line(self):
        """Toggle between step and line plot"""
        self.is_step_plot = not self.is_step_plot
        self.step_spec.set_visible(self.is_step_plot)
        self.line_spec.set_visible(not self.is_step_plot)
        
        # Redraw error spectrum for both Default and Shaded modes
        # (Shaded mode needs to be cleared and redrawn to maintain correct styling)
        if self.err is not None:
            self.redraw_error_spectrum()
        
        self.spectrum_line = self.step_spec if self.is_step_plot else self.line_spec
        
        # Redraw residual panel if visible (to use correct step/line style)
        if self.is_residual_shown:
            self.redraw_residual_panel()
        
        self.fig.canvas.draw_idle()
        print("Plot style toggled:", "Step plot" if self.is_step_plot else "Line plot")

    def on_smoothing_interactive_mode_changed(self, state):
        """Handle toggling of interactive smoothing mode"""
        self.smoothing_interactive_mode = (state == QtCore.Qt.Checked)
        if self.smoothing_interactive_mode:
            print("Interactive smoothing mode ENABLED")
            print("  Up-Down drag: adjust Gaussian sigma")
            print("  Left-Right drag: adjust Median kernel")
        else:
            print("Interactive smoothing mode DISABLED")

    def on_joystick_x_moved(self, x_pos):
        """Handle joystick X movement (left-right) - controls Median kernel"""
        # x_pos is 0-1, centered at 0.5
        # Symmetric mapping: center (0.5) = 1, edges (0 or 1) = 15
        min_kernel = 1
        max_kernel = 15
        
        # Only update if not at center (deadzone)
        if abs(x_pos - 0.5) > 0.05:
            # Distance from center (0 to 0.5)
            distance = abs(x_pos - 0.5)
            # Map distance to kernel value (symmetric, both sides increase)
            kernel_val = min_kernel + (distance / 0.5) * (max_kernel - min_kernel)
            # Make it odd (required for median)
            kernel_val = int(kernel_val) if int(kernel_val) % 2 == 1 else int(kernel_val) + 1
            kernel_val = max(min_kernel, min(max_kernel, kernel_val))
            
            self.smoothing_median_input.setText(str(kernel_val))
            self.on_smoothing_apply()
        else:
            # At center (deadzone) = no smoothing
            self.smoothing_median_input.setText("1")
            self.on_smoothing_apply()

    def on_joystick_y_moved(self, y_pos):
        """Handle joystick Y movement (up-down) - controls Gaussian sigma"""
        # y_pos is 0-1, centered at 0.5
        # Symmetric mapping: center (0.5) = 0.0, edges (0 or 1) = 5.0
        min_sigma = 0.0
        max_sigma = 5.0
        
        # Only update if not at center (deadzone)
        if abs(y_pos - 0.5) > 0.05:
            # Distance from center (0 to 0.5)
            distance = abs(y_pos - 0.5)
            # Map distance to sigma value (symmetric, both sides increase)
            sigma_val = min_sigma + (distance / 0.5) * (max_sigma - min_sigma)
            sigma_val = max(min_sigma, min(max_sigma, sigma_val))
            
            self.smoothing_gaussian_input.setText(f"{sigma_val:.2f}")
            self.on_smoothing_apply()
        else:
            # At center (deadzone) = no smoothing
            self.smoothing_gaussian_input.setText("0.0")
            self.on_smoothing_apply()

    # Check if there is an existing fitted continuum covering the current bounds
    def get_existing_continuum(self, left_bound, right_bound):
        try:
            for continuum_fit in self.continuum_fits:
                if continuum_fit['bounds'][0] <= left_bound and continuum_fit['bounds'][1] >= right_bound:
                    x_range = self.x_data[(self.x_data >= left_bound) & (self.x_data <= right_bound)]
                    if 'coeffs' in continuum_fit:
                        # New format with polynomial coefficients
                        # Coefficients stored as [c_N, c_{N-1}, ..., c_1, c_0] (high-to-low order, as np.polyval expects)
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
    
    def get_continuum_fit_dict(self, left_bound, right_bound):
        """Get the continuum fit dictionary for a given bound region
        
        Returns the full continuum fit dict (with coeffs, covariance, etc.)
        if one exists that encompasses the bounds, None otherwise
        """
        try:
            for continuum_fit in self.continuum_fits:
                if continuum_fit['bounds'][0] <= left_bound and continuum_fit['bounds'][1] >= right_bound:
                    return continuum_fit
            return None
        except (ValueError, KeyError):
            return None
    
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
            ew = (self.rest_wavelength/c_in_km_per_s) * trapz_compat(1 - (profile_values + continuum_values) / continuum_values, x_values)
        else:
            ew = trapz_compat(1 - (profile_values + continuum_values) / continuum_values, x_values)
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
        ew = trapz_compat(1 - (model_y + cont_y) / cont_y, comp_x)
        ew_r = ew / (1 + redshift)
        return ew, ew_r

    def calculate_and_plot_residuals(self):
        """Legacy method - delegates to redraw_residual_panel for consistent plot styling"""
        self.residuals = self.calculate_residuals()
        # Use the new redraw method to ensure consistent styling with plot mode
        self.redraw_residual_panel()
    
        
    def toggle_residual_panel(self):
        """Toggle residual panel on/off. When shown, spectrum shrinks to make room."""
        if not self.is_residual_shown:
            # Show residual panel
            # Shrink main spectrum plot to make room for residual below it
            # Original position: left=0.125, bottom=0.1, width=0.775, height=0.8
            # Split: spectrum gets 60% height at top, residual gets 20% height at bottom, with NO gap between them
            
            spectrum_position = [0.125, 0.3, 0.775, 0.6]   # Spectrum: top 60% (from 0.3 to 0.9)
            residual_position = [0.125, 0.1, 0.775, 0.2]   # Residual: bottom 20% (from 0.1 to 0.3)
            
            # Resize spectrum plot
            self.ax.set_position(spectrum_position)
            
            # Create residual panel only if it doesn't exist
            if self.residual_ax is None:
                self.residual_ax = self.fig.add_axes(residual_position)
            else:
                # Resize existing residual axes
                self.residual_ax.set_position(residual_position)
            
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
            print("Residual panel shown")
        else:
            # Hide residual panel
            # Restore spectrum plot to full size
            spectrum_position = [0.125, 0.1, 0.775, 0.8]  # Full height
            self.ax.set_position(spectrum_position)
            
            # Hide residual axes
            if self.residual_ax is not None:
                self.residual_ax.clear()
                self.residual_ax.set_visible(False)
            
            self.is_residual_shown = False
            
            # Restore x-ticks to main plot
            self.ax.set_xticks(self.ax.get_xticks())
            print("Residual panel hidden")

        self.fig.canvas.draw_idle()  # Refresh plot to show/hide residual panel

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
                
                # Handle Chebyshev vs polynomial
                if continuum_fit.get('type') == 'chebyshev':
                    # CHEBYSHEV: Use chebval with domain rescaling
                    lam_min = continuum_fit.get('lam_min', left_bound)
                    lam_max = continuum_fit.get('lam_max', right_bound)
                    x_rescaled = 2 * (comp_x - lam_min) / (lam_max - lam_min) - 1
                    continuum_sum[mask] += np.polynomial.chebyshev.chebval(x_rescaled, coeffs)
                else:
                    # POLYNOMIAL: Coefficients stored as [c_N, ..., c_1, c_0] (high-to-low), use directly with polyval
                    continuum_sum[mask] += np.polyval(coeffs, comp_x)

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
                
                # Add polynomial and Chebyshev components from the listfit (already filtered - deleted ones removed)
                for comp in components:
                    if comp['type'] == 'polynomial':
                        order = comp.get('order', 1)
                        # Get stored coefficients from component (precomputed during fit)
                        poly_coeffs = comp.get('coeffs', [])
                        if poly_coeffs:
                            # Coefficients stored as [c_N, ..., c_1, c_0] (high-to-low), use directly with polyval
                            y_poly = np.polyval(poly_coeffs, comp_x)
                            listfit_poly_sum[mask] += y_poly
                    elif comp['type'] == 'chebyshev':
                        # CHEBYSHEV: Use chebval with domain rescaling
                        degree = comp.get('degree', 1)
                        cheb_coeffs = comp.get('coeffs', [])
                        if cheb_coeffs:
                            lam_min = comp.get('lam_min', left_bound)
                            lam_max = comp.get('lam_max', right_bound)
                            x_rescaled = 2 * (comp_x - lam_min) / (lam_max - lam_min) - 1
                            y_cheb = np.polynomial.chebyshev.chebval(x_rescaled, cheb_coeffs)
                            listfit_poly_sum[mask] += y_cheb

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
    def gaussian(self, x, amp, mu, sigma):
        y = amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        # return y
        return self.apply_lsf(y)

    def multi_gaussian(self, x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, mu, sigma = params[i:i+3]
            y += self.gaussian(x, amp, mu, sigma)
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

    # ============================================================================
    # Fit Diagnostics & Management Methods
    # ============================================================================
    
    def next_fit_id(self):
        """Increment and return next unique fit ID"""
        self.fit_counter += 1
        return self.fit_counter
    
    def assign_fit_color(self, fit_id):
        """Assign a unique color to a fit (cycles through palette)"""
        if fit_id not in self.fit_colors:
            color_idx = (fit_id - 1) % len(self.FIT_COLORS)
            self.fit_colors[fit_id] = self.FIT_COLORS[color_idx]
        return self.fit_colors[fit_id]
    
    def _register_loaded_fit_metadata(self, fit_id, fit_type_str, quality_metrics):
        """Register metadata for a loaded fit into Fit Diagnostics
        
        Args:
            fit_id: Unique fit identifier
            fit_type_str: String describing fit type ('Single Gaussian', 'Listfit', etc.)
            quality_metrics: Dict with keys chi2, chi2_reduced, r_squared, akaike, bayesian, n_params, n_data
        """
        # Extract from quality metrics (from FIT_DIAGNOSTICS in QSAP)
        chi2 = quality_metrics.get('chi2')
        chi2_reduced = quality_metrics.get('chi2_reduced')
        r_squared = quality_metrics.get('r_squared')
        akaike = quality_metrics.get('akaike')
        bayesian = quality_metrics.get('bayesian')
        n_params = quality_metrics.get('n_params')
        n_data = quality_metrics.get('n_data')
        
        # If n_data not from diagnostics, use spectrum length
        if n_data is None:
            n_data = len(self.wav) if hasattr(self, 'wav') and len(self.wav) > 0 else None
        
        print(f"[DEBUG] _register_loaded_fit_metadata: fit_id={fit_id}, type={fit_type_str}, n_params={n_params}, n_data={n_data}")
        print(f"[DEBUG]   chi2={chi2}, chi2_reduced={chi2_reduced}, r_squared={r_squared}, akaike={akaike}, bayesian={bayesian}")
        
        # Build diagnostics dict with what we have
        diagnostics_dict = {
            'r_squared': r_squared,
            'chi2_reduced': chi2_reduced,
            'akaike': akaike,
            'bayesian': bayesian,
            'n_params': n_params,
            'n_data': n_data,
            'covariance': None,  # Will be populated later by _distribute_covariance_to_fits
        }
        
        # Call register_fit_metadata to populate fit_metadata and Fit Diagnostics panel
        self.register_fit_metadata(fit_id, fit_type_str, diagnostics_dict)
    
    def register_fit_metadata(self, fit_id, fit_type, diagnostics_dict):
        """Store fit metadata for Fit Diagnostics tab
        
        Args:
            fit_id: Unique fit identifier
            fit_type: String describing fit type ('Single Gaussian', 'Multi-Gaussian', 'Listfit', etc.)
            diagnostics_dict: Dict with keys:
                - r_squared: R² goodness of fit
                - chi2_reduced: Reduced χ²
                - akaike: AIC value
                - bayesian: BIC value
                - n_params: Number of free parameters
                - n_data: Number of data points
                - covariance: Covariance matrix (for condition number calculation)
        """
        from datetime import datetime
        
        print(f"[DEBUG] register_fit_metadata called: fit_id={fit_id}, fit_type={fit_type}")
        print(f"[DEBUG] fit_diagnostics_panel exists: {hasattr(self, 'fit_diagnostics_panel')}")
        if hasattr(self, 'fit_diagnostics_panel'):
            print(f"[DEBUG] fit_diagnostics_panel value: {self.fit_diagnostics_panel}")
        
        # Calculate condition number if covariance available
        cond_num = None
        if diagnostics_dict.get('covariance') is not None:
            try:
                cov_array = np.array(diagnostics_dict['covariance'], dtype=float)
                if cov_array.size > 0:
                    eigenvals = np.linalg.eigvalsh(cov_array)
                    eigenvals = eigenvals[eigenvals > 1e-15]  # Filter near-zero eigenvalues
                    if len(eigenvals) > 0:
                        cond_num = np.max(eigenvals) / (np.min(eigenvals) + 1e-30)
            except (np.linalg.LinAlgError, ValueError):
                cond_num = None
        
        # Determine flag (Green/Yellow/Red)
        flag = self._compute_fit_flag(
            diagnostics_dict.get('r_squared'),
            diagnostics_dict.get('chi2_reduced'),
            cond_num,
            diagnostics_dict.get('n_params'),
            diagnostics_dict.get('n_data')
        )
        
        self.fit_metadata[fit_id] = {
            'type': fit_type,
            'r_squared': diagnostics_dict.get('r_squared'),
            'chi2_reduced': diagnostics_dict.get('chi2_reduced'),
            'akaike': diagnostics_dict.get('akaike'),
            'bayesian': diagnostics_dict.get('bayesian'),
            'n_params': diagnostics_dict.get('n_params'),
            'n_data': diagnostics_dict.get('n_data'),
            'condition_num': cond_num,
            'flag': flag,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Assign color if not already assigned
        if fit_id not in self.fit_colors:
            self.assign_fit_color(fit_id)
        
        # Update Fit Diagnostics panel
        diagnostics_for_panel = {
            'r_squared': diagnostics_dict.get('r_squared'),
            'chi2_reduced': diagnostics_dict.get('chi2_reduced'),
            'akaike': diagnostics_dict.get('akaike'),
            'bayesian': diagnostics_dict.get('bayesian'),
            'n_params': diagnostics_dict.get('n_params'),
            'n_data': diagnostics_dict.get('n_data'),
            'condition_num': cond_num,
            'flag': flag,
        }
        
        # Extract covariance and parameter names for visualization
        covariance = diagnostics_dict.get('covariance')
        parameter_names = diagnostics_dict.get('parameter_names')
        
        print(f"[DEBUG] About to call update_fit_diagnostics with:")
        print(f"[DEBUG]   fit_id={fit_id}")
        print(f"[DEBUG]   fit_type={fit_type}")
        print(f"[DEBUG]   diagnostics_for_panel keys={list(diagnostics_for_panel.keys())}")
        print(f"[DEBUG]   fit_colors[fit_id]={self.fit_colors.get(fit_id, 'NOT FOUND')}")
        
        self.fit_diagnostics_panel.update_fit_diagnostics(
            fit_id, fit_type, diagnostics_for_panel, self.fit_colors[fit_id],
            covariance=covariance, parameter_names=parameter_names
        )
        
        print(f"[DEBUG] update_fit_diagnostics call completed")
        print(f"[DEBUG] fit_diagnostics_panel.fits_data now contains: {list(self.fit_diagnostics_panel.fits_data.keys())}")

    
    def _compute_fit_flag(self, r2, chi2_red, cond_num, n_params, n_data):
        """Compute diagnostic flag (Green/Yellow/Red) based on fit metrics"""
        flags = []
        
        # R² check
        if r2 is not None:
            if r2 < 0.5:
                flags.append('Red')  # Poor fit
            elif r2 < 0.8:
                flags.append('Yellow')  # Marginal
            # else Green (good fit)
        
        # Condition number check (indicates ill-conditioning)
        if cond_num is not None and cond_num > 100:
            flags.append('Yellow' if len(flags) == 0 else ('Red' if 'Red' in flags else 'Yellow'))
        
        # Degrees of freedom check
        if n_params is not None and n_data is not None:
            dof = n_data - n_params
            if dof < 1:
                flags.append('Red')  # Insufficient degrees of freedom
        
        # Return most severe flag, default to Green
        if 'Red' in flags:
            return '🔴 Red'
        elif 'Yellow' in flags:
            return '🟡 Yellow'
        else:
            return '🟢 Green'
    
    def add_fit_item(self, fit_id, item_id):
        """Track that an item belongs to a fit"""
        if fit_id not in self.fit_items:
            self.fit_items[fit_id] = set()
        self.fit_items[fit_id].add(item_id)
    
    def remove_fit_if_empty(self, fit_id):
        """Remove fit from Fit Diagnostics if it has no items left"""
        if fit_id in self.fit_items:
            if len(self.fit_items[fit_id]) == 0:
                # Remove from tracking dicts
                self.fit_items.pop(fit_id, None)
                self.fit_metadata.pop(fit_id, None)
                self.fit_colors.pop(fit_id, None)
                # Notify Fit Diagnostics panel to remove the fit
                if hasattr(self, 'fit_diagnostics_panel') and self.fit_diagnostics_panel:
                    self.fit_diagnostics_panel.remove_fit(fit_id)

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
            # Check if this line list should be updated with redshift (default: True for backward compatibility)
            apply_redshift = linelist_info.get('apply_redshift', True)
            
            for line in linelist.lines:
                # Apply redshift to the line wavelength only if apply_redshift is True
                if apply_redshift:
                    shifted_wl = line.wave * (1 + self.redshift)
                else:
                    # Use rest-frame wavelength without redshift
                    shifted_wl = line.wave
                
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
        
    def register_item(self, item_type, name, fit_dict=None, line_obj=None, patch_obj=None, position='', color='gray', bounds=None, fit_id=None):
        """Register an item with the tracker
        
        Args:
            item_type: Type of item ('gaussian', 'voigt', 'polynomial', 'marker', etc.)
            name: Display name
            fit_dict: Associated fit data dictionary
            line_obj: Line object reference
            patch_obj: Patch object reference
            position: Position/bounds description
            color: Line/patch color
            bounds: Wavelength bounds
            fit_id: Optional fit ID to group items. If None, creates new fit.
        """
        item_id = f"{item_type}_{self.item_id_counter}"
        self.item_id_counter += 1
        
        # Assign or create fit_id
        if fit_id is None:
            fit_id = self.next_fit_id()
        self.add_fit_item(fit_id, item_id)
        
        # Get fit color
        fit_color = self.assign_fit_color(fit_id)
        
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
            'original_linewidth': original_linewidth,
            'original_zorder': line_obj.get_zorder() if line_obj else None,
            'fit_id': fit_id
        }
        # Store fit_id in the fit_dict so we can retrieve it later for redraws
        if fit_dict is not None:
            fit_dict['_fit_id'] = fit_id
        self.item_tracker.add_item(item_id, item_type, name, position=position, color=color, line_obj=line_obj, 
                                   fit_id=fit_id, fit_color=fit_color)
        # Also add to Fit Information window
        self.fit_information_window.add_fit(item_id, item_type, fit_dict, name)
        return item_id
    
    def _extract_listfit_diagnostics(self, lmfit_result):
        """Extract diagnostic metrics directly from lmfit result object
        
        Args:
            lmfit_result: lmfit.model.ModelResult object
            
        Returns:
            Dictionary with diagnostic metrics for registration
        """
        diagnostics = {}
        
        # Extract reduced chi-squared
        if hasattr(lmfit_result, 'redchi') and lmfit_result.redchi is not None:
            diagnostics['chi2_reduced'] = float(lmfit_result.redchi)
        else:
            diagnostics['chi2_reduced'] = None
        
        # Extract R-squared
        if hasattr(lmfit_result, 'rsquared') and lmfit_result.rsquared is not None:
            diagnostics['r_squared'] = float(lmfit_result.rsquared)
        else:
            diagnostics['r_squared'] = None
        
        # Extract AIC and BIC
        if hasattr(lmfit_result, 'aic') and lmfit_result.aic is not None:
            diagnostics['akaike'] = float(lmfit_result.aic)
        else:
            diagnostics['akaike'] = None
        
        if hasattr(lmfit_result, 'bic') and lmfit_result.bic is not None:
            diagnostics['bayesian'] = float(lmfit_result.bic)
        else:
            diagnostics['bayesian'] = None
        
        # Extract number of parameters and data points
        if hasattr(lmfit_result, 'nvarys') and lmfit_result.nvarys is not None:
            diagnostics['n_params'] = int(lmfit_result.nvarys)
        else:
            diagnostics['n_params'] = 0
        
        if hasattr(lmfit_result, 'ndata') and lmfit_result.ndata is not None:
            diagnostics['n_data'] = int(lmfit_result.ndata)
        else:
            diagnostics['n_data'] = 0
        
        # Extract covariance matrix
        if hasattr(lmfit_result, 'covar') and lmfit_result.covar is not None:
            diagnostics['covariance'] = lmfit_result.covar
        else:
            diagnostics['covariance'] = None
        
        # Extract parameter names from the result
        # IMPORTANT: Use var_names (free parameters only) to match covariance matrix size
        if hasattr(lmfit_result, 'var_names') and lmfit_result.var_names is not None:
            # var_names contains only FREE parameters, which matches covariance dimensions
            diagnostics['parameter_names'] = list(lmfit_result.var_names)
        elif hasattr(lmfit_result, 'params') and lmfit_result.params is not None:
            # Fallback: extract free parameter names from params
            diagnostics['parameter_names'] = [name for name in lmfit_result.params.keys() 
                                            if lmfit_result.params[name].vary]
        else:
            diagnostics['parameter_names'] = None
        
        return diagnostics
    
    def _extract_fit_diagnostics(self, fit_data, fit_type):
        """Extract diagnostic metrics from fit data dict
        
        Args:
            fit_data: Dictionary with fit parameters (chi2, chi2_nu, covariance, etc.)
            fit_type: Type of fit ('Gaussian', 'Voigt', 'Continuum', 'Listfit')
            
        Returns:
            Dictionary with diagnostic metrics for registration
        """
        diagnostics = {}
        
        # Extract chi2 reduced
        diagnostics['chi2_reduced'] = fit_data.get('chi2_nu', fit_data.get('chi2', None))
        
        # Calculate R² from chi2_nu if possible
        # R² = 1 - (chi2 / (n_data - 1)) / (variance of y)
        # For now, estimate from chi2_nu: lower chi2_nu = higher R²
        chi2_nu = fit_data.get('chi2_nu', fit_data.get('chi2', 0))
        if chi2_nu is not None and chi2_nu > 0:
            # Rough estimate: R² ≈ 1 - (chi2_nu / 100) clamped to [0, 1]
            # Better estimate would need the actual y values
            diagnostics['r_squared'] = max(0, min(1, 1 - chi2_nu / 100))
        else:
            diagnostics['r_squared'] = None
        
        # Extract or calculate AIC/BIC
        n_params = 0
        if fit_type in ['Gaussian', 'Voigt']:
            n_params = 3  # amplitude, mean, stddev/FWHM
        elif fit_type == 'Continuum':
            n_params = fit_data.get('poly_order', 1) + 1
        elif fit_type == 'Listfit':
            # For Listfit, count parameters by looking at the 'type' field
            comp_type = fit_data.get('type', 'unknown').lower()
            if comp_type == 'gaussian' or comp_type == 'voigt':
                n_params = 3
            elif comp_type == 'polynomial':
                n_params = fit_data.get('poly_order', 1) + 1
        
        diagnostics['n_params'] = n_params
        
        # Estimate n_data from bounds if available
        n_data = None
        if 'bounds' in fit_data:
            left, right = fit_data['bounds']
            mask = (self.x_data >= left) & (self.x_data <= right)
            n_data = np.sum(mask)
        diagnostics['n_data'] = n_data if n_data else 0
        
        # Calculate AIC and BIC if we have chi2 and n_data
        chi2 = fit_data.get('chi2', 0)
        if n_data and n_params:
            k = n_params
            n = n_data
            diagnostics['akaike'] = chi2 + 2 * k  # AIC = chi2 + 2k
            diagnostics['bayesian'] = chi2 + k * np.log(n)  # BIC = chi2 + k*ln(n)
        else:
            diagnostics['akaike'] = None
            diagnostics['bayesian'] = None
        
        # Extract covariance matrix
        diagnostics['covariance'] = fit_data.get('covariance', None)
        
        # Generate parameter names based on fit type
        parameter_names = self._generate_parameter_names(fit_type, fit_data)
        diagnostics['parameter_names'] = parameter_names
        
        return diagnostics
    
    def _generate_parameter_names(self, fit_type, fit_data):
        """Generate parameter names based on fit type
        
        Args:
            fit_type: Type of fit ('Gaussian', 'Voigt', 'Continuum', 'Listfit')
            fit_data: Fit data dictionary
            
        Returns:
            List of parameter names
        """
        names = []
        
        if fit_type.lower() == 'gaussian':
            names = ['amplitude', 'mean', 'sigma']
        elif fit_type.lower() == 'voigt':
            names = ['amplitude', 'center', 'sigma', 'gamma']
        elif fit_type.lower() == 'continuum':
            poly_order = fit_data.get('poly_order', 1)
            # Generate polynomial coefficient names (c0, c1, c2, etc.)
            names = [f'c{i}' for i in range(poly_order + 1)]
        elif fit_type.lower() == 'listfit':
            # For listfit, get parameter names from fit_data if available
            if 'param_names' in fit_data:
                names = fit_data['param_names']
            else:
                # Fallback: generate based on component types
                comp_type = fit_data.get('type', 'unknown').lower()
                if comp_type == 'gaussian':
                    names = ['amplitude', 'mean', 'sigma']
                elif comp_type == 'voigt':
                    names = ['amplitude', 'center', 'sigma', 'gamma']
                elif comp_type == 'polynomial':
                    poly_order = fit_data.get('poly_order', 1)
                    names = [f'c{i}' for i in range(poly_order + 1)]
        
        return names if names else None
    
    def unregister_item(self, item_id):
        """Remove item from tracker and cleanup fits if empty"""
        if item_id in self.item_id_map:
            # Get fit_id before deleting
            fit_id = self.item_id_map[item_id].get('fit_id')
            
            del self.item_id_map[item_id]
            # Only call item_tracker.remove_item if the item is still in the tracker
            # (to avoid double-removal during clear_all operations)
            if item_id in self.item_tracker.items:
                self.item_tracker.remove_item(item_id)
            self.fit_information_window.remove_fit(item_id)
            
            # Remove item from fit tracking and check if fit is now empty
            if fit_id is not None:
                if fit_id in self.fit_items:
                    self.fit_items[fit_id].discard(item_id)
                
                # Clean up lmfit result object if this is the last item from this fit
                if fit_id in self.fit_items and len(self.fit_items[fit_id]) == 0:
                    # Remove the fit's lmfit result to avoid memory leaks
                    if fit_id in self.lmfit_results:
                        del self.lmfit_results[fit_id]
                # Remove fit from diagnostics if no items left
                self.remove_fit_if_empty(fit_id)
        
        # Clean up highlighting tracking if this item was highlighted
        if item_id in self.highlighted_item_ids:
            self.highlighted_item_ids.discard(item_id)
    
    def save_and_print_qsap_fit(self, fit_data, fit_type, fit_mode='Single', lmfit_result=None):
        """Save fit to .qsap file and print contents to terminal
        
        Args:
            fit_data: Single dict or list of dicts with fit parameters
            fit_type: 'Gaussian', 'Voigt', 'Continuum', or 'Listfit'
            fit_mode: 'Single', 'Multi-Gaussian', 'Listfit', etc.
            lmfit_result: Optional lmfit result object for extracting covariance and tie info
            
        Returns:
            Tuple of (filepath, file_content)
        """
        # Initialize continuum_fit_dict for all paths (needed for EW calculation)
        continuum_fit_dict = None
        
        # Calculate equivalent width using Monte Carlo error propagation
        # Only if "Calculate EW automatically" checkbox is enabled
        if self.calculate_ew_enabled and fit_type in ['Gaussian', 'Voigt', 'Listfit']:
            if isinstance(fit_data, list):
                # For Listfit, extract continuum from the component list
                if fit_type == 'Listfit':
                    # Find the polynomial or chebyshev component in the listfit_fit_data
                    for component in fit_data:
                        if component.get('type') in ['polynomial', 'chebyshev']:
                            continuum_fit_dict = component
                            break
                else:
                    # For non-listfit types, use the most recent continuum fit from self.continuum_fits
                    if self.continuum_fits:
                        continuum_fit_dict = self.continuum_fits[-1]
                
                for fit in fit_data:
                    # For Listfit, extract the actual profile type from the component
                    if fit_type == 'Listfit':
                        component_type = fit.get('type', '').lower()
                        print(f"[DEBUG_EW_LOOP] Processing component type: {component_type}")
                        # Skip non-profile components (polynomial, masks, diagnostics)
                        if component_type not in ['gaussian', 'voigt']:
                            print(f"[DEBUG_EW_LOOP] Skipping non-profile component type: {component_type}")
                            continue
                        # For Listfit, only attempt EW if we have proper continuum info
                        component_fit_type = component_type
                        # Set component_id if not already present
                        if 'component_id' not in fit:
                            fit['component_id'] = fit.get('symbol', '?')
                    else:
                        # For non-listfit types (Multi-Gaussian, Multi-Voigt), use the provided fit_type
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
                # For single Gaussian/Voigt fits, get the most recent continuum fit
                if fit_type in ['Gaussian', 'Voigt']:
                    if self.continuum_fits:
                        continuum_fit_dict = self.continuum_fits[-1]
                
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
                fit_data, self.fits_file, spectrum_info, lmfit_result=lmfit_result
            )
        else:
            raise ValueError(f"Unknown fit type: {fit_type}")
        
        # Print the file contents to terminal
        print("\n" + "="*70)
        print(f"FIT SAVED TO: {os.path.basename(filepath)}")
        print("="*70)
        print(content)
        print("="*70 + "\n")
        
        # Register fit metadata for Fit Diagnostics panel
        if fit_type == 'Listfit' and lmfit_result is not None:
            # For Listfit, extract diagnostics from lmfit_result and use current fit_counter
            diagnostics = self._extract_listfit_diagnostics(lmfit_result)
            self.register_fit_metadata(self.fit_counter, fit_type, diagnostics)
        elif isinstance(fit_data, list):
            # Multiple components (Multi-Gaussian, Multi-Voigt)
            for fit in fit_data:
                if fit.get('fit_id') is not None:
                    diagnostics = self._extract_fit_diagnostics(fit, fit_type)
                    self.register_fit_metadata(fit['fit_id'], fit_type, diagnostics)
        else:
            # Single component
            if fit_data.get('fit_id') is not None:
                diagnostics = self._extract_fit_diagnostics(fit_data, fit_type)
                self.register_fit_metadata(fit_data['fit_id'], fit_type, diagnostics)
        
        return filepath, content
    
    def _convert_curve_fit_to_lmfit_like(self, param_names, param_values, pcov):
        """Convert scipy.optimize.curve_fit results to lmfit-like structure
        
        For storing covariance from single/multi-mode fits (which use curve_fit, not lmfit).
        Creates a minimal mock result object with the same interface as lmfit.
        
        Args:
            param_names: List of parameter names (e.g., ['amp', 'mean', 'stddev'])
            param_values: Array of parameter values from curve_fit
            pcov: Covariance matrix from curve_fit
            
        Returns:
            Mock result object with var_names, covar, and params attributes
        """
        class MockCurveFitResult:
            pass
        
        result = MockCurveFitResult()
        result.var_names = list(param_names)
        result.covar = np.array(pcov, dtype=float) if pcov is not None else None
        
        # Create mock parameter objects similar to lmfit
        result.params = {}
        for pname, pval in zip(param_names, param_values):
            class MockParam:
                pass
            p = MockParam()
            p.value = float(pval)
            p.expr = None  # No tied parameters in curve_fit mode
            result.params[pname] = p
        
        return result
    
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
                                # Coefficients stored as [c_N, c_{N-1}, ..., c_1, c_0] (high-to-low order, as np.polyval expects)
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
            x_int = np.linspace(bounds[0], bounds[1], 2000)
            
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
            ew = trapz_compat(normalized_diff, x_int)
            
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
            # Try 'amp' first (from single Voigt mode), fall back to 'amplitude' (from other modes)
            amp = fit_dict.get('amp') if fit_dict.get('amp') is not None else fit_dict.get('amplitude')
            params = [amp, fit_dict.get('center', fit_dict.get('mean')), 
                     fit_dict.get('sigma'), fit_dict.get('gamma')]
            names = ['amp', 'center', 'sigma', 'gamma']
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
    
    def _reconstruct_tied_parameters(self, result, free_param_sample, free_param_names):
        """Reconstruct all parameter values from a sample of free parameters.
        
        When parameters are tied (e.g., g0_sigma = g1_sigma * (1+Z1)), this method:
        1. Takes a sample of only the free parameters
        2. Reconstructs values of all tied parameters using their tie expressions
        3. Returns dict mapping all parameter names to their values
        
        This handles complex tie expressions including redshift-tied parameters like:
        - g0_center = (1 + redshift) * 1215.24
        - g1_sigma = g0_sigma * (1240.81 / 1215.24)
        
        Args:
            result: lmfit fit result object (contains tie expressions)
            free_param_sample: array of sampled free parameter values (one-to-one with free_param_names)
            free_param_names: list of free parameter names in sample order (from result.var_names)
        
        Returns:
            Dictionary mapping all parameter names (free + tied) to their values
        """
        # Build initial dict with free parameters from the sample
        all_params = {}
        for i, name in enumerate(free_param_names):
            all_params[name] = free_param_sample[i]
        
        # Iteratively reconstruct tied parameters
        # Use multiple passes in case tied params depend on other tied params
        max_iterations = 10
        iteration = 0
        unreconstructed = set(result.params.keys()) - set(all_params.keys())
        
        while unreconstructed and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for param_name in list(unreconstructed):  # Iterate over copy since we'll modify set
                param = result.params[param_name]
                if param.expr is not None:
                    # Evaluate the tie expression using the parameters we have so far
                    try:
                        # Create a namespace for eval() using currently available parameters
                        namespace = {name: all_params[name] for name in all_params}
                        # Also add common math functions
                        namespace.update({
                            'sqrt': np.sqrt,
                            'exp': np.exp,
                            'log': np.log,
                            'sin': np.sin,
                            'cos': np.cos,
                            'abs': abs,
                            'pi': np.pi
                        })
                        # Evaluate the expression
                        value = float(eval(param.expr, {"__builtins__": {}}, namespace))
                        all_params[param_name] = value
                        unreconstructed.remove(param_name)
                        made_progress = True
                    except Exception as e:
                        # Parameter not yet available (depends on another tied param)
                        # Will retry in next iteration
                        pass
                else:
                    # Fixed parameter - use its value
                    all_params[param_name] = result.params[param_name].value if result.params[param_name].value is not None else 0.0
                    unreconstructed.remove(param_name)
                    made_progress = True
            
            if not made_progress and unreconstructed:
                # No progress made and still parameters unreconstructed - there's a problem
                print(f"[MC] WARNING: Could not reconstruct {len(unreconstructed)} tied parameters:")
                for param_name in unreconstructed:
                    param = result.params[param_name]
                    print(f"[MC]   {param_name} = {param.expr}")
                    print(f"[MC]     Available params: {list(all_params.keys())}")
                
                # Fall back to initial guess values for remaining params
                for param_name in unreconstructed:
                    all_params[param_name] = result.params[param_name].value if result.params[param_name].value is not None else 0.0
                break
        
        return all_params
    
    def _reconstruct_tied_parameters_from_qsap(self, free_param_sample, free_param_names, tie_expressions):
        """Reconstruct tied parameter values from QSAP tie expressions and free parameter samples.
        
        For loaded Listfits with tied parameters, this evaluates tie expressions like:
        - g1_sigma = g0_sigma * (1240.81 / 1215.24)
        - g0_center = (1 + z1) * 1215.24
        
        Args:
            free_param_sample: array of sampled free parameter values
            free_param_names: list of free parameter names in sample order
            tie_expressions: dict mapping param names to expression strings (from QSAP)
        
        Returns:
            Dictionary mapping all parameter names (free + tied) to their values
        """
        # Build initial dict with free parameters from the sample
        all_params = {}
        for i, name in enumerate(free_param_names):
            all_params[name] = free_param_sample[i]
        
        # Evaluate tie expressions to get tied parameter values
        # Iterate multiple times in case tied params depend on other tied params
        max_iterations = 10
        iteration = 0
        unreconstructed = set(tie_expressions.keys()) - set(all_params.keys())
        
        while unreconstructed and iteration < max_iterations:
            iteration += 1
            made_progress = False
            
            for param_name in list(unreconstructed):
                expr_str = tie_expressions[param_name]
                try:
                    # Create a safe namespace for eval()
                    namespace = dict(all_params)
                    namespace.update({
                        'sqrt': np.sqrt,
                        'exp': np.exp,
                        'log': np.log,
                        'sin': np.sin,
                        'cos': np.cos,
                        'abs': abs,
                        'pi': np.pi
                    })
                    # Evaluate the tie expression
                    value = float(eval(expr_str, {"__builtins__": {}}, namespace))
                    all_params[param_name] = value
                    unreconstructed.remove(param_name)
                    made_progress = True
                except Exception as e:
                    # Parameter not yet available (depends on another tied param)
                    # Will retry in next iteration
                    pass
            
            if not made_progress and unreconstructed:
                # No progress made - missing dependencies
                print(f"[MC] WARNING: Could not evaluate {len(unreconstructed)} tie expressions")
                print(f"[MC]   Available params: {list(all_params.keys())}")
                for param_name in unreconstructed:
                    print(f"[MC]   {param_name} = {tie_expressions[param_name]}")
                break
        
        return all_params
    
    def _calculate_equivalent_width_monte_carlo(self, fit_dict, continuum_fit_dict, fit_type='gaussian', n_samples=1000):
        """Calculate equivalent width using Monte Carlo error propagation
        
        This method samples from the joint posterior distribution of:
        - Gaussian/Voigt profile parameters and their covariance
        - Continuum polynomial and its covariance
        
        Then calculates EW, sigma, and 3-sigma credible intervals from the resulting distribution.
        
        IMPORTANT NOTES ON LISTFIT WITH REDSHIFT:
        - When using redshift-tied Gaussians (e.g., g1_sigma = g0_sigma * (λ_rest/λ_rest_ref)),
          the component covariance matrix becomes singular because tied parameters have zero degrees
          of freedom. MC sampling requires at least one free parameter with non-zero variance.
        - This is a mathematical constraint, not a bug: tied parameters cannot vary independently.
        - WORKAROUND: Fit without redshift-tying, or measure EW manually using the Gaussian components.
        - Fit quality is unaffected; only the EW uncertainty estimation fails.
        
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
            print(f"[MC] _calculate_equivalent_width_monte_carlo called with fit_type='{fit_type}'")
            # Refuse to calculate EW for polynomials - they have no equivalent width
            if fit_type.lower() in ['polynomial', 'continuum', 'poly']:
                print(f"[MC] ERROR: Cannot calculate EW for profile type '{fit_type}' - only Gaussian/Voigt profiles have meaningful EW")
                return None
            
            # Require continuum fit to proceed
            if continuum_fit_dict is None or 'coeffs' not in continuum_fit_dict:
                print("[MC] ERROR: Continuum fit required for MC EW calculation")
                return None
            
            bounds = fit_dict.get('bounds')
            if bounds is None or bounds[0] is None or bounds[1] is None:
                print(f"[MC] ERROR: Invalid or missing bounds in fit_dict: {bounds}")
                print(f"[MC]   fit_dict keys: {list(fit_dict.keys())}")
                return None
            
            # Check for tied parameters FIRST - if present, use free parameter covariance from result or stored full covariance
            # Try to get result object from separate storage (not from fit_dict to avoid cleanup issues)
            fit_id = fit_dict.get('fit_id')
            result_obj = self.lmfit_results.get(fit_id) if fit_id else None
            # Fallback to fit_dict (for backward compatibility with older loaded fits)
            if result_obj is None:
                result_obj = fit_dict.get('result')
            has_result_obj = result_obj is not None
            has_stored_ties = 'tie_expressions' in fit_dict and 'full_covariance' in fit_dict
            
            # Debug: Show what's available in fit_dict
            print(f"[MC] DEBUG fit_dict keys: {list(fit_dict.keys())}")
            print(f"[MC] DEBUG: fit_id={fit_id}, has_result_obj={has_result_obj}, has_tie_expressions={'tie_expressions' in fit_dict}, has_full_covariance={'full_covariance' in fit_dict}")
            
            # For tied parameter reconstruction, we need either:
            # 1. A lmfit result object (from interactive fitting), OR
            # 2. Stored tie_expressions AND full_covariance (from loaded QSAP)
            # BUT: If it's a Listfit component with its own covariance but NO ties, 
            #      use non-tied path with component covariance
            is_listfit_component = 'component_prefix' in fit_dict and 'param_names' in fit_dict
            has_component_covariance = 'covariance' in fit_dict
            
            will_use_tied_path = (has_result_obj or has_stored_ties)
            # Special case: Listfit component with covariance but no ties -> use non-tied path
            if is_listfit_component and has_component_covariance and not has_stored_ties:
                will_use_tied_path = False
            
            print(f"[MC] Tie check: has_result_obj={has_result_obj}, has_stored_ties={has_stored_ties}, is_listfit={is_listfit_component}, will_use_tied_path={will_use_tied_path}")
            
            # Get covariance matrices
            # CRITICAL: For tied parameters, we use the free parameter covariance from lmfit or full stored covariance
            # For non-tied parameters, we use the component covariance
            if will_use_tied_path:
                # Will use full covariance for tied parameter reconstruction below
                if has_result_obj:
                    profile_cov = np.array(result_obj.covar, dtype=float) if result_obj.covar is not None else None
                    if profile_cov is None:
                        print("[MC] ERROR: Free parameter covariance not available in lmfit result")
                        return None
                else:
                    # For loaded fits, keep profile_cov as None - will use full_covariance in tied path
                    profile_cov = None
            else:
                # Use component covariance for non-tied fits
                profile_cov = fit_dict.get('covariance')
                if profile_cov is None:
                    print(f"[MC] ERROR: No component covariance available in fit_dict")
                    print(f"[MC]   fit_dict keys: {list(fit_dict.keys())}")
                    print(f"[MC]   This may indicate loaded fits need covariance extraction from QSAP")
                    return None
                if isinstance(profile_cov, list):
                    profile_cov = np.array(profile_cov, dtype=float)
                elif not isinstance(profile_cov, np.ndarray):
                    profile_cov = np.array(profile_cov, dtype=float)
            
            cont_cov = continuum_fit_dict.get('covariance') if continuum_fit_dict else None
            
            # Get continuum polynomial or Chebyshev if available
            cont_coeffs = None
            cont_type = None  # Track continuum type: 'polynomial' or 'chebyshev'
            cont_domain_min = None
            cont_domain_max = None
            
            if continuum_fit_dict and 'coeffs' in continuum_fit_dict:
                cont_type = continuum_fit_dict.get('type', 'polynomial')  # Default to polynomial for backward compat
                
                if cont_type == 'polynomial':
                    # Coefficients stored as [c_N, c_{N-1}, ..., c_1, c_0] (high-to-low order, as np.polyval expects)
                    cont_coeffs = np.array(continuum_fit_dict['coeffs'], dtype=float)
                elif cont_type == 'chebyshev':
                    # Chebyshev coefficients are stored in normalized [-1, 1] frame
                    # NO REVERSAL needed - chebval takes them in ascending order [c0, c1, ...]
                    cont_coeffs = np.array(continuum_fit_dict['coeffs'], dtype=float)
                    cont_domain_min = continuum_fit_dict.get('lam_min')
                    cont_domain_max = continuum_fit_dict.get('lam_max')
                    
                    if cont_domain_min is None or cont_domain_max is None:
                        print("[MC] ERROR: Chebyshev continuum missing domain bounds (lam_min/lam_max)")
                        return None
                
                if isinstance(cont_cov, list):
                    cont_cov = np.array(cont_cov, dtype=float)
                elif cont_cov is not None and not isinstance(cont_cov, np.ndarray):
                    cont_cov = np.array(cont_cov, dtype=float)
                
                # Validate covariance matrix shape and non-singularity
                if cont_cov is None:
                    print("[MC] ERROR: Continuum covariance matrix is missing")
                    return None
                
                # Check shape matches coefficients
                expected_shape = (len(cont_coeffs), len(cont_coeffs))
                if cont_cov.shape != expected_shape:
                    print(f"[MC] ERROR: Covariance shape {cont_cov.shape} doesn't match coeffs count {len(cont_coeffs)}")
                    # Try to fix by rebuilding diagonal covariance
                    cont_cov = np.diag(np.diag(cont_cov)) if cont_cov.ndim == 2 else np.diag([1e-10] * len(cont_coeffs))
                    print(f"[MC]   Rebuilt as diagonal: {cont_cov.shape}")
                
                # Check for zero/near-zero variance
                cov_diag = np.diag(cont_cov) if cont_cov.ndim == 2 else cont_cov
                if np.any(cov_diag <= 0):
                    print("[MC] WARNING: Covariance matrix has zero or negative diagonal elements")
                    # Replace with small default values
                    cont_cov = np.diag(np.maximum(cov_diag, 1e-20))
                    print(f"[MC]   Replaced with small defaults to avoid singular matrix")
                
                # Check for severely imbalanced covariance (sign of ill-conditioning)
                # This happens when polynomial coefficients have vastly different scales
                # NOTE: For Chebyshev, this should be much better conditioned
                cov_diag_nonzero = cov_diag[cov_diag > 0]
                if len(cov_diag_nonzero) > 1:
                    diag_ratio = np.max(cov_diag_nonzero) / (np.min(cov_diag_nonzero) + 1e-30)
                    if diag_ratio > 1e6:
                        print(f"[MC] WARNING: Continuum covariance is severely imbalanced (ratio={diag_ratio:.1e})")
                        print(f"[MC]   Diagonal: {cov_diag}")
                        if cont_type == 'polynomial':
                            print(f"[MC]   This indicates polynomial coefficients with very different scales")
                            print(f"[MC]   Consider using Chebyshev polynomials instead")
                        print(f"[MC]   MC sampling from this matrix will likely produce garbage values")
                        print(f"[MC]   Consider: (1) checking the continuum fit quality, (2) using Chebyshev basis")
                        # Still continue, but samples will be validated during MC loop
            
            # Integration grid
            x_int = np.linspace(bounds[0], bounds[1], 2000)
            
            ew_samples = []
            profile_samples = []  # Store all realized profiles
            continuum_samples = []  # Store all realized continua
            
            # Extract profile parameters FIRST (needed for singularity check and MC loop)
            profile_params, param_names = self._get_profile_params_from_dict(fit_dict, fit_type)
            if profile_params is None:
                return None
            
            # Convert to float array to ensure proper dtype
            profile_params = np.array(profile_params, dtype=float)
            
            # Determine which path to use for MC sampling
            tried_tied_path = False  # Track whether we attempted tied path
            
            if will_use_tied_path:
                # Try MC sampling with tied parameter reconstruction
                tried_tied_path = True
                try:
                    print("[MC] ========== TIED PARAMETER PATH ==========")
                    print("[MC] Sampling with tied parameter reconstruction")
                    
                    # PATH A: lmfit result object (from interactive fitting)
                    if has_result_obj and result_obj is not None:
                        print("[MC] Using lmfit result object for tied parameter reconstruction")
                    # PATH B: Loaded Listfit with ties (from QSAP file)
                    elif has_stored_ties:
                        print("[MC] Using loaded Listfit with tie expressions")
                        # For loaded fits, we need to reconstruct parameters from full covariance
                        # The key insight: sample ALL free parameters, then apply tie expressions
                        # to get the specific component's parameters
                    
                    # Get free parameter names and covariance
                    if result_obj is not None:
                        free_param_names = result_obj.var_names
                        free_param_covariance = np.array(result_obj.covar, dtype=float) if result_obj.covar is not None else None
                        free_param_values = np.array([result_obj.params[name].value for name in free_param_names], dtype=float)
                    else:
                        # Loaded Listfit - use stored full covariance and all free param names
                        free_param_covariance = fit_dict.get('full_covariance')
                        all_free_names = fit_dict.get('all_free_param_names', [])
                        free_param_names = all_free_names
                        
                        # Check if we have pre-extracted free parameter values from covariance_data
                        stored_free_param_names = fit_dict.get('free_param_names_all', [])
                        stored_free_param_values = fit_dict.get('free_param_values_all', [])
                        
                        # Build a lookup dict from stored values if available
                        stored_param_dict = {}
                        if stored_free_param_names and stored_free_param_values:
                            if len(stored_free_param_names) == len(stored_free_param_values):
                                stored_param_dict = dict(zip(stored_free_param_names, stored_free_param_values))
                        
                        # Get current best-fit values for all free parameters
                        # Strategy: use stored values first (from FREE_PARAMETER_VALUES in QSAP),
                        # then fall back to component fit_dict lookups
                        free_param_values = []
                        for pname in free_param_names:
                            val = None
                            
                            # FIRST TRY: Use stored parameter values from QSAP file
                            if pname in stored_param_dict:
                                val = stored_param_dict[pname]
                            
                            # FALLBACK: POLYNOMIAL PARAMETERS - Extract from continuum_fit_dict
                            elif pname.startswith('p') and '_c' in pname:
                                if continuum_fit_dict and 'coeffs' in continuum_fit_dict:
                                    parts = pname.split('_c')
                                    coeff_idx = int(parts[1])
                                    coeffs = continuum_fit_dict.get('coeffs')
                                    if coeffs and coeff_idx < len(coeffs):
                                        val = coeffs[coeff_idx]
                            
                            # FALLBACK: CURRENT COMPONENT PARAMETERS (gaussian_0 in this EW calculation)
                            elif pname.startswith('g') and '_' in pname:
                                prefix, suffix = pname.split('_', 1)
                                comp_idx = int(prefix[1:])  # Extract index from g0, g1, etc.
                                
                                # Check if this is the current component (from fit_dict)
                                current_comp_id = fit_dict.get('component_id')
                                if f'gaussian_{comp_idx}' == current_comp_id or f'g{comp_idx}' == current_comp_id:
                                    # This is the current component - get from fit_dict
                                    if suffix == 'amp':
                                        val = fit_dict.get('amp')
                                    elif suffix in ['sigma', 'std', 'stddev']:
                                        val = fit_dict.get('stddev')
                                    elif suffix in ['mu', 'mean', 'center']:
                                        val = fit_dict.get('mean')
                                else:
                                    # This is a different gaussian - look it up in self.gaussian_fits
                                    for gfit in self.gaussian_fits:
                                        if gfit.get('is_listfit_component') and gfit.get('_fit_id') == fit_dict.get('_fit_id'):
                                            gcomp_id = gfit.get('component_id')
                                            if f'gaussian_{comp_idx}' == gcomp_id or f'g{comp_idx}' == gcomp_id:
                                                if suffix == 'amp':
                                                    val = gfit.get('amp')
                                                elif suffix in ['sigma', 'std', 'stddev']:
                                                    val = gfit.get('stddev')
                                                elif suffix in ['mu', 'mean', 'center']:
                                                    val = gfit.get('mean')
                                                break
                            
                            # FALLBACK: VOIGT PARAMETERS - Similar to Gaussian
                            elif pname.startswith('v') and '_' in pname:
                                prefix, suffix = pname.split('_', 1)
                                comp_idx = int(prefix[1:])
                                
                                current_comp_id = fit_dict.get('component_id')
                                if f'voigt_{comp_idx}' == current_comp_id or f'v{comp_idx}' == current_comp_id:
                                    if suffix == 'amp':
                                        val = fit_dict.get('amp')
                                    elif suffix == 'sigma':
                                        val = fit_dict.get('stddev')
                                    elif suffix == 'gamma':
                                        val = fit_dict.get('gamma')
                                    elif suffix in ['center', 'mu', 'mean']:
                                        val = fit_dict.get('mean')
                                else:
                                    for vfit in self.voigt_fits:
                                        if vfit.get('is_listfit_component') and vfit.get('_fit_id') == fit_dict.get('_fit_id'):
                                            vcomp_id = vfit.get('component_id')
                                            if f'voigt_{comp_idx}' == vcomp_id or f'v{comp_idx}' == vcomp_id:
                                                if suffix == 'amp':
                                                    val = vfit.get('amp')
                                                elif suffix == 'sigma':
                                                    val = vfit.get('stddev')
                                                elif suffix == 'gamma':
                                                    val = vfit.get('gamma')
                                                elif suffix in ['center', 'mu', 'mean']:
                                                    val = vfit.get('mean')
                                                break
                            
                            # REDSHIFT PARAMETER - Check stored values first
                            elif pname.startswith('z'):
                                # NOTE: z parameters are already in stored_param_dict if loaded from QSAP
                                if pname not in stored_param_dict:
                                    print(f"[MC] ERROR: Redshift parameter {pname} not found in stored_param_dict")
                                    print(f"[MC]   stored_param_dict keys: {list(stored_param_dict.keys())}")
                                    print(f"[MC]   stored_free_param_names: {stored_free_param_names}")
                                    print(f"[MC]   stored_free_param_values: {stored_free_param_values}")
                            
                            # CRITICAL: Fail if value is None (don't use fallback)
                            if val is None:
                                print(f"[MC] ERROR: Could not extract value for parameter {pname}")
                                print(f"[MC]   pname={pname}, type check: starts_p={pname.startswith('p')}, starts_g={pname.startswith('g')}, starts_v={pname.startswith('v')}, starts_z={pname.startswith('z')}")
                                if pname.startswith('g') and '_' in pname:
                                    print(f"[MC]   Gaussian check: stored_in_dict={pname in stored_param_dict}, current_fit_component_id={fit_dict.get('component_id')}")
                                if pname.startswith('z'):
                                    print(f"[MC]   Redshift check: stored_param_dict has z keys: {[k for k in stored_param_dict if k.startswith('z')]}")
                                raise RuntimeError(f"Could not extract free parameter value for {pname}")
                            
                            free_param_values.append(val)
                        
                        free_param_values = np.array(free_param_values, dtype=float)

                    
                    if free_param_covariance is None:
                        print("[MC] ERROR: Free parameter covariance not available")
                        raise RuntimeError("Free parameter covariance is None")
                    
                    # Check if free-parameter covariance is singular
                    try:
                        cov_det = np.linalg.det(free_param_covariance)
                        cond_num = np.linalg.cond(free_param_covariance)
                        eigenvalues = np.linalg.eigvals(free_param_covariance)
                        print(f"[MC] Covariance matrix diagnostics:")
                        print(f"[MC]   Shape: {free_param_covariance.shape}")
                        print(f"[MC]   Determinant: {cov_det}")
                        print(f"[MC]   Condition number: {cond_num}")
                        print(f"[MC]   Eigenvalues: {eigenvalues}")
                        print(f"[MC]   Min eigenvalue: {np.min(eigenvalues)}")
                        print(f"[MC]   Max eigenvalue: {np.max(eigenvalues)}")
                        if abs(cov_det) < 1e-10:
                            print("[MC] WARNING: Free-parameter covariance matrix is singular!")
                            print("[MC]   This is unexpected - should only include free params in covar")
                        if cond_num > 1e10:
                            print(f"[MC] WARNING: Covariance matrix is ill-conditioned (cond={cond_num:.2e})")
                            print("[MC]   This will cause numerical instability in MC sampling")
                    except (np.linalg.LinAlgError, ValueError) as e:
                        print(f"[MC] WARNING: Could not analyze covariance matrix: {e}")
                    
                    print(f"[MC] Free parameters: {free_param_names}")
                    print(f"[MC] Free param values: {free_param_values}")
                    print(f"[MC] Free param covariance shape: {free_param_covariance.shape}")
                    print(f"[MC] Free param covariance diag: {np.diag(free_param_covariance)}")
                    
                    # Monte Carlo loop with tied parameter reconstruction
                    for sample_idx in range(n_samples):
                        try:
                            # Sample free parameters from their covariance
                            try:
                                free_param_sample = np.random.multivariate_normal(free_param_values, free_param_covariance)
                            except (np.linalg.LinAlgError, ValueError) as e:
                                print(f"[MC] ERROR: Cannot sample from free-parameter covariance on iteration {sample_idx}: {e}")
                                raise  # Re-raise to exit tied path and try non-tied
                            
                            # Diagnostic: log first few samples
                            if sample_idx < 3:
                                print(f"[MC] Sample {sample_idx}: free_param_sample = {free_param_sample}")
                                print(f"[MC]   Deviation from mean: {free_param_sample - free_param_values}")
                            
                            if result_obj is not None:
                                # Reconstruct all parameters (free + tied) using lmfit tie expressions
                                all_params_dict = self._reconstruct_tied_parameters(result_obj, free_param_sample, free_param_names)
                            else:
                                # Reconstruct all parameters using QSAP tie expressions
                                all_params_dict = self._reconstruct_tied_parameters_from_qsap(
                                    free_param_sample, 
                                    free_param_names,
                                    fit_dict.get('tie_expressions', {})
                                )
                            
                            # Extract parameters for this specific component
                            # For lmfit: use component_prefix + param_names_list
                            # For loaded: extract by component ID from all_params_dict
                            if result_obj is not None:
                                component_prefix = fit_dict.get('component_prefix')
                                param_names_list = fit_dict.get('param_names', [])
                                
                                component_params = []
                                for pname in param_names_list:
                                    full_name = f'{component_prefix}{pname}'
                                    if full_name in all_params_dict:
                                        component_params.append(all_params_dict[full_name])
                                    else:
                                        print(f"[MC] WARNING: Parameter {full_name} not found in reconstructed params")
                                        raise RuntimeError(f"Missing parameter: {full_name}")
                                component_params = np.array(component_params, dtype=float)
                            else:
                                # Loaded Listfit - extract this component's parameters from all_params_dict
                                component_id = fit_dict.get('component_id')  # e.g., 'gaussian_0' or 'g0'
                                
                                # Determine component type and parameter order
                                if fit_type.lower() == 'gaussian':
                                    # Gaussian: need [amp, mean, sigma]
                                    # Look for g0_amp, g0_mu (or mean), g0_sigma in all_params_dict
                                    amp_val = None
                                    mean_val = None
                                    sigma_val = None
                                    
                                    # Try to find the parameters in reconstructed dict
                                    # First, figure out the component index from component_id
                                    comp_idx = 0
                                    if 'gaussian_' in component_id:
                                        comp_idx = int(component_id.split('_')[1])
                                    
                                    for pname, pval in all_params_dict.items():
                                        if pname == f'g{comp_idx}_amp':
                                            amp_val = pval
                                        elif pname in [f'g{comp_idx}_mu', f'g{comp_idx}_mean', f'g{comp_idx}_center']:
                                            mean_val = pval
                                        elif pname in [f'g{comp_idx}_sigma', f'g{comp_idx}_std']:
                                            sigma_val = pval
                                    
                                    # Use reconstructed values if available, else use fit_dict values
                                    if amp_val is None:
                                        amp_val = fit_dict.get('amp', 1.0)
                                    if mean_val is None:
                                        mean_val = fit_dict.get('mean', 5000.0)
                                    if sigma_val is None:
                                        sigma_val = fit_dict.get('stddev', 1.0)
                                    
                                    component_params = np.array([amp_val, mean_val, sigma_val], dtype=float)
                                    
                                elif fit_type.lower() == 'voigt':
                                    # Voigt: need [amp, center, sigma, gamma]
                                    comp_idx = 0
                                    if 'voigt_' in component_id:
                                        comp_idx = int(component_id.split('_')[1])
                                    
                                    amp_val = all_params_dict.get(f'v{comp_idx}_amp', fit_dict.get('amp', 1.0))
                                    center_val = all_params_dict.get(f'v{comp_idx}_center', fit_dict.get('mean', 5000.0))
                                    sigma_val = all_params_dict.get(f'v{comp_idx}_sigma', fit_dict.get('stddev', 1.0))
                                    gamma_val = all_params_dict.get(f'v{comp_idx}_gamma', fit_dict.get('gamma', 0.1))
                                    
                                    component_params = np.array([amp_val, center_val, sigma_val, gamma_val], dtype=float)
                                else:
                                    raise RuntimeError(f"Unknown fit_type: {fit_type}")
                            
                            # CRITICAL: Validate reconstructed sample for NaN/inf and physical validity
                            if np.any(np.isnan(component_params)) or np.any(np.isinf(component_params)):
                                raise RuntimeError(f"Reconstructed component parameters contain NaN/inf: {component_params}")
                            
                            # Check for physical validity based on fit type
                            if fit_type.lower() == 'gaussian':
                                # stddev MUST be positive
                                if component_params[2] <= 0:
                                    raise RuntimeError(f"Invalid Gaussian stddev: {component_params[2]}")
                            elif fit_type.lower() == 'voigt':
                                # sigma MUST be positive, gamma MUST be non-negative
                                if component_params[2] <= 0 or component_params[3] < 0:
                                    raise RuntimeError(f"Invalid Voigt sigma/gamma: {component_params[2]}/{component_params[3]}")
                            
                            # Evaluate profile with reconstructed parameters
                            profile = self._evaluate_profile(x_int, fit_type, component_params)
                            
                            # Sample continuum polynomial if available
                            if cont_coeffs is not None and cont_cov is not None:
                                try:
                                    cont_sample = np.random.multivariate_normal(cont_coeffs, cont_cov)
                                except (np.linalg.LinAlgError, ValueError) as e:
                                    print(f"[MC] ERROR in continuum sampling (sample {sample_idx}): {e}")
                                    print(f"[MC]   cont_coeffs={cont_coeffs}, cont_cov shape={cont_cov.shape}")
                                    raise  # Re-raise to exit tied path
                                
                                # Diagnostic: log first few continuum samples
                                if sample_idx < 3:
                                    print(f"[MC] Sample {sample_idx} continuum ({cont_type}): coeffs={cont_sample}")
                                    if cont_type == 'polynomial':
                                        eval_x = x_int[0]
                                        eval_y = np.polyval(cont_sample, eval_x)
                                        print(f"[MC]   Evaluation at x={eval_x:.1f}: y={eval_y:.3e}")
                                        eval_x = x_int[-1]
                                        eval_y = np.polyval(cont_sample, eval_x)
                                        print(f"[MC]   Evaluation at x={eval_x:.1f}: y={eval_y:.3e}")
                                    elif cont_type == 'chebyshev':
                                        eval_x = x_int[0]
                                        x_rescaled = 2 * (eval_x - cont_domain_min) / (cont_domain_max - cont_domain_min) - 1
                                        eval_y = np.polynomial.chebyshev.chebval(x_rescaled, cont_sample)
                                        print(f"[MC]   Evaluation at x={eval_x:.1f} (rescaled={x_rescaled:.4f}): y={eval_y:.3e}")
                                        eval_x = x_int[-1]
                                        x_rescaled = 2 * (eval_x - cont_domain_min) / (cont_domain_max - cont_domain_min) - 1
                                        eval_y = np.polynomial.chebyshev.chebval(x_rescaled, cont_sample)
                                        print(f"[MC]   Evaluation at x={eval_x:.1f} (rescaled={x_rescaled:.4f}): y={eval_y:.3e}")
                                
                                # Validate sampled polynomial coefficients
                                if np.any(np.isnan(cont_sample)) or np.any(np.isinf(cont_sample)):
                                    raise RuntimeError(f"Continuum sample contains NaN/inf: {cont_sample}")
                                
                                # Evaluate continuum at integration grid based on type
                                if cont_type == 'polynomial':
                                    continuum = np.polyval(cont_sample, x_int)
                                elif cont_type == 'chebyshev':
                                    # Rescale wavelengths to [-1, 1]
                                    x_rescaled = 2 * (x_int - cont_domain_min) / (cont_domain_max - cont_domain_min) - 1
                                    continuum = np.polynomial.chebyshev.chebval(x_rescaled, cont_sample)
                                else:
                                    raise RuntimeError(f"Unknown continuum type: {cont_type}")
                                
                                # Validate continuum for pathological behavior
                                if np.any(np.isnan(continuum)) or np.any(np.isinf(continuum)):
                                    raise RuntimeError("Continuum evaluation produced NaN/inf")
                                
                                # Check if continuum varies wildly (sign of singular matrix)
                                # For a polynomial fit, large variations at the edges are expected,
                                # but variance should not change by orders of magnitude
                                cont_min = np.min(continuum)
                                cont_max = np.max(continuum)
                                cont_range_ratio = cont_max / (cont_min + 1e-10)
                                
                                # If continuum varies by more than 100x, likely singular matrix
                                if cont_range_ratio > 100:
                                    raise RuntimeError(f"Continuum varies wildly (ratio={cont_range_ratio:.1f}), suggests singular covariance")
                            else:
                                print("[MC] ERROR: Cannot compute MC EW without continuum covariance matrix")
                                raise RuntimeError("Continuum covariance missing")

                            
                            # Ensure non-zero continuum to avoid division issues
                            continuum = np.maximum(continuum, 1e-10)
                            
                            # Store samples for later plotting
                            profile_samples.append(profile)
                            continuum_samples.append(continuum)
                            
                            # Calculate EW for this sample
                            normalized = -profile / continuum
                            ew = trapz_compat(normalized, x_int)
                            
                            # Diagnostic: log first few EW calculations
                            if sample_idx < 3:
                                print(f"[MC] Sample {sample_idx} EW calc:")
                                print(f"[MC]   profile: min={np.min(profile):.3e}, max={np.max(profile):.3e}")
                                print(f"[MC]   continuum: min={np.min(continuum):.3e}, max={np.max(continuum):.3e}")
                                print(f"[MC]   normalized: min={np.min(normalized):.3e}, max={np.max(normalized):.3e}")
                                print(f"[MC]   EW={ew:.3e}")

                            ew_samples.append(ew)
                        
                        except Exception as e:
                            print(f"[MC] Exception in sample {sample_idx}: {type(e).__name__}: {e}")
                            if sample_idx < 5:  # Only print first few
                                import traceback
                                traceback.print_exc()
                            # Continue trying remaining samples
                            continue
                
                except Exception as e:
                    # Tied parameter path failed - will try non-tied below
                    print(f"[MC] Tied parameter path failed: {type(e).__name__}: {e}")
                    print("[MC]   Will attempt non-tied parameter path...")
                    ew_samples = []  # Clear any partial samples
                    profile_samples = []
                    continuum_samples = []
            
            # If tied path was attempted and produced samples, we're done
            # Otherwise, try the non-tied path (but ONLY if we have profile covariance)
            if not ew_samples and profile_cov is not None:
                # Original MC sampling without tied parameters (component covariance is non-singular)
                print("[MC] ========== NON-TIED PARAMETER PATH ==========")
                print("[MC] Sampling without tied parameters")
                
                try:
                    # Check if profile covariance is singular
                    # Do this BEFORE trying any numpy operations on it
                    profile_cov_array = np.array(profile_cov, dtype=float)
                    
                    # Check for all-zero rows/columns or very small diagonal (signs of singularity)
                    cov_diag = np.diag(profile_cov_array) if profile_cov_array.ndim == 2 else profile_cov_array
                    num_zero_diag = np.sum(cov_diag <= 1e-15)
                    
                    # Try to detect singularity without calling det() which might crash
                    is_singular = False
                    if num_zero_diag > 0:
                        is_singular = True
                        print(f"[MC] ERROR: {num_zero_diag} parameters have zero/near-zero variance (tied/fixed)")
                        print("[MC]   This means the profile covariance is singular - cannot do MC sampling")
                        print("[MC]   Check that tied parameters were properly handled in the fit")
                        raise RuntimeError("Profile covariance is singular")
                    else:
                        # Check determinant only if diagonal looks OK
                        try:
                            cov_det = np.linalg.det(profile_cov_array)
                            if abs(cov_det) < 1e-10:
                                is_singular = True
                                print("[MC] ERROR: Covariance matrix has near-zero determinant (singular)")
                                print("[MC]   Cannot perform MC sampling with singular covariance matrix")
                                raise RuntimeError("Covariance matrix is singular (det ≈ 0)")
                        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
                            is_singular = True
                            print(f"[MC] ERROR: Could not compute determinant: {e}")
                            raise RuntimeError(f"Determinant check failed: {e}")
                    
                    # No longer checking is_singular - just let MC loop run
                    # If there are issues, the loop will catch them and return None
                    
                    # Monte Carlo loop (standard path for non-tied parameters)
                    invalid_sample_count = 0
                    for sample_idx in range(n_samples):
                        try:
                            # Sample profile parameters from multivariate normal
                            try:
                                profile_sample = np.random.multivariate_normal(profile_params, profile_cov)
                            except (np.linalg.LinAlgError, ValueError) as e:
                                print(f"[MC] ERROR: Cannot sample from singular covariance matrix: {e}")
                                print(f"[MC]   This typically happens with tied parameters")
                                print(f"[MC]   MC sampling failed - cannot calculate EW with proper uncertainties")
                                raise  # Re-raise to exit non-tied path
                            
                            # CRITICAL: Validate sample for NaN/inf and physical validity
                            # (NaN/inf can occur with singular/near-singular covariance matrices)
                            if np.any(np.isnan(profile_sample)) or np.any(np.isinf(profile_sample)):
                                invalid_sample_count += 1
                                if sample_idx < 3:  # Log first few occurrences
                                    print(f"[MC] WARNING: Sample {sample_idx} contains NaN/inf: {profile_sample}")
                                continue  # Skip this invalid sample
                            
                            # Check for physical validity based on fit type
                            if fit_type.lower() == 'gaussian':
                                # amplitude can be negative (absorption), mean is positive wavelength, stddev MUST be positive
                                if profile_sample[2] <= 0:  # stddev must be positive
                                    invalid_sample_count += 1
                                    if sample_idx < 3:
                                        print(f"[MC] WARNING: Sample {sample_idx} has invalid stddev: {profile_sample[2]}")
                                    continue
                            elif fit_type.lower() == 'voigt':
                                # sigma MUST be positive, gamma MUST be non-negative
                                if profile_sample[2] <= 0 or profile_sample[3] < 0:
                                    invalid_sample_count += 1
                                    if sample_idx < 3:
                                        print(f"[MC] WARNING: Sample {sample_idx} has invalid sigma/gamma: {profile_sample[2]}/{profile_sample[3]}")
                                    continue
                            
                            # Evaluate profile at high resolution
                            profile = self._evaluate_profile(x_int, fit_type, profile_sample)
                            
                            # Sample or use continuum polynomial/Chebyshev
                            if cont_coeffs is not None:
                                if cont_cov is not None:
                                    # Sample continuum from covariance (when continuum has free parameters)
                                    try:
                                        cont_sample = np.random.multivariate_normal(cont_coeffs, cont_cov)
                                    except (np.linalg.LinAlgError, ValueError) as e:
                                        print(f"[MC] ERROR in continuum sampling (sample {sample_idx}): {e}")
                                        print(f"[MC]   Coefficients: {cont_coeffs}")
                                        print(f"[MC]   Covariance shape: {cont_cov.shape}, diagonal: {np.diag(cont_cov)}")
                                        raise  # Re-raise to exit non-tied path
                                else:
                                    # Use fixed continuum coefficients (when all continuum parameters are tied)
                                    cont_sample = cont_coeffs
                                
                                # Validate sampled polynomial coefficients
                                if np.any(np.isnan(cont_sample)) or np.any(np.isinf(cont_sample)):
                                    invalid_sample_count += 1
                                    if sample_idx < 3:
                                        print(f"[MC] WARNING: Sample {sample_idx} continuum has NaN/inf coefficients")
                                    continue
                                
                                if cont_type == 'polynomial':
                                    continuum = np.polyval(cont_sample, x_int)
                                elif cont_type == 'chebyshev':
                                    # Rescale wavelengths to [-1, 1]
                                    x_rescaled = 2 * (x_int - cont_domain_min) / (cont_domain_max - cont_domain_min) - 1
                                    continuum = np.polynomial.chebyshev.chebval(x_rescaled, cont_sample)
                                else:
                                    raise RuntimeError(f"Unknown continuum type: {cont_type}")
                                
                                # Validate continuum for pathological behavior
                                if np.any(np.isnan(continuum)) or np.any(np.isinf(continuum)):
                                    invalid_sample_count += 1
                                    if sample_idx < 3:
                                        print(f"[MC] WARNING: Sample {sample_idx} continuum evaluation produced NaN/inf")
                                    continue
                                
                                # Check if continuum varies wildly (sign of singular matrix)
                                cont_min = np.min(continuum)
                                cont_max = np.max(continuum)
                                cont_range_ratio = cont_max / (cont_min + 1e-10)
                                
                                # If continuum varies by more than 100x, likely singular matrix
                                if cont_range_ratio > 100:
                                    invalid_sample_count += 1
                                    if sample_idx < 3:
                                        print(f"[MC] WARNING: Sample {sample_idx} continuum varies wildly (ratio={cont_range_ratio:.1f})")
                                    continue
                            else:
                                # Cannot proceed without continuum coefficients
                                print("[MC] ERROR: Cannot compute MC EW without continuum coefficients")
                                raise RuntimeError("Continuum coefficients missing")
                            
                            # NOTE: We do NOT artificially floor negative continuum values to 1e-10
                            # If the polynomial covariance is ill-conditioned, some MC samples will
                            # naturally produce negative continuum. This is an honest representation of
                            # the underlying singular covariance matrix and must be fixed at the source
                            # (not masked by sample rejection or artificial flooring).
                            # The fix is Chebyshev polynomials (Step 3), which have well-conditioned
                            # orthogonal basis. Until then, we show the true pathology.
                            
                            # Store samples for later plotting
                            profile_samples.append(profile)
                            continuum_samples.append(continuum)
                            
                            # Calculate EW for this sample
                            # Note: profile is the residual (flux - continuum) from the fit
                            # EW = ∫(continuum - flux)/continuum dλ = -∫residual/continuum dλ
                            normalized = -profile / continuum
                            ew = trapz_compat(normalized, x_int)
                            
                            ew_samples.append(ew)
                        
                        except Exception as e:
                            print(f"[MC] Exception in sample {sample_idx}: {type(e).__name__}: {e}")
                            if sample_idx < 5:  # Only print first few
                                import traceback
                                traceback.print_exc()
                            # Continue with remaining samples
                            continue
                    
                    # Check if too many samples were invalid (sign of singular covariance)
                    if invalid_sample_count > n_samples * 0.1:  # More than 10% invalid
                        print(f"[MC] ERROR: {invalid_sample_count}/{n_samples} samples were invalid (NaN/inf or unphysical)")
                        print(f"[MC]   This indicates a singular/near-singular covariance matrix")
                        print(f"[MC]   Likely cause: tied parameters with zero variance not properly reconstructed")
                        raise RuntimeError(f"Too many invalid samples ({invalid_sample_count}/{n_samples})")
                    elif invalid_sample_count > 0:
                        print(f"[MC] WARNING: {invalid_sample_count} samples were invalid and skipped")
                
                except Exception as e:
                    # Non-tied path also failed
                    print(f"[MC] Non-tied parameter path also failed: {type(e).__name__}: {e}")
                    print("[MC]   Both tied and non-tied paths have failed")
            
            # At this point, ew_samples should be populated from either tied or non-tied path
            if not ew_samples:
                print("[MC] ERROR: No EW samples generated")
                print("[MC]")
                print("[MC] ===== DIAGNOSIS =====")
                print("[MC] This typically occurs when:")
                print("[MC]   1. Listfit with redshift: Tied parameters cause singular covariance")
                print("[MC]   2. Continuum polynomial: Coefficients have vastly different scales")
                print("[MC]   3. Too few free parameters: Complex ties reduce degrees of freedom")
                print("[MC]   4. Missing or None covariance matrices")
                print("[MC]")
                return None
            
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
            
            # ============= STEP 1 DIAGNOSTICS: Fat-tail detection =============
            print("[MC] ========== STEP 1 DIAGNOSTICS: EW Distribution Analysis ==========")
            
            # 1. Percentile distribution and max
            percentiles = np.percentile(valid_ew_samples, [1, 16, 50, 84, 99])
            max_ew = np.max(valid_ew_samples)
            min_ew = np.min(valid_ew_samples)
            p99_ew = percentiles[4]
            ratio_max_to_p99 = abs(max_ew) / (abs(p99_ew) + 1e-30)
            
            print(f"[MC] 1. EW Percentiles:")
            print(f"[MC]   1st percentile:  {percentiles[0]:.6e}")
            print(f"[MC]   16th percentile: {percentiles[1]:.6e}")
            print(f"[MC]   50th (median):   {percentiles[2]:.6e}")
            print(f"[MC]   84th percentile: {percentiles[3]:.6e}")
            print(f"[MC]   99th percentile: {percentiles[4]:.6e}")
            print(f"[MC]   Min EW:          {min_ew:.6e}")
            print(f"[MC]   Max EW:          {max_ew:.6e}")
            print(f"[MC]   Max/99th ratio:  {ratio_max_to_p99:.2e}")
            if ratio_max_to_p99 > 10:
                print(f"[MC]   FAT TAIL DETECTED: max is {ratio_max_to_p99:.1f}x larger than 99th percentile")
            
            # 2. Count continuum samples hitting the 1e-10 floor
            if len(continuum_samples_array) > 0:
                floor_value = 1e-10
                n_clipped_points = 0
                n_clipped_samples = 0
                
                for cont_sample in continuum_samples_array:
                    n_below_floor = np.sum(cont_sample <= floor_value)
                    if n_below_floor > 0:
                        n_clipped_points += n_below_floor
                        n_clipped_samples += 1
                
                print(f"[MC] 2. Continuum Clipping at 1e-10 floor:")
                print(f"[MC]   Samples with clipped points: {n_clipped_samples}/{len(continuum_samples_array)}")
                print(f"[MC]   Total clipped grid points: {n_clipped_points}/{len(continuum_samples_array) * len(x_int)}")
                print(f"[MC]   Fraction of samples clipped: {100*n_clipped_samples/len(continuum_samples_array):.1f}%")
                if n_clipped_samples > len(continuum_samples_array) * 0.01:
                    print(f"[MC]   WARNING: >1% of samples had clipped continuum points")
            
            # 3. Continuum covariance conditioning
            if cont_cov is not None:
                try:
                    cont_cond = np.linalg.cond(cont_cov)
                    print(f"[MC] 3. Continuum Polynomial Covariance Conditioning:")
                    print(f"[MC]   Condition number: {cont_cond:.2e}")
                    if cont_cond > 1e8:
                        print(f"[MC]   WARNING: Highly ill-conditioned (cond > 1e8)")
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"[MC] 3. Could not compute continuum covariance condition number: {e}")
            
            # 4. Eigenvalue analysis
            if cont_cov is not None:
                try:
                    eigvals = np.linalg.eigvalsh(cont_cov)
                    min_eig = np.min(eigvals)
                    max_eig = np.max(eigvals)
                    eig_ratio = max_eig / (abs(min_eig) + 1e-30)
                    print(f"[MC] 4. Continuum Covariance Eigenvalue Analysis:")
                    print(f"[MC]   Eigenvalues: {eigvals}")
                    print(f"[MC]   Min eigenvalue: {min_eig:.6e}")
                    print(f"[MC]   Max eigenvalue: {max_eig:.6e}")
                    print(f"[MC]   Max/min ratio: {eig_ratio:.2e}")
                    if min_eig < 0:
                        print(f"[MC]   ERROR: Negative eigenvalue detected (non-positive-definite matrix)")
                    elif min_eig < 1e-15 * max_eig:
                        print(f"[MC]   WARNING: Smallest eigenvalue near machine epsilon relative to largest")
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"[MC] 4. Could not compute eigenvalues: {e}")
            
            # 5. Bounds vs fit window extrapolation check
            if 'bounds' in fit_dict and continuum_fit_dict:
                ew_bounds = fit_dict.get('bounds')
                cont_bounds = continuum_fit_dict.get('bounds')
                if ew_bounds and cont_bounds:
                    if ew_bounds[0] < cont_bounds[0] or ew_bounds[1] > cont_bounds[1]:
                        print(f"[MC] 5. BOUNDS EXTRAPOLATION DETECTED:")
                        print(f"[MC]   EW integration bounds:     {ew_bounds[0]:.2f} - {ew_bounds[1]:.2f} Å")
                        print(f"[MC]   Continuum fit bounds:      {cont_bounds[0]:.2f} - {cont_bounds[1]:.2f} Å")
                        print(f"[MC]   EW extends beyond continuum fit region")
                        print(f"[MC]   LEFT:  fit={cont_bounds[0]:.2f}, EW={ew_bounds[0]:.2f}, extrapolate={max(0, cont_bounds[0] - ew_bounds[0]):.2f} Å")
                        print(f"[MC]   RIGHT: fit={cont_bounds[1]:.2f}, EW={ew_bounds[1]:.2f}, extrapolate={max(0, ew_bounds[1] - cont_bounds[1]):.2f} Å")
                    else:
                        print(f"[MC] 5. Bounds OK: EW integration within continuum fit region")
            
            print("[MC] ========== END STEP 1 DIAGNOSTICS ==========")
            
            # Calculate statistics from the distribution
            median_ew = np.percentile(valid_ew_samples, 50)
            mean_ew = np.mean(valid_ew_samples)
            
            # Calculate "best" EW from the fitted parameters (no MC)
            profile_params, _ = self._get_profile_params_from_dict(fit_dict, fit_type)
            best_profile = self._evaluate_profile(x_int, fit_type, profile_params)
            if cont_coeffs is not None:
                # Check if continuum is Chebyshev (has type and domain fields)
                if continuum_fit_dict and continuum_fit_dict.get('type') == 'chebyshev':
                    # Chebyshev evaluation with domain rescaling
                    lam_min = continuum_fit_dict.get('lam_min', bounds[0])
                    lam_max = continuum_fit_dict.get('lam_max', bounds[1])
                    x_rescaled = 2 * (x_int - lam_min) / (lam_max - lam_min) - 1
                    best_continuum = np.polynomial.chebyshev.chebval(x_rescaled, cont_coeffs)
                else:
                    # Standard polynomial evaluation (coefficients reversed for polyval)
                    best_continuum = np.polyval(cont_coeffs, x_int)
            else:
                continuum_level = self._get_continuum_level_estimate(bounds)
                if continuum_level is None or continuum_level <= 0:
                    continuum_level = 1.0
                best_continuum = np.ones_like(x_int) * continuum_level
            best_continuum = np.maximum(best_continuum, 1e-10)
            best_normalized = -best_profile / best_continuum
            best_ew = trapz_compat(best_normalized, x_int)
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
            
            print(f"[MC] ========== RESULTS ==========")
            print(f"[MC] Median EW: {median_ew:.6f}")
            print(f"[MC] 1-sigma: -{median_ew - p16_1:.6f}/+{p84_1 - median_ew:.6f}")
            print(f"[MC] 2-sigma: -{median_ew - p228_2:.6f}/+{p9772_2 - median_ew:.6f}")
            print(f"[MC] 3-sigma: -{median_ew - p0135_3:.6f}/+{p99865_3 - median_ew:.6f}")
            print(f"[MC] Generated {len(valid_ew_samples)} valid samples")
            
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
                        # Remove component by matching type and index
                        for i in range(len(components) - 1, -1, -1):  # Iterate backwards to avoid index shifting
                            comp = components[i]
                            if (comp.get('type') == 'polynomial' and 
                                comp.get('index') == poly_index):
                                components.pop(i)
                                print(f"[DEBUG] Removed polynomial (index={poly_index}) from listfit components")
                                break
                        break
        
        # Handle Chebyshev deletion - remove from listfit components list
        elif item_type == 'chebyshev':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            cheb_index = fit_dict.get('cheb_index')
            
            # Find the listfit this chebyshev belongs to and remove the component
            if listfit_bounds is not None and cheb_index is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        # Remove the component from the listfit's components list
                        components = listfit.get('components', [])
                        # Remove component by matching type and index
                        for i in range(len(components) - 1, -1, -1):  # Iterate backwards to avoid index shifting
                            comp = components[i]
                            if (comp.get('type') == 'chebyshev' and 
                                comp.get('index') == cheb_index):
                                components.pop(i)
                                print(f"[DEBUG] Removed chebyshev (index={cheb_index}) from listfit components")
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
            
            if listfit_bounds is not None and min_lambda is not None and max_lambda is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i in range(len(components) - 1, -1, -1):  # Iterate backwards for safe removal
                            comp = components[i]
                            if (comp.get('type') == 'polynomial_guess_mask' and 
                                comp.get('min_lambda') == min_lambda and
                                comp.get('max_lambda') == max_lambda):
                                components.pop(i)
                                print(f"[DEBUG] Removed polynomial_guess_mask ({min_lambda:.2f}-{max_lambda:.2f}) from listfit components")
                                break
                        break
        
        # Handle Data Mask deletion - remove from listfit
        elif item_type == 'data_mask':
            fit_dict = item_info.get('fit_dict', {})
            listfit_bounds = fit_dict.get('listfit_bounds')
            min_lambda = fit_dict.get('min_lambda')
            max_lambda = fit_dict.get('max_lambda')
            
            if listfit_bounds is not None and min_lambda is not None and max_lambda is not None:
                for listfit in self.listfit_fits:
                    if listfit.get('bounds') == listfit_bounds:
                        components = listfit.get('components', [])
                        for i in range(len(components) - 1, -1, -1):  # Iterate backwards for safe removal
                            comp = components[i]
                            if (comp.get('type') == 'data_mask' and 
                                comp.get('min_lambda') == min_lambda and
                                comp.get('max_lambda') == max_lambda):
                                components.pop(i)
                                print(f"[DEBUG] Removed data_mask ({min_lambda:.2f}-{max_lambda:.2f}) from listfit components")
                                break
                        break
        
        # Handle line objects (gaussian, voigt, continuum, polynomial, listfit_total)
        line_obj = item_info.get('line_obj')
        if line_obj:
            try:
                line_obj.remove()
                print(f"[DEBUG] Removed line object for {item_type}")
            except Exception as e:
                # Object may have already been removed or cannot be removed
                # Log the error but don't crash
                print(f"[DEBUG] Warning: Could not remove line object for {item_type}: {type(e).__name__}: {e}")
        
        # Handle patches (continuum regions, masks)
        patch_obj = item_info.get('patch_obj')
        if patch_obj:
            try:
                patch_obj.remove()
                print(f"[DEBUG] Removed patch object for {item_type}")
            except Exception as e:
                # Object may have already been removed or cannot be removed
                # Log the error but don't crash
                print(f"[DEBUG] Warning: Could not remove patch object for {item_type}: {type(e).__name__}: {e}")
        
        # If it's a continuum region, also remove from continuum_patches list
        if item_type == 'continuum_region' and 'bounds' in item_info:
            bounds = item_info['bounds']
            self.continuum_patches = [p for p in self.continuum_patches if p.get('bounds') != bounds]
        
        # Additional cleanup for mask items - ensure they're removed from listfit_fits components if present
        if item_type in ['polynomial_guess_mask', 'data_mask']:
            print(f"[DEBUG] Completed cleanup for {item_type} item")
        
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
        print(f"[DEBUG_ON_EW] on_calculate_ew_from_tracker called with item_id={item_id}")
        if item_id not in self.item_id_map:
            print(f"[DEBUG_ON_EW] item_id {item_id} not in item_id_map, returning")
            return
        
        item_info = self.item_id_map[item_id]
        item_type = item_info.get('type')
        fit_dict = item_info.get('fit_dict')
        
        print(f"[DEBUG_ON_EW] item_type={item_type}, has_fit_dict={fit_dict is not None}")
        
        if not fit_dict:
            print("[EW] No fit dictionary found for this item")
            return
        
        # Skip EW calculation for polynomials - they don't have meaningful equivalent width
        if item_type == 'polynomial' or item_type == 'continuum':
            print(f"[EW] Skipping EW calculation for {item_type} - not a spectral feature")
            print(f"[DEBUG_TRACE] on_calculate_ew_from_tracker called for {item_type} item - printing traceback:")
            import traceback
            traceback.print_stack()
            return
        
        # Determine if this is from a listfit or a regular fit
        is_listfit = 'listfit_bounds' in fit_dict
        
        # Get the corresponding fit type and continuum
        if item_type == 'gaussian':
            fit_type = 'gaussian'
        elif item_type == 'voigt':
            fit_type = 'voigt'
        else:
            print(f"[EW] Cannot calculate EW for item type: {item_type}")
            return
        
        # Get the continuum fit
        if is_listfit:
            # For listfit profiles, get continuum from the same listfit
            continuum_fit_dict = self._get_listfit_continuum(fit_dict)
            if continuum_fit_dict is None:
                print("[EW] No continuum polynomial found in listfit (exactly 1 required)")
                return
        else:
            # For regular fits, use the stored continuum_fit_dict if available (stored during fitting)
            # Otherwise fall back to the last continuum fit
            continuum_fit_dict = fit_dict.get('continuum_fit_dict')
            if continuum_fit_dict is None:
                continuum_fit_dict = self.continuum_fits[-1] if self.continuum_fits else None
        
        # Calculate EW
        try:
            ew_result = self._calculate_equivalent_width_monte_carlo(fit_dict, continuum_fit_dict, fit_type)
        except Exception as e:
            print(f"[EW] Exception during MC calculation: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            ew_result = None
        
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
    
    def _get_listfit_continuum(self, listfit_profile_dict):
        """Extract the continuum polynomial from a listfit
        
        Returns continuum fit dict if there's exactly 1 polynomial, None otherwise
        """
        print(f"[DEBUG_CONT] _get_listfit_continuum called, printing traceback:")
        import traceback
        traceback.print_stack()
        
        listfit_bounds = listfit_profile_dict.get('listfit_bounds')
        if listfit_bounds is None:
            return None
        
        # FIRST: Check if polynomial coefficients are directly stored in the profile dict
        # (this happens when a Gaussian/Voigt from listfit stores its continuum reference)
        if 'coeffs' in listfit_profile_dict and 'covariance' in listfit_profile_dict:
            # Polynomial coefficients are directly available in the dict
            print("[EW] Found polynomial coefficients stored in profile dict")
            return listfit_profile_dict
        
        # SECOND: Look in continuum_fits for polynomials from this listfit
        listfit_continua = [c for c in self.continuum_fits 
                           if c.get('bounds') == listfit_bounds and c.get('listfit_source')]
        
        if len(listfit_continua) == 1:
            continuum = listfit_continua[0]
            cont_type = continuum.get('type', 'polynomial')
            print(f"[EW] Found {cont_type} in continuum_fits: order={continuum.get('poly_order')}")
            # Return as continuum dict for MC calculation
            # Use stored covariance if available (from loaded .qsap), otherwise build from coeffs_err
            covariance = continuum.get('covariance')
            if covariance is None and continuum.get('coeffs_err') is not None:
                covariance = np.diag([(e**2 if e > 0 else 1e-10) for e in continuum.get('coeffs_err', [])])
            elif covariance is None:
                covariance = np.diag([1e-10] * len(continuum.get('coeffs', [])))
            
            result_dict = {
                'coeffs': continuum.get('coeffs', []),
                'covariance': covariance,
                'bounds': listfit_bounds,
                'type': cont_type  # Include type for proper evaluation in MC
            }
            # Add domain bounds for Chebyshev
            if cont_type == 'chebyshev':
                result_dict['lam_min'] = continuum.get('lam_min')
                result_dict['lam_max'] = continuum.get('lam_max')
            return result_dict
        elif len(listfit_continua) > 1:
            print(f"[EW] Found {len(listfit_continua)} polynomials in continuum_fits (need exactly 1)")
            return None
        
        # THIRD: Fall back to looking in listfit_fits for polynomial or Chebyshev components
        for listfit in self.listfit_fits:
            if listfit.get('bounds') == listfit_bounds:
                # Extract all continuum components (polynomial or Chebyshev) from this listfit
                components = listfit.get('components', [])
                continuum_comps = [c for c in components if c.get('type') in ['polynomial', 'chebyshev']]
                
                # Return the continuum only if there's exactly 1 continuum component
                if len(continuum_comps) == 1:
                    cont_comp = continuum_comps[0]
                    
                    # Get coefficients from the stored continuum component
                    coeffs = cont_comp.get('coeffs', [])
                    coeffs_err = cont_comp.get('coeffs_err', [])
                    
                    if not coeffs:
                        return None
                    
                    # Build covariance matrix from errors (diagonal approximation)
                    covariance = np.diag([e**2 if e > 0 else 1e-10 for e in coeffs_err]) if coeffs_err else np.diag([1e-10] * len(coeffs))
                    
                    continuum_dict = {
                        'coeffs': coeffs,
                        'covariance': covariance,
                        'bounds': listfit.get('bounds'),
                        'type': cont_comp.get('type'),  # Store type so MC calculation knows if it's Chebyshev
                        'lam_min': cont_comp.get('lam_min'),
                        'lam_max': cont_comp.get('lam_max')
                    }
                    comp_type = cont_comp.get('type')
                    print(f"[EW] Found {comp_type} in listfit_fits components")
                    return continuum_dict
                else:
                    print(f"[EW] Listfit has {len(continuum_comps)} continuum components (need exactly 1)")
                    return None
        
        print("[EW] No continuum polynomial found for this listfit")
        return None
        
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
    
    def on_error_spectrum_mode_changed(self):
        """Handle change in error spectrum display mode (Default or Shaded)"""
        if self.error_spectrum_default_radio.isChecked():
            self.error_spectrum_mode = "Default"
            print("Error spectrum mode: Default (red dashed line)")
        else:
            self.error_spectrum_mode = "Shaded"
            print("Error spectrum mode: Shaded (gray band ±error)")
        
        # Redraw error spectrum
        self.redraw_error_spectrum()
    
    def on_residual_display_mode_changed(self):
        """Handle change in residual display mode (None, Sigma, or Shaded)"""
        if self.residual_none_radio.isChecked():
            self.residual_display_mode = "None"
            print("Residual display mode: None (default)")
        elif self.residual_sigma_radio.isChecked():
            self.residual_display_mode = "Sigma"
            print("Residual display mode: Sigma (residual / error)")
        else:
            self.residual_display_mode = "Shaded"
            print("Residual display mode: Shaded")
        
        # Redraw residual panel if visible
        if self.is_residual_shown:
            self.redraw_residual_panel()
    
    def redraw_error_spectrum(self):
        """Redraw error spectrum based on current mode (Default or Shaded)"""
        try:
            if not hasattr(self, 'ax') or self.ax is None or self.err is None:
                return
            
            # Get error color configuration
            error_cfg = self.colors['spectrum']['error']
            
            # Remove old error spectrum visualizations
            if self.error_band_fill is not None:
                try:
                    self.error_band_fill.remove()
                except (ValueError, AttributeError):
                    pass
                self.error_band_fill = None
            
            if self.error_line is not None:
                try:
                    self.error_line.remove()
                except (ValueError, AttributeError):
                    pass
                self.error_line = None
            
            if self.step_error is not None:
                try:
                    self.step_error.remove()
                except (ValueError, AttributeError):
                    pass
                self.step_error = None
            
            if self.line_error is not None:
                try:
                    self.line_error.remove()
                except (ValueError, AttributeError):
                    pass
                self.line_error = None
            
            # Draw error spectrum based on mode
            if self.error_spectrum_mode == "Shaded":
                # Draw shaded band (spectrum ± error) with semi-transparent gray
                if self.is_step_plot:
                    self.error_band_fill = self.ax.fill_between(
                        self.x_data, 
                        self.spec - self.err, 
                        self.spec + self.err,
                        step='mid',
                        alpha=0.15,
                        color='gray',
                        label='Error',
                        zorder=1
                    )
                else:
                    self.error_band_fill = self.ax.fill_between(
                        self.x_data, 
                        self.spec - self.err, 
                        self.spec + self.err,
                        alpha=0.15,
                        color='gray',
                        label='Error',
                        zorder=1
                    )
            else:  # Default mode - plot error values as red dashed line (original behavior)
                if self.is_step_plot:
                    # Step plot version
                    self.step_error, = self.ax.step(
                        self.x_data, self.err,
                        where='mid', 
                        color=error_cfg['color'], 
                        linestyle=error_cfg['linestyle'], 
                        linewidth=1.0,
                        alpha=error_cfg['alpha'], 
                        label='Error',
                        zorder=0
                    )
                    self.error_line = self.step_error
                else:
                    # Line plot version
                    self.line_error, = self.ax.plot(
                        self.x_data, self.err,
                        color=error_cfg['color'], 
                        linestyle=error_cfg['linestyle'], 
                        linewidth=1.0,
                        alpha=error_cfg['alpha'], 
                        label='Error',
                        zorder=0
                    )
                    self.error_line = self.line_error
            
            # Update legend
            if hasattr(self, 'ax') and self.ax is not None:
                self.update_legend()
            
            # Redraw canvas
            self.ax.figure.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error redrawing error spectrum: {e}")
            import traceback
            traceback.print_exc()
    
    def redraw_residual_panel(self):
        """Redraw residual panel based on current display mode"""
        try:
            if not hasattr(self, 'residual_ax') or self.residual_ax is None:
                return
            
            # Clear residual panel
            self.residual_ax.clear()
            
            # Get residuals
            residuals = self.calculate_residuals()
            if residuals is None or len(residuals) == 0:
                return
            
            # Get color configuration for residuals
            residual_cfg = self.colors['residual']
            residual_color = residual_cfg['color']
            
            # Plot based on mode
            if self.residual_display_mode == "Sigma":
                # Plot residual / error
                if self.err is not None and len(self.err) == len(residuals):
                    # Avoid division by zero
                    sigma_residuals = np.divide(
                        residuals, 
                        self.err,
                        out=np.zeros_like(residuals),
                        where=self.err!=0
                    )
                    if self.is_step_plot:
                        self.residual_ax.step(
                            self.x_data, sigma_residuals,
                            where='mid', color=residual_color, linewidth=1.0
                        )
                    else:
                        self.residual_ax.plot(
                            self.x_data, sigma_residuals,
                            color=residual_color, linewidth=1.0
                        )
                    # Set y-axis label for sigma
                    self.residual_ax.set_ylabel(r"Residuals ($\sigma$)", fontsize=10)
                else:
                    # Fallback to regular residuals if no error
                    if self.is_step_plot:
                        self.residual_ax.step(
                            self.x_data, residuals,
                            where='mid', color=residual_color, linewidth=1.0
                        )
                    else:
                        self.residual_ax.plot(
                            self.x_data, residuals,
                            color=residual_color, linewidth=1.0
                        )
                    self.residual_ax.set_ylabel("Residuals", fontsize=10)
            elif self.residual_display_mode == "Shaded":
                # Plot residuals with shaded confidence band
                if self.is_step_plot:
                    self.residual_ax.step(
                        self.x_data, residuals,
                        where='mid', color=residual_color, linewidth=1.0
                    )
                else:
                    self.residual_ax.plot(
                        self.x_data, residuals,
                        color=residual_color, linewidth=1.0
                    )
                # Add shaded band around residuals using error spectrum
                if self.err is not None and len(self.err) == len(residuals):
                    if self.is_step_plot:
                        self.residual_ax.fill_between(
                            self.x_data,
                            residuals - self.err,
                            residuals + self.err,
                            step='mid',
                            alpha=0.15,
                            color='lightgray',
                            zorder=0
                        )
                    else:
                        self.residual_ax.fill_between(
                            self.x_data,
                            residuals - self.err,
                            residuals + self.err,
                            alpha=0.15,
                            color='lightgray',
                            zorder=0
                        )
                self.residual_ax.set_ylabel("Residuals", fontsize=10)
            else:  # "None" mode (default)
                # Regular residuals plot
                if self.is_step_plot:
                    self.residual_ax.step(
                        self.x_data, residuals,
                        where='mid', color=residual_color, linewidth=1.0
                    )
                else:
                    self.residual_ax.plot(
                        self.x_data, residuals,
                        color=residual_color, linewidth=1.0
                    )
                self.residual_ax.set_ylabel("Residuals", fontsize=10)
            
            # Draw zero reference line
            ref_cfg = self.colors['reference_lines']
            self.residual_ax.axhline(y=0, color=ref_cfg['color'], linestyle=ref_cfg['linestyle'], linewidth=ref_cfg['linewidth'])
            
            # Set x-axis label based on velocity mode
            if self.is_velocity_mode:
                self.residual_ax.set_xlabel(r"Velocity (km s$^{-1}$)", fontsize=10)
            else:
                self.residual_ax.set_xlabel(self._get_wavelength_unit_label(), fontsize=10)
            
            # Sync x-axis bounds with spectrum
            if self.ax is not None:
                self.residual_ax.set_xlim(self.ax.get_xlim())
            
            # Update formatting and limits
            self.update_residual_ticks()
            self.update_residual_ybounds()
            
            # Update x-data for velocity mode if needed
            if self.is_velocity_mode and hasattr(self, 'velocities') and len(self.velocities) == len(self.x_data):
                # Residual lines already plotted, but update if needed
                pass
            
            # Redraw canvas
            self.residual_ax.figure.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error redrawing residual panel: {e}")
            import traceback
            traceback.print_exc()
    
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
            ew = trapz_compat(flux_diff, x)
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

        print(f"Gaussian EW 2-sigma confidence interval (obs): [{ew_gauss_16th:.4f}, {ew_gauss_50th:.4f}, {ew_gauss_84th:.4f}]")
        print(f"Total EW 2-sigma confidence interval (obs): [{ew_total_16th:.4f}, {ew_total_50th:.4f}, {ew_total_84th:.4f}]")
        print(f"Gaussian EW 2-sigma confidence interval (rest): [{ew_gauss_16th_rest:.4f}, {ew_gauss_50th_rest:.4f}, {ew_gauss_84th_rest:.4f}]")
        print(f"Total EW 2-sigma confidence interval (rest): [{ew_total_16th_rest:.4f}, {ew_total_50th_rest:.4f}, {ew_total_84th_rest:.4f}]")

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

    def on_mouse_press(self, event):
        """Handle mouse button press events - start drag for guess drawing, mask drawing, or interactive smoothing"""
        # Set focus to the canvas when clicked to deselect text fields
        if event.inaxes or event.xdata is not None:
            self.canvas.setFocus()
        
        # Handle interactive smoothing mode - click-drag to adjust
        if self.smoothing_interactive_mode and event.inaxes == self.ax and event.xdata is not None:
            self.smoothing_drag_start_x = event.xdata
            self.smoothing_drag_start_y = event.ydata

            return  # Don't process other mouse events in smoothing mode
        
        # Handle constraint bounds setting mode - click-drag to set bounds
        if self.constraint_bounds_mode and event.inaxes == self.ax and event.xdata is not None:
            self.constraint_bounds_drag_start_x = event.xdata
            return
        
        # Handle mask drawing mode - click-drag to draw regions
        if self.mask_drawing_mode and event.inaxes == self.ax and event.xdata is not None:
            self.mask_drag_start_x = event.xdata
            return
        
        # Handle guess drawing mode - click-drag to draw, or multi-point click for polynomial/chebyshev
        if self.guess_drawing_mode and event.inaxes == self.ax and event.xdata is not None:
            comp_type = self.current_component_for_guess.get('type', '').lower()
            
            # Polynomial: multi-point click mode
            if comp_type == 'polynomial':
                self._add_polynomial_point(event.xdata, event.ydata)
                return  # Don't start a drag for polynomial
            
            # Chebyshev: multi-point click mode (same as polynomial)
            if comp_type == 'chebyshev':
                self._add_chebyshev_point(event.xdata, event.ydata)
                return  # Don't start a drag for chebyshev
            
            # Gaussian/Voigt: drag mode
            self.guess_mouse_down = True
            self.guess_drag_start_x = event.xdata
            self.guess_drag_start_y = event.ydata
            self.guess_drag_end_x = event.xdata
            self.guess_drag_end_y = event.ydata
            print(f"[Guess] Drag started at λ={event.xdata:.2f} Å, flux={event.ydata:.4f}")
    
    def on_mouse_release(self, event):
        """Handle mouse button release events - end drag for guess drawing, mask drawing, or smoothing"""
        # End interactive smoothing drag
        if self.smoothing_interactive_mode and self.smoothing_drag_start_x is not None:
            self.smoothing_drag_start_x = None
            self.smoothing_drag_start_y = None
            try:
                median_kernel = int(self.smoothing_median_input.text() or 1)
                gaussian_sigma = float(self.smoothing_gaussian_input.text() or 0.0)
                # Finalize smoothing
                self.smooth_spectrum(median_kernel, gaussian_sigma)
                self._update_spectrum_display(self.smoothed_spec)
                self.last_applied_median = median_kernel
                self.last_applied_gaussian = gaussian_sigma
            except ValueError:
                pass
            return  # Don't process other release events
        
        # End constraint bounds setting drag
        if self.constraint_bounds_mode and self.constraint_bounds_drag_start_x is not None and event.inaxes == self.ax:
            if event.xdata is not None:
                x_min = min(self.constraint_bounds_drag_start_x, event.xdata)
                x_max = max(self.constraint_bounds_drag_start_x, event.xdata)
                
                # Store bounds in the constraint editor
                self._apply_constraint_bounds(x_min, x_max)
                
                # Exit constraint bounds mode
                self.constraint_bounds_mode = False
                self.constraint_parameter = None
                self.constraint_bounds_drag_start_x = None
                self.constraint_bounds_drag_end_x = None
                
                # Remove preview line
                if self.constraint_bounds_preview_line is not None:
                    try:
                        self.constraint_bounds_preview_line.remove()
                    except (ValueError, RuntimeError):
                        pass
                self.constraint_bounds_preview_line = None
                
                print(f"[Constraint] Bounds set: {x_min:.2f} to {x_max:.2f} Å")
                self.canvas.draw_idle()
            
            return  # Don't process other release events
        
        # End mask drawing drag
        if self.mask_drawing_mode and self.mask_drag_start_x is not None and event.inaxes == self.ax:
            if event.xdata is not None:
                x_start = min(self.mask_drag_start_x, event.xdata)
                x_end = max(self.mask_drag_start_x, event.xdata)
                self.mask_regions.append((x_start, x_end))
                
                print(f"[Mask] Region {len(self.mask_regions)} drawn: λ={x_start:.2f} to {x_end:.2f} Å")
                print(f"  Drag to draw another region or press ENTER to confirm")
            
            # Clear preview rectangles and redraw with updated rectangles
            for rect in self.mask_preview_rects:
                try:
                    rect.remove()
                except (ValueError, RuntimeError):
                    pass
            self.mask_preview_rects.clear()
            
            # Redraw preview for all regions
            from matplotlib.patches import Rectangle
            for x_start, x_end in self.mask_regions:
                y_min, y_max = self.ax.get_ylim()
                rect = Rectangle((x_start, y_min), x_end - x_start, y_max - y_min,
                               linewidth=1, edgecolor='red', facecolor='red', alpha=0.15, zorder=1)
                self.ax.add_patch(rect)
                self.mask_preview_rects.append(rect)
            
            self.canvas.draw_idle()
            self.mask_drag_start_x = None
            return  # Don't process other release events
        
        # End guess drawing drag
        if self.guess_drawing_mode and self.guess_mouse_down and event.inaxes == self.ax:
            self.guess_mouse_down = False
            if event.xdata is not None and event.ydata is not None:
                self.guess_drag_end_x = event.xdata
                self.guess_drag_end_y = event.ydata
                
                # Process the completed drag
                comp_type = self.current_component_for_guess.get('type', '').lower()
                if comp_type in ['gaussian', 'voigt']:
                    self._process_gaussian_voigt_drag()
                elif comp_type == 'polynomial':
                    self._process_polynomial_drag()
    
    def _process_gaussian_voigt_drag(self):
        """Process a completed drag for Gaussian/Voigt guess"""
        x1, y1 = self.guess_drag_start_x, self.guess_drag_start_y
        x2, y2 = self.guess_drag_end_x, self.guess_drag_end_y
        
        # Calculate guess parameters from drag
        self.guess_center = (x1 + x2) / 2.0
        # Preserve sign: positive if dragging up (y2 > y1), negative if dragging down (y2 < y1)
        self.guess_amp = y2 - y1
        # Convert FWHM to sigma: FWHM = 2.355 * sigma
        fwhm = abs(x2 - x1)
        self.guess_sigma = max(0.01, fwhm / 2.355)
        
        print(f"[Guess] Drag ended. Calculated parameters:")
        print(f"  Center (λ): {self.guess_center:.2f} Å")
        print(f"  Amplitude: {self.guess_amp:.4f}")
        print(f"  Width (sigma): {self.guess_sigma:.2f} Å (FWHM: {fwhm:.2f} Å)")
        
        # Update preview
        self._update_guess_preview_salmon()
    
    def _process_polynomial_drag(self):
        """Process a completed drag for polynomial guess"""
        x1, y1 = self.guess_drag_start_x, self.guess_drag_start_y
        x2, y2 = self.guess_drag_end_x, self.guess_drag_end_y
        
        # Calculate linear fit: y = mx + b
        if abs(x2 - x1) < 1e-10:
            print("[Guess] Error: Cannot draw vertical line. Try dragging horizontally.")
            return
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        print(f"[Guess] Polynomial line drawn from λ={x1:.2f} to λ={x2:.2f}")
        print(f"  Slope (m): {slope:.6f}")
        print(f"  Intercept (b): {intercept:.4f}")
        print(f"  Equation: y = {slope:.6f} * x + {intercept:.4f}")
        
        # Store polynomial parameters in guess
        self.current_component_for_guess['guess'] = {
            'slope': slope,
            'intercept': intercept,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }
        
        # Plot the polynomial line
        self._plot_polynomial_guess_line(x1, x2, y1, y2)
        
        print("[Guess] Press ENTER to confirm polynomial guess, or drag again to redraw")

    def on_mouse_move(self, event):
        # Check if the cursor is within the axes bounds
        if event.inaxes == self.ax:
            self.x_lower_bound, self.x_upper_bound = self.ax.get_xlim()
            self.y_lower_bound, self.y_upper_bound = self.ax.get_ylim()
            if not (self.x_lower_bound <= event.xdata <= self.x_upper_bound and
                    self.y_lower_bound <= event.ydata <= self.y_upper_bound):
                return  # Exit if the cursor is outside the plot area
        
        # Handle interactive smoothing drag
        if self.smoothing_interactive_mode and self.smoothing_drag_start_x is not None and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                # Calculate drag distances
                dx = event.xdata - self.smoothing_drag_start_x
                dy = event.ydata - self.smoothing_drag_start_y
                
                # Determine primary drag direction
                abs_dx = abs(dx)
                abs_dy = abs(dy)
                
                try:
                    # Up-down drag (> 2x larger vertical component): adjust Gaussian sigma
                    if abs_dy > abs_dx * 2:
                        # Map vertical drag to Gaussian sigma change
                        sigma_delta = dy * 0.01  # Sensitivity factor (10x less aggressive)
                        current_sigma = float(self.smoothing_gaussian_input.text() or 0.0)
                        new_sigma = max(0.0, current_sigma + sigma_delta)
                        self.smoothing_gaussian_input.setText(f"{new_sigma:.2f}")
                    
                    # Left-right drag (> 2x larger horizontal component): adjust Median kernel
                    elif abs_dx > abs_dy * 2:
                        # Map horizontal drag to median kernel change
                        kernel_delta = int(dx * 0.005)  # Sensitivity factor (10x less aggressive)
                        current_kernel = int(self.smoothing_median_input.text() or 1)
                        new_kernel = max(1, current_kernel + kernel_delta)
                        # Ensure odd number
                        if new_kernel % 2 == 0:
                            new_kernel += 1
                        self.smoothing_median_input.setText(str(new_kernel))
                    
                    # Apply current smoothing values during drag for live preview
                    try:
                        median_kernel = int(self.smoothing_median_input.text() or 1)
                        gaussian_sigma = float(self.smoothing_gaussian_input.text() or 0.0)
                        self.smooth_spectrum(median_kernel, gaussian_sigma)
                        self._update_spectrum_display(self.smoothed_spec)
                    except ValueError:
                        pass  # Ignore parsing errors during typing
                
                except (ValueError, AttributeError):
                    pass  # Ignore errors
        
        # Handle live preview during constraint bounds setting drag
        if self.constraint_bounds_mode and event.inaxes == self.ax and self.constraint_bounds_drag_start_x is not None:
            if event.xdata is not None:
                # Remove old preview line
                if self.constraint_bounds_preview_line is not None:
                    try:
                        self.constraint_bounds_preview_line.remove()
                    except (ValueError, RuntimeError):
                        pass
                
                # Draw two vertical lines showing min and max bounds
                x_min = min(self.constraint_bounds_drag_start_x, event.xdata)
                x_max = max(self.constraint_bounds_drag_start_x, event.xdata)
                y_min, y_max = self.ax.get_ylim()
                
                # Draw a vertical span to show the bounds
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=1, edgecolor='green', facecolor='green', alpha=0.1, zorder=1)
                self.ax.add_patch(rect)
                self.constraint_bounds_preview_line = rect
                
                # Also draw vertical lines at min and max
                line_min = self.ax.axvline(x=x_min, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                line_max = self.ax.axvline(x=x_max, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                
                self.canvas.draw_idle()
        
        # Handle live preview during mask drawing drag
        if self.mask_drawing_mode and event.inaxes == self.ax and self.mask_drag_start_x is not None:
            if event.xdata is not None:
                # Clear old preview rectangles
                for rect in self.mask_preview_rects:
                    try:
                        rect.remove()
                    except (ValueError, RuntimeError):
                        pass
                self.mask_preview_rects.clear()
                
                # Draw preview rectangles for all regions (existing + current drag)
                from matplotlib.patches import Rectangle
                
                # Draw existing regions
                for x_start, x_end in self.mask_regions:
                    y_min, y_max = self.ax.get_ylim()
                    rect = Rectangle((x_start, y_min), x_end - x_start, y_max - y_min,
                                   linewidth=1, edgecolor='red', facecolor='red', alpha=0.15, zorder=1)
                    self.ax.add_patch(rect)
                    self.mask_preview_rects.append(rect)
                
                # Draw current drag preview
                x_start = min(self.mask_drag_start_x, event.xdata)
                x_end = max(self.mask_drag_start_x, event.xdata)
                y_min, y_max = self.ax.get_ylim()
                current_rect = Rectangle((x_start, y_min), x_end - x_start, y_max - y_min,
                                       linewidth=2, edgecolor='red', facecolor='red', alpha=0.25, zorder=1)
                self.ax.add_patch(current_rect)
                self.mask_preview_rects.append(current_rect)
                
                self.canvas.draw_idle()
        
        # Handle live preview during guess drawing drag
        if self.guess_drawing_mode and event.inaxes == self.ax and self.guess_mouse_down:
            if event.xdata is not None and event.ydata is not None:
                self.guess_drag_end_x = event.xdata
                self.guess_drag_end_y = event.ydata
                
                # Update preview in real-time based on component type
                comp_type = self.current_component_for_guess.get('type', '').lower()
                if comp_type in ['gaussian', 'voigt']:
                    # Update live preview of Gaussian/Voigt as user drags
                    x1, y1 = self.guess_drag_start_x, self.guess_drag_start_y
                    x2, y2 = self.guess_drag_end_x, self.guess_drag_end_y
                    
                    center = (x1 + x2) / 2.0
                    # Preserve sign: positive if dragging up (y2 > y1), negative if dragging down (y2 < y1)
                    amp = y2 - y1
                    fwhm = abs(x2 - x1)
                    sigma = max(0.01, fwhm / 2.355)
                    
                    # Show live preview
                    x_preview = np.linspace(center - 3*sigma, center + 3*sigma, 200)
                    if comp_type == 'gaussian':
                        y_preview = amp * np.exp(-((x_preview - center)**2) / (2 * sigma**2))
                    else:  # voigt
                        gamma = sigma * 0.01
                        s2pi = np.sqrt(2*np.pi)
                        amp_compensated = amp * (sigma * s2pi)
                        y_preview = self.voigt(x_preview, amp_compensated, center, sigma, gamma)
                    
                    # Add polynomial or chebyshev baseline to preview (floor for profiles to stand on)
                    poly_guesses = [c for c in self.listfit_components if c.get('type') in ['polynomial', 'chebyshev'] and c.get('guess')]
                    poly_baseline_for_profile = np.zeros_like(x_preview)
                    
                    # Remove old polynomial baseline line preview
                    if self.guess_polynomial_baseline_line is not None:
                        try:
                            self.guess_polynomial_baseline_line.remove()
                        except (ValueError, RuntimeError):
                            pass
                        self.guess_polynomial_baseline_line = None
                    
                    # If exactly one polynomial guess exists, visualize it with the profile
                    if len(poly_guesses) == 1:
                        poly_comp = poly_guesses[0]
                        poly_guess = poly_comp.get('guess', {})
                        try:
                            # Get x range for polynomial baseline visualization (use full visible range)
                            x_poly_range = np.linspace(self.ax.get_xlim()[0], self.ax.get_xlim()[1], 300)
                            poly_baseline_for_profile = np.zeros_like(x_preview)
                            x_poly_baseline = None
                            
                            if 'coefficients' in poly_guess:
                                coeffs = poly_guess.get('coefficients')
                                if coeffs:
                                    print(f"[DEBUG] Using polynomial coefficients for baseline: {coeffs}")
                                    poly_baseline_for_profile = np.polyval(coeffs, x_preview)
                                    x_poly_baseline = np.polyval(coeffs, x_poly_range)
                            elif poly_guess.get('x1') is not None and poly_guess.get('x2') is not None:
                                # Old format: linear interpolation (extrapolate to full range)
                                x1_poly = poly_guess.get('x1')
                                x2_poly = poly_guess.get('x2')
                                y1_poly = poly_guess.get('y1')
                                y2_poly = poly_guess.get('y2')
                                print(f"[DEBUG] Using linear polynomial: ({x1_poly}, {y1_poly}) to ({x2_poly}, {y2_poly})")
                                
                                # Calculate slope and extrapolate across entire x_preview range
                                if abs(x2_poly - x1_poly) > 1e-10:
                                    slope = (y2_poly - y1_poly) / (x2_poly - x1_poly)
                                    # Use point-slope form: y - y1 = slope * (x - x1)
                                    poly_baseline_for_profile = y1_poly + slope * (x_preview - x1_poly)
                                    x_poly_baseline = y1_poly + slope * (x_poly_range - x1_poly)
                                    print(f"[DEBUG] Linear baseline slope={slope}, extended to full x_preview range")
                                else:
                                    # Vertical line - use y1
                                    poly_baseline_for_profile[:] = y1_poly
                                    x_poly_baseline = np.full_like(x_poly_range, y1_poly)
                            
                            # Draw the polynomial baseline line as a visual reference (gray, lighter)
                            if x_poly_baseline is not None:
                                self.guess_polynomial_baseline_line, = self.ax.plot(x_poly_range, x_poly_baseline, 
                                                                                   color='gray', linestyle='--', 
                                                                                   linewidth=1.5, alpha=0.5, zorder=10,
                                                                                   label='Polynomial Baseline')
                                print(f"[DEBUG] Plotted polynomial baseline line")
                        except Exception as e:
                            print(f"[Debug] Error visualizing polynomial baseline: {e}")
                            import traceback
                            traceback.print_exc()
                            pass
                        
                        # Add the baseline to the profile so it sits on top (OUTSIDE the try block to ensure it always happens)
                        print(f"[DEBUG] Before adding baseline: y_preview range [{np.min(y_preview):.2f}, {np.max(y_preview):.2f}]")
                        print(f"[DEBUG] Polynomial baseline range [{np.min(poly_baseline_for_profile):.2f}, {np.max(poly_baseline_for_profile):.2f}]")
                        y_preview = y_preview + poly_baseline_for_profile
                        print(f"[DEBUG] After adding baseline: y_preview range [{np.min(y_preview):.2f}, {np.max(y_preview):.2f}]")
                    
                    # Remove old preview
                    if self.guess_preview_line is not None:
                        try:
                            self.guess_preview_line.remove()
                        except (ValueError, RuntimeError):
                            pass
                    
                    # Draw new salmon-colored preview with higher z-order so it appears on top
                    self.guess_preview_line, = self.ax.plot(x_preview, y_preview, color='salmon', 
                                                           linestyle='-', linewidth=2, alpha=0.7, zorder=15,
                                                           label=f'{comp_type.title()} Guess Preview')
                    self.canvas.draw_idle()
                
                elif comp_type == 'polynomial':
                    # Update live preview of polynomial line as user drags
                    # Calculate line equation from the two clicked points
                    x1, y1 = self.guess_drag_start_x, self.guess_drag_start_y
                    x2, y2 = self.guess_drag_end_x, self.guess_drag_end_y
                    
                    # Calculate slope and intercept
                    if abs(x2 - x1) > 1e-10:
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        
                        # Extend to full listfit fitting range (or use current bounds as fallback)
                        if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                            x_min = min(self.listfit_bounds)
                            x_max = max(self.listfit_bounds)
                        else:
                            x_min = self.x_lower_bound
                            x_max = self.x_upper_bound
                        
                        # Generate extended line across full range
                        x_line = np.linspace(x_min, x_max, 200)
                        y_line = slope * x_line + intercept
                        
                        # Also mark the clicked points on the line
                        x_points = [x1, x2]
                        y_points = [y1, y2]
                    else:
                        # Fallback to simple two-point line if vertical
                        x_line = [x1, x2]
                        y_line = [y1, y2]
                        x_points = [x1, x2]
                        y_points = [y1, y2]
                    
                    # Remove old preview
                    if self.guess_polynomial_line is not None:
                        try:
                            self.guess_polynomial_line.remove()
                        except (ValueError, RuntimeError):
                            pass
                    
                    # Draw new salmon-colored line preview (extended across full range)
                    self.guess_polynomial_line, = self.ax.plot(x_line, y_line, color='salmon',
                                                              linestyle='-', linewidth=2.5, alpha=0.7,
                                                              label='Polynomial Guess Preview')
                    
                    # Overlay the clicked points as markers on the extended line
                    # Store markers so they can be removed later
                    if self.guess_polynomial_clicked_points is not None:
                        try:
                            self.guess_polynomial_clicked_points.remove()
                        except (ValueError, RuntimeError):
                            pass
                    
                    if len(x_points) == 2:
                        self.guess_polynomial_clicked_points, = self.ax.plot(x_points, y_points, 'o', color='salmon', markersize=7, 
                                    markeredgewidth=1.5, markeredgecolor='darkred', zorder=20)
                    
                    self.canvas.draw_idle()

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
        """Draw the total line by summing all plotted profile lines from the item tracker."""
        # Remove any existing total lines first to avoid duplicates
        existing_total_lines = [line for line in self.ax.get_lines() if line.get_label() == "Total"]
        for line in existing_total_lines:
            line.remove()
        
        # Generate x values for plotting at high resolution
        x_plot = np.linspace(self.x_data.min(), self.x_data.max(), 10000)
        total_y = np.zeros_like(x_plot)
        
        print(f"[DEBUG_TOTAL_LINE] Starting total line calculation, x_plot range: {x_plot.min():.2f} to {x_plot.max():.2f}")
        
        # Simple approach: sum all profile lines from the item tracker
        # The item tracker's items dict contains all plotted profiles (gaussian, voigt, continuum, polynomial, chebyshev, etc.)
        if hasattr(self, 'item_tracker') and self.item_tracker and hasattr(self.item_tracker, 'items'):
            for item_id, item_data in self.item_tracker.items.items():
                item_type = item_data.get('type')
                line_obj = item_data.get('line_obj')
                
                # Only sum actual profile lines (not masks or other non-profile items)
                if item_type in ['gaussian', 'voigt', 'continuum', 'polynomial', 'chebyshev'] and line_obj is not None:
                    try:
                        # Get the plotted data from the line object
                        fit_x = line_obj.get_xdata()
                        fit_y = line_obj.get_ydata()
                        
                        print(f"[DEBUG_TOTAL_LINE] Adding {item_type} line: x range {fit_x.min():.2f}-{fit_x.max():.2f}, y range {fit_y.min():.6f}-{fit_y.max():.6f}")
                        
                        if len(fit_x) > 0 and len(fit_y) > 0:
                            # Interpolate to common x grid
                            from scipy.interpolate import interp1d
                            interp_func = interp1d(fit_x, fit_y, kind='linear', bounds_error=False, fill_value=0)
                            interpolated = interp_func(x_plot)
                            total_y += interpolated
                            print(f"[DEBUG_TOTAL_LINE]   Interpolated y range: {interpolated.min():.6f}-{interpolated.max():.6f}")
                    except Exception as e:
                        # Skip lines that can't be interpolated
                        print(f"[DEBUG] Warning: Could not add {item_type} line to total: {e}")
                        continue
        
        print(f"[DEBUG_TOTAL_LINE] Final total_y range: {total_y.min():.6f} to {total_y.max():.6f}")
        
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
                if self.mask_drawing_mode:
                    self._cancel_mask_drawing()
                    return
                # Cancel guess drawing first (removes polynomial points and preview lines)
                if self.guess_drawing_mode:
                    self._cancel_guess()
                    return
                self.on_deactivate_all()
                return
            
            # Handle Enter/Return to confirm guess or mask drawing
            if event.key in ['enter', 'return']:
                if self.mask_drawing_mode:
                    self._confirm_mask_regions()
                    return
                elif self.guess_drawing_mode:
                    self._confirm_guess()
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
                self.current_continuum_fit_id = None  # Clear fit tracking
                self.continuum_regions = []  # Clear any defined regions
                # Update dropdown to blank
                self.continuum_mode_dropdown.blockSignals(True)
                self.continuum_mode_dropdown.setCurrentIndex(0)
                self.continuum_mode_dropdown.blockSignals(False)
                print('Exiting continuum mode.')
            else:
                # Enter continuum mode
                self.continuum_mode = True
                self.current_continuum_fit_id = self.next_fit_id()  # Assign new fit_id for this session
                self.assign_fit_color(self.current_continuum_fit_id)
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
                self.current_continuum_fit_id = None  # Clear fit tracking
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
                self.current_continuum_fit_id = None  # Clear fit tracking
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
            # Register with ItemTracker - use same fit_id as the regions
            bounds_str = f"λ: {region_bounds[0]:.2f}-{region_bounds[1]:.2f} Å"
            continuum_cfg = self.colors['profiles']['continuum_line']
            self.register_item('continuum', f'Continuum (order {self.poly_order})', fit_dict=continuum_fit,
                             line_obj=continuum_line, position=bounds_str, color=continuum_cfg['color'],
                             fit_id=self.current_continuum_fit_id)
            
            # Record action for undo/redo
            self.record_action('fit_continuum', f'Fit Continuum (order {self.poly_order})')
            
            # Exit continuum mode and clear fit tracking
            self.continuum_mode = False
            self.current_continuum_fit_id = None
            
            # Save fit to .qsap file and print
            # Convert curve_fit covariance to lmfit-like format
            pcov_from_dict = continuum_fit.get('covariance')
            if pcov_from_dict is not None:
                param_names = [f'p0_c{i}' for i in range(len(coeffs))]
                param_values = list(coeffs)
                mock_result = self._convert_curve_fit_to_lmfit_like(param_names, param_values, pcov_from_dict)
                self.save_and_print_qsap_fit(continuum_fit, 'Continuum', 'Single', lmfit_result=mock_result)
            else:
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
                                 position=position_str, color=continuum_region_cfg['color'], bounds=region_bounds,
                                 fit_id=self.current_continuum_fit_id)
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
            print("            Or press ENTER/RETURN to use the full spectral range.")

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

        # Use full spectrum range for listfit with Enter/Return
        if event.key in ('enter', 'return') and self.listfit_mode and len(self.listfit_bounds) == 0:
            # Use full spectrum x-range as bounds
            left_bound = float(np.min(self.x_data))
            right_bound = float(np.max(self.x_data))
            self.listfit_bounds = [left_bound, right_bound]
            print(f"Listfit bounds set to full spectrum range: {left_bound:.2f} to {right_bound:.2f}. Opening component selection dialog...")
            self.record_action('set_listfit_bounds_full_range', f'Set Listfit bounds to full spectrum range λ={left_bound:.2f}-{right_bound:.2f} Å')
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
        
        # Exit constraint bounds setting mode with Escape
        if event.key == 'escape' and self.constraint_bounds_mode:
            print("[Constraint] Bounds setting cancelled.")
            self.constraint_bounds_mode = False
            self.constraint_parameter = None
            self.constraint_bounds_drag_start_x = None
            self.constraint_bounds_drag_end_x = None
            
            # Remove preview line
            if self.constraint_bounds_preview_line is not None:
                try:
                    self.constraint_bounds_preview_line.remove()
                except (ValueError, RuntimeError):
                    pass
            self.constraint_bounds_preview_line = None
            
            # Keep constraint editor dialog open so user can try again
            self.current_constraint_editor = None
            self.current_constraint_editor_dialog = None
            
            plt.draw()

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
                    y_fit_full = self.gaussian(x_fit, amp, mean, stddev) + continuum_for_plot  # For residuals
                    y_fit_plot = self.gaussian(x_fit, amp, mean, stddev)  # For display (without continuum offset)
                    residuals = comp_y - y_fit_full
                    # Calculate chi2 with optional errors
                    if comp_err is not None:
                        chi2 = np.sum((residuals ** 2) / comp_err) # Calculate chi2
                    else:
                        chi2 = np.sum(residuals ** 2) # Chi2 without errors
                    chi2_nu = chi2 / (len(x_fit) - len(params))# Calculate chi2 d.o.f.
                    
                    # Lazy import of scipy for interpolation
                    from scipy.interpolate import interp1d
                    interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                    # Convert curve_fit covariance to lmfit-like format
                    fit_gaussian = self.gaussian_fits[-1]
                    pcov_from_dict = fit_gaussian.get('covariance')
                    if pcov_from_dict is not None:
                        param_names = ['amp', 'mean', 'stddev']
                        param_values = [fit_gaussian['amp'], fit_gaussian['mean'], fit_gaussian['stddev']]
                        mock_result = self._convert_curve_fit_to_lmfit_like(param_names, param_values, pcov_from_dict)
                        self.save_and_print_qsap_fit(fit_gaussian, 'Gaussian', 'Single', lmfit_result=mock_result)
                    else:
                        self.save_and_print_qsap_fit(fit_gaussian, 'Gaussian', 'Single')

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
                
                # Store full pcov for later use when saving to .qsap
                self._multi_gaussian_full_pcov = pcov
                self._multi_gaussian_param_names = []
                
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
                    y_fit_plot = self.gaussian(x_fit, amp, mean, stddev)  # For display (without continuum offset)
                    continuum_sub_data = comp_ys[i // 3] - continuum_ys[i // 3]
                    residuals = continuum_sub_data - self.gaussian(x_fit, amp, mean, stddev)
                    
                    # Extract this component's covariance from the full covariance matrix
                    comp_cov_indices = [i, i+1, i+2]
                    comp_cov = pcov[np.ix_(comp_cov_indices, comp_cov_indices)]
                    
                    # Track parameter names for covariance storage
                    g_idx = i // 3
                    self._multi_gaussian_param_names.extend([f'g{g_idx}_amp', f'g{g_idx}_mu', f'g{g_idx}_sigma'])
                    
                    # Calculate chi2 or SSR depending on whether errors are available
                    if sigma_param is not None and i // 3 < len(component_boundaries):
                        start_idx, end_idx = component_boundaries[i // 3]
                        component_errors = sigma_param[start_idx:end_idx]
                        chi2 = np.sum((residuals / component_errors) ** 2)  # Proper chi-squared
                    else:
                        chi2 = np.sum(residuals ** 2)  # SSR without errors
                    chi2_nu = chi2 / (len(x_fit) - 3)  # 3 params per component
                    interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                    # Convert full curve_fit covariance to lmfit-like format
                    if hasattr(self, '_multi_gaussian_full_pcov') and self._multi_gaussian_full_pcov is not None:
                        param_names = self._multi_gaussian_param_names if hasattr(self, '_multi_gaussian_param_names') else []
                        param_values = list(params)  # All parameter values from curve_fit
                        if param_names and len(param_names) == len(param_values):
                            mock_result = self._convert_curve_fit_to_lmfit_like(param_names, param_values, self._multi_gaussian_full_pcov)
                            self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian', lmfit_result=mock_result)
                        else:
                            self.save_and_print_qsap_fit(multi_gaussian_components, 'Gaussian', 'Multi-Gaussian')
                    else:
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
                continuum_fit_dict = self.get_continuum_fit_dict(left_bound, right_bound)
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
                
                # Lazy imports for lmfit model fitting
                from lmfit import Model, Parameters
                
                initial_params = Parameters()
                if np.mean(continuum_subtracted_y) > 0:
                    initial_params.add('amp', value=max(continuum_subtracted_y) - min(continuum_subtracted_y))
                else:
                    initial_params.add('amp', value=min(continuum_subtracted_y) - max(continuum_subtracted_y))
                initial_params.add('center', value=center_guess)
                initial_params.add('sigma', value=np.std(comp_x)/10, min=0, max=sigma_max)  # Constrain sigma with upper bound
                initial_params.add('gamma', value=np.std(comp_x)/10, min=0, max=sigma_max)  # Constrain gamma similarly

                # Create the Voigt model and perform the fit
                voigt_model = Model(self.voigt)
                test_output = voigt_model.eval(params=initial_params, x=comp_x)
                if np.isnan(test_output).any():
                    print("Warning: Model function generated NaN values with initial parameters.")
                # Use weights if errors available, otherwise None
                weights = 1/comp_err if comp_err is not None else None
                result = voigt_model.fit(continuum_subtracted_y, x=comp_x, params=initial_params, weights=weights)
                
                # Extract covariance matrix from lmfit result
                # If result has full covariance, use it; otherwise build diagonal from stderr
                if result.covar is not None:
                    covariance = result.covar
                else:
                    # Build diagonal covariance matrix from parameter errors
                    param_names = [name for name in result.params.keys()]
                    param_errors = [result.params[name].stderr if result.params[name].stderr is not None else 1e-10 for name in param_names]
                    covariance = np.diag([e**2 for e in param_errors])
                
                # Convert to list for storage
                if isinstance(covariance, np.ndarray):
                    covariance = covariance.tolist()

                # Clear bounds
                for line in self.bound_lines:
                    line.remove()
                self.bound_lines.clear()
                self.bounds = []
                self.ax.figure.canvas.draw_idle()  # Redraw to show bound lines removed
                # Visualize the fit
                x_fit = np.linspace(left_bound, right_bound, len(continuum_for_plot))
                y_fit_full = result.eval(x=x_fit) + continuum_for_plot  # For residuals
                y_fit_plot = result.eval(x=x_fit)  # For display (without continuum offset)
                residuals = comp_y - y_fit_full
                # Calculate chi2 with optional errors
                if comp_err is not None:
                    chi2 = np.sum((residuals ** 2) / comp_err) # Calculate combined chi2
                else:
                    chi2 = np.sum(residuals ** 2) # Chi2 without errors
                chi2_nu = chi2 / (len(x_fit) - len(result.params)) # Calculate chi2 d.o.f.
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                    'N': np.log10(N) if N else None,
                    'continuum_fit_dict': continuum_fit_dict,  # Store the continuum fit for EW calculation
                    'covariance': covariance  # Store covariance matrix for error propagation
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
                voigt_cfg = self.colors['profiles']['voigt']
                self.register_item('voigt', f'Voigt', fit_dict=fit_results, line_obj=fit_results.get('line'),
                                 position=position_str, color=voigt_cfg['color'])
                
                # Record action for undo/redo
                self.record_action('fit_voigt', f'Fit Voigt at λ={fit_results.get("center", fit_results.get("mean", 0)):.2f} Å')
                
                # Save fit to .qsap file and print
                self.save_and_print_qsap_fit(fit_results, 'Voigt', 'Single', lmfit_result=result)
                
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
                y_fit_full = self.voigt(x_fit, amp, center, sigma, gamma) + existing_continuum  # For residuals
                y_fit_plot = self.voigt(x_fit, amp, center, sigma, gamma)  # For display
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                voigt_cfg = self.colors['profiles']['voigt']
                self.register_item('voigt', f'Voigt', fit_dict=fit_results, line_obj=fit_results.get('line'),
                                 position=position_str, color=voigt_cfg['color'])
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
                y_fit_full = self.gaussian(x_fit, amp, mean, stddev) + existing_continuum  # For residuals
                y_fit_plot = self.gaussian(x_fit, amp, mean, stddev)  # For display
                from scipy.interpolate import interp1d
                interpolator = interp1d(x_fit, y_fit_plot, kind='cubic', bounds_error=False, fill_value='extrapolate')
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
                    # Smoothing with keys '1'-'9' (quick Gaussian smoothing)
                    if event.key in '123456789':
                        try:
                            print(f"Key pressed: {event.key}")
                            key_num = int(event.key)
                            # Quick Gaussian smoothing via keyboard (no median)
                            self.smooth_spectrum(median_kernel=1, gaussian_sigma=key_num)
                            if self.spectrum_line:
                                self._update_spectrum_display(self.smoothed_spec)
                            self.fig.canvas.draw_idle()
                            self.last_applied_gaussian = key_num
                            self.last_applied_median = 1
                        except Exception as e:
                            print(f"Error: {e}")
                    # Reset to original spectrum with '0'
                    elif event.key == '0':
                        print("Key pressed:", event.key)
                        print("Going back to unsmoothed spectrum.")
                        if self.spectrum_line:
                            self._update_spectrum_display(self.original_spec)
                            self.last_applied_gaussian = 0.0
                            self.last_applied_median = 1
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
        self.listfit_window = ListfitWindow(self.listfit_bounds, self.resources_dir)
        self.listfit_window.fit_requested.connect(self.perform_listfit)
        self.listfit_window.bounds_cleared.connect(self.clear_listfit_bounds)
        self.listfit_window.cleanup_guesses.connect(self._remove_guess_lines)  # Clean up all drawn guesses when listfit closes
        self.listfit_window.components_changed.connect(self._on_listfit_components_changed)
        self.listfit_window.request_set_guess.connect(self.on_request_set_guess)
        self.listfit_window.request_remove_guess.connect(self._remove_guess)
        self.listfit_window.request_draw_data_mask_regions.connect(self.on_request_draw_data_mask_regions)
        self.listfit_window.request_draw_polynomial_mask_regions.connect(self.on_request_draw_polynomial_mask_regions)
        self.listfit_window.request_set_constraint_bounds.connect(self.on_request_set_constraint_bounds)
        self.listfit_window.show()
    
    def on_request_set_guess(self, row, component):
        """Handle request to set guess for a listfit component - click-drag to draw"""
        self.guess_drawing_mode = True
        self.current_component_for_guess = component
        self.current_component_row = row
        
        # Reset all drag tracking
        self.guess_mouse_down = False
        self.guess_drag_start_x = None
        self.guess_drag_start_y = None
        self.guess_drag_end_x = None
        self.guess_drag_end_y = None
        self.guess_center = None
        self.guess_sigma = None
        self.guess_amp = None
        
        # Initialize polynomial multi-point click mode
        self.polynomial_points = []
        self.polynomial_preview_points = []
        self.polynomial_order = component.get('order', 1)  # Get order from component, default to 1
        
        # Remove any old preview/polynomial lines
        if self.guess_preview_line is not None:
            try:
                self.guess_preview_line.remove()
            except (ValueError, RuntimeError):
                pass
        if self.guess_polynomial_line is not None:
            try:
                self.guess_polynomial_line.remove()
            except (ValueError, RuntimeError):
                pass
        self.guess_preview_line = None
        self.guess_polynomial_line = None
        
        comp_type = component.get('type', '').lower()
        print(f"\n{'='*70}")
        print(f"[Guess Mode] Setting guess for {comp_type.upper()} #{row + 1}")
        
        if comp_type in ['gaussian', 'voigt']:
            print(f"  • CLICK & DRAG on the plot to draw the guess:")
            print(f"    - Drag UP/DOWN to set AMPLITUDE")
            print(f"    - Drag LEFT/RIGHT to set WIDTH (sigma)")
            print(f"  • Press ENTER/RETURN to confirm guess")
            print(f"  • Press ESC to cancel")
        elif comp_type == 'polynomial':
            points_needed = self.polynomial_order + 1
            print(f"  • CLICK on the plot to set points for polynomial guess:")
            print(f"    - Polynomial order: {self.polynomial_order}")
            print(f"    - Points needed: {points_needed}")
            print(f"    - The polynomial will be fit through the clicked points")
            print(f"  • Press ENTER/RETURN to confirm guess")
            print(f"  • Press ESC to cancel")
        elif comp_type == 'chebyshev':
            degree = component.get('degree', 1)
            points_needed = degree + 2  # Need at least degree+1 points, use degree+2 for robustness
            print(f"  • CLICK on the plot to set points for Chebyshev guess:")
            print(f"    - Chebyshev degree: {degree}")
            print(f"    - Points needed: at least {degree + 1} (recommended {points_needed})")
            print(f"    - The Chebyshev polynomial will be fit through the clicked points")
            print(f"  • Press ENTER/RETURN to confirm guess")
            print(f"  • Press ESC to cancel")
        print(f"{'='*70}\n")
    
    def on_request_set_constraint_bounds(self, parameter):
        """Handle request to set constraint bounds by clicking and dragging on plot"""
        self.constraint_bounds_mode = True
        self.constraint_parameter = parameter
        self.constraint_bounds_drag_start_x = None
        self.constraint_bounds_drag_end_x = None
        
        # Remove any old preview line
        if self.constraint_bounds_preview_line is not None:
            try:
                self.constraint_bounds_preview_line.remove()
            except (ValueError, RuntimeError):
                pass
        self.constraint_bounds_preview_line = None
        
        param_display = {
            'amp': 'Amplitude',
            'mu': 'Center (mu)',
            'center': 'Center',
            'sigma': 'Width (sigma)',
            'gamma': 'Gamma',
        }
        
        param_name = param_display.get(parameter, parameter.upper())
        
        print(f"\n{'='*70}")
        print(f"[Constraint Bounds] Setting bounds for: {param_name}")
        print(f"  • CLICK & DRAG on the plot to set bounds:")
        print(f"    - Start position sets MINIMUM bound")
        print(f"    - End position sets MAXIMUM bound")
        print(f"  • Bounds will be applied to the constraint dialog")
        print(f"  • Press ESC to cancel bounds setting")
        print(f"{'='*70}\n")
    
    def _apply_constraint_bounds(self, x_min, x_max):
        """Apply the dragged bounds to the current constraint editor"""
        if not self.current_constraint_editor:
            print("[Constraint] Error: No constraint editor reference found")
            return
        
        # Map parameter name to constraint editor field
        parameter = self.constraint_parameter
        
        # Format bounds as strings
        min_str = f"{x_min:.2f}"
        max_str = f"{x_max:.2f}"
        
        # Update the appropriate fields in the constraint editor
        if parameter == 'amp':
            if hasattr(self.current_constraint_editor, 'amp_min'):
                self.current_constraint_editor.amp_min.setText(min_str)
            if hasattr(self.current_constraint_editor, 'amp_max'):
                self.current_constraint_editor.amp_max.setText(max_str)
            print(f"[Constraint] Amplitude bounds set: {min_str} to {max_str}")
        elif parameter == 'mu':
            if hasattr(self.current_constraint_editor, 'mean_min'):
                self.current_constraint_editor.mean_min.setText(min_str)
            if hasattr(self.current_constraint_editor, 'mean_max'):
                self.current_constraint_editor.mean_max.setText(max_str)
            print(f"[Constraint] Center (mu) bounds set: {min_str} to {max_str}")
        elif parameter == 'center':
            if hasattr(self.current_constraint_editor, 'center_min'):
                self.current_constraint_editor.center_min.setText(min_str)
            if hasattr(self.current_constraint_editor, 'center_max'):
                self.current_constraint_editor.center_max.setText(max_str)
            print(f"[Constraint] Center bounds set: {min_str} to {max_str}")
        elif parameter == 'sigma':
            if hasattr(self.current_constraint_editor, 'sigma_min'):
                self.current_constraint_editor.sigma_min.setText(min_str)
            if hasattr(self.current_constraint_editor, 'sigma_max'):
                self.current_constraint_editor.sigma_max.setText(max_str)
            print(f"[Constraint] Width (sigma) bounds set: {min_str} to {max_str}")
        elif parameter == 'gamma':
            if hasattr(self.current_constraint_editor, 'gamma_min'):
                self.current_constraint_editor.gamma_min.setText(min_str)
            if hasattr(self.current_constraint_editor, 'gamma_max'):
                self.current_constraint_editor.gamma_max.setText(max_str)
            print(f"[Constraint] Gamma bounds set: {min_str} to {max_str}")
        
        # Clear references
        self.current_constraint_editor = None
        self.current_constraint_editor_dialog = None
    
    def on_request_draw_data_mask_regions(self):
        """Handle request to draw data mask regions - click and drag to define regions"""
        self.mask_drawing_mode = True
        self.mask_type = 'data_mask'
        self.mask_regions = []
        self.mask_preview_rects = []
        self.mask_drag_start_x = None
        
        print(f"\n{'='*70}")
        print(f"[Mask Drawing Mode] Data Mask Regions")
        print(f"  • CLICK & DRAG on the plot to draw mask regions:")
        print(f"    - Drag LEFT to RIGHT to set the masked wavelength range")
        print(f"    - You can draw multiple regions")
        print(f"  • Press ENTER/RETURN to confirm all regions")
        print(f"  • Press ESC to cancel")
        print(f"{'='*70}\n")
    
    def on_request_draw_polynomial_mask_regions(self):
        """Handle request to draw polynomial mask regions - click and drag to define regions"""
        self.mask_drawing_mode = True
        self.mask_type = 'polynomial_mask'
        self.mask_regions = []
        self.mask_preview_rects = []
        self.mask_drag_start_x = None
        
        print(f"\n{'='*70}")
        print(f"[Mask Drawing Mode] Polynomial Exclude Regions")
        print(f"  • CLICK & DRAG on the plot to draw mask regions:")
        print(f"    - Drag LEFT to RIGHT to set the excluded wavelength range")
        print(f"    - You can draw multiple regions")
        print(f"    - Note: These don't apply to polynomial guesses from clicked points")
        print(f"  • Press ENTER/RETURN to confirm all regions")
        print(f"  • Press ESC to cancel")
        print(f"{'='*70}\n")
    
    def _confirm_mask_regions(self):
        """Confirm mask regions and add them to listfit window"""
        if not self.mask_regions:
            print("[Mask] No regions drawn - cancelled")
            self._cancel_mask_drawing()
            return
        
        print(f"\n{'='*70}")
        print(f"[Mask] ✓ Confirmed {len(self.mask_regions)} region(s) for {self.mask_type}:")
        for i, (x_min, x_max) in enumerate(self.mask_regions):
            print(f"  Region {i+1}: λ={x_min:.2f} to {x_max:.2f} Å")
        print(f"{'='*70}\n")
        
        # Add the regions to listfit window
        if self.mask_type == 'data_mask':
            self.listfit_window.add_drawn_data_mask_regions(self.mask_regions)
        elif self.mask_type == 'polynomial_mask':
            self.listfit_window.add_drawn_polynomial_mask_regions(self.mask_regions)
        
        self._cancel_mask_drawing()
    
    def _cancel_mask_drawing(self):
        """Cancel mask drawing mode and clean up"""
        # Remove preview rectangles
        for rect in self.mask_preview_rects:
            try:
                rect.remove()
            except (ValueError, RuntimeError):
                pass
        
        self.mask_preview_rects.clear()
        self.mask_regions.clear()
        self.mask_drawing_mode = False
        self.mask_type = None
        self.mask_drag_start_x = None
        
        self.canvas.draw()
    
    def on_item_display_toggled(self, item_id, display_state):
        """Handle Display checkbox toggle in Item Tracker"""
        if item_id not in self.item_id_map:
            return
        
        item_info = self.item_id_map[item_id]
        line_obj = item_info.get('line_obj')
        
        if line_obj is None:
            return
        
        # Toggle visibility
        if display_state:
            # Turning on - restore visibility and original zorder
            line_obj.set_visible(True)
            # Restore zorder if it was saved
            if 'original_zorder' in item_info:
                line_obj.set_zorder(item_info['original_zorder'])
        else:
            # Turning off - hide the line but save zorder first
            if 'original_zorder' not in item_info:
                item_info['original_zorder'] = line_obj.get_zorder()
            line_obj.set_visible(False)
        
        # Redraw canvas to reflect changes
        self.canvas.draw_idle()
    
    def _update_guess_preview_salmon(self):
        """Update the live preview of the Gaussian/Voigt with salmon color (deprecated - use on_mouse_move instead)"""
        # This method is now handled in on_mouse_move for direct drag preview
        pass
    
    def _confirm_guess(self):
        """Confirm the guess and update the component, creating a persistent line"""
        if not self.current_component_for_guess:
            print("[Guess] No component selected - cancelled")
            self._cancel_guess()
            return
        
        comp_type = self.current_component_for_guess.get('type', '').lower()
        
        if comp_type in ['gaussian', 'voigt']:
            # Confirm Gaussian/Voigt guess
            if self.guess_center is None or self.guess_amp is None or self.guess_sigma is None:
                print("[Guess] Incomplete Gaussian/Voigt guess - cancelled")
                self._cancel_guess()
                return
            
            # Update the component's guess dictionary
            guess_dict = {
                'center': self.guess_center,
                'amp': self.guess_amp,
            }
            
            if comp_type == 'gaussian':
                guess_dict['stddev'] = self.guess_sigma
            else:  # voigt
                guess_dict['sigma'] = self.guess_sigma
                guess_dict['gamma'] = self.guess_sigma * 0.01  # Small Lorentzian contribution
            
            self.current_component_for_guess['guess'] = guess_dict
            
            print(f"\n{'='*70}")
            print(f"[Guess] ✓ Confirmed for {comp_type.upper()}:")
            print(f"  Center (λ): {self.guess_center:.2f} Å")
            print(f"  Amplitude: {self.guess_amp:.4f}")
            print(f"  Width (sigma): {self.guess_sigma:.2f} Å")
            print(f"{'='*70}\n")
            
            # Create persistent guess line on the plot (salmon color)
            self._plot_guess_line(self.current_component_for_guess)
        
        elif comp_type == 'polynomial':
            # Confirm polynomial guess
            if 'guess' not in self.current_component_for_guess or not self.current_component_for_guess['guess']:
                print("[Guess] Incomplete polynomial guess - cancelled")
                self._cancel_guess()
                return
            
            guess = self.current_component_for_guess['guess']
            print(f"\n{'='*70}")
            print(f"[Guess] ✓ Confirmed for POLYNOMIAL:")
            
            # Handle both old format (x1, y1, x2, y2) and new format (coefficients)
            if 'coefficients' in guess:
                # New multi-point format
                order = guess.get('order', 1)
                coeffs = guess.get('coefficients', [])
                x_points = guess.get('x_points', [])
                y_points = guess.get('y_points', [])
                print(f"  Order: {order}")
                print(f"  Points: {len(x_points)}")
                
                # Extend x_min and x_max to cover the FULL listfit fitting range
                if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                    x_min = min(self.listfit_bounds)
                    x_max = max(self.listfit_bounds)
                    print(f"  Fitted range extended to listfit bounds: λ={x_min:.2f} to {x_max:.2f} Å")
                else:
                    # Fallback to original points range if no listfit bounds
                    x_min = guess.get('x_min', 0)
                    x_max = guess.get('x_max', 0)
                    print(f"  Range: λ={x_min:.2f} to {x_max:.2f} Å")
                
                # Update guess with full range
                guess['x_min'] = x_min
                guess['x_max'] = x_max
                
                # Format polynomial with coefficients in scientific notation for clarity
                poly_str = "y = "
                for i, coeff in enumerate(coeffs):
                    power = len(coeffs) - 1 - i
                    if power == 0:
                        poly_str += f"{coeff:.3e}"
                    elif power == 1:
                        poly_str += f"{coeff:.3e}*x + "
                    else:
                        poly_str += f"{coeff:.3e}*x^{power} + "
                print(f"  Equation: {poly_str}")
            else:
                # Old drag format
                print(f"  Equation: y = {guess.get('slope', 0):.6f} * x + {guess.get('intercept', 0):.4f}")
                print(f"  From: λ={guess.get('x1', 0):.2f}, y={guess.get('y1', 0):.4f}")
                print(f"  To:   λ={guess.get('x2', 0):.2f}, y={guess.get('y2', 0):.4f}")
            
            print(f"{'='*70}\n")
            
            # Exit guess drawing mode BEFORE plotting to prevent on_mouse_move() from redrawing preview
            self.guess_drawing_mode = False
            
            # Remove temporary polynomial preview line before plotting extended persistent line
            if self.guess_polynomial_line is not None:
                try:
                    self.guess_polynomial_line.remove()
                    print("[DEBUG] Removed temporary polynomial preview line")
                except (ValueError, RuntimeError):
                    pass
            
            # Also remove temporary clicked points markers
            if self.guess_polynomial_clicked_points is not None:
                try:
                    self.guess_polynomial_clicked_points.remove()
                    print("[DEBUG] Removed temporary clicked points markers")
                except (ValueError, RuntimeError):
                    pass
            
            # Create persistent polynomial guess line on the plot (extended across full range)
            self._plot_guess_line(self.current_component_for_guess)
            
            # Mark that polynomial guess was confirmed (so we don't remove it in _cancel_guess)
            self._polynomial_guess_confirmed = True
        
        elif comp_type == 'chebyshev':
            # Confirm Chebyshev guess
            if 'guess' not in self.current_component_for_guess or not self.current_component_for_guess['guess']:
                print("[Guess] Incomplete Chebyshev guess - cancelled")
                self._cancel_guess()
                return
            
            guess = self.current_component_for_guess['guess']
            print(f"\n{'='*70}")
            print(f"[Guess] ✓ Confirmed for CHEBYSHEV:")
            
            # Handle multi-point Chebyshev format
            if 'coefficients' in guess:
                degree = guess.get('degree', 1)
                coeffs = guess.get('coefficients', [])
                x_points = guess.get('x_points', [])
                y_points = guess.get('y_points', [])
                domain = guess.get('domain', [x_points[0] if x_points else 0, x_points[-1] if x_points else 0])
                print(f"  Degree: {degree}")
                print(f"  Points: {len(x_points)}")
                print(f"  Domain: [{domain[0]:.2f}, {domain[1]:.2f}] Å")
                
                # Extend x_min and x_max to cover the FULL listfit fitting range
                if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                    x_min = min(self.listfit_bounds)
                    x_max = max(self.listfit_bounds)
                    print(f"  Fitted range extended to listfit bounds: λ={x_min:.2f} to {x_max:.2f} Å")
                else:
                    x_min = guess.get('x_min', domain[0])
                    x_max = guess.get('x_max', domain[1])
                
                # Update guess with full range
                guess['x_min'] = x_min
                guess['x_max'] = x_max
                
                # Format Chebyshev coefficients for display
                coeffs_str = ', '.join([f"{c:.3e}" for c in coeffs])
                print(f"  Coefficients: [{coeffs_str}]")
            else:
                print(f"  No Chebyshev data found")
            
            print(f"{'='*70}\n")
            
            # Exit guess drawing mode BEFORE plotting
            self.guess_drawing_mode = False
            
            # Remove temporary Chebyshev preview line
            if hasattr(self, 'guess_chebyshev_line') and self.guess_chebyshev_line is not None:
                try:
                    self.guess_chebyshev_line.remove()
                    print("[DEBUG] Removed temporary Chebyshev preview line")
                except (ValueError, RuntimeError):
                    pass
            
            # Remove temporary clicked points markers
            if hasattr(self, 'chebyshev_preview_points'):
                for marker in self.chebyshev_preview_points:
                    try:
                        marker.remove()
                    except (ValueError, RuntimeError):
                        pass
                self.chebyshev_preview_points.clear()
            
            # Create persistent Chebyshev guess line on the plot
            self._plot_guess_line(self.current_component_for_guess)
            
            # Mark that Chebyshev guess was confirmed
            self._chebyshev_guess_confirmed = True
        
        # Update the table display in listfit_window
        if hasattr(self, 'listfit_window') and self.listfit_window:
            self.listfit_window.component_list._update_guess_indicator(self.current_component_row)
        
        self._cancel_guess()
    
    def _plot_guess_line(self, component):
        """Plot a persistent salmon-colored line for a confirmed Gaussian/Voigt/Polynomial guess
        
        For polynomial guesses: Plot across full Listfit x-range after guess is confirmed.
        This line will be removed after fitting completes or fails.
        """
        if not component or 'guess' not in component:
            return
        
        guess = component.get('guess', {})
        comp_id = component.get('id')
        comp_type = component.get('type', '').lower()
        
        # Handle polynomial guesses (plot across full Listfit x-range)
        if comp_type == 'polynomial':
            print(f"[DEBUG] Plotting polynomial guess #{comp_id}")
            print(f"[DEBUG] listfit_bounds: {self.listfit_bounds}")
            print(f"[DEBUG] guess dict keys: {guess.keys()}")
            
            # Support both old format (x1, y1, x2, y2) and new format (coefficients, x_points, y_points)
            if 'coefficients' in guess:
                # New multi-point format
                try:
                    coeffs = guess.get('coefficients', [])
                    print(f"[DEBUG] coeffs: {coeffs}")
                    # Use listfit_bounds for full x-range
                    if coeffs and self.listfit_bounds and len(self.listfit_bounds) >= 2:
                        x_min, x_max = self.listfit_bounds[0], self.listfit_bounds[1]
                        x_line = np.linspace(x_min, x_max, 200)
                        y_line = np.polyval(coeffs, x_line)
                        label_str = f'Poly Guess #{comp_id} (order {guess.get("order", 1)})'
                        print(f"[DEBUG] Plotting polynomial with coefficients across x_range [{x_min}, {x_max}]")
                    else:
                        print(f"[DEBUG] Skipping: coeffs={bool(coeffs)}, listfit_bounds={bool(self.listfit_bounds)}")
                        return
                except Exception as e:
                    print(f"[Guess] Error plotting polynomial with coefficients: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            elif guess.get('x1') is not None and guess.get('x2') is not None:
                # Old drag format (linear only)
                print(f"[DEBUG] Old format polynomial: x1={guess.get('x1')}, x2={guess.get('x2')}, y1={guess.get('y1')}, y2={guess.get('y2')}")
                if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                    x_min, x_max = self.listfit_bounds[0], self.listfit_bounds[1]
                    # Create linear polynomial across full range
                    x1 = guess.get('x1')
                    x2 = guess.get('x2')
                    y1 = guess.get('y1')
                    y2 = guess.get('y2')
                    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                    x_line = np.linspace(x_min, x_max, 200)
                    y_line = y1 + slope * (x_line - x1)
                    label_str = f'Poly Guess #{comp_id}'
                    print(f"[DEBUG] Plotting linear polynomial across x_range [{x_min}, {x_max}], slope={slope}")
                else:
                    print(f"[DEBUG] Skipping old format: listfit_bounds not properly set")
                    return
            else:
                print(f"[DEBUG] No polynomial data found in guess")
                return
            
            # Remove old line if exists
            if comp_id in self.guess_lines:
                try:
                    self.guess_lines[comp_id].remove()
                except (ValueError, RuntimeError):
                    pass
            
            # Plot salmon-colored line across full Listfit x-range (z-order 5 so profiles appear on top)
            line, = self.ax.plot(x_line, y_line, color='salmon', linestyle='-',
                               linewidth=2.5, alpha=0.8,
                               label=label_str, zorder=5)
            self.guess_lines[comp_id] = line
            print(f"[DEBUG] Successfully plotted polynomial guess #{comp_id}")
            self.canvas.draw_idle()  # Use canvas.draw_idle() to refresh the plot properly
            return
        
        # Handle Chebyshev guesses
        if comp_type == 'chebyshev':
            if 'guess' not in component or 'coefficients' not in component.get('guess', {}):
                return
            
            guess = component.get('guess', {})
            coeffs = guess.get('coefficients', [])
            degree = guess.get('degree', 1)
            domain = guess.get('domain', [guess.get('x_min', 0), guess.get('x_max', 0)])
            
            try:
                # Use listfit_bounds for full x-range if available
                if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                    x_min, x_max = self.listfit_bounds[0], self.listfit_bounds[1]
                    x_line = np.linspace(x_min, x_max, 200)
                    label_str = f'Cheb Guess #{comp_id} (deg {degree})'
                    print(f"[DEBUG] Plotting Chebyshev guess #{comp_id} across x_range [{x_min}, {x_max}]")
                else:
                    x_min, x_max = domain[0], domain[1]
                    x_line = np.linspace(x_min, x_max, 200)
                    label_str = f'Cheb Guess #{comp_id}'
                    print(f"[DEBUG] Plotting Chebyshev guess #{comp_id} across domain [{x_min}, {x_max}]")
                
                # Rescale x to [-1, 1] for Chebyshev evaluation
                x_rescaled = 2 * (x_line - domain[0]) / (domain[1] - domain[0]) - 1
                y_line = np.polynomial.chebyshev.chebval(x_rescaled, coeffs)
                
                # Remove old line if exists
                if comp_id in self.guess_lines:
                    try:
                        self.guess_lines[comp_id].remove()
                    except (ValueError, RuntimeError):
                        pass
                
                # Plot medium sea green colored line (distinct from polynomial salmon)
                line, = self.ax.plot(x_line, y_line, color='mediumseagreen', linestyle='-',
                                    linewidth=2.5, alpha=0.8,
                                    label=label_str, zorder=5)
                self.guess_lines[comp_id] = line
                print(f"[DEBUG] Successfully plotted Chebyshev guess #{comp_id}")
                self.canvas.draw_idle()
                return
            except Exception as e:
                print(f"[Guess] Error plotting Chebyshev guess: {e}")
                import traceback
                traceback.print_exc()
                return
        
        # Handle Gaussian/Voigt guesses
        if guess.get('center') is None or guess.get('amp') is None:
            return
        
        guess_center = guess.get('center')
        guess_amp = guess.get('amp')
        
        # Determine width parameter name
        if comp_type == 'gaussian':
            guess_width = guess.get('stddev', 0.5)
        else:  # voigt
            guess_width = guess.get('sigma', 0.5)
        
        # Remove old line if it exists
        if comp_id in self.guess_lines:
            try:
                self.guess_lines[comp_id].remove()
            except (ValueError, RuntimeError):
                pass
        
        # Create new persistent line
        x_line = np.linspace(guess_center - 3*guess_width, guess_center + 3*guess_width, 200)
        
        try:
            if comp_type == 'gaussian':
                y_line = guess_amp * np.exp(-((x_line - guess_center)**2) / (2 * guess_width**2))
            elif comp_type == 'voigt':
                # Compensate for Voigt normalization: voigt divides by (sigma * sqrt(2*pi))
                gamma = guess.get('gamma', guess_width * 0.01)  # Use stored gamma or default
                s2pi = np.sqrt(2*np.pi)
                amp_compensated = guess_amp * (guess_width * s2pi)
                y_line = self.voigt(x_line, amp_compensated, guess_center, guess_width, gamma)
            else:
                return
            
            # Check if there's exactly ONE polynomial or chebyshev guess in listfit_components
            # If so, add the polynomial/chebyshev baseline to the profile (floor for profiles to stand on)
            poly_guesses = [c for c in self.listfit_components if c.get('type') in ['polynomial', 'chebyshev'] and c.get('guess')]
            if len(poly_guesses) == 1:
                poly_comp = poly_guesses[0]
                poly_guess = poly_comp.get('guess', {})
                try:
                    # Get polynomial coefficients (new format)
                    if 'coefficients' in poly_guess:
                        coeffs = poly_guess.get('coefficients')
                        if coeffs:
                            # Evaluate polynomial at x_line points and add to profile
                            poly_baseline = np.polyval(coeffs, x_line)
                            y_line = y_line + poly_baseline
                            print(f"[Guess] Added polynomial baseline from component #{poly_comp.get('id')}")
                    # Also handle old linear format
                    elif poly_guess.get('x1') is not None and poly_guess.get('x2') is not None:
                        x1_poly = poly_guess.get('x1')
                        x2_poly = poly_guess.get('x2')
                        y1_poly = poly_guess.get('y1')
                        y2_poly = poly_guess.get('y2')
                        # Only add baseline for x values within the polynomial range
                        in_range = (x_line >= min(x1_poly, x2_poly)) & (x_line <= max(x1_poly, x2_poly))
                        poly_baseline = y1_poly + (y2_poly - y1_poly) * (x_line - x1_poly) / (x2_poly - x1_poly)
                        y_line[in_range] = y_line[in_range] + poly_baseline[in_range]
                        print(f"[Guess] Added linear polynomial baseline from component #{poly_comp.get('id')}")
                except Exception as e:
                    print(f"[Guess] Warning: Could not add polynomial baseline: {e}")
            
            # Plot as salmon-colored solid line with HIGHER z-order so it appears on top of polynomial
            line, = self.ax.plot(x_line, y_line, color='salmon', linestyle='-',
                                linewidth=2, alpha=0.8, label=f'Guess #{comp_id}', zorder=15)
            self.guess_lines[comp_id] = line
            plt.draw()
        except Exception as e:
            print(f"Error plotting guess line: {e}")
    
    def _plot_polynomial_guess_line(self, x1, x2, y1, y2):
        """Plot a salmon-colored polynomial guess line during confirmation, extended across full fitting range"""
        # Remove old polynomial preview line
        if self.guess_polynomial_line is not None:
            try:
                self.guess_polynomial_line.remove()
            except (ValueError, RuntimeError):
                pass
        
        # Calculate line equation from the two clicked points
        if abs(x2 - x1) > 1e-10:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Extend to full listfit fitting range (or use current bounds as fallback)
            if self.listfit_bounds and len(self.listfit_bounds) >= 2:
                x_min = min(self.listfit_bounds)
                x_max = max(self.listfit_bounds)
            else:
                x_min = self.x_lower_bound
                x_max = self.x_upper_bound
            
            # Generate extended line across full range
            x_line = np.linspace(x_min, x_max, 200)
            y_line = slope * x_line + intercept
            x_points = [x1, x2]
            y_points = [y1, y2]
        else:
            # Fallback to simple two-point line if vertical
            x_line = [x1, x2]
            y_line = [y1, y2]
            x_points = [x1, x2]
            y_points = [y1, y2]
        
        # Draw salmon-colored line (extended across full range)
        self.guess_polynomial_line, = self.ax.plot(x_line, y_line, color='salmon',
                                                   linestyle='-', linewidth=2.5, alpha=0.8,
                                                   label='Polynomial Guess')
        
        # Overlay the clicked points as markers on the extended line
        if len(x_points) == 2:
            self.ax.plot(x_points, y_points, 'o', color='salmon', markersize=7,
                        markeredgewidth=1.5, markeredgecolor='darkred', zorder=20)
        
        self.canvas.draw_idle()
    
    def _remove_guess(self, component_id):
        """Remove a guess line from the plot and clear the guess dictionary"""
        # Remove the line from plot
        if component_id in self.guess_lines:
            try:
                self.guess_lines[component_id].remove()
                print(f"[DEBUG] Removed guess line for component #{component_id}")
            except (ValueError, RuntimeError) as e:
                print(f"[DEBUG] Warning: Could not remove guess line for component #{component_id}: {e}")
            del self.guess_lines[component_id]
        
        # Find the component and clear its guess
        for component in self.listfit_components:
            if component.get('id') == component_id:
                comp_type = component.get('type', '').lower()
                # Clear the guess dictionary based on component type
                if comp_type == 'polynomial':
                    component['guess'] = {}
                    print(f"[Guess] Removed polynomial guess for component #{component_id}")
                elif comp_type == 'gaussian':
                    component['guess'] = {'center': None, 'amp': None, 'stddev': None}
                    print(f"[Guess] Removed Gaussian guess for component #{component_id}")
                elif comp_type == 'voigt':
                    component['guess'] = {'center': None, 'amp': None, 'sigma': None, 'gamma': None}
                    print(f"[Guess] Removed Voigt guess for component #{component_id}")
                break
        
        plt.draw()
    
    def public_remove_guess(self, component_id):
        """Public method for removing a guess - called from listfit_window context menu"""
        self._remove_guess(component_id)
    
    def _add_polynomial_point(self, x, y):
        """Add a point for polynomial multi-point click mode"""
        # Determine polynomial order from component type
        if not hasattr(self, 'polynomial_points'):
            self.polynomial_points = []
        if not hasattr(self, 'polynomial_order'):
            self.polynomial_order = 1
        
        # Add the point
        self.polynomial_points.append((x, y))
        points_needed = self.polynomial_order + 1
        points_have = len(self.polynomial_points)
        points_remaining = max(0, points_needed - points_have)
        
        print(f"[Polynomial Guess] Point {points_have} clicked at λ={x:.2f} Å, flux={y:.4f}")
        print(f"  Points needed: {points_needed}, Points remaining: {points_remaining}")
        
        # Plot the clicked point as a small marker
        point_marker, = self.ax.plot([x], [y], 'o', color='salmon', markersize=8, zorder=15)
        self.polynomial_preview_points.append(point_marker)
        
        # If we have enough points, fit the polynomial
        if points_have >= points_needed:
            self._fit_polynomial_from_points()
        
        self.canvas.draw()
    
    def _fit_polynomial_from_points(self):
        """Fit a polynomial through the collected points"""
        if len(self.polynomial_points) < 2:
            print("[Polynomial Guess] Need at least 2 points to fit polynomial")
            return
        
        # Extract x and y coordinates
        x_pts = np.array([pt[0] for pt in self.polynomial_points])
        y_pts = np.array([pt[1] for pt in self.polynomial_points])
        
        # Fit polynomial
        try:
            coeffs = np.polyfit(x_pts, y_pts, self.polynomial_order)
            
            # Generate smooth curve for display
            x_line = np.linspace(x_pts.min(), x_pts.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            
            # Store in guess
            self.current_component_for_guess['guess'] = {
                'coefficients': coeffs.tolist(),
                'order': self.polynomial_order,
                'x_points': x_pts.tolist(),
                'y_points': y_pts.tolist(),
                'x_min': float(x_pts.min()),
                'x_max': float(x_pts.max())
            }
            
            # Plot the fitted curve
            self._plot_polynomial_fitted_curve(x_line, y_line)
            
            # Print info with coefficients in scientific notation for clarity
            poly_str = "y = "
            for i, coeff in enumerate(coeffs):
                power = len(coeffs) - 1 - i
                # Use scientific notation for clarity, especially for tiny coefficients
                if power == 0:
                    poly_str += f"{coeff:.3e}"
                elif power == 1:
                    poly_str += f"{coeff:.3e}*x + "
                else:
                    poly_str += f"{coeff:.3e}*x^{power} + "
            
            print(f"[Polynomial Guess] ✓ Fitted polynomial (order {self.polynomial_order}):")
            print(f"  {poly_str}")
            print(f"[Polynomial Guess] Press ENTER to confirm, or ESC to cancel")
            
        except np.linalg.LinAlgError as e:
            print(f"[Polynomial Guess] Error fitting polynomial: {e}")
    
    def _plot_polynomial_fitted_curve(self, x_line, y_line):
        """Plot the fitted polynomial curve"""
        # Remove old guess polynomial line
        if self.guess_polynomial_line is not None:
            try:
                self.guess_polynomial_line.remove()
            except (ValueError, RuntimeError):
                pass
        
        # Draw salmon-colored curve
        self.guess_polynomial_line, = self.ax.plot(x_line, y_line, color='salmon',
                                                   linestyle='-', linewidth=2.5, alpha=0.8,
                                                   label='Polynomial Guess', zorder=14)
        self.canvas.draw()
    
    def _add_chebyshev_point(self, x, y):
        """Add a point for Chebyshev multi-point click mode"""
        # Initialize Chebyshev points list if needed
        if not hasattr(self, 'chebyshev_points'):
            self.chebyshev_points = []
        if not hasattr(self, 'chebyshev_preview_points'):
            self.chebyshev_preview_points = []
        
        # Get degree from component
        degree = self.current_component_for_guess.get('degree', 1)
        points_needed = degree + 1
        
        # Add the point
        self.chebyshev_points.append((x, y))
        points_have = len(self.chebyshev_points)
        points_remaining = max(0, points_needed - points_have)
        
        print(f"[Chebyshev Guess] Point {points_have} clicked at λ={x:.2f} Å, flux={y:.4f}")
        print(f"  Points needed: {points_needed}, Points remaining: {points_remaining}")
        
        # Plot the clicked point as a small marker
        point_marker, = self.ax.plot([x], [y], 'o', color='mediumseagreen', markersize=8, zorder=15)
        self.chebyshev_preview_points.append(point_marker)
        
        # If we have enough points, fit the Chebyshev
        if points_have >= points_needed:
            self._fit_chebyshev_from_points()
        
        self.canvas.draw()
    
    def _fit_chebyshev_from_points(self):
        """Fit a Chebyshev polynomial through the collected points"""
        if len(self.chebyshev_points) < 2:
            print("[Chebyshev Guess] Need at least 2 points to fit Chebyshev")
            return
        
        # Extract x and y coordinates
        x_pts = np.array([pt[0] for pt in self.chebyshev_points])
        y_pts = np.array([pt[1] for pt in self.chebyshev_points])
        
        # Fit Chebyshev polynomial
        try:
            degree = self.current_component_for_guess.get('degree', 1)
            from numpy.polynomial import chebyshev
            cheb_fit = chebyshev.Chebyshev.fit(x_pts, y_pts, degree, domain=[x_pts.min(), x_pts.max()])
            
            # Generate smooth curve for display
            x_line = np.linspace(x_pts.min(), x_pts.max(), 100)
            # Evaluate Chebyshev using the fitted object's call method
            y_line = cheb_fit(x_line)
            
            # Store in guess
            self.current_component_for_guess['guess'] = {
                'coefficients': cheb_fit.coef.tolist(),
                'degree': degree,
                'domain': [float(cheb_fit.domain[0]), float(cheb_fit.domain[1])],
                'x_points': x_pts.tolist(),
                'y_points': y_pts.tolist(),
                'x_min': float(x_pts.min()),
                'x_max': float(x_pts.max())
            }
            
            # Plot the fitted curve
            self._plot_chebyshev_fitted_curve(x_line, y_line)
            
            # Print info with coefficients
            coeffs_str = ', '.join([f"{c:.3e}" for c in cheb_fit.coef])
            print(f"[Chebyshev Guess] ✓ Fitted Chebyshev (degree {degree}):")
            print(f"  Coefficients: [{coeffs_str}]")
            print(f"  Domain: [{cheb_fit.domain[0]:.2f}, {cheb_fit.domain[1]:.2f}] Å")
            print(f"[Chebyshev Guess] Press ENTER to confirm, or ESC to cancel")
            
        except Exception as e:
            print(f"[Chebyshev Guess] Error fitting Chebyshev: {e}")
    
    def _plot_chebyshev_fitted_curve(self, x_line, y_line):
        """Plot the fitted Chebyshev curve"""
        # Remove old guess Chebyshev line
        if not hasattr(self, 'guess_chebyshev_line'):
            self.guess_chebyshev_line = None
        if self.guess_chebyshev_line is not None:
            try:
                self.guess_chebyshev_line.remove()
            except (ValueError, RuntimeError):
                pass
        
        # Draw medium sea green colored curve (distinct from salmon polynomial)
        self.guess_chebyshev_line, = self.ax.plot(x_line, y_line, color='mediumseagreen',
                                                   linestyle='-', linewidth=2.5, alpha=0.8,
                                                   label='Chebyshev Guess', zorder=14)
        self.canvas.draw()
    
    def _cancel_guess(self):
        """Cancel guess drawing mode (removes temporary preview only, NOT persistent lines)"""
        # Remove preview lines ONLY (not persistent lines in guess_lines dict)
        if self.guess_preview_line is not None:
            try:
                self.guess_preview_line.remove()
            except (ValueError, RuntimeError):
                pass
        
        # Remove polynomial preview markers (clicked points)
        if not hasattr(self, 'polynomial_preview_points'):
            self.polynomial_preview_points = []
        for marker in self.polynomial_preview_points:
            try:
                marker.remove()
            except (ValueError, RuntimeError):
                pass
        self.polynomial_preview_points.clear()
        if hasattr(self, 'polynomial_points'):
            self.polynomial_points.clear()
        
        # Remove Chebyshev preview markers (clicked points)
        if not hasattr(self, 'chebyshev_preview_points'):
            self.chebyshev_preview_points = []
        for marker in self.chebyshev_preview_points:
            try:
                marker.remove()
            except (ValueError, RuntimeError):
                pass
        self.chebyshev_preview_points.clear()
        if hasattr(self, 'chebyshev_points'):
            self.chebyshev_points.clear()
        
        # NOTE: Do NOT remove self.guess_polynomial_line here if a guess was confirmed
        # The persistent line should already be in self.guess_lines and will be drawn
        # Only remove the temporary preview if it wasn't confirmed
        if self.guess_polynomial_line is not None and not hasattr(self, '_polynomial_guess_confirmed'):
            try:
                self.guess_polynomial_line.remove()
            except (ValueError, RuntimeError):
                pass
        self._polynomial_guess_confirmed = False
        
        # Remove Chebyshev guess line preview if not confirmed
        if not hasattr(self, 'guess_chebyshev_line'):
            self.guess_chebyshev_line = None
        if self.guess_chebyshev_line is not None and not hasattr(self, '_chebyshev_guess_confirmed'):
            try:
                self.guess_chebyshev_line.remove()
            except (ValueError, RuntimeError):
                pass
        self._chebyshev_guess_confirmed = False
        
        # Clean up polynomial baseline line preview
        if self.guess_polynomial_baseline_line is not None:
            try:
                self.guess_polynomial_baseline_line.remove()
            except (ValueError, RuntimeError):
                pass
            self.guess_polynomial_baseline_line = None
        
        # Reset variables
        self.guess_drawing_mode = False
        self.current_component_for_guess = None
        self.current_component_row = None
        self.guess_center = None
        self.guess_sigma = None
        self.guess_amp = None
        self.guess_preview_line = None
        
        # Reset drag tracking
        self.guess_mouse_down = False
        self.guess_drag_start_x = None
        self.guess_drag_start_y = None
        self.guess_drag_end_x = None
        self.guess_drag_end_y = None
        
        plt.draw()
    
    def on_remove_all_plotted_features(self):
        """Remove all plotted features except spectrum with confirmation dialog"""
        # Show confirmation dialog
        reply = QtWidgets.QMessageBox.question(
            self,
            "Remove All Plotted Features",
            "Are you sure you want to remove all plotted features (except for the plotted spectrum)? Progress may be lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            self.remove_all_plotted_features()
    
    def remove_all_plotted_features(self):
        """Internally remove all plotted features (guesses, fits, bounds, etc.)"""
        print("[DEBUG] Removing all plotted features...")
        
        try:
            # Cancel any active guess drawing
            if self.guess_drawing_mode:
                self._cancel_guess()
            
            # Remove all guess lines
            for comp_id in list(self.guess_lines.keys()):
                try:
                    self.guess_lines[comp_id].remove()
                except (ValueError, RuntimeError):
                    pass
            self.guess_lines.clear()
            
            # Remove all polynomial preview points
            for marker in self.polynomial_preview_points:
                try:
                    marker.remove()
                except (ValueError, RuntimeError):
                    pass
            self.polynomial_preview_points.clear()
            self.polynomial_points.clear()
            
            # Remove all listfit bound lines and clear components
            for line in self.listfit_bound_lines:
                try:
                    line.remove()
                except (ValueError, RuntimeError):
                    pass
            self.listfit_bound_lines.clear()
            self.listfit_bounds = []
            self.listfit_components = []
            
            # Remove all continuum fits and patches
            for fit in self.continuum_fits:
                try:
                    if 'line' in fit and fit['line'] is not None:
                        fit['line'].remove()
                except (ValueError, RuntimeError):
                    pass
            self.continuum_fits.clear()
            
            for patch_info in self.continuum_patches:
                try:
                    patch = patch_info.get('patch')
                    if patch is not None and self.ax is not None and patch in self.ax.patches:
                        patch.remove()
                except (ValueError, RuntimeError, AttributeError):
                    pass
            self.continuum_patches.clear()
            
            # Remove all Gaussian fits
            for fit in self.gaussian_fits:
                try:
                    if 'line' in fit and fit['line'] is not None:
                        fit['line'].remove()
                except (ValueError, RuntimeError):
                    pass
            self.gaussian_fits.clear()
            
            # Remove all Voigt fits
            for fit in self.voigt_fits:
                try:
                    if 'line' in fit and fit['line'] is not None:
                        fit['line'].remove()
                except (ValueError, RuntimeError):
                    pass
            self.voigt_fits.clear()
            
            # Remove all Bayes fits and bounds
            for line in self.bayes_bound_lines:
                try:
                    line.remove()
                except (ValueError, RuntimeError):
                    pass
            self.bayes_bound_lines.clear()
            self.bayes_bounds.clear()
            
            # Remove all markers and labels
            for marker in self.markers:
                try:
                    marker.remove()
                except (ValueError, RuntimeError):
                    pass
            self.markers.clear()
            
            for label in self.labels:
                try:
                    label.remove()
                except (ValueError, RuntimeError):
                    pass
            self.labels.clear()
            
            # Clear item tracker
            if hasattr(self, 'item_id_map'):
                self.item_id_map.clear()
            if hasattr(self, 'item_tracker'):
                self.item_tracker.clear_all()
            
            # Deactivate modes
            self.guess_drawing_mode = False
            self.continuum_mode = False
            self.current_continuum_fit_id = None  # Clear fit tracking
            self.gaussian_mode = False
            self.voigt_mode = False
            self.multi_gaussian_mode = False
            self.multi_gaussian_mode_old = False
            self.multi_voigt_mode = False
            self.mask_drawing_mode = False
            self.bayes_mode = False
            self.listfit_mode = False
            self.redshift_estimation_mode = False
            self.calculate_ew_selection_mode = False
            
            # Reset dropdown menus
            self.continuum_mode_dropdown.blockSignals(True)
            self.continuum_mode_dropdown.setCurrentIndex(0)
            self.continuum_mode_dropdown.blockSignals(False)
            
            self.gaussian_mode_dropdown.blockSignals(True)
            self.gaussian_mode_dropdown.setCurrentIndex(0)
            self.gaussian_mode_dropdown.blockSignals(False)
            
            self.advanced_mode_dropdown.blockSignals(True)
            self.advanced_mode_dropdown.setCurrentIndex(0)
            self.advanced_mode_dropdown.blockSignals(False)
            
            self.calculate_mode_dropdown.blockSignals(True)
            self.calculate_mode_dropdown.setCurrentIndex(0)
            self.calculate_mode_dropdown.blockSignals(False)
            
            self.calculate_ew_mode_dropdown.blockSignals(True)
            self.calculate_ew_mode_dropdown.setCurrentIndex(0)
            self.calculate_ew_mode_dropdown.blockSignals(False)
            
            # Disable relevant buttons
            self.continuum_enter_button.setEnabled(False)
            self.gaussian_enter_button.setEnabled(False)
            
            # Redraw the plot if figure exists
            if self.fig is not None and hasattr(self.fig, 'canvas'):
                try:
                    self.fig.canvas.draw_idle()
                except Exception as e:
                    print(f"[DEBUG] Warning: Could not redraw plot: {e}")
            
            print("[DEBUG] All plotted features removed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to remove plotted features: {e}")
            import traceback
            traceback.print_exc()
    
    def clear_listfit_bounds(self):
        """Clear listfit bounds and remove bound lines from plot"""
        for line in self.listfit_bound_lines:
            try:
                line.remove()
            except (ValueError, NotImplementedError):
                pass
        self.listfit_bound_lines.clear()
        self.listfit_bounds = []
        
        # Also clear all guess lines
        for comp_id in list(self.guess_lines.keys()):
            try:
                self.guess_lines[comp_id].remove()
            except (ValueError, RuntimeError):
                pass
        self.guess_lines.clear()
        
        # Also clear all mask regions when user escapes/closes listfit
        self._clear_mask_regions()
        
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


    def perform_listfit(self, fit_data):
        """Perform multi-component fitting
        
        Args:
            fit_data: dict with 'components' and 'tied_parameters' keys
        """
        # Handle both dict (new format) and list (legacy format for backward compatibility)
        if isinstance(fit_data, dict):
            components = fit_data.get('components', [])
            tied_parameters = fit_data.get('tied_parameters', [])
        else:
            # Legacy format: fit_data is just components list
            components = fit_data
            tied_parameters = []
        
        # FIRST: Clear all temporary guess visualizations (salmon-colored lines)
        # The guesses themselves remain in components dict to use as initial values
        self._remove_guess_lines()
        
        if not self.listfit_bounds or len(self.listfit_bounds) < 2:
            print("Error: Invalid bounds for listfit")
            self._clear_mask_regions()  # Clear masks on error
            return
        
        left_bound, right_bound = sorted(self.listfit_bounds)
        
        # Extract data within bounds
        mask = (self.x_data >= left_bound) & (self.x_data <= right_bound)
        x_fit = self.x_data[mask]
        y_fit = self.spec[mask]
        err_fit = self.err[mask] if self.err is not None else None
        
        if len(x_fit) == 0:
            print("Error: No data within bounds")
            self._remove_guess_lines()
            self._clear_mask_regions()  # Clear masks on error
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
            self._remove_guess_lines()
            self._clear_mask_regions()  # Clear masks on error
            return
        
        # Build the composite model
        try:
            composite_model = self.build_composite_model(components, x_fit_masked, y_fit_masked, err_fit_masked, tied_parameters)
        except Exception as e:
            print(f"Error building composite model: {e}")
            import traceback
            traceback.print_exc()
            self._remove_guess_lines()
            self._clear_mask_regions()  # Clear masks on error
            return
        
        if composite_model is None:
            print("Error: Could not build composite model")
            self._remove_guess_lines()
            self._clear_mask_regions()  # Clear masks on error
            return
        
        # Perform the fit with optional weights
        params = None  # Initialize to None - will be created during fitting
        try:
            # **COMPREHENSIVE DEBUG OUTPUT BEFORE FITTING**
            print("\n" + "="*80)
            print("[DEBUG] COMPREHENSIVE LISTFIT MODEL SETUP")
            print("="*80)
            print(f"[DEBUG] Fit data: {len(x_fit_masked)} data points (from {len(x_fit)} total in bounds)")
            print(f"[DEBUG] Composite model structure:\n{composite_model}")
            print("\n[DEBUG] Parameter Summary BEFORE fit:")
            if hasattr(composite_model, 'make_params'):
                params = composite_model.make_params()
                
                # **APPLY TIED PARAMETERS NOW THAT PARAMETERS EXIST**
                if tied_parameters:
                    print(f"[DEBUG] Applying {len(tied_parameters)} tied parameter(s) to created parameters...")
                    for tie in tied_parameters:
                        param1 = tie.get('param1', '').strip()
                        param2_expr = tie.get('param2', '').strip()
                        
                        if not param1 or not param2_expr:
                            continue
                        
                        # Parse param1 to get the actual parameter name
                        if '=' in param1:
                            param1 = param1.split('=')[0].strip()
                        
                        if param1 in params:
                            # **CRITICAL FIX**: Skip if parameter already has an expression (e.g., from redshift tying)
                            if params[param1].expr:
                                print(f"[DEBUG] SKIPPING tie for {param1} (already has expression: {params[param1].expr})")
                                continue
                            
                            print(f"[DEBUG] Setting tie: {param1} = {param2_expr}")
                            try:
                                params[param1].expr = param2_expr
                                params[param1].vary = False  # Dependent parameters don't vary
                            except Exception as e:
                                print(f"[DEBUG] ERROR applying tie {param1} = {param2_expr}: {e}")
                        else:
                            print(f"[DEBUG] ERROR: Parameter '{param1}' not found in model")
                
                for pname in sorted(params.keys()):
                    p = params[pname]
                    bounds_str = f"[{p.min if p.min is not None else '-∞'}, {p.max if p.max is not None else '+∞'}]"
                    expr_str = f", expr='{p.expr}'" if p.expr else ""
                    print(f"  {pname:20s}: value={p.value:12.6f}, vary={p.vary!s:5s}, bounds={bounds_str}{expr_str}")
            
            # **ADDITIONAL DEBUG: Show which parameters have tied expressions**
            print("\n[DEBUG] Parameters with expressions (tied parameters):")
            if params:
                for pname in sorted(params.keys()):
                    if params[pname].expr:
                        print(f"  {pname:20s} = {params[pname].expr}")
                print(f"\n[DEBUG] All parameters in params object: {list(params.keys())}")
            else:
                print("  (None - params not yet created or empty)")
            print("\n[DEBUG] Fit method: leastsq")
            if err_fit_masked is not None:
                print(f"[DEBUG] Using error spectrum (weights=1/err)")
            else:
                print(f"[DEBUG] No error spectrum (unweighted fit)")
            print("="*80 + "\n")
            
            # Use the params object we created (which has ties applied)
            if hasattr(composite_model, 'make_params') and params is not None:
                if err_fit_masked is not None:
                    result = composite_model.fit(y_fit_masked, params, x=x_fit_masked, weights=1.0/err_fit_masked)
                else:
                    result = composite_model.fit(y_fit_masked, params, x=x_fit_masked)
            else:
                # Fallback if params wasn't created properly
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
            self._remove_guess_lines()
            QtWidgets.QMessageBox.critical(self, "Fit Failed - Invalid Constraints", error_msg)
            return
        except Exception as e:
            error_msg = f"Fit failed with error: {str(e)}\n\nPlease check your constraints and try again."
            print(f"Error: {error_msg}")
            self._remove_guess_lines()
            QtWidgets.QMessageBox.critical(self, "Fit Failed", error_msg)
            return
        
        # Check if fit succeeded OR if it converged but just failed on error estimation
        fit_converged = result.success or (result.nfree > 0 and result.ndata > result.nfree)
        
        if not fit_converged:
            print(f"Fit failed: {result.message}")
            self._remove_guess_lines()
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
        
        # Store the fit (IMPORTANT: Do NOT store 'result' object as it can have corrupted state)
        print("[DEBUG] Storing fit results...")
        try:
            self.listfit_fits.append({
                'bounds': (left_bound, right_bound),
                'components': components,
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
        
        # Extract initial guesses from lmfit and merge with user-drawn guesses
        print("[DEBUG] Extracting and merging guess parameters...")
        
        gauss_count = 0
        voigt_count = 0
        
        for comp in components:
            if comp['type'] == 'gaussian':
                # User-drawn guesses are already in comp['guess']
                # No need to extract from lmfit to avoid accessing corrupted parameter state
                gauss_count += 1
            
            elif comp['type'] == 'voigt':
                # User-drawn guesses are already in comp['guess']
                # No need to extract from lmfit to avoid accessing corrupted parameter state
                voigt_count += 1
        
        print("[DEBUG] Successfully extracted initial guess parameters")
        
        # Extract fit data for .qsap file (store components separately)
        listfit_fit_data = []
        
        # Track component counters to match what build_composite_model uses
        gauss_count = 0
        voigt_count = 0
        poly_count = 0
        
        try:
            print("[DEBUG] Starting fit data extraction...")
            
            for comp in components:
                comp_dict = {'type': comp['type']}
                
                if comp['type'] == 'gaussian':
                    prefix = f'g{gauss_count}_'
                    amp_name = f'{prefix}amp'
                    mu_name = f'{prefix}mu'
                    sigma_name = f'{prefix}sigma'
                    
                    if all(name in result.params for name in [amp_name, mu_name, sigma_name]):
                        try:
                            # Store best fit parameters with errors - extract carefully
                            comp_dict['amp'] = float(result.params[amp_name].value) if result.params[amp_name].value is not None else None
                            comp_dict['amp_err'] = float(result.params[amp_name].stderr) if result.params[amp_name].stderr is not None else None
                            comp_dict['mean'] = float(result.params[mu_name].value) if result.params[mu_name].value is not None else None
                            comp_dict['mean_err'] = float(result.params[mu_name].stderr) if result.params[mu_name].stderr is not None else None
                            comp_dict['stddev'] = float(result.params[sigma_name].value) if result.params[sigma_name].value is not None else None
                            comp_dict['stddev_err'] = float(result.params[sigma_name].stderr) if result.params[sigma_name].stderr is not None else None
                            comp_dict['bounds'] = (float(left_bound), float(right_bound))
                            print(f"[DEBUG] Gaussian {gauss_count}: bounds={comp_dict['bounds']}")
                            comp_dict['is_velocity_mode'] = self.is_velocity_mode
                            
                            # Include guess parameters if they exist (v1.3+)
                            guess = comp.get('guess')
                            if guess and any(v is not None for v in guess.values()):
                                comp_dict['guess'] = guess
                            
                            # CRITICAL for MC EW with tied parameters:
                            # Store the result object, component prefix, and parameter names
                            comp_dict['result'] = result
                            comp_dict['component_prefix'] = prefix
                            comp_dict['param_names'] = ['amp', 'mu', 'sigma']
                            
                            # Extract covariance matrix from result
                            # For Gaussian: correlate (amp, mu, sigma) which are positions 0, 1, 2 in the free params
                            # Build mapping of parameter names to their indices in var_names
                            if result.covar is not None:
                                param_indices = []
                                for pname in ['amp', 'mu', 'sigma']:
                                    full_name = f'{prefix}{pname}'
                                    if full_name in result.var_names:
                                        param_indices.append(result.var_names.index(full_name))
                                
                                if len(param_indices) == 3:
                                    # Extract the 3x3 submatrix for this component
                                    comp_cov = result.covar[np.ix_(param_indices, param_indices)]
                                    comp_dict['covariance'] = comp_cov
                        except Exception as e:
                            print(f"[DEBUG] Warning: Could not store Gaussian {gauss_count} fit data: {e}")
                    
                    gauss_count += 1
                
                elif comp['type'] == 'voigt':
                    prefix = f'v{voigt_count}_'
                    amp_name = f'{prefix}amp'
                    mean_name = f'{prefix}center'
                    sigma_name = f'{prefix}sigma'
                    gamma_name = f'{prefix}gamma'
                    
                    if all(name in result.params for name in [amp_name, mean_name, sigma_name, gamma_name]):
                        try:
                            # Store best fit parameters with errors - extract carefully
                            comp_dict['amplitude'] = float(result.params[amp_name].value) if result.params[amp_name].value is not None else None
                            comp_dict['amplitude_err'] = float(result.params[amp_name].stderr) if result.params[amp_name].stderr is not None else None
                            comp_dict['mean'] = float(result.params[mean_name].value) if result.params[mean_name].value is not None else None
                            comp_dict['mean_err'] = float(result.params[mean_name].stderr) if result.params[mean_name].stderr is not None else None
                            comp_dict['sigma'] = float(result.params[sigma_name].value) if result.params[sigma_name].value is not None else None
                            comp_dict['sigma_err'] = float(result.params[sigma_name].stderr) if result.params[sigma_name].stderr is not None else None
                            comp_dict['gamma'] = float(result.params[gamma_name].value) if result.params[gamma_name].value is not None else None
                            comp_dict['gamma_err'] = float(result.params[gamma_name].stderr) if result.params[gamma_name].stderr is not None else None
                            comp_dict['bounds'] = (float(left_bound), float(right_bound))
                            print(f"[DEBUG] Voigt {voigt_count}: bounds={comp_dict['bounds']}")
                            comp_dict['is_velocity_mode'] = self.is_velocity_mode
                            
                            # Include guess parameters if they exist (v1.3+)
                            guess = comp.get('guess')
                            if guess and any(v is not None for v in guess.values()):
                                comp_dict['guess'] = guess
                            
                            # CRITICAL for MC EW with tied parameters:
                            # Store the result object, component prefix, and parameter names
                            comp_dict['result'] = result
                            comp_dict['component_prefix'] = prefix
                            comp_dict['param_names'] = ['amp', 'center', 'sigma', 'gamma']
                            
                            # Extract covariance matrix from result for Voigt
                            # For Voigt: (amp, center, sigma, gamma) correlations
                            if result.covar is not None:
                                param_indices = []
                                for pname in ['amp', 'center', 'sigma', 'gamma']:
                                    full_name = f'{prefix}{pname}'
                                    if full_name in result.var_names:
                                        param_indices.append(result.var_names.index(full_name))
                                
                                if len(param_indices) == 4:
                                    # Extract the 4x4 submatrix for this component
                                    comp_cov = result.covar[np.ix_(param_indices, param_indices)]
                                    comp_dict['covariance'] = comp_cov
                        except Exception as e:
                            print(f"[DEBUG] Warning: Could not store Voigt {voigt_count} fit data: {e}")
                    
                    voigt_count += 1
                
                elif comp['type'] == 'polynomial':
                    prefix = f'p{poly_count}_'
                    coeffs = []
                    coeffs_err = []
                    order = comp.get('order', 1)
                    
                    try:
                        for i in range(order + 1):
                            coeff_name = f'{prefix}c{i}'
                            if coeff_name in result.params:
                                try:
                                    coeff_val = float(result.params[coeff_name].value) if result.params[coeff_name].value is not None else None
                                    coeff_err = float(result.params[coeff_name].stderr) if result.params[coeff_name].stderr is not None else None
                                    coeffs.append(coeff_val)
                                    coeffs_err.append(coeff_err)
                                except Exception as e:
                                    print(f"[DEBUG] Warning: Could not extract coefficient {coeff_name}: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not extract polynomial coefficients: {e}")
                    
                    if coeffs:
                        comp_dict['poly_order'] = order
                        comp_dict['coeffs'] = coeffs
                        comp_dict['coeffs_err'] = coeffs_err
                        comp_dict['bounds'] = (float(left_bound), float(right_bound))
                        print(f"[DEBUG] Polynomial: bounds={comp_dict['bounds']}")
                        
                        # Extract covariance matrix for polynomial coefficients
                        if result.covar is not None:
                            param_indices = []
                            for i in range(order + 1):
                                coeff_name = f'{prefix}c{i}'
                                if coeff_name in result.var_names:
                                    param_indices.append(result.var_names.index(coeff_name))
                            
                            if len(param_indices) == len(coeffs):
                                # Extract the submatrix for all polynomial coefficients
                                poly_cov = result.covar[np.ix_(param_indices, param_indices)]
                                comp_dict['covariance'] = poly_cov
                        
                        poly_count += 1
                
                elif comp['type'] == 'chebyshev':
                    prefix = f'c{poly_count}_'
                    coeffs = []
                    coeffs_err = []
                    degree = comp.get('degree', 1)
                    
                    try:
                        for i in range(degree + 1):
                            coeff_name = f'{prefix}c{i}'
                            if coeff_name in result.params:
                                try:
                                    coeff_val = float(result.params[coeff_name].value) if result.params[coeff_name].value is not None else None
                                    coeff_err = float(result.params[coeff_name].stderr) if result.params[coeff_name].stderr is not None else None
                                    coeffs.append(coeff_val)
                                    coeffs_err.append(coeff_err)
                                except Exception as e:
                                    print(f"[DEBUG] Warning: Could not extract coefficient {coeff_name}: {e}")
                    except Exception as e:
                        print(f"[DEBUG] Warning: Could not extract Chebyshev coefficients: {e}")
                    
                    if coeffs:
                        comp_dict['degree'] = degree
                        comp_dict['coeffs'] = coeffs
                        comp_dict['coeffs_err'] = coeffs_err
                        comp_dict['bounds'] = (float(left_bound), float(right_bound))
                        comp_dict['lam_min'] = comp.get('lam_min', float(left_bound))
                        comp_dict['lam_max'] = comp.get('lam_max', float(right_bound))
                        print(f"[DEBUG] Chebyshev: bounds={comp_dict['bounds']}, domain=[{comp_dict['lam_min']}, {comp_dict['lam_max']}]")
                        
                        # Extract covariance matrix for Chebyshev coefficients
                        if result.covar is not None:
                            param_indices = []
                            for i in range(degree + 1):
                                coeff_name = f'{prefix}c{i}'
                                if coeff_name in result.var_names:
                                    param_indices.append(result.var_names.index(coeff_name))
                            
                            if len(param_indices) == len(coeffs):
                                # Extract the submatrix for all Chebyshev coefficients
                                cheb_cov = result.covar[np.ix_(param_indices, param_indices)]
                                comp_dict['covariance'] = cheb_cov
                        
                        poly_count += 1
                
                elif comp['type'] == 'data_mask':
                    # Store mask parameters as-is (no fit results needed)
                    comp_dict['min_lambda'] = comp.get('min_lambda')
                    comp_dict['max_lambda'] = comp.get('max_lambda')
                
                # Only add non-empty component dictionaries
                if len(comp_dict) > 1:  # More than just 'type'
                    listfit_fit_data.append(comp_dict)
            
            # Extract redshift parameters from result and add them to listfit_fit_data
            # Also update profile dicts with TIED_REDSHIFT field
            print("[DEBUG] Extracting redshift components...")
            redshift_components = [c for c in components if c.get('type') == 'redshift']
            
            if redshift_components:
                # Build a map: profile_index -> redshift_index
                profile_to_redshift = {}
                for idx, comp in enumerate(components):
                    if comp.get('tied_to_redshift') and 'redshift_index' in comp:
                        redshift_idx = comp['redshift_index']
                        if redshift_idx not in profile_to_redshift:
                            profile_to_redshift[redshift_idx] = []
                        profile_to_redshift[redshift_idx].append(idx)
                
                # Extract each redshift
                for redshift_num, redshift_comp in enumerate(redshift_components, 1):
                    z_param_name = f"z{redshift_num}"
                    
                    if z_param_name in result.params:
                        try:
                            z_value = float(result.params[z_param_name].value) if result.params[z_param_name].value is not None else None
                            z_error = float(result.params[z_param_name].stderr) if result.params[z_param_name].stderr is not None else None
                            
                            # Create redshift component dict
                            redshift_dict = {
                                'type': 'redshift',
                                'redshift': z_value,
                                'error_redshift': z_error,
                                'redshift_number': redshift_num,
                                'label': redshift_comp.get('label', f'Redshift #{redshift_num}'),
                            }
                            
                            # Include initial guess if available
                            z_initial = redshift_comp.get('guess', {}).get('z')
                            if z_initial is not None:
                                redshift_dict['z_initial'] = float(z_initial)
                            
                            # Count profiles tied to this redshift
                            num_tied_profiles = len(profile_to_redshift.get(redshift_components.index(redshift_comp), []))
                            if num_tied_profiles > 0:
                                redshift_dict['num_profiles'] = num_tied_profiles
                            
                            listfit_fit_data.append(redshift_dict)
                            print(f"[DEBUG] Extracted redshift {z_param_name}: z = {z_value} ± {z_error}")
                        except Exception as e:
                            print(f"[DEBUG] Warning: Could not extract redshift {z_param_name}: {e}")
                
                # Update profile dicts with TIED_REDSHIFT field
                # Iterate through listfit_fit_data and add tied_redshift for profiles
                for profile_idx, (comp_idx, comp) in enumerate(enumerate(components)):
                    if comp.get('tied_to_redshift') and 'redshift_index' in comp:
                        redshift_idx = comp['redshift_index']
                        redshift_comp = components[redshift_idx]
                        if redshift_comp.get('type') == 'redshift':
                            redshift_number = redshift_comp.get('redshift_number', 1)
                            z_symbol = f"Z{redshift_number}"
                            
                            # Find the corresponding profile dict in listfit_fit_data and update it
                            # Count which profile number this is (accounting for other non-profile components before it)
                            profile_count = sum(1 for c in components[:comp_idx] if c.get('type') in ('gaussian', 'voigt'))
                            
                            # Find this profile in listfit_fit_data
                            gaussian_voigt_count = 0
                            for fit_dict in listfit_fit_data:
                                if fit_dict.get('type') in ('gaussian', 'voigt'):
                                    if gaussian_voigt_count == profile_count:
                                        fit_dict['tied_redshift'] = z_symbol
                                        print(f"[DEBUG] Added TIED_REDSHIFT={z_symbol} to {fit_dict.get('type')} component")
                                        break
                                    gaussian_voigt_count += 1
            
            print("[DEBUG] Redshift extraction complete")
            print("[DEBUG] Component data extraction complete. Adding diagnostics...")
            
            # Add fit diagnostics as a special metadata entry
            try:
                diagnostics_dict = {
                    'type': 'fit_diagnostics',
                    'ssr': None,
                    'ssr_nu': None,
                    'chi2': None,
                    'chi2_reduced': None,
                    'akaike_info_criterion': None,
                    'bayesian_info_criterion': None,
                    'r_squared': None,
                    'n_data_points': None,
                    'n_parameters': None,
                    'n_degrees_freedom': None,
                    'fit_success': None
                }
                
                # Safely extract diagnostics
                if result.residual is not None:
                    try:
                        diagnostics_dict['ssr'] = float(np.sum(result.residual ** 2))
                    except:
                        pass
                
                if hasattr(result, 'redchi') and result.redchi is not None:
                    try:
                        diagnostics_dict['ssr_nu'] = float(result.redchi)
                        diagnostics_dict['chi2_reduced'] = float(result.redchi)
                    except:
                        pass
                
                if hasattr(result, 'chisqr') and result.chisqr is not None:
                    try:
                        diagnostics_dict['chi2'] = float(result.chisqr)
                    except:
                        pass
                
                if hasattr(result, 'aic') and result.aic is not None:
                    try:
                        diagnostics_dict['akaike_info_criterion'] = float(result.aic)
                    except:
                        pass
                
                if hasattr(result, 'bic') and result.bic is not None:
                    try:
                        diagnostics_dict['bayesian_info_criterion'] = float(result.bic)
                    except:
                        pass
                
                if hasattr(result, 'ndata'):
                    try:
                        diagnostics_dict['n_data_points'] = int(result.ndata)
                    except:
                        pass
                
                if hasattr(result, 'nvarys'):
                    try:
                        diagnostics_dict['n_parameters'] = int(result.nvarys)
                    except:
                        pass
                
                if hasattr(result, 'nfree'):
                    try:
                        diagnostics_dict['n_degrees_freedom'] = int(result.nfree)
                    except:
                        pass
                
                if hasattr(result, 'success'):
                    try:
                        diagnostics_dict['fit_success'] = bool(result.success)
                    except:
                        pass
                
                # Calculate R-squared if possible
                try:
                    if result.residual is not None:
                        ss_res = float(np.sum(result.residual ** 2))
                    else:
                        ss_res = float(np.sum((y_fit - result.best_fit) ** 2))
                    ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
                    diagnostics_dict['r_squared'] = float(r_squared)
                except Exception as e:
                    print(f"[DEBUG] Could not calculate R-squared: {e}")
                
                listfit_fit_data.append(diagnostics_dict)
                print("[DEBUG] Diagnostics added successfully")
            except Exception as e:
                print(f"[DEBUG] Error adding diagnostics: {e}")
                import traceback
                traceback.print_exc()
            
            # Add constraints information as a special metadata entry (v1.3+)
            if constraints_info:
                try:
                    constraints_dict = {
                        'type': 'constraints',
                        'constraints': constraints_info
                    }
                    listfit_fit_data.append(constraints_dict)
                    print("[DEBUG] Constraints added successfully")
                except Exception as e:
                    print(f"[DEBUG] Error adding constraints: {e}")
            
            print("[DEBUG] Fit data extraction complete. Saving to .qsap file...")
            
        except Exception as e:
            print(f"[ERROR] Unexpected error during fit data extraction: {e}")
            import traceback
            traceback.print_exc()
        
        # **Auto-calculate equivalent widths if conditions are met**
        # EW is calculated if and only if:
        # 1. Exactly 1 polynomial continuum is fitted
        # 2. At least 1 non-polynomial profile (Gaussian or Voigt) is fitted
        # NOTE: Temporarily disabled due to crashes - set ENABLE_AUTO_EW_LISTFIT=True to re-enable
        ENABLE_AUTO_EW_LISTFIT = False
        if ENABLE_AUTO_EW_LISTFIT:
            try:
                ew_results = self._auto_calculate_listfit_ew(result, components, left_bound, right_bound, x_fit, y_fit)
                if ew_results:
                    listfit_fit_data.append(ew_results)
            except Exception as e:
                print(f"[AUTO_EW] ERROR: {e}")
                import traceback
                traceback.print_exc()
                # Continue without EW - don't crash the entire fit
        
        # Save fit to .qsap file
        if listfit_fit_data:
            print("[DEBUG] About to save fit to .qsap file...")
            try:
                print(f"[DEBUG] Fit data has {len(listfit_fit_data)} components")
                
                # Create a clean copy of the fit data to avoid any corrupted references
                import json
                print("[DEBUG] Serializing fit data to check for issues...")
                
                # Try to serialize to JSON to catch any problematic objects
                try:
                    clean_data = []
                    for item in listfit_fit_data:
                        try:
                            # Try to serialize this item
                            json_str = json.dumps(item, default=str)
                            # If successful, add it back
                            clean_data.append(item)
                            print(f"[DEBUG]   Item type '{item.get('type', 'unknown')}' serialized OK")
                        except (TypeError, ValueError) as se:
                            print(f"[DEBUG]   Warning: Could not fully serialize item type '{item.get('type', 'unknown')}': {se}")
                            # Still add it - lmfit should handle it
                            clean_data.append(item)
                    
                    print("[DEBUG] All fit data items prepared")
                except Exception as e:
                    print(f"[DEBUG] Warning: Pre-serialization check failed: {e}")
                    clean_data = listfit_fit_data
                
                print("[DEBUG] Calling save_and_print_qsap_fit...")
                self.save_and_print_qsap_fit(clean_data, 'Listfit', 'Listfit', lmfit_result=result)
                print("[DEBUG] Successfully saved fit to .qsap file")
            except Exception as e:
                print(f"[ERROR] Failed to save fit to .qsap file: {e}")
                import traceback
                traceback.print_exc()
                # Continue anyway - don't let save errors crash the entire UI
        
        # **Remove persistent guess lines from plot after successful fit**
        print("[DEBUG] About to remove guess lines...")
        try:
            self._remove_guess_lines()
            print("[DEBUG] Guess lines removed")
        except Exception as e:
            print(f"[ERROR] Error removing guess lines: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear mask regions now that fit is complete
        print("[DEBUG] About to clear mask regions...")
        try:
            self._clear_mask_regions()
            print("[DEBUG] Mask regions cleared")
        except Exception as e:
            print(f"[ERROR] Error clearing mask regions: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear listfit mode
        print("[DEBUG] Clearing listfit mode...")
        try:
            self.listfit_mode = False
            for line in self.listfit_bound_lines:
                try:
                    line.remove()
                except:
                    pass
            self.listfit_bound_lines.clear()
            self.listfit_bounds = []
            print("[DEBUG] Listfit mode cleared")
        except Exception as e:
            print(f"[ERROR] Error clearing listfit mode: {e}")
            import traceback
            traceback.print_exc()
        
        # Reset Advanced dropdown after successful fit
        print("[DEBUG] Resetting advanced dropdown...")
        try:
            self.reset_advanced_dropdown()
            print("[DEBUG] Advanced dropdown reset")
        except Exception as e:
            print(f"[ERROR] Error resetting advanced dropdown: {e}")
            import traceback
            traceback.print_exc()
        
        print("[DEBUG] About to redraw canvas...")
        try:
            self.fig.canvas.draw_idle()  # Redraw to show listfit results
            print("[DEBUG] Canvas redrawn successfully")
        except Exception as e:
            print(f"[ERROR] Error redrawing canvas: {e}")
            import traceback
            traceback.print_exc()
        
        print("[DEBUG] _perform_listfit completed successfully!")

    def _remove_guess_lines(self):
        """Remove all guess profile lines from the plot"""
        print("[DEBUG] Removing guess profile lines from plot...")
        for comp_id in list(self.guess_lines.keys()):
            try:
                self.guess_lines[comp_id].remove()
                print(f"[DEBUG] Removed guess line for component #{comp_id}")
            except (ValueError, RuntimeError) as e:
                print(f"[DEBUG] Warning: Could not remove guess line for component #{comp_id}: {e}")
        self.guess_lines.clear()
        
        # Also remove temporary polynomial clicked points markers
        if self.guess_polynomial_clicked_points is not None:
            try:
                self.guess_polynomial_clicked_points.remove()
                print(f"[DEBUG] Removed polynomial clicked points markers")
            except (ValueError, RuntimeError):
                pass
            self.guess_polynomial_clicked_points = None
        
        print("[DEBUG] All guess profile lines removed")

    def _clear_mask_regions(self):
        """Remove all mask region patches (data_mask and polynomial_guess_mask) from the plot.
        
        This is called after listfit completes (successfully or with error) to clean up
        the gray mask regions that were displayed during the fitting process.
        """
        print("[DEBUG] Clearing mask regions from plot...")
        # Find and remove all mask-type items from ItemTracker
        items_to_remove = []
        for item_id, item_info in self.item_id_map.items():
            if item_info['type'] in ['data_mask', 'polynomial_guess_mask']:
                items_to_remove.append(item_id)
        
        for item_id in items_to_remove:
            try:
                patch_obj = self.item_id_map[item_id].get('patch_obj')
                if patch_obj:
                    patch_obj.remove()
                    print(f"[DEBUG] Removed mask patch: {item_id}")
                self.unregister_item(item_id)
                print(f"[DEBUG] Unregistered mask item: {item_id}")
            except (ValueError, RuntimeError) as e:
                print(f"[DEBUG] Warning: Could not remove mask {item_id}: {e}")
        
        print(f"[DEBUG] Cleared {len(items_to_remove)} mask region(s)")

    def build_composite_model(self, components, x_fit, y_fit, err_fit, tied_parameters=None):
        """Build a composite lmfit Model from component list with improved initial guesses for blended profiles
        
        Args:
            components: List of component dicts
            x_fit: Wavelength array
            y_fit: Flux array
            err_fit: Error array (optional)
            tied_parameters: List of {'param1': str, 'param2': str} dicts for tied parameters
        """
        from scipy.signal import find_peaks
        from lmfit import Model, Parameters
        
        if tied_parameters is None:
            tied_parameters = []
        
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
        # (excluding redshift-tied ones which have their center determined by redshift parameter)
        num_gaussians = len([c for c in components if c['type'] == 'gaussian' and not c.get('tied_to_redshift')])
        num_voigts = len([c for c in components if c['type'] == 'voigt' and not c.get('tied_to_redshift')])
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
        for i, c in enumerate(components):
            print(f"[DEBUG]   Component {i}: type={c.get('type')}, label={c.get('label', 'N/A')}")
        
        for comp in components:
            # Skip mask features and redshift parameters - they're not fitted as spectral components
            # Redshift parameters are handled after the main model is built
            if comp['type'] in ['polynomial_guess_mask', 'data_mask', 'redshift']:
                print(f"[DEBUG] Skipping {comp['type']} component: {comp.get('label', 'N/A')}")
                continue
            
            if comp['type'] == 'gaussian':
                print(f"[DEBUG] Processing Gaussian #{gauss_count}: label={comp.get('label', 'N/A')}")
                # Check if user provided a guess first (interactive guess drawing)
                user_guess = comp.get('guess', {})
                if user_guess.get('center') is not None and user_guess.get('amp') is not None and user_guess.get('stddev') is not None:
                    # Use user-provided guess values
                    amp_guess = user_guess['amp']
                    center_guess = user_guess['center']
                    sigma_guess = user_guess['stddev']
                    print(f"[DEBUG] Using user-provided guess for Gaussian {gauss_count}: center={center_guess:.2f}, amp={amp_guess:.2f}, stddev={sigma_guess:.2f}")
                else:
                    # Fall back to auto-detection if no user guess
                    if peak_index_for_component < len(peaks):
                        amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(
                            x_fit, y_fit, peak_idx=peaks[peak_index_for_component]
                        )
                        peak_index_for_component += 1
                    else:
                        # Fallback if we run out of detected peaks
                        amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(x_fit, y_fit)
                
                prefix = f'g{gauss_count}_'
                print(f"[DEBUG] Creating model with prefix '{prefix}' for Gaussian {gauss_count}")
                gauss_model = Model(self.gaussian, prefix=prefix, independent_vars=['x'])
                
                # Set initial guesses FIRST with the detected peak positions
                gauss_model.set_param_hint(f'{prefix}amp', value=amp_guess)
                gauss_model.set_param_hint(f'{prefix}mu', value=center_guess)
                gauss_model.set_param_hint(f'{prefix}sigma', value=sigma_guess, min=1e-6)
                print(f"[DEBUG] Set initial guesses: {prefix}amp={amp_guess}, {prefix}mu={center_guess}, {prefix}sigma={sigma_guess}")
                
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
                print(f"[DEBUG] Gaussian {gauss_count - 1} added. gauss_count is now {gauss_count}")
            
            elif comp['type'] == 'voigt':
                # Check if user provided a guess first (interactive guess drawing)
                user_guess = comp.get('guess', {})
                if user_guess.get('center') is not None and user_guess.get('amp') is not None and user_guess.get('sigma') is not None:
                    # Use user-provided guess values
                    amp_guess = user_guess['amp']
                    center_guess = user_guess['center']
                    sigma_guess = user_guess['sigma']
                    gamma_guess = user_guess.get('gamma', sigma_guess * 0.01)  # Use user's gamma or default
                    print(f"[DEBUG] Using user-provided guess for Voigt {voigt_count}: center={center_guess:.2f}, amp={amp_guess:.2f}, sigma={sigma_guess:.2f}, gamma={gamma_guess:.2f}")
                else:
                    # Fall back to auto-detection if no user guess
                    if peak_index_for_component < len(peaks):
                        amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(
                            x_fit, y_fit, peak_idx=peaks[peak_index_for_component]
                        )
                        peak_index_for_component += 1
                    else:
                        # Fallback if we run out of detected peaks
                        amp_guess, center_guess, sigma_guess = self._estimate_gaussian_params(x_fit, y_fit)
                    
                    gamma_guess = sigma_guess * 0.01  # Small Lorentzian contribution
                
                prefix = f'v{voigt_count}_'
                voigt_model = Model(self.voigt, prefix=prefix, independent_vars=['x'])
                
                # Set initial guesses FIRST with the detected peak positions (user's or auto-detected)
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
                
                # Check if user provided a polynomial guess (drawn line)
                user_guess = comp.get('guess', {})
                
                # Create polynomial model using lmfit's built-in PolynomialModel
                from lmfit.models import PolynomialModel
                poly_model = PolynomialModel(degree=order, prefix=prefix, independent_vars=['x'])
                
                # **Use user-drawn polynomial guess if available (only for order=1)**
                if order == 1 and user_guess.get('slope') is not None and user_guess.get('intercept') is not None:
                    slope = user_guess['slope']
                    intercept = user_guess['intercept']
                    # For PolynomialModel with degree=1: y = c0 + c1*x
                    # Our line is: y = slope*x + intercept
                    # So: c0 = intercept, c1 = slope
                    print(f"[DEBUG] Using user-drawn polynomial guess for {prefix}: y = {slope:.6f}*x + {intercept:.4f}")
                    poly_model.set_param_hint(f'{prefix}c0', value=intercept)
                    poly_model.set_param_hint(f'{prefix}c1', value=slope)
                
                # **Use pre-fitted polynomial parameters if available (Stage 1 results)**
                elif prefix in polynomial_fits and polynomial_fits[prefix] is not None:
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
            
            elif comp['type'] == 'chebyshev':
                degree = comp.get('degree', 1)
                prefix = f'c{poly_count}_'
                
                print(f"[DEBUG] Processing Chebyshev #{poly_count}: label={comp.get('label', 'N/A')}, degree={degree}")
                
                # Create Chebyshev model using lmfit's Model wrapper with custom evaluator
                # Get domain bounds from x_fit
                lam_min = x_fit.min()
                lam_max = x_fit.max()
                
                # Create a factory function that captures domain bounds and degree
                def make_chebyshev_model_func(degree_val, lam_min_val, lam_max_val):
                    """Factory to create Chebyshev evaluator with captured parameters"""
                    def chebyshev_model(x, **params):
                        """Evaluate Chebyshev with rescaling to [-1, 1]"""
                        # Build coefficient array from parameters
                        # NOTE: lmfit STRIPS the prefix before passing to the evaluator,
                        # so we receive unprefixed keys like 'c0', 'c1', 'c2' not 'c0_c0', etc.
                        coeffs = []
                        for j in range(degree_val + 1):
                            key = f'c{j}'  # Unprefixed - this is what lmfit passes
                            if key in params:
                                coeffs.append(params[key])
                            else:
                                coeffs.append(0.0)
                        
                        # Rescale x to [-1, 1]
                        x_rescaled = 2.0 * (x - lam_min_val) / (lam_max_val - lam_min_val) - 1.0
                        # Evaluate Chebyshev polynomial
                        return np.polynomial.chebyshev.chebval(x_rescaled, coeffs)
                    
                    return chebyshev_model
                
                # Create the model function
                cheb_func = make_chebyshev_model_func(degree, lam_min, lam_max)
                
                # Create lmfit Model
                cheb_model = Model(cheb_func, prefix=prefix, independent_vars=['x'])
                
                # Check if user provided a polynomial guess (can be used as Chebyshev approximation)
                user_guess = comp.get('guess', {})
                
                # Use pre-fitted polynomial parameters if available (Stage 1 results, reuse poly fits)
                # Chebyshev can use polynomial fit as initial guess, converting coefficients
                if prefix in polynomial_fits and polynomial_fits[prefix] is not None:
                    print(f"[DEBUG] Using pre-fitted polynomial parameters as Chebyshev guess for {prefix}")
                    # Use first two coefficients from polynomial as starting point for Chebyshev
                    for i in range(degree + 1):
                        param_name = f'{prefix}c{i}'
                        if i < 2:
                            # Use polynomial coefficients for first two Chebyshev coefficients
                            poly_param_name = f'{prefix}c{i}'
                            if poly_param_name in polynomial_fits[prefix]:
                                guess_value = polynomial_fits[prefix][poly_param_name].value
                            else:
                                guess_value = 0.0
                        else:
                            guess_value = 0.0
                        cheb_model.set_param_hint(param_name, value=guess_value)
                        print(f"[DEBUG]   {param_name} = {guess_value}")
                else:
                    # Estimate Chebyshev coefficients from data using numpy
                    print(f"[DEBUG] Using estimated coefficients for Chebyshev {prefix}")
                    try:
                        # Fit Chebyshev directly to data
                        from numpy.polynomial import chebyshev
                        cheb_fit = chebyshev.Chebyshev.fit(x_fit, y_fit, degree, domain=[lam_min, lam_max])
                        
                        # Extract coefficients
                        for i in range(degree + 1):
                            param_name = f'{prefix}c{i}'
                            guess_value = cheb_fit.coef[i] if i < len(cheb_fit.coef) else 0.0
                            cheb_model.set_param_hint(param_name, value=guess_value)
                            print(f"[DEBUG]   {param_name} = {guess_value}")
                    except Exception as e:
                        print(f"[DEBUG] Warning: Chebyshev coefficient estimation failed: {e}")
                        # Fallback: set small initial values
                        for i in range(degree + 1):
                            param_name = f'{prefix}c{i}'
                            guess_value = np.mean(y_fit) if i == 0 else 0.0
                            cheb_model.set_param_hint(param_name, value=guess_value)
                
                # Store domain bounds in the component for later use (MC EW calculation)
                comp['lam_min'] = lam_min
                comp['lam_max'] = lam_max
                
                if model is None:
                    model = cheb_model
                else:
                    model = model + cheb_model
                poly_count += 1
        
        # **HANDLE REDSHIFT PARAMETERS AND TIED EXPRESSIONS**
        # Add redshift parameters and tie line centers to them
        redshift_components = [comp for comp in components if comp['type'] == 'redshift']
        if redshift_components:
            print(f"[DEBUG] Processing {len(redshift_components)} redshift component(s)...")
            
            for redshift_idx, redshift_comp in enumerate(redshift_components):
                z_param_name = f'z{redshift_idx + 1}'  # Use z1, z2, z3 (1-indexed, not 0-indexed)
                redshift_guess = redshift_comp.get('guess', {}).get('z')
                
                if model is None:
                    # Need to create a dummy model if no other components exist
                    # (unlikely, but handle it just in case)
                    from lmfit import Parameters
                    model = Model(lambda x: np.zeros_like(x), prefix='dummy_')
                
                # Add redshift parameter
                print(f"[DEBUG] Adding redshift parameter '{z_param_name}'")
                if redshift_guess is not None:
                    model.set_param_hint(z_param_name, value=redshift_guess, min=-1.0, max=10.0)
                    print(f"[DEBUG]   Initial guess: z = {redshift_guess}")
                else:
                    model.set_param_hint(z_param_name, value=0.0, min=-1.0, max=10.0)
                    print(f"[DEBUG]   No guess provided, using default z = 0.0")
                
                # Find all lines tied to this redshift and create tie expressions
                tied_line_indices = redshift_comp.get('tied_lines', [])
                for line_idx in tied_line_indices:
                    if line_idx >= len(components):
                        continue
                    
                    line_comp = components[line_idx]
                    if line_comp.get('tied_to_redshift') != True:
                        continue
                    
                    rest_wavelength = line_comp.get('rest_wavelength')
                    if rest_wavelength is None:
                        print(f"[DEBUG] Warning: Line component {line_idx} missing rest_wavelength")
                        continue
                    
                    # Determine parameter prefix (g for gaussian, v for voigt)
                    line_type = line_comp.get('type')
                    if line_type == 'gaussian':
                        # Find which gaussian this is (count ALL gaussians up to this component, including redshift-tied)
                        gauss_idx = 0
                        for i, c in enumerate(components[:line_idx]):
                            if c.get('type') == 'gaussian':  # Count ALL Gaussians
                                gauss_idx += 1
                        prefix = f'g{gauss_idx}_'
                    elif line_type == 'voigt':
                        # Find which voigt this is (count ALL voigts up to this component, including redshift-tied)
                        voigt_idx = 0
                        for i, c in enumerate(components[:line_idx]):
                            if c.get('type') == 'voigt':  # Count ALL Voigts
                                voigt_idx += 1
                        prefix = f'v{voigt_idx}_'
                    else:
                        print(f"[DEBUG] Warning: Unknown line type {line_type}")
                        continue
                    
                    # Set tie expression: center = (1 + z) * rest_wavelength
                    # IMPORTANT: Use 'mu' for Gaussian (not 'mean'), 'center' for Voigt
                    center_param = f'{prefix}mu' if line_type == 'gaussian' else f'{prefix}center'
                    tie_expression = f'(1 + {z_param_name}) * {rest_wavelength}'
                    print(f"[DEBUG] Tying {center_param} to redshift: {center_param} = {tie_expression}")
                    print(f"[DEBUG]   z_param_name = '{z_param_name}' (type={type(z_param_name)})")
                    print(f"[DEBUG]   rest_wavelength = {rest_wavelength} (type={type(rest_wavelength)})")
                    model.set_param_hint(center_param, expr=tie_expression)
        
        # **APPLY TIED PARAMETERS FROM TIED_PARAMETERS LIST**
        # NOTE: Don't apply ties via set_param_hint here - apply them after parameters are created
        # Return both model and tied_parameters so they can be applied properly during fitting
        
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
            try:
                # Use safe getter that handles constrained parameters
                init_val = param.init_value if hasattr(param, 'init_value') and param.init_value is not None else param.value
                if init_val is not None:
                    guesses[param_name] = init_val
            except (AttributeError, ValueError, TypeError):
                # Skip parameters that cause issues
                pass
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
            "  • Sum of Squared Residuals (SSR) = Sum[(data - model)²]\n"
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
        """Apply constraints to a Gaussian model
        
        IMPORTANT: Combines all constraints into single set_param_hint calls
        to avoid lmfit state corruption from multiple calls on same parameter.
        
        NOTE: If a parameter has a tied constraint (expression), bounds are NOT applied
        since tied parameters are determined by their expression, not by direct fitting.
        """
        if not constraints:
            return
        
        print(f"\n[DEBUG] Applying Gaussian constraints for {prefix}")
        print(f"[DEBUG]   Raw constraint dict: {constraints}")
        
        # Check which parameters have linked constraints (these will use expressions)
        linked_constraints = constraints.get('linked_constraints', [])
        tied_parameters = {lc.get('parameter', '').lower() for lc in linked_constraints}
        print(f"[DEBUG]   Tied parameters (with expressions): {tied_parameters}")
        
        # **AMPLITUDE CONSTRAINTS**
        amp_hints = {}
        if constraints.get('amplitude_fixed'):
            fixed_val = constraints.get('amplitude_fixed_value')
            if fixed_val:
                amp_hints['value'] = float(fixed_val)
            amp_hints['vary'] = False
            print(f"[DEBUG]   Amplitude: FIXED to {fixed_val}")
        else:
            amp_min, amp_max = constraints.get('amplitude_bounds', ('', ''))
            if amp_min or amp_max:
                if amp_min:
                    amp_hints['min'] = float(amp_min)
                if amp_max:
                    amp_hints['max'] = float(amp_max)
                print(f"[DEBUG]   Amplitude bounds: min={amp_min if amp_min else 'none'}, max={amp_max if amp_max else 'none'}")
            else:
                print(f"[DEBUG]   Amplitude: no bounds")
        
        # Apply amplitude hints in single call if any exist
        if amp_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}amp', {amp_hints})")
            model.set_param_hint(f'{prefix}amp', **amp_hints)
        
        # **MEAN/CENTER CONSTRAINTS (parameter name is now 'mu')**
        # SKIP bounds if mu is a tied parameter (has an expression)
        mu_hints = {}
        if 'mu' not in tied_parameters:
            if constraints.get('mean_fixed'):
                fixed_val = constraints.get('mean_fixed_value')
                if fixed_val:
                    mu_hints['value'] = float(fixed_val)
                mu_hints['vary'] = False
                print(f"[DEBUG]   Mu: FIXED to {fixed_val}")
            else:
                mean_min, mean_max = constraints.get('mean_bounds', ('', ''))
                if mean_min or mean_max:
                    if mean_min:
                        mu_hints['min'] = float(mean_min)
                    if mean_max:
                        mu_hints['max'] = float(mean_max)
                    print(f"[DEBUG]   Mu bounds: min={mean_min if mean_min else 'none'}, max={mean_max if mean_max else 'none'}")
                else:
                    print(f"[DEBUG]   Mu: no bounds")
        else:
            print(f"[DEBUG]   Mu: TIED (bounds skipped, will use expression)")
        
        # Apply mu hints in single call if any exist
        if mu_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}mu', {mu_hints})")
            model.set_param_hint(f'{prefix}mu', **mu_hints)
        
        # **SIGMA/STDDEV CONSTRAINTS (parameter name is now 'sigma')**
        # SKIP bounds if sigma is a tied parameter (has an expression)
        sigma_hints = {}
        if 'sigma' not in tied_parameters:
            if constraints.get('sigma_fixed'):
                fixed_val = constraints.get('sigma_fixed_value')
                if fixed_val:
                    sigma_hints['value'] = float(fixed_val)
                sigma_hints['vary'] = False
                print(f"[DEBUG]   Sigma: FIXED to {fixed_val}")
            else:
                sigma_min, sigma_max = constraints.get('sigma_bounds', ('', ''))
                if sigma_min or sigma_max:
                    if sigma_min:
                        sigma_hints['min'] = float(sigma_min)
                    if sigma_max:
                        sigma_hints['max'] = float(sigma_max)
                    print(f"[DEBUG]   Sigma bounds: min={sigma_min if sigma_min else 'none'}, max={sigma_max if sigma_max else 'none'}")
                else:
                    print(f"[DEBUG]   Sigma: no bounds")
        else:
            print(f"[DEBUG]   Sigma: TIED (bounds skipped, will use expression)")
        
        # Apply sigma hints in single call if any exist
        if sigma_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}sigma', {sigma_hints})")
            model.set_param_hint(f'{prefix}sigma', **sigma_hints)
        
        # Apply linked constraints (multiple parameter linking)
        for linked in linked_constraints:
            parameter = linked.get('parameter', 'mu')
            expression = linked.get('expression', '')
            if expression:
                # Extract right-hand side if expression contains '='
                if '=' in expression:
                    rhs = expression.split('=', 1)[1].strip()
                else:
                    rhs = expression.strip()
                
                # Map parameter name to parameter key (use mu and sigma now)
                param_map = {'mean': 'mu', 'mu': 'mu', 'stddev': 'sigma', 'sigma': 'sigma', 'amp': 'amp'}
                param_key = param_map.get(parameter, parameter)
                
                # Check if this is a plain number or an expression
                # If it's a literal number with no operators/variables, use value= instead of expr=
                try:
                    # Try to parse as a float - if successful and no operators, it's a literal
                    float_val = float(rhs)
                    # Check if rhs contains any operators or variable references (contains letters, _, +, -, *, /, etc.)
                    import re
                    if not re.search(r'[a-zA-Z_+\*/(\-)\s]', rhs) or rhs == str(float_val):
                        # It's a literal number, use value= with vary=False
                        print(f"[DEBUG]   {parameter}: TIED to literal value {float_val}")
                        print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', value={float_val}, vary=False)")
                        model.set_param_hint(f'{prefix}{param_key}', value=float_val, vary=False)
                    else:
                        # It's an expression, use expr=
                        print(f"[DEBUG]   {parameter}: TIED to expression '{rhs}'")
                        print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', expr='{rhs}')")
                        model.set_param_hint(f'{prefix}{param_key}', expr=rhs)
                except (ValueError, TypeError):
                    # Not a number, treat as expression
                    print(f"[DEBUG]   {parameter}: TIED to expression '{rhs}'")
                    print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', expr='{rhs}')")
                    model.set_param_hint(f'{prefix}{param_key}', expr=rhs)
    
    def _apply_voigt_constraints(self, model, prefix, constraints):
        """Apply constraints to a Voigt model
        
        IMPORTANT: Combines all constraints into single set_param_hint calls
        to avoid lmfit state corruption from multiple calls on same parameter.
        
        NOTE: If a parameter has a tied constraint (expression), bounds are NOT applied
        since tied parameters are determined by their expression, not by direct fitting.
        """
        if not constraints:
            return
        
        print(f"\n[DEBUG] Applying Voigt constraints for {prefix}")
        print(f"[DEBUG]   Raw constraint dict: {constraints}")
        
        # Check which parameters have linked constraints (these will use expressions)
        linked_constraints = constraints.get('linked_constraints', [])
        tied_parameters = {lc.get('parameter', '').lower() for lc in linked_constraints}
        print(f"[DEBUG]   Tied parameters (with expressions): {tied_parameters}")
        
        # **AMPLITUDE CONSTRAINTS**
        amp_hints = {}
        if constraints.get('amplitude_fixed'):
            fixed_val = constraints.get('amplitude_fixed_value')
            if fixed_val:
                amp_hints['value'] = float(fixed_val)
            amp_hints['vary'] = False
            print(f"[DEBUG]   Amplitude: FIXED to {fixed_val}")
        else:
            amp_min, amp_max = constraints.get('amplitude_bounds', ('', ''))
            if amp_min or amp_max:
                if amp_min:
                    amp_hints['min'] = float(amp_min)
                if amp_max:
                    amp_hints['max'] = float(amp_max)
                print(f"[DEBUG]   Amplitude bounds: min={amp_min if amp_min else 'none'}, max={amp_max if amp_max else 'none'}")
            else:
                print(f"[DEBUG]   Amplitude: no bounds")
        
        # Apply amplitude hints in single call if any exist
        if amp_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}amp', {amp_hints})")
            model.set_param_hint(f'{prefix}amp', **amp_hints)
        
        # **CENTER CONSTRAINTS**
        # SKIP bounds if center is a tied parameter (has an expression)
        center_hints = {}
        if 'center' not in tied_parameters:
            if constraints.get('center_fixed'):
                fixed_val = constraints.get('center_fixed_value')
                if fixed_val:
                    center_hints['value'] = float(fixed_val)
                center_hints['vary'] = False
                print(f"[DEBUG]   Center: FIXED to {fixed_val}")
            else:
                center_min, center_max = constraints.get('center_bounds', ('', ''))
                if center_min or center_max:
                    if center_min:
                        center_hints['min'] = float(center_min)
                    if center_max:
                        center_hints['max'] = float(center_max)
                    print(f"[DEBUG]   Center bounds: min={center_min if center_min else 'none'}, max={center_max if center_max else 'none'}")
                else:
                    print(f"[DEBUG]   Center: no bounds")
        else:
            print(f"[DEBUG]   Center: TIED (bounds skipped, will use expression)")
        
        # Apply center hints in single call if any exist
        if center_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}center', {center_hints})")
            model.set_param_hint(f'{prefix}center', **center_hints)
        
        # **SIGMA CONSTRAINTS**
        # SKIP bounds if sigma is a tied parameter (has an expression)
        sigma_hints = {}
        if 'sigma' not in tied_parameters:
            if constraints.get('sigma_fixed'):
                fixed_val = constraints.get('sigma_fixed_value')
                if fixed_val:
                    sigma_hints['value'] = float(fixed_val)
                sigma_hints['vary'] = False
                print(f"[DEBUG]   Sigma: FIXED to {fixed_val}")
            else:
                sigma_min, sigma_max = constraints.get('sigma_bounds', ('', ''))
                if sigma_min or sigma_max:
                    if sigma_min:
                        sigma_hints['min'] = float(sigma_min)
                    if sigma_max:
                        sigma_hints['max'] = float(sigma_max)
                    print(f"[DEBUG]   Sigma bounds: min={sigma_min if sigma_min else 'none'}, max={sigma_max if sigma_max else 'none'}")
                else:
                    print(f"[DEBUG]   Sigma: no bounds")
        else:
            print(f"[DEBUG]   Sigma: TIED (bounds skipped, will use expression)")
        
        # Apply sigma hints in single call if any exist
        if sigma_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}sigma', {sigma_hints})")
            model.set_param_hint(f'{prefix}sigma', **sigma_hints)
        
        # **GAMMA CONSTRAINTS**
        # SKIP bounds if gamma is a tied parameter (has an expression)
        gamma_hints = {}
        if 'gamma' not in tied_parameters:
            if constraints.get('gamma_fixed'):
                fixed_val = constraints.get('gamma_fixed_value')
                if fixed_val:
                    gamma_hints['value'] = float(fixed_val)
                gamma_hints['vary'] = False
                print(f"[DEBUG]   Gamma: FIXED to {fixed_val}")
            else:
                gamma_min, gamma_max = constraints.get('gamma_bounds', ('', ''))
                if gamma_min or gamma_max:
                    if gamma_min:
                        gamma_hints['min'] = float(gamma_min)
                    if gamma_max:
                        gamma_hints['max'] = float(gamma_max)
                    print(f"[DEBUG]   Gamma bounds: min={gamma_min if gamma_min else 'none'}, max={gamma_max if gamma_max else 'none'}")
                else:
                    print(f"[DEBUG]   Gamma: no bounds")
        else:
            print(f"[DEBUG]   Gamma: TIED (bounds skipped, will use expression)")
        
        # Apply gamma hints in single call if any exist
        if gamma_hints:
            print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}gamma', {gamma_hints})")
            model.set_param_hint(f'{prefix}gamma', **gamma_hints)
        
        # Apply linked constraints (multiple parameter linking)
        for linked in linked_constraints:
            parameter = linked.get('parameter', 'center')
            expression = linked.get('expression', '')
            if expression:
                # Extract right-hand side if expression contains '='
                if '=' in expression:
                    rhs = expression.split('=', 1)[1].strip()
                else:
                    rhs = expression.strip()
                
                # Map parameter name to parameter key
                param_map = {'center': 'center', 'sigma': 'sigma', 'amp': 'amp', 'gamma': 'gamma'}
                param_key = param_map.get(parameter, parameter)
                
                # Check if this is a plain number or an expression
                # If it's a literal number with no operators/variables, use value= instead of expr=
                try:
                    # Try to parse as a float - if successful and no operators, it's a literal
                    float_val = float(rhs)
                    # Check if rhs contains any operators or variable references (contains letters, _, +, -, *, /, etc.)
                    import re
                    if not re.search(r'[a-zA-Z_+\*/(\-)\s]', rhs) or rhs == str(float_val):
                        # It's a literal number, use value= with vary=False
                        print(f"[DEBUG]   {parameter}: TIED to literal value {float_val}")
                        print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', value={float_val}, vary=False)")
                        model.set_param_hint(f'{prefix}{param_key}', value=float_val, vary=False)
                    else:
                        # It's an expression, use expr=
                        print(f"[DEBUG]   {parameter}: TIED to expression '{rhs}'")
                        print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', expr='{rhs}')")
                        model.set_param_hint(f'{prefix}{param_key}', expr=rhs)
                except (ValueError, TypeError):
                    # Not a number, treat as expression
                    print(f"[DEBUG]   {parameter}: TIED to expression '{rhs}'")
                    print(f"[DEBUG]   → lmfit call: model.set_param_hint('{prefix}{param_key}', expr='{rhs}')")
                    model.set_param_hint(f'{prefix}{param_key}', expr=rhs)

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
    
    def _extract_component_covariance(self, result, prefix, param_names):
        """Extract covariance matrix for a specific component from lmfit result
        
        This builds a FULL covariance matrix that includes all component parameters.
        For tied/constrained parameters, the covariance entries are zero (no independent uncertainty).
        
        Args:
            result: lmfit fit result object
            prefix: component prefix (e.g., 'v0_', 'g0_')
            param_names: list of parameter names for this component (e.g., ['amp', 'center', 'sigma', 'gamma'])
        
        Returns:
            Full NxN covariance matrix (N = len(param_names)) or None if not available
            - Free parameters use values from result.covar
            - Tied/constrained parameters have zero covariance entries
        """
        if result.covar is None:
            return None
        
        # Get free parameters list from lmfit result
        free_param_names = [name for name in result.params.keys() if result.params[name].vary]
        
        # Build full NxN covariance matrix (N = number of component parameters)
        n_params = len(param_names)
        full_cov = np.zeros((n_params, n_params), dtype=float)
        
        # Map each parameter to its index in free_params (or None if tied)
        param_to_free_idx = {}
        for param_name in param_names:
            full_name = f'{prefix}{param_name}'
            if full_name in free_param_names:
                try:
                    idx = free_param_names.index(full_name)
                    param_to_free_idx[param_name] = idx
                except ValueError:
                    param_to_free_idx[param_name] = None
            else:
                param_to_free_idx[param_name] = None  # Parameter is tied/constrained
        
        # Fill in covariance entries from lmfit result
        cov_array = np.array(result.covar, dtype=float)
        for i, param_i in enumerate(param_names):
            idx_i = param_to_free_idx[param_i]
            if idx_i is not None:  # This parameter is free
                for j, param_j in enumerate(param_names):
                    idx_j = param_to_free_idx[param_j]
                    if idx_j is not None:  # This parameter is free
                        # Check bounds to avoid index error
                        if idx_i < cov_array.shape[0] and idx_j < cov_array.shape[1]:
                            full_cov[i, j] = cov_array[idx_i, idx_j]
        
        # Tied parameters keep zero covariance (already initialized to 0)
        
        return full_cov

    def _auto_calculate_listfit_ew(self, result, components, left_bound, right_bound, x_fit, y_fit):
        """Automatically calculate equivalent widths for listfit if conditions are met.
        
        EW is calculated if and only if:
        1. Exactly 1 polynomial continuum is fitted
        2. At least 1 non-polynomial profile (Gaussian or Voigt) is fitted
        
        Args:
            result: lmfit fit result object
            components: list of component dicts from listfit
            left_bound: left wavelength bound
            right_bound: right wavelength bound
            x_fit: wavelength data
            y_fit: flux data
        
        Returns:
            Dictionary with EW results (type='equivalent_widths', with EWs for each component) 
            or None if conditions not met
        """
        try:
            # Count component types
            polynomial_components = [c for c in components if c['type'] == 'polynomial']
            profile_components = [c for c in components if c['type'] in ['gaussian', 'voigt']]
            
            # Check conditions: exactly 1 polynomial AND at least 1 profile
            if len(polynomial_components) != 1 or len(profile_components) == 0:
                # Don't calculate EW if conditions not met
                return None
            
            print("[AUTO_EW] Conditions met: 1 polynomial + " + str(len(profile_components)) + " profile(s). Calculating EW...")
            
            # Extract polynomial continuum from the lmfit result directly
            # (it's not in self.continuum_fits, but in the listfit components)
            poly_comp = polynomial_components[0]
            poly_order = poly_comp.get('order', 1)
            
            # The polynomial index is always 0 (we checked for exactly 1 polynomial)
            poly_index = 0
            prefix = f'p{poly_index}_'
            
            # Extract polynomial coefficients from lmfit result
            poly_coeffs = []
            for i in range(poly_order + 1):
                coeff_name = f'{prefix}c{i}'
                if coeff_name not in result.params:
                    print(f"[AUTO_EW] Error: Polynomial coefficient {coeff_name} not found in fit result")
                    return None
                poly_coeffs.append(float(result.params[coeff_name].value))
            
            # Extract covariance for the polynomial (same method as Gaussian/Voigt)
            poly_param_names = [f'c{i}' for i in range(poly_order + 1)]
            poly_covariance = self._extract_component_covariance(result, prefix, poly_param_names)
            
            if poly_covariance is None:
                print("[AUTO_EW] Warning: Polynomial covariance not available. Skipping EW calculation.")
                return None
            
            # Ensure covariance is a proper array
            if isinstance(poly_covariance, list):
                poly_covariance = np.array(poly_covariance, dtype=float)
            
            # Build continuum_fit_dict from extracted polynomial
            continuum_fit_dict = {
                'coeffs': poly_coeffs,  # In ascending order (c0, c1, ..., cn)
                'covariance': poly_covariance,
                'bounds': (left_bound, right_bound)
            }
            
            # Calculate EW for each profile component
            ew_data = {'type': 'equivalent_widths', 'ew_results': {}}
            
            gauss_count = 0
            voigt_count = 0
            
            for comp in components:
                if comp['type'] == 'gaussian':
                    prefix = f'g{gauss_count}_'
                    comp_name = f"Gaussian {gauss_count}"
                    
                    # Get covariance for this component
                    gaussian_param_names = ['amp', 'mean', 'stddev']
                    component_covariance = self._extract_component_covariance(result, prefix, gaussian_param_names)
                    
                    if component_covariance is None:
                        print(f"[AUTO_EW] Warning: No covariance for {comp_name}. Skipping EW.")
                        gauss_count += 1
                        continue
                    
                    # Get profile parameters
                    if all(f'{prefix}{pname}' in result.params for pname in gaussian_param_names):
                        try:
                            # Build fit_dict structure for _calculate_equivalent_width_monte_carlo
                            fit_dict = {
                                'amp': float(result.params[f'{prefix}amp'].value),
                                'mean': float(result.params[f'{prefix}mean'].value),
                                'stddev': float(result.params[f'{prefix}stddev'].value),
                                'bounds': (left_bound, right_bound),
                                'covariance': component_covariance,
                                # Add metadata for MC sampling of tied parameters
                                'result': result,
                                'component_prefix': prefix,
                                'fit_type': 'gaussian',
                                'param_names': gaussian_param_names
                            }
                            
                            # Calculate EW
                            ew_result = self._calculate_equivalent_width_monte_carlo(
                                fit_dict, continuum_fit_dict, fit_type='gaussian'
                            )
                            
                            if ew_result is not None:
                                ew_dict = ew_result
                                ew_median = ew_dict.get('ew')
                                ew_1sigma = (ew_dict.get('ew_1sigma_lower', 0), ew_dict.get('ew_1sigma_upper', 0))
                                ew_2sigma = (ew_dict.get('ew_2sigma_lower', 0), ew_dict.get('ew_2sigma_upper', 0))
                                ew_3sigma = (ew_dict.get('ew_3sigma_lower', 0), ew_dict.get('ew_3sigma_upper', 0))
                                
                                ew_data['ew_results'][comp_name] = {
                                    'ew_median': float(ew_median) if ew_median is not None else None,
                                    'ew_1sigma': [float(ew_1sigma[0]), float(ew_1sigma[1])],
                                    'ew_2sigma': [float(ew_2sigma[0]), float(ew_2sigma[1])],
                                    'ew_3sigma': [float(ew_3sigma[0]), float(ew_3sigma[1])],
                                    'profile_type': 'gaussian'
                                }
                                if ew_median is not None:
                                    ew_err = (ew_1sigma[1] - ew_1sigma[0]) / 2
                                    print(f"[AUTO_EW] {comp_name}: EW = {ew_median:.4f} ± {ew_err:.4f} Å")
                        except Exception as ew_e:
                            print(f"[AUTO_EW] Error calculating EW for {comp_name}: {ew_e}")
                    
                    gauss_count += 1
                
                elif comp['type'] == 'voigt':
                    prefix = f'v{voigt_count}_'
                    comp_name = f"Voigt {voigt_count}"
                    
                    # Get covariance for this component
                    voigt_param_names = ['amp', 'center', 'sigma', 'gamma']
                    component_covariance = self._extract_component_covariance(result, prefix, voigt_param_names)
                    
                    if component_covariance is None:
                        print(f"[AUTO_EW] Warning: No covariance for {comp_name}. Skipping EW.")
                        voigt_count += 1
                        continue
                    
                    # Get profile parameters
                    if all(f'{prefix}{pname}' in result.params for pname in voigt_param_names):
                        try:
                            # Build fit_dict structure for _calculate_equivalent_width_monte_carlo
                            fit_dict = {
                                'amp': float(result.params[f'{prefix}amp'].value),
                                'center': float(result.params[f'{prefix}center'].value),
                                'sigma': float(result.params[f'{prefix}sigma'].value),
                                'gamma': float(result.params[f'{prefix}gamma'].value),
                                'bounds': (left_bound, right_bound),
                                'covariance': component_covariance,
                                # Add metadata for MC sampling of tied parameters
                                'result': result,
                                'component_prefix': prefix,
                                'fit_type': 'voigt',
                                'param_names': voigt_param_names
                            }
                            
                            # Calculate EW
                            ew_result = self._calculate_equivalent_width_monte_carlo(
                                fit_dict, continuum_fit_dict, fit_type='voigt'
                            )
                            
                            if ew_result is not None:
                                ew_dict = ew_result
                                ew_median = ew_dict.get('ew')
                                ew_1sigma = (ew_dict.get('ew_1sigma_lower', 0), ew_dict.get('ew_1sigma_upper', 0))
                                ew_2sigma = (ew_dict.get('ew_2sigma_lower', 0), ew_dict.get('ew_2sigma_upper', 0))
                                ew_3sigma = (ew_dict.get('ew_3sigma_lower', 0), ew_dict.get('ew_3sigma_upper', 0))
                                
                                ew_data['ew_results'][comp_name] = {
                                    'ew_median': float(ew_median) if ew_median is not None else None,
                                    'ew_1sigma': [float(ew_1sigma[0]), float(ew_1sigma[1])],
                                    'ew_2sigma': [float(ew_2sigma[0]), float(ew_2sigma[1])],
                                    'ew_3sigma': [float(ew_3sigma[0]), float(ew_3sigma[1])],
                                    'profile_type': 'voigt'
                                }
                                if ew_median is not None:
                                    ew_err = (ew_1sigma[1] - ew_1sigma[0]) / 2
                                    print(f"[AUTO_EW] {comp_name}: EW = {ew_median:.4f} ± {ew_err:.4f} Å")
                        except Exception as ew_e:
                            print(f"[AUTO_EW] Error calculating EW for {comp_name}: {ew_e}")
                    
                    voigt_count += 1
            
            # Return the EW data if we calculated any
            if ew_data['ew_results']:
                print(f"[AUTO_EW] Completed: Calculated EW for {len(ew_data['ew_results'])} component(s)")
                return ew_data
            else:
                print("[AUTO_EW] No EW results to save")
                return None
        
        except Exception as e:
            print(f"[AUTO_EW] Exception during EW calculation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_listfit_components(self, result, components, x_fit, y_fit, err_fit, left_bound, right_bound):
        """Plot the fitted components with different colors"""
        # Assign single fit_id for all components of this listfit
        listfit_fit_id = self.next_fit_id()
        self.assign_fit_color(listfit_fit_id)
        
        x_smooth = np.linspace(x_fit.min(), x_fit.max(), len(x_fit) * 50)
        
        # Color mapping for components
        # Get colors from config
        gaussian_cfg = self.colors['profiles']['gaussian']
        voigt_cfg = self.colors['profiles']['voigt']
        continuum_cfg = self.colors['profiles']['continuum_line']
        colors = {'gaussian': gaussian_cfg['color'], 'voigt': voigt_cfg['color'], 'polynomial': continuum_cfg['color'], 'chebyshev': continuum_cfg['color']}
        
        # Plot mask regions as gray fill patches
        data_masks = [comp for comp in components if comp['type'] == 'data_mask']
        polynomial_guess_masks = [comp for comp in components if comp['type'] == 'polynomial_guess_mask']
        
        mask_count = 0
        for mask in data_masks:
            min_lambda = mask.get('min_lambda')
            max_lambda = mask.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                patch = self.ax.axvspan(min_lambda, max_lambda, alpha=0.2, color='gray', zorder=1)
                # Register with ItemTracker - include fit_dict with required fields for deletion handling
                position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                mask_fit_dict = {
                    'listfit_bounds': (left_bound, right_bound),
                    'min_lambda': min_lambda,
                    'max_lambda': max_lambda,
                    'component_obj': mask  # Store reference to the actual mask component
                }
                self.register_item('data_mask', f'Data Mask {mask_count+1} ({min_lambda:.2f}-{max_lambda:.2f} Å)', 
                                 fit_dict=mask_fit_dict, patch_obj=patch, position=position_str, color='gray',
                                 fit_id=listfit_fit_id)
                mask_count += 1
        
        poly_mask_count = 0
        for mask in polynomial_guess_masks:
            min_lambda = mask.get('min_lambda')
            max_lambda = mask.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                patch = self.ax.axvspan(min_lambda, max_lambda, alpha=0.1, color='lightgray', zorder=0.5, linestyle='--', edgecolor='gray', linewidth=1)
                # Register with ItemTracker - include fit_dict with required fields for deletion handling
                position_str = f"λ: {min_lambda:.2f}-{max_lambda:.2f} Å"
                poly_mask_fit_dict = {
                    'listfit_bounds': (left_bound, right_bound),
                    'min_lambda': min_lambda,
                    'max_lambda': max_lambda,
                    'component_obj': mask  # Store reference to the actual mask component
                }
                self.register_item('polynomial_guess_mask', f'Poly Guess Mask {poly_mask_count+1} ({min_lambda:.2f}-{max_lambda:.2f} Å)', 
                                 fit_dict=poly_mask_fit_dict, patch_obj=patch, position=position_str, color='lightgray',
                                 fit_id=listfit_fit_id)
                poly_mask_count += 1
        
        # Plot individual components
        gauss_count = 0
        voigt_count = 0
        poly_count = 0
        
        # Pre-collect polynomial info from this listfit for all profiles to use
        listfit_continuum = None
        polynomials = [c for c in components if c.get('type') == 'polynomial']
        chebyshevs = [c for c in components if c.get('type') == 'chebyshev']
        
        print(f"[DEBUG] plot_listfit_components: Found {len(polynomials)} polynomial component(s), {len(chebyshevs)} Chebyshev component(s)")
        
        # Extract polynomial continuum if present
        if len(polynomials) == 1:
            # Get polynomial coefficients and errors from fit result
            poly_comp = polynomials[0]
            prefix = f'p0_'
            order = poly_comp.get('order', 1)
            
            print(f"[DEBUG] Polynomial: order={order}, checking for params with prefix '{prefix}'")
            
            if f'{prefix}c0' in result.params:
                poly_coeffs = []
                poly_coeffs_err = []
                for i in range(order + 1):
                    coeff_val = result.params[f'{prefix}c{i}'].value
                    coeff_err = result.params[f'{prefix}c{i}'].stderr if result.params[f'{prefix}c{i}'].stderr is not None else 0.0
                    poly_coeffs.append(coeff_val)
                    poly_coeffs_err.append(coeff_err)
                
                # Reverse coefficients for np.polyval (expects highest order first)
                poly_coeffs_reversed = poly_coeffs[::-1]
                poly_coeffs_err_reversed = poly_coeffs_err[::-1]
                covariance = np.diag([e**2 if e > 0 else 1e-10 for e in poly_coeffs_err_reversed])
                
                # Store as continuum dict for all profiles to use
                listfit_continuum = {
                    'type': 'polynomial',
                    'coeffs': poly_coeffs_reversed,
                    'covariance': covariance,
                    'bounds': (left_bound, right_bound)
                }
                print(f"[DEBUG] Successfully extracted polynomial continuum: coeffs={poly_coeffs_reversed}, cov_diag={np.diag(covariance)}")
            else:
                print(f"[DEBUG] WARNING: Polynomial parameter '{prefix}c0' not found in result.params!")
                print(f"[DEBUG]   Available params: {list(result.params.keys())}")
        elif len(polynomials) > 1:
            print(f"[DEBUG] WARNING: Found {len(polynomials)} polynomials, only using first one")
            # TODO: handle multiple polynomials
        
        # Extract Chebyshev continuum if present
        if len(chebyshevs) == 1 and listfit_continuum is None:
            # Get Chebyshev coefficients and errors from fit result
            cheb_comp = chebyshevs[0]
            prefix = f'c0_'
            degree = cheb_comp.get('degree', 1)
            
            print(f"[DEBUG] Chebyshev: degree={degree}, checking for params with prefix '{prefix}'")
            
            if f'{prefix}c0' in result.params:
                cheb_coeffs = []
                cheb_coeffs_err = []
                for i in range(degree + 1):
                    coeff_val = result.params[f'{prefix}c{i}'].value
                    coeff_err = result.params[f'{prefix}c{i}'].stderr if result.params[f'{prefix}c{i}'].stderr is not None else 0.0
                    cheb_coeffs.append(coeff_val)
                    cheb_coeffs_err.append(coeff_err)
                
                # Get domain bounds stored during model building
                lam_min = cheb_comp.get('lam_min', left_bound)
                lam_max = cheb_comp.get('lam_max', right_bound)
                
                # Create covariance matrix from errors
                covariance = np.diag([e**2 if e > 0 else 1e-10 for e in cheb_coeffs_err])
                
                # Store as continuum dict for all profiles to use
                listfit_continuum = {
                    'type': 'chebyshev',
                    'coeffs': cheb_coeffs,
                    'covariance': covariance,
                    'bounds': (left_bound, right_bound),
                    'degree': degree,
                    'lam_min': lam_min,
                    'lam_max': lam_max
                }
                print(f"[DEBUG] Successfully extracted Chebyshev continuum: degree={degree}, coeffs={cheb_coeffs}, domain=[{lam_min}, {lam_max}]")
            else:
                print(f"[DEBUG] WARNING: Chebyshev parameter '{prefix}c0' not found in result.params!")
                print(f"[DEBUG]   Available params: {list(result.params.keys())}")
        elif len(chebyshevs) > 1:
            print(f"[DEBUG] WARNING: Found {len(chebyshevs)} Chebyshev components, only using first one")
            # TODO: handle multiple Chebyshev
        
        for comp in components:
            comp_type = comp['type']
            
            # Skip mask types and redshift components (they're handled separately)
            # Chebyshev IS handled here in this loop with polynomial
            if comp_type in ['polynomial_guess_mask', 'data_mask', 'redshift']:
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
                g_mu = params[f'{prefix}mu'].value
                g_sigma = params[f'{prefix}sigma'].value
                
                # Extract errors from lmfit results
                g_amp_err = params[f'{prefix}amp'].stderr if params[f'{prefix}amp'].stderr is not None else 0.0
                g_mu_err = params[f'{prefix}mu'].stderr if params[f'{prefix}mu'].stderr is not None else 0.0
                g_sigma_err = params[f'{prefix}sigma'].stderr if params[f'{prefix}sigma'].stderr is not None else 0.0
                
                y_component = self.gaussian(x_smooth, g_amp, g_mu, g_sigma)
                # Add label only for the first listfit gaussian
                label = 'Gaussian' if 'gaussian' not in self.legend_profile_types else None
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle=gaussian_cfg['linestyle'], linewidth=gaussian_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('gaussian')
                
                # Extract covariance for this Gaussian component
                gaussian_param_names = ['amp', 'mu', 'sigma']
                component_covariance = self._extract_component_covariance(result, prefix, gaussian_param_names)
                
                # Extract tie expressions from result object for MC EW reconstruction
                tie_expressions = {}
                all_free_param_names = []
                free_param_values_dict = {}
                
                if result is not None and hasattr(result, 'params'):
                    # Collect all parameter names and extract ties
                    for pname, param in result.params.items():
                        if param.expr is not None:  # This parameter is tied
                            tie_expressions[pname] = param.expr
                        elif param.vary:  # This is a free parameter
                            all_free_param_names.append(pname)
                            free_param_values_dict[pname] = param.value
                
                # Add to gaussian_fits for redshift mode
                gaussian_fit = {
                    'fit_id': listfit_fit_id,  # Use the Listfit fit_id, not self.fit_id
                    '_fit_id': self.fit_id,  # Keep original for backward compatibility
                    'is_velocity_mode': self.is_velocity_mode,
                    'component_id': self.component_id,
                    'amp': g_amp, 'amp_err': g_amp_err,
                    'mean': g_mu, 'mean_err': g_mu_err,
                    'stddev': g_sigma, 'stddev_err': g_sigma_err,
                    'bounds': (left_bound, right_bound),
                    'line_id': None,
                    'line_wavelength': None,
                    'line': line,
                    'rest_wavelength': self.rest_wavelength,
                    'rest_id': self.rest_id,
                    'z_sys': self.redshift,
                    'listfit_bounds': (left_bound, right_bound),
                    'gauss_index': gauss_count,
                    'covariance': component_covariance,
                    'continuum_fit_dict': listfit_continuum,  # Store the listfit continuum for EW calculation
                    'component_prefix': f'g{gauss_count}_',  # For extracting params from reconstructed dict
                    'param_names': ['amp', 'mu', 'sigma'],  # Parameter names after prefix
                    'tie_expressions': tie_expressions,  # For MC EW with tied parameters
                    'all_free_param_names': all_free_param_names,  # For MC EW reconstruction
                    'free_param_values_all': [free_param_values_dict.get(pn, 0.0) for pn in all_free_param_names],  # For MC EW reconstruction
                    'free_param_names_all': all_free_param_names,  # For MC EW reconstruction
                }
                
                # Add full covariance if available (for MC EW with tied parameters)
                if result is not None and result.covar is not None:
                    gaussian_fit['full_covariance'] = np.array(result.covar, dtype=float)
                # Store lmfit result separately to avoid cleanup issues (result objects have circular refs)
                if listfit_fit_id not in self.lmfit_results and result is not None:
                    self.lmfit_results[listfit_fit_id] = result
                if listfit_continuum is None:
                    print(f"[DEBUG] Gaussian {gauss_count}: continuum_fit_dict is None!")
                else:
                    print(f"[DEBUG] Gaussian {gauss_count}: continuum_fit_dict = {{'coeffs': ..., 'covariance': ...}}")
                self.gaussian_fits.append(gaussian_fit)
                # Register with ItemTracker
                position_str = f"λ: {g_mu:.2f} Å"
                item_id = self.register_item('gaussian', f'Gaussian {gauss_count+1}', fit_dict=gaussian_fit, 
                                           line_obj=line, position=position_str, color=color,
                                           fit_id=listfit_fit_id)
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
                
                # Extract covariance for this Voigt component
                voigt_param_names = ['amp', 'center', 'sigma', 'gamma']
                component_covariance = self._extract_component_covariance(result, prefix, voigt_param_names)
                
                # Extract tie expressions from result object for MC EW reconstruction
                tie_expressions = {}
                all_free_param_names = []
                free_param_values_dict = {}
                
                if result is not None and hasattr(result, 'params'):
                    # Collect all parameter names and extract ties
                    for pname, param in result.params.items():
                        if param.expr is not None:  # This parameter is tied
                            tie_expressions[pname] = param.expr
                        elif param.vary:  # This is a free parameter
                            all_free_param_names.append(pname)
                            free_param_values_dict[pname] = param.value
                
                # Add to voigt_fits for redshift mode
                voigt_fit = {
                    'fit_id': listfit_fit_id,  # Use the Listfit fit_id, not self.fit_id
                    '_fit_id': self.fit_id,  # Keep original for backward compatibility
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
                    'z_sys': self.redshift,
                    'listfit_bounds': (left_bound, right_bound),
                    'voigt_index': voigt_count,
                    'covariance': component_covariance,
                    'continuum_fit_dict': listfit_continuum,  # Store the listfit continuum for EW calculation
                    'component_prefix': f'v{voigt_count}_',  # For extracting params from reconstructed dict
                    'param_names': ['amp', 'center', 'sigma', 'gamma'],  # Parameter names after prefix
                    'tie_expressions': tie_expressions,  # For MC EW with tied parameters
                    'all_free_param_names': all_free_param_names,  # For MC EW reconstruction
                    'free_param_values_all': [free_param_values_dict.get(pn, 0.0) for pn in all_free_param_names],  # For MC EW reconstruction
                    'free_param_names_all': all_free_param_names,  # For MC EW reconstruction
                }
                
                # Add full covariance if available (for MC EW with tied parameters)
                if result is not None and result.covar is not None:
                    voigt_fit['full_covariance'] = np.array(result.covar, dtype=float)
                # Store lmfit result separately to avoid cleanup issues (result objects have circular refs)
                if listfit_fit_id not in self.lmfit_results and result is not None:
                    self.lmfit_results[listfit_fit_id] = result
                if listfit_continuum is None:
                    print(f"[DEBUG] Voigt {voigt_count}: continuum_fit_dict is None!")
                else:
                    print(f"[DEBUG] Voigt {voigt_count}: continuum_fit_dict = {{'coeffs': ..., 'covariance': ...}}")
                self.voigt_fits.append(voigt_fit)
                # Register with ItemTracker
                position_str = f"λ: {v_center:.2f} Å"
                item_id = self.register_item('voigt', f'Voigt {voigt_count+1}', fit_dict=voigt_fit,
                                           line_obj=line, position=position_str, color=color,
                                           fit_id=listfit_fit_id)
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
                poly_coeffs_err = []
                for i in range(order + 1):
                    coeff_val = params[f'{prefix}c{i}'].value
                    coeff_err = params[f'{prefix}c{i}'].stderr if params[f'{prefix}c{i}'].stderr is not None else 0.0
                    poly_coeffs.append(coeff_val)
                    poly_coeffs_err.append(coeff_err)
                # Reverse coefficients for np.polyval (expects highest order first)
                poly_coeffs_reversed = poly_coeffs[::-1]
                poly_coeffs_err_reversed = poly_coeffs_err[::-1]
                y_component = np.polyval(poly_coeffs_reversed, x_smooth)
                # Add label only for the first listfit polynomial
                label = 'Continuum' if 'continuum' not in self.legend_profile_types else None
                line, = self.ax.plot(x_smooth, y_component, color=color, linestyle=continuum_cfg['linestyle'], linewidth=continuum_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('continuum')
                
                # Extract covariance for this polynomial component
                poly_param_names = [f'c{i}' for i in range(order + 1)]
                component_covariance = self._extract_component_covariance(result, prefix, poly_param_names)
                
                # Build covariance matrix from errors (diagonal approximation)
                covariance = np.diag([e**2 if e > 0 else 1e-10 for e in poly_coeffs_err_reversed]) if poly_coeffs_err_reversed else np.diag([1e-10] * len(poly_coeffs_reversed))
                
                # CRITICAL FIX: Add polynomial components to continuum_fits so they appear in the total line
                # This ensures listfit polynomials are included when toggling the total line
                # IMPORTANT: Store coefficients REVERSED (highest to lowest degree) for np.polyval compatibility
                continuum_fit = {
                    'bounds': (left_bound, right_bound),
                    'coeffs': poly_coeffs_reversed,  # ← REVERSED to match np.polyval expectations
                    'coeffs_err': poly_coeffs_err_reversed,  # ← Also reverse errors to maintain alignment
                    'poly_order': order,
                    'line': line,
                    'is_velocity_mode': self.is_velocity_mode,
                    'listfit_source': True  # Mark this as coming from listfit
                }
                self.continuum_fits.append(continuum_fit)
                
                # Register with ItemTracker - store metadata for deletion handling AND polynomial coefficients for EW calculation
                # Store the polynomial component object as well for identity matching
                position_str = f"λ: {left_bound:.2f}-{right_bound:.2f} Å"
                poly_fit_dict = {
                    'listfit_bounds': (left_bound, right_bound),
                    'poly_index': poly_count,
                    'order': order,
                    'coeffs': poly_coeffs_reversed,  # Store coefficients for EW calculation
                    'coeffs_err': poly_coeffs_err_reversed,  # Store coefficient errors
                    'covariance': covariance,  # Store covariance matrix for EW Monte Carlo
                    'component_obj': comp  # Store reference to the actual component for deletion
                }
                item_id = self.register_item('polynomial', f'Polynomial (order={order})', fit_dict=poly_fit_dict, line_obj=line,
                                           position=position_str, color=color,
                                           fit_id=listfit_fit_id)
                poly_count += 1
            
            elif comp_type == 'chebyshev':
                prefix = f'c{poly_count}_'
                degree = comp.get('degree', 1)
                
                # Check if this component's parameters are in the fit result
                if f'{prefix}c0' not in params:
                    print(f"[DEBUG] Warning: Chebyshev component {poly_count} not found in fit result. Skipping...")
                    poly_count += 1
                    continue
                
                cheb_coeffs = []
                cheb_coeffs_err = []
                for i in range(degree + 1):
                    coeff_val = params[f'{prefix}c{i}'].value
                    coeff_err = params[f'{prefix}c{i}'].stderr if params[f'{prefix}c{i}'].stderr is not None else 0.0
                    cheb_coeffs.append(coeff_val)
                    cheb_coeffs_err.append(coeff_err)
                
                # Get domain bounds stored during model building
                lam_min = comp.get('lam_min', left_bound)
                lam_max = comp.get('lam_max', right_bound)
                
                # CRITICAL FIX: Only plot Chebyshev within the domain where it was fitted
                # Chebyshev polynomials are unstable outside [-1, 1]
                mask = (x_smooth >= lam_min) & (x_smooth <= lam_max)
                x_plot = x_smooth[mask]
                
                # Rescale x for Chebyshev evaluation (only within domain)
                x_rescaled = 2 * (x_plot - lam_min) / (lam_max - lam_min) - 1
                y_plot = np.polynomial.chebyshev.chebval(x_rescaled, cheb_coeffs)
                
                # Add label only for the first listfit Chebyshev
                label = 'Continuum' if 'continuum' not in self.legend_profile_types else None
                line, = self.ax.plot(x_plot, y_plot, color=color, linestyle=continuum_cfg['linestyle'], linewidth=continuum_cfg['linewidth'], label=label)
                if label:
                    self.legend_profile_types.add('continuum')
                
                # Build covariance matrix from errors (diagonal approximation)
                covariance = np.diag([e**2 if e > 0 else 1e-10 for e in cheb_coeffs_err]) if cheb_coeffs_err else np.diag([1e-10] * len(cheb_coeffs))
                
                # Add Chebyshev component to continuum_fits
                continuum_fit = {
                    'bounds': (left_bound, right_bound),
                    'type': 'chebyshev',
                    'coeffs': cheb_coeffs,  # NOT reversed for Chebyshev (uses chebval directly)
                    'coeffs_err': cheb_coeffs_err,
                    'degree': degree,
                    'lam_min': lam_min,
                    'lam_max': lam_max,
                    'line': line,
                    'is_velocity_mode': self.is_velocity_mode,
                    'listfit_source': True  # Mark this as coming from listfit
                }
                self.continuum_fits.append(continuum_fit)
                
                # Register with ItemTracker - store metadata for deletion handling
                position_str = f"λ: {left_bound:.2f}-{right_bound:.2f} Å"
                cheb_fit_dict = {
                    'listfit_bounds': (left_bound, right_bound),
                    'cheb_index': poly_count,
                    'degree': degree,
                    'coeffs': cheb_coeffs,
                    'coeffs_err': cheb_coeffs_err,
                    'covariance': covariance,
                    'lam_min': lam_min,
                    'lam_max': lam_max,
                    'component_obj': comp  # Store reference to the actual component for deletion
                }
                item_id = self.register_item('chebyshev', f'Chebyshev (degree={degree})', fit_dict=cheb_fit_dict, line_obj=line,
                                           position=position_str, color=color,
                                           fit_id=listfit_fit_id)
                poly_count += 1




