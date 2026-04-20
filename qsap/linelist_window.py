"""
LineListWindow - Line list dialog window with dual-panel interface
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from typing import List, Optional
from .linelist import LineList, Line
from qasap.ui_utils import get_qasap_icon


class LineListWindow(QtWidgets.QWidget):
    """
    Dual-panel line selection window.
    
    Left panel: Available line lists
    Right panel: Lines from selected line list
    """
    selected_line = QtCore.pyqtSignal(str, float)  # Signal: line_id, wavelength
    closed = pyqtSignal()

    def __init__(self, available_line_lists: Optional[List[LineList]] = None, 
                 line_wavelengths=None, line_ids=None):
        """
        Initialize the LineListWindow.
        
        Args:
            available_line_lists: List of LineList objects (new interface)
            line_wavelengths: Legacy parameter for backward compatibility
            line_ids: Legacy parameter for backward compatibility
        """
        super().__init__()
        self.available_line_lists = available_line_lists or []
        
        # Handle legacy interface
        if line_wavelengths and line_ids and not available_line_lists:
            # Create a single LineList from legacy parameters
            lines = [Line(wave=w, name=line_id) 
                    for w, line_id in zip(line_wavelengths, line_ids)]
            self.available_line_lists = [LineList(name="Lines", lines=lines)]
        
        self.selected_list: Optional[LineList] = None
        self.init_ui()

    def closeEvent(self, event):
        """Emit signal when window closes."""
        self.closed.emit()
        event.accept()

    def init_ui(self):
        """Initialize the user interface with dual panels."""
        self.setWindowTitle("QASAP - Line List")
        # Load and set window icon
        self.setWindowIcon(get_qasap_icon())
        self.setGeometry(100, 100, 700, 500)

        # Main layout - horizontal split
        main_layout = QtWidgets.QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create both panels first
        left_panel = self._create_left_panel()
        main_layout.addWidget(left_panel, 1)

        right_panel = self._create_right_panel()
        main_layout.addWidget(right_panel, 1)

        # Now connect the signal after both panels exist
        self.linelist_widget.itemSelectionChanged.connect(self._on_linelist_selected)
        
        # Trigger initial population of right panel
        if self.linelist_widget.count() > 0:
            self.linelist_widget.setCurrentRow(0)

        self.setLayout(main_layout)

    def _create_left_panel(self) -> QtWidgets.QWidget:
        """Create the left panel for line list selection."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QtWidgets.QLabel("Line Lists:")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        # List widget for available line lists
        self.linelist_widget = QtWidgets.QListWidget()
        
        # Populate with available line lists
        for line_list in self.available_line_lists:
            display_name = f"{line_list.name} ({len(line_list.lines)} lines)"
            item = QtWidgets.QListWidgetItem(display_name)
            item.setData(QtCore.Qt.UserRole, line_list)  # Store the LineList object
            self.linelist_widget.addItem(item)
        
        layout.addWidget(self.linelist_widget)
        panel.setLayout(layout)
        return panel

    def _create_right_panel(self) -> QtWidgets.QWidget:
        """Create the right panel for displaying lines from selected list."""
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QtWidgets.QLabel("Lines:")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        # List widget for lines
        self.lines_widget = QtWidgets.QListWidget()
        self.lines_widget.itemDoubleClicked.connect(self._on_line_selected)
        layout.addWidget(self.lines_widget)

        panel.setLayout(layout)
        return panel

    def _on_linelist_selected(self):
        """Handle line list selection change."""
        current_item = self.linelist_widget.currentItem()
        if not current_item:
            self.lines_widget.clear()
            return

        # Get the selected LineList
        self.selected_list = current_item.data(QtCore.Qt.UserRole)
        
        # Update right panel with lines from selected list
        self.lines_widget.clear()
        if self.selected_list:
            for line in self.selected_list.lines:
                item_text = f"{line.name}: {line.wave:.2f} Å"
                self.lines_widget.addItem(item_text)

    def _on_line_selected(self, item):
        """Handle line selection from the lines list."""
        if not self.selected_list:
            return

        # Extract wavelength from item text
        text = item.text()
        try:
            # Format: "name: wavelength Å"
            parts = text.split(': ')
            line_name = parts[0]
            wavelength_str = parts[1].replace(' Å', '')
            wavelength = float(wavelength_str)
            
            # Emit signal and close
            self.selected_line.emit(line_name, wavelength)
            self.close()
        except (ValueError, IndexError):
            print(f"Error parsing line: {text}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='QASAP - Quick Analysis of Spectra and Profiles (v0.1)')
    parser.add_argument('fits_file', type=str, nargs='?', help='Path to the spectrum (FITS) file')
    parser.add_argument('--redshift', type=float, default=0.0, help='Initial redshift value (default: 0.0)')
    parser.add_argument('--zoom_factor', type=float, default=0.1, help='Initial zoom factor (default: 0.1)')
    parser.add_argument('--file_flag', type=int, default=0, help='File flag (default: 0)')
    parser.add_argument('--lsf', type=str, default="10", help='Line spread function: float width in km/s or path to LSF file')
    parser.add_argument('--gui', action='store_true', help='Launch the GUI for interactive input')
    args = parser.parse_args()

    # Initialize QApplication (importantly, initialize this before the SpectrumPlotter)
    # ... and ensure a QApplication is running
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()

    # Launch the GUI if the --gui flag is used
    if args.gui:
        plotter = SpectrumPlotter(args.fits_file, args.redshift, args.zoom_factor, args.file_flag)
        opener_window = SpectrumPlotterApp(plotter)
        opener_window.show()

    # Handle file processing from command line if not using the GUI
    if args.fits_file:
        print(f"Opening file: {args.fits_file}")
        plotter = SpectrumPlotter(args.fits_file, args.redshift, args.zoom_factor, args.file_flag)
        plotter.plot_spectrum()
    else:
        print("No FITS file provided. Please specify a file or use the GUI.")
        sys.exit(1)  # Exit with an error code if no file is specified

    # Close the Qt
    sys.exit(app.exec_())  # Call exec_ only once at the end

if __name__ == '__main__':
    main()
