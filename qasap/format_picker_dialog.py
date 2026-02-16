"""
Format Picker Dialog - PyQt5 based format selection UI

Displays detected spectrum formats and allows user
to select the format before loading the spectrum.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHeaderView
from typing import Optional, Dict, Any, List, Tuple


class FormatPickerDialog(QtWidgets.QDialog):
    """
    Dialog for selecting spectrum file format from detected candidates.
    
    Shows a list of auto-detected formats, descriptions,
    and allows user to select which one to use. For ASCII formats, provides
    delimiter and column mapping options.
    """
    
    def __init__(self, filepath: str, candidates: List[Dict[str, Any]], parent=None):
        """
        Initialize the format picker dialog.
        
        Parameters
        ----------
        filepath : str
            Path to the spectrum file
        candidates : list of dict
            List of detected format candidates from detect_spectrum_format()
            Each dict has: {"key": "...", "score": int, "notes": str, "options": dict}
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.filepath = filepath
        self.candidates = candidates
        self.selected_format = None
        self.selected_options = None
        
        self.setWindowTitle("Select Spectrum Format")
        self.setGeometry(100, 100, 600, 400)
        self.setModal(True)
        
        # Sort candidates by score (highest first)
        self.candidates.sort(key=lambda c: c["score"], reverse=True)
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the user interface."""
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # File label
        file_label = QtWidgets.QLabel()
        file_label.setText(f"<b>File:</b> {self.filepath}")
        file_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(file_label)
        
        # Format list label
        list_label = QtWidgets.QLabel("Detected formats:")
        layout.addWidget(list_label)
        
        # Format listbox
        self.format_list = QtWidgets.QListWidget()
        
        for i, candidate in enumerate(self.candidates):
            key = candidate["key"]
            notes = candidate.get("notes", "")
            item_text = f"{key}  —  {notes}"
            item = QtWidgets.QListWidgetItem(item_text)
            self.format_list.addItem(item)
        
        # Select first item by default
        if self.format_list.count() > 0:
            self.format_list.setCurrentRow(0)
        
        self.format_list.setMaximumHeight(150)
        layout.addWidget(self.format_list, 0)
        
        # Columns information table
        columns_label = QtWidgets.QLabel("Column Mapping:")
        layout.addWidget(columns_label)
        
        self.columns_table = QtWidgets.QTableWidget()
        self.columns_table.setColumnCount(4)
        self.columns_table.setHorizontalHeaderLabels(["Col", "Description", "Import", "Used"])
        self.columns_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.columns_table.setMaximumHeight(120)
        self.columns_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.columns_table, 0)
        
        # ASCII Options Frame (create BEFORE connecting signals)
        ascii_group = QtWidgets.QGroupBox("ASCII Options")
        ascii_layout = QtWidgets.QGridLayout()
        
        # Delimiter
        ascii_layout.addWidget(QtWidgets.QLabel("Delimiter (auto if blank):"), 0, 0)
        self.delim_input = QtWidgets.QLineEdit()
        self.delim_input.setMaximumWidth(80)
        ascii_layout.addWidget(self.delim_input, 0, 1)
        ascii_layout.addItem(QtWidgets.QSpacerItem(100, 0), 0, 2)
        
        # Column mapping
        ascii_layout.addWidget(QtWidgets.QLabel("Columns (0-based): wave, flux, err (optional)"), 1, 0, 1, 3)
        
        # Wave column
        ascii_layout.addWidget(QtWidgets.QLabel("wave:"), 2, 0)
        self.col_wave = QtWidgets.QSpinBox()
        self.col_wave.setValue(0)
        self.col_wave.setMaximumWidth(60)
        ascii_layout.addWidget(self.col_wave, 2, 1)
        
        # Flux column
        ascii_layout.addWidget(QtWidgets.QLabel("flux:"), 3, 0)
        self.col_flux = QtWidgets.QSpinBox()
        self.col_flux.setValue(1)
        self.col_flux.setMaximumWidth(60)
        ascii_layout.addWidget(self.col_flux, 3, 1)
        
        # Error column
        ascii_layout.addWidget(QtWidgets.QLabel("err:"), 4, 0)
        self.col_err = QtWidgets.QLineEdit()
        self.col_err.setMaximumWidth(60)
        self.col_err.setPlaceholderText("blank for none")
        ascii_layout.addWidget(self.col_err, 4, 1)
        
        # Wavelength unit selection
        ascii_layout.addWidget(QtWidgets.QLabel("Wavelength units:"), 5, 0)
        
        # Radio buttons for unit selection
        unit_group = QtWidgets.QGroupBox()
        unit_layout = QtWidgets.QHBoxLayout()
        
        self.unit_angstrom = QtWidgets.QRadioButton("Ångström (Å)")
        self.unit_angstrom.setChecked(True)
        unit_layout.addWidget(self.unit_angstrom)
        
        self.unit_nanometer = QtWidgets.QRadioButton("Nanometer (nm)")
        unit_layout.addWidget(self.unit_nanometer)
        
        self.unit_micron = QtWidgets.QRadioButton("Micron (μm)")
        unit_layout.addWidget(self.unit_micron)
        
        unit_layout.addStretch()
        unit_group.setLayout(unit_layout)
        ascii_layout.addWidget(unit_group, 5, 1, 1, 2)
        
        ascii_layout.addItem(QtWidgets.QSpacerItem(0, 0), 6, 0, 1, 3)
        ascii_group.setLayout(ascii_layout)
        layout.addWidget(ascii_group, 0)
        
        self.ascii_frame = ascii_group
        self._update_ascii_visibility()
        self._update_columns_display()  # Populate columns for initial selection
        
        # NOW connect the signal AFTER frame is created
        self.format_list.itemSelectionChanged.connect(self._on_format_selected)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()
        
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setFixedWidth(80)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        load_btn = QtWidgets.QPushButton("Load")
        load_btn.setFixedWidth(80)
        load_btn.setDefault(True)
        load_btn.clicked.connect(self._on_load)
        button_layout.addWidget(load_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def _on_format_selected(self):
        """Handle format selection from list."""
        self._update_ascii_visibility()
        self._update_columns_display()
    
    def _update_columns_display(self):
        """Update the columns table based on selected format."""
        if self.format_list.currentRow() < 0:
            self.columns_table.setRowCount(0)
            return
        
        current_idx = self.format_list.currentRow()
        candidate = self.candidates[current_idx]
        options = candidate.get("options", {})
        
        # Clear table
        self.columns_table.setRowCount(0)
        
        # Map format to column info
        fmt_key = candidate["key"]
        
        if fmt_key.startswith("ascii:"):
            # ASCII format - show detected columns
            colmap = options.get("colmap", {})
            delimiter = options.get("delimiter", "\t")
            
            row_data = [
                ("0", "Wave/Lambda", "wav", "✓" if colmap.get("wave") == 0 else ""),
                ("1", "Flux/Spectrum", "flux", "✓" if colmap.get("flux") == 1 else ""),
                ("2", "Error", "err", "✓" if colmap.get("err") == 2 else ""),
            ]
            
            self.columns_table.setRowCount(len(row_data))
            for i, (col_idx, desc, import_name, used) in enumerate(row_data):
                self.columns_table.setItem(i, 0, QtWidgets.QTableWidgetItem(col_idx))
                self.columns_table.setItem(i, 1, QtWidgets.QTableWidgetItem(desc))
                self.columns_table.setItem(i, 2, QtWidgets.QTableWidgetItem(import_name))
                self.columns_table.setItem(i, 3, QtWidgets.QTableWidgetItem(used))
        
        elif fmt_key == "fits:image1d":
            # FITS 1D image - show wavelength info
            row_data = [
                ("0", "Primary HDU", "wav", "✓"),
                ("1", "Primary HDU", "flux", "✓"),
            ]
            self.columns_table.setRowCount(len(row_data))
            for i, (col_idx, desc, import_name, used) in enumerate(row_data):
                self.columns_table.setItem(i, 0, QtWidgets.QTableWidgetItem(col_idx))
                self.columns_table.setItem(i, 1, QtWidgets.QTableWidgetItem(desc))
                self.columns_table.setItem(i, 2, QtWidgets.QTableWidgetItem(import_name))
                self.columns_table.setItem(i, 3, QtWidgets.QTableWidgetItem(used))
        
        elif "table" in fmt_key:
            # FITS table - show detected columns
            hdu = options.get("hdu", 1)
            row_data = [
                ("Wave", f"HDU {hdu}", "wav", "✓"),
                ("Flux", f"HDU {hdu}", "flux", "✓"),
                ("Error", f"HDU {hdu}", "err", "✓"),
            ]
            self.columns_table.setRowCount(len(row_data))
            for i, (col_name, desc, import_name, used) in enumerate(row_data):
                self.columns_table.setItem(i, 0, QtWidgets.QTableWidgetItem(col_name))
                self.columns_table.setItem(i, 1, QtWidgets.QTableWidgetItem(desc))
                self.columns_table.setItem(i, 2, QtWidgets.QTableWidgetItem(import_name))
                self.columns_table.setItem(i, 3, QtWidgets.QTableWidgetItem(used))
    
    def _update_ascii_visibility(self):
        """Show/hide ASCII options based on selected format."""
        if self.format_list.currentRow() < 0:
            self.ascii_frame.hide()
            return
        
        current_idx = self.format_list.currentRow()
        fmt_key = self.candidates[current_idx]["key"]
        
        if fmt_key.startswith("ascii:"):
            self.ascii_frame.show()
        else:
            self.ascii_frame.hide()
    
    def _on_load(self):
        """Handle load button - validate and prepare result."""
        if self.format_list.currentRow() < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a format")
            return
        
        current_idx = self.format_list.currentRow()
        candidate = self.candidates[current_idx]
        fmt = candidate["key"]
        options = dict(candidate.get("options", {}))
        
        # Handle ASCII options if needed
        if fmt.startswith("ascii:"):
            delim = self.delim_input.text().strip()
            if delim == "":
                delim = options.get("delimiter", "\t")
            
            try:
                wave_col = self.col_wave.value()
                flux_col = self.col_flux.value()
                err_str = self.col_err.text().strip()
                err_col = int(err_str) if err_str else None
                
                # Get selected wavelength unit
                if self.unit_angstrom.isChecked():
                    wave_unit = "angstrom"
                elif self.unit_nanometer.isChecked():
                    wave_unit = "nanometer"
                elif self.unit_micron.isChecked():
                    wave_unit = "micron"
                else:
                    wave_unit = "angstrom"  # Default
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Columns must be integers (err can be blank)")
                return
            
            options.update({
                "delimiter": delim,
                "colmap": {"wave": wave_col, "flux": flux_col, "err": err_col},
                "wave_unit": wave_unit
            })
            # Force flexible reader if user edited mapping
            fmt = "ascii:flex"
        
        self.selected_format = fmt
        self.selected_options = options
        self.accept()
    
    def get_selection(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the selected format and options.
        
        Returns
        -------
        tuple or None
            (format_key, options_dict) if user clicked Load, None if cancelled
        """
        if self.selected_format:
            return (self.selected_format, self.selected_options or {})
        return None

