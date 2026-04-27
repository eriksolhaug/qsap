"""
Format Picker Dialog - PyQt5 based format selection UI

Displays detected spectrum formats and allows user
to select the format before loading the spectrum.
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHeaderView
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path


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
        self.detected_wave_unit = None  # Store detected unit from file
        
        # Try to detect wavelength unit from file
        self._detect_file_wave_unit()
        
        self.setWindowTitle("Select Spectrum Format")
        self.setGeometry(100, 100, 600, 400)
        self.setModal(True)
        
        # Sort candidates by score (highest first)
        self.candidates.sort(key=lambda c: c["score"], reverse=True)
        
        self._init_ui()
    
    def _detect_file_wave_unit(self):
        """Detect wavelength unit from FITS header if available."""
        from pathlib import Path
        path = Path(self.filepath)
        
        if path.suffix.lower() in (".fits", ".fit", ".fts"):
            try:
                from astropy.io import fits
                with fits.open(path, memmap=True) as hdul:
                    # Check primary HDU header
                    if len(hdul) > 0:
                        header = hdul[0].header
                        cunit = header.get("CUNIT1", "").upper()
                        if cunit:
                            # Map common FITS wavelength units
                            unit_map = {
                                "ANGSTROM": "Ångström",
                                "A": "Ångström",
                                "NM": "Nanometer",
                                "NANOMETER": "Nanometer",
                                "UM": "Micron",
                                "MICRON": "Micron",
                                "µM": "Micron",
                            }
                            for key, val in unit_map.items():
                                if key in cunit:
                                    self.detected_wave_unit = val
                                    return
            except Exception:
                pass
    
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
        
        # Detected wavelength unit info (always visible)
        if self.detected_wave_unit:
            unit_info_label = QtWidgets.QLabel(f"<i>Detected wavelength unit: <b>{self.detected_wave_unit}</b></i>")
            unit_info_label.setStyleSheet("color: #666666; font-size: 9px;")
            layout.addWidget(unit_info_label)
        
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
        
        ascii_layout.addItem(QtWidgets.QSpacerItem(0, 0), 5, 0, 1, 3)
        ascii_group.setLayout(ascii_layout)
        layout.addWidget(ascii_group, 0)
        
        self.ascii_frame = ascii_group
        self._update_ascii_visibility()
        self._update_columns_display()  # Populate columns for initial selection
        
        # NOW connect the signal AFTER frame is created
        self.format_list.itemSelectionChanged.connect(self._on_format_selected)
        
        # Scaling Factor Group
        scaling_group = QtWidgets.QGroupBox("Preprocessing (optional)")
        scaling_layout = QtWidgets.QGridLayout()
        
        scaling_label = QtWidgets.QLabel("Scaling Factor:")
        scaling_layout.addWidget(scaling_label, 0, 0)
        
        self.scaling_input = QtWidgets.QLineEdit()
        self.scaling_input.setPlaceholderText("1.0 (no scaling)")
        self.scaling_input.setText("1.0")
        self.scaling_input.setMaximumWidth(100)
        scaling_layout.addWidget(self.scaling_input, 0, 1)
        
        scaling_info = QtWidgets.QLabel(
            "Multiply all spectrum y-values by this factor.\n"
            "Useful for spectra with very small values (e.g., 1e-17).\n"
            "Set to 1.0 for no scaling."
        )
        scaling_info.setStyleSheet("color: #666; font-size: 10px;")
        scaling_layout.addWidget(scaling_info, 1, 0, 1, 2)
        
        scaling_layout.addItem(QtWidgets.QSpacerItem(0, 0), 2, 0, 1, 2)
        scaling_group.setLayout(scaling_layout)
        layout.addWidget(scaling_group)
        
        # Replace Values Group
        replace_values_group = QtWidgets.QGroupBox("Replace Values (NaN/Inf)")
        replace_values_layout = QtWidgets.QGridLayout()
        
        # Status label (will be updated dynamically)
        self.replace_values_status = QtWidgets.QLabel("Checking for NaN/Inf values...")
        self.replace_values_status.setStyleSheet("color: #666; font-size: 10px;")
        replace_values_layout.addWidget(self.replace_values_status, 0, 0, 1, 2)
        
        # Replacement value input (hidden by default)
        replace_label = QtWidgets.QLabel("Replace with value:")
        replace_values_layout.addWidget(replace_label, 1, 0)
        
        self.replace_values_input = QtWidgets.QLineEdit()
        self.replace_values_input.setPlaceholderText("0.0")
        self.replace_values_input.setText("0.0")
        self.replace_values_input.setMaximumWidth(100)
        replace_values_layout.addWidget(self.replace_values_input, 1, 1)
        
        # Initially hide the input
        replace_label.hide()
        self.replace_values_input.hide()
        
        self.replace_values_label = replace_label
        
        replace_values_layout.addItem(QtWidgets.QSpacerItem(0, 0), 2, 0, 1, 2)
        replace_values_group.setLayout(replace_values_layout)
        layout.addWidget(replace_values_group)
        
        self.replace_values_group = replace_values_group
        
        # Wavelength Unit Selection Group (always visible for all formats)
        unit_group = QtWidgets.QGroupBox("Wavelength Units")
        unit_layout = QtWidgets.QVBoxLayout()
        
        # Radio buttons in a horizontal layout
        radio_layout = QtWidgets.QHBoxLayout()
        
        self.unit_angstrom = QtWidgets.QRadioButton("Ångström (Å)")
        self.unit_angstrom.setChecked(True)
        radio_layout.addWidget(self.unit_angstrom)
        
        self.unit_nanometer = QtWidgets.QRadioButton("Nanometer (nm)")
        radio_layout.addWidget(self.unit_nanometer)
        
        self.unit_micron = QtWidgets.QRadioButton("Micron (μm)")
        radio_layout.addWidget(self.unit_micron)
        
        radio_layout.addStretch()
        unit_layout.addLayout(radio_layout)
        
        # Add detected unit label (small gray text)
        if self.detected_wave_unit:
            detected_label = QtWidgets.QLabel(f"File contains: {self.detected_wave_unit}")
            detected_label.setStyleSheet("color: gray; font-size: 9px;")
            unit_layout.addWidget(detected_label)
            
            # Pre-select the radio button based on detected unit
            if self.detected_wave_unit == "Ångström":
                self.unit_angstrom.setChecked(True)
            elif self.detected_wave_unit == "Nanometer":
                self.unit_nanometer.setChecked(True)
            elif self.detected_wave_unit == "Micron":
                self.unit_micron.setChecked(True)
        else:
            # Show message that unit was not detected
            not_detected_label = QtWidgets.QLabel("File wavelength unit not detected, using Ångström")
            not_detected_label.setStyleSheet("color: gray; font-size: 9px;")
            unit_layout.addWidget(not_detected_label)
        
        unit_layout.addItem(QtWidgets.QSpacerItem(0, 0))
        unit_group.setLayout(unit_layout)
        layout.addWidget(unit_group)
        
        # Wavelength Conversion Group
        wav_group = QtWidgets.QGroupBox("Wavelength Conversion")
        wav_layout = QtWidgets.QVBoxLayout()
        
        conversion_button_layout = QtWidgets.QHBoxLayout()
        
        self.wav_none = QtWidgets.QRadioButton("No conversion")
        self.wav_none.setChecked(True)
        conversion_button_layout.addWidget(self.wav_none)
        
        self.wav_air_to_vac = QtWidgets.QRadioButton("Air → Vacuum (Å)")
        conversion_button_layout.addWidget(self.wav_air_to_vac)
        
        self.wav_vac_to_air = QtWidgets.QRadioButton("Vacuum → Air (Å)")
        conversion_button_layout.addWidget(self.wav_vac_to_air)
        
        conversion_button_layout.addStretch()
        wav_layout.addLayout(conversion_button_layout)
        
        # Citation for the conversion formula
        citation_label = QtWidgets.QLabel(
            "Source: Donald Morton (2000, ApJ. Suppl., 130, 403)\n"
            "https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion"
        )
        citation_label.setStyleSheet("color: gray; font-size: 9px;")
        citation_label.setOpenExternalLinks(True)
        wav_layout.addWidget(citation_label)
        
        wav_layout.addItem(QtWidgets.QSpacerItem(0, 0))
        wav_group.setLayout(wav_layout)
        layout.addWidget(wav_group)
        
        # Trigger initial format selection check AFTER all UI elements are created
        self._on_format_selected()
        
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
        self._check_and_auto_scale_spectrum()
        self._check_for_nan_inf()
    
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
        
        elif fmt_key == "fits:image1d" or fmt_key == "fits:image1d:ext_data":
            # FITS 1D image - show wavelength info and error extension if present
            row_data = [
                ("primary", "Primary HDU", "wav", "✓"),
                (f"ext {options.get('hdu_flux', 1)}", f"Extension {options.get('hdu_flux', 1)}", "flux", "✓"),
            ]
            # Check if error extension is mentioned in notes or options
            notes = candidate.get("notes", "")
            error_ext = options.get("hdu_error")
            if error_ext is not None or "ERROR" in notes.upper():
                error_ext_num = error_ext if error_ext is not None else None
                if error_ext_num:
                    row_data.append((f"ext {error_ext_num}", f"Extension {error_ext_num}", "err", "✓"))
                else:
                    row_data.append(("ext", "ERROR/VARIANCE extension", "err", "✓"))
            
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
        
        # Handle scaling factor
        scaling_text = self.scaling_input.text().strip()
        if scaling_text and scaling_text != "1.0":  # Only process if different from default
            try:
                scaling_factor = float(scaling_text)
                if scaling_factor <= 0:
                    QtWidgets.QMessageBox.warning(self, "Error", "Scaling factor must be positive")
                    return
                options["scaling_factor"] = scaling_factor
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Scaling factor must be a number (e.g., 1.0, 1e17)")
                return
        
        # Handle NaN/Inf replacement value
        if self.replace_values_input.isVisible():
            replace_text = self.replace_values_input.text().strip()
            if replace_text:
                try:
                    replace_value = float(replace_text)
                    options["replace_nan_inf"] = replace_value
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Error", "Replacement value must be a number (e.g., 0.0, -999.0)")
                    return
            else:
                # If input is visible but empty, use default
                options["replace_nan_inf"] = 0.0
        
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
        
        # Handle wavelength conversion option
        if self.wav_air_to_vac.isChecked():
            options["wav_conversion"] = "air_to_vac"
        elif self.wav_vac_to_air.isChecked():
            options["wav_conversion"] = "vac_to_air"
        # else: no conversion (default)
        
        # Handle wavelength unit for ALL formats (not just ASCII)
        if self.unit_angstrom.isChecked():
            options["wave_unit"] = "angstrom"
        elif self.unit_nanometer.isChecked():
            options["wave_unit"] = "nanometer"
        elif self.unit_micron.isChecked():
            options["wave_unit"] = "micron"
        
        self.selected_format = fmt
        self.selected_options = options
        self.accept()
    
    def _check_and_auto_scale_spectrum(self):
        """
        Load a preview of the spectrum and auto-set scaling factor based on median flux.
        
        Process:
        1. Iteratively reject 1-sigma outliers until convergence
        2. If cleaned median is in range (0, 1e-6), calculate nearest 1eXX scaling factor
        """
        try:
            if self.format_list.currentRow() < 0:
                return
            
            current_idx = self.format_list.currentRow()
            candidate = self.candidates[current_idx]
            fmt_key = candidate["key"]
            options = dict(candidate.get("options", {}))
            
            # Get current user selections
            if fmt_key.startswith("ascii:"):
                delim = self.delim_input.text().strip()
                if delim == "":
                    delim = options.get("delimiter", "\t")
                
                try:
                    wave_col = self.col_wave.value()
                    flux_col = self.col_flux.value()
                    err_str = self.col_err.text().strip()
                    err_col = int(err_str) if err_str else None
                except ValueError:
                    return
                
                options.update({
                    "delimiter": delim,
                    "colmap": {"wave": wave_col, "flux": flux_col, "err": err_col},
                })
                fmt_key = "ascii:flex"
            
            # Try to load spectrum preview
            flux_data = self._load_spectrum_preview(self.filepath, fmt_key, options)
            if flux_data is None or len(flux_data) == 0:
                return
            
            # Get valid finite flux values
            flux_valid = flux_data[np.isfinite(flux_data)]
            if len(flux_valid) == 0:
                return
            
            # Take absolute value
            flux_abs = np.abs(flux_valid)
            
            # Iteratively reject 1-sigma outliers until convergence
            flux_cleaned = flux_abs.copy()
            prev_count = len(flux_cleaned)
            
            while True:
                mean_flux = np.mean(flux_cleaned)
                std_flux = np.std(flux_cleaned)
                
                # Mask for values within mean ± 1*std
                one_sigma_mask = np.abs(flux_cleaned - mean_flux) <= std_flux
                flux_cleaned = flux_cleaned[one_sigma_mask]
                
                # Check for convergence (no more values rejected)
                if len(flux_cleaned) == prev_count:
                    break
                
                prev_count = len(flux_cleaned)
            
            if len(flux_cleaned) == 0:
                return
            
            # Calculate median of converged data
            median_flux = np.median(flux_cleaned)
            
            # Check if scaling is needed: 0 < median < 1e-6
            if median_flux > 0 and median_flux < 1e-6:
                # Calculate power of 10 to bring median to ~1
                # scaling_factor = 10^x where x makes median * 10^x ≈ 1
                # So x ≈ -log10(median)
                power_of_ten = -np.floor(np.log10(median_flux))
                scaling_factor = f"1e{int(power_of_ten)}"
                
                self.scaling_input.setText(scaling_factor)
                scaled_median = median_flux * (10 ** power_of_ten)
                print(f"[Spectrum Loader] Median flux = {median_flux:.2e} (1σ-iterative), "
                      f"scaling to {scaling_factor} → median becomes {scaled_median:.2f}")
        
        except Exception as e:
            # Silently ignore errors during preview - don't interrupt user workflow
            print(f"[Spectrum Loader] Preview calculation skipped: {e}")
    
    def _load_spectrum_preview(self, filepath: str, fmt: str, options: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Load a preview of spectrum flux data (first 1000 rows).
        
        Returns
        -------
        np.ndarray or None
            Flux array if successful, None if failed
        """
        from qsap.spectrum_io import SpectrumIO
        
        try:
            # Load spectrum using SpectrumIO
            wav, spec, err, meta = SpectrumIO.read_spectrum(filepath, fmt=fmt, options=options)
            return spec
        except Exception as e:
            print(f"[Spectrum Loader] Could not load preview: {e}")
            return None
    
    def _check_for_nan_inf(self):
        """
        Check if loaded spectrum contains NaN or Inf values.
        Update UI to show replacement controls if values exist.
        """
        try:
            if self.format_list.currentRow() < 0:
                return
            
            current_idx = self.format_list.currentRow()
            candidate = self.candidates[current_idx]
            fmt_key = candidate["key"]
            options = dict(candidate.get("options", {}))
            
            # Get current user selections for ASCII formats
            if fmt_key.startswith("ascii:"):
                delim = self.delim_input.text().strip()
                if delim == "":
                    delim = options.get("delimiter", "\t")
                
                try:
                    wave_col = self.col_wave.value()
                    flux_col = self.col_flux.value()
                    err_str = self.col_err.text().strip()
                    err_col = int(err_str) if err_str else None
                except ValueError:
                    return
                
                options.update({
                    "delimiter": delim,
                    "colmap": {"wave": wave_col, "flux": flux_col, "err": err_col},
                })
                fmt_key = "ascii:flex"
            
            # Load spectrum preview
            flux_data = self._load_spectrum_preview(self.filepath, fmt_key, options)
            if flux_data is None:
                self.replace_values_status.setText("Unable to preview spectrum for NaN/Inf check")
                self.replace_values_label.hide()
                self.replace_values_input.hide()
                return
            
            # Check for NaN and Inf values
            nan_count = np.isnan(flux_data).sum()
            inf_count = np.isinf(flux_data).sum()
            total_bad = nan_count + inf_count
            
            if total_bad > 0:
                # Show message and input field
                msg = f"Found {total_bad} bad values: {nan_count} NaN, {inf_count} Inf"
                self.replace_values_status.setText(msg)
                self.replace_values_label.show()
                self.replace_values_input.show()
            else:
                # No bad values
                self.replace_values_status.setText("No NaN/Inf values detected ✓")
                self.replace_values_label.hide()
                self.replace_values_input.hide()
        
        except Exception as e:
            # Silently ignore errors during check
            print(f"[Spectrum Loader] NaN/Inf check skipped: {e}")
            self.replace_values_status.setText("Unable to check for NaN/Inf values")
            self.replace_values_label.hide()
            self.replace_values_input.hide()
    
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

