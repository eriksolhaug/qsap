"""
FitInformationWindow - Display detailed information about all fitted profiles
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from qsap.ui_utils import get_qsap_icon


class FitInformationWindow(QtWidgets.QWidget):
    """Window displaying detailed parameters for all fitted profiles"""
    
    item_selected = pyqtSignal(str)  # Emits item_id when a row is selected
    item_deselected = pyqtSignal()   # Emits when no rows are selected
    
    def __init__(self):
        super().__init__()
        self.fits_table = None
        self.item_id_map = {}  # Maps item_id to row index for quick lookup
        self.row_to_item_id = {}  # Maps row index to item_id
        self.currently_selected_item_id = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QSAP - Fit Information")
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(100, 100, 2000, 600)  # Wider window to show all information
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Fitted Profiles - Detailed Parameters (press 'S' to save all):"))
        
        # Table widget with scroll area
        self.fits_table = QtWidgets.QTableWidget()
        self.fits_table.setColumnCount(0)  # Will dynamically add columns based on fit types
        self.fits_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fits_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.fits_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.fits_table.horizontalHeader().setStretchLastSection(True)
        self.fits_table.setColumnCount(3)  # Name, Type, Parameters
        layout.addWidget(self.fits_table)
        
        # Info label
        self.info_label = QtWidgets.QLabel("Click on a row to highlight the corresponding item in the tracker")
        self.info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
    
    def add_fit(self, item_id, item_type, fit_dict, name):
        """Add a fit to the information window"""
        # Initialize columns if needed
        if self.fits_table.rowCount() == 0:
            self._setup_columns()
        
        # Add a new row
        row = self.fits_table.rowCount()
        self.fits_table.insertRow(row)
        
        # Store mapping
        self.item_id_map[item_id] = row
        self.row_to_item_id[row] = item_id
        
        # Populate columns based on fit type
        self._populate_row(row, item_id, item_type, fit_dict, name)
    
    def _setup_columns(self):
        """Setup column headers (will be called on first fit)"""
        columns = ['Name', 'Type', 'Parameters']
        self.fits_table.setColumnCount(len(columns))
        self.fits_table.setHorizontalHeaderLabels(columns)
        self.fits_table.setColumnWidth(0, 120)
        self.fits_table.setColumnWidth(1, 100)
        self.fits_table.setColumnWidth(2, 1700)  # Much wider for all parameters
    
    def _populate_row(self, row, item_id, item_type, fit_dict, name):
        """Populate a row with fit information based on type"""
        # Name column
        name_item = QtWidgets.QTableWidgetItem(name)
        name_item.setData(Qt.UserRole, item_id)
        self.fits_table.setItem(row, 0, name_item)
        
        # Type column
        type_item = QtWidgets.QTableWidgetItem(item_type.capitalize())
        self.fits_table.setItem(row, 1, type_item)
        
        # Parameters column - formatted based on type
        params_text = self._format_parameters(item_type, fit_dict)
        params_item = QtWidgets.QTableWidgetItem(params_text)
        params_item.setToolTip(params_text)  # Tooltip for full text
        self.fits_table.setItem(row, 2, params_item)
    
    def _format_parameters(self, item_type, fit_dict):
        """Format parameters for display based on fit type - show ALL information"""
        if not fit_dict:
            return "No parameters"
        
        params = []
        
        if item_type == 'gaussian':
            # Gaussian: all fit information
            params.append(f"Fit ID: {fit_dict.get('fit_id', 'N/A')}")
            params.append(f"Component ID: {fit_dict.get('component_id', 'N/A')}")
            params.append(f"μ = {fit_dict.get('mean', 0):.4f} ± {fit_dict.get('mean_err', 0):.4e}")
            params.append(f"σ = {fit_dict.get('stddev', 0):.4f} ± {fit_dict.get('stddev_err', 0):.4e}")
            params.append(f"A = {fit_dict.get('amp', 0):.4e} ± {fit_dict.get('amp_err', 0):.4e}")
            params.append(f"χ² = {fit_dict.get('chi2', 0):.4f}")
            params.append(f"χ²/ν = {fit_dict.get('chi2_nu', 0):.4f}")
            bounds = fit_dict.get('bounds', ())
            if bounds:
                params.append(f"Bounds: λ {bounds[0]:.2f}-{bounds[1]:.2f}")
            if fit_dict.get('line_id'):
                params.append(f"Line ID: {fit_dict.get('line_id')}")
                params.append(f"Line λ: {fit_dict.get('line_wavelength', 0):.4f}")
            params.append(f"Rest λ: {fit_dict.get('rest_wavelength', 'N/A')}")
            params.append(f"z_sys: {fit_dict.get('z_sys', 0):.6f}")
            params.append(f"Velocity mode: {fit_dict.get('is_velocity_mode', False)}")
            
        elif item_type == 'voigt':
            # Voigt: all fit information
            params.append(f"Fit ID: {fit_dict.get('fit_id', 'N/A')}")
            params.append(f"Component ID: {fit_dict.get('component_id', 'N/A')}")
            center = fit_dict.get('center', fit_dict.get('mean', 0))
            params.append(f"λ_center = {center:.4f} ± {fit_dict.get('center_err', fit_dict.get('mean_err', 0)):.4e}")
            
            sigma_l = fit_dict.get('sigma_l', 0)
            sigma_g = fit_dict.get('sigma_g', 0)
            sigma = fit_dict.get('sigma', 0)
            if sigma_l:
                params.append(f"σ_L = {sigma_l:.4e} ± {fit_dict.get('sigma_l_err', 0):.4e}")
            if sigma_g:
                params.append(f"σ_G = {sigma_g:.4e} ± {fit_dict.get('sigma_g_err', 0):.4e}")
            if sigma and not sigma_l:
                params.append(f"σ = {sigma:.4e} ± {fit_dict.get('sigma_err', 0):.4e}")
            
            gamma = fit_dict.get('gamma', 0)
            if gamma:
                params.append(f"γ = {gamma:.4e} ± {fit_dict.get('gamma_err', 0):.4e}")
            
            amp = fit_dict.get('amplitude', fit_dict.get('amp', 0))
            amp_err = fit_dict.get('amplitude_err', fit_dict.get('amp_err', 0))
            params.append(f"A = {amp:.4e} ± {amp_err:.4e}")
            
            params.append(f"χ² = {fit_dict.get('chi2', 0):.4f}")
            params.append(f"χ²/ν = {fit_dict.get('chi2_nu', 0):.4f}")
            
            bounds = fit_dict.get('bounds', ())
            if bounds:
                params.append(f"Bounds: λ {bounds[0]:.2f}-{bounds[1]:.2f}")
            
            if fit_dict.get('line_id'):
                params.append(f"Line ID: {fit_dict.get('line_id')}")
                params.append(f"Line λ: {fit_dict.get('line_wavelength', 0):.4f}")
            
            params.append(f"Rest λ: {fit_dict.get('rest_wavelength', 'N/A')}")
            params.append(f"z_sys: {fit_dict.get('z_sys', 0):.6f}")
            
            # Temperature info if available
            if 'logT_eff' in fit_dict:
                params.append(f"log(T_eff) = {fit_dict['logT_eff']:.2f}")
            if 'b' in fit_dict:
                params.append(f"b = {fit_dict['b']:.4e} cm/s")
            if 'N' in fit_dict:
                params.append(f"log(N) = {fit_dict['N']:.2f}")
            
            params.append(f"Velocity mode: {fit_dict.get('is_velocity_mode', False)}")
        
        elif item_type == 'continuum':
            # Continuum: all fit information
            poly_order = fit_dict.get('poly_order', 0)
            params.append(f"Polynomial order: {poly_order}")
            
            bounds = fit_dict.get('bounds', ())
            if bounds:
                params.append(f"Bounds: λ {bounds[0]:.2f}-{bounds[1]:.2f}")
            
            coeffs = fit_dict.get('coeffs', [])
            coeffs_err = fit_dict.get('coeffs_err', [])
            for i, (c, err) in enumerate(zip(coeffs, coeffs_err)):
                params.append(f"c_{i} = {c:.4e} ± {err:.4e}")
            
            params.append(f"Velocity mode: {fit_dict.get('is_velocity_mode', False)}")
        
        elif item_type == 'listfit':
            # Listfit: all fit information
            left_bound = fit_dict.get('left_bound', 0)
            right_bound = fit_dict.get('right_bound', 0)
            if not left_bound or not right_bound:
                bounds = fit_dict.get('bounds', ())
                if bounds:
                    left_bound, right_bound = bounds[0], bounds[1]
            if left_bound and right_bound:
                params.append(f"Bounds: λ {left_bound:.2f}-{right_bound:.2f}")
            
            chi2_nu = fit_dict.get('chi2_nu', fit_dict.get('chi2', 0))
            if chi2_nu > 100:  # Assume it's chi2 not chi2_nu
                params.append(f"χ² = {chi2_nu:.2f}")
            else:
                params.append(f"χ²/ν = {chi2_nu:.4f}")
            
            n_components = fit_dict.get('n_components', 0)
            if n_components:
                params.append(f"Components: {n_components}")
            
            # Display component information if available
            components = fit_dict.get('components', [])
            if components:
                params.append(f"Component details:")
                for i, comp in enumerate(components):
                    comp_type = comp.get('type', 'unknown')
                    params.append(f"  [{i}] Type: {comp_type}")
                    if comp_type == 'gaussian':
                        params.append(f"       μ₀={comp.get('mu0', 0):.4f}, σ₀={comp.get('sigma0', 0):.4e}")
                    elif comp_type == 'voigt':
                        params.append(f"       center₀={comp.get('center0', 0):.4f}, σ₀={comp.get('sigma0', 0):.4e}")
            
            # Result info if available
            result = fit_dict.get('result')
            if result:
                params.append(f"Fit result: {len(result.params)} parameters")
        
        return " | ".join(params) if params else "No parameters"
    
    def update_fit(self, item_id, fit_dict, name, item_type):
        """Update an existing fit's information"""
        if item_id not in self.item_id_map:
            # If not in table, add it
            self.add_fit(item_id, item_type, fit_dict, name)
            return
        
        row = self.item_id_map[item_id]
        
        # Update name
        name_item = self.fits_table.item(row, 0)
        if name_item:
            name_item.setText(name)
        
        # Update parameters
        params_text = self._format_parameters(item_type, fit_dict)
        params_item = QtWidgets.QTableWidgetItem(params_text)
        params_item.setToolTip(params_text)
        self.fits_table.setItem(row, 2, params_item)
    
    def remove_fit(self, item_id):
        """Remove a fit from the information window"""
        if item_id not in self.item_id_map:
            return
        
        row = self.item_id_map[item_id]
        self.fits_table.removeRow(row)
        
        # Update mappings
        del self.item_id_map[item_id]
        del self.row_to_item_id[row]
        
        # Rebuild row_to_item_id mapping since rows have shifted
        new_row_to_item_id = {}
        for r in range(self.fits_table.rowCount()):
            item = self.fits_table.item(r, 0)
            if item:
                new_item_id = item.data(Qt.UserRole)
                new_row_to_item_id[r] = new_item_id
                self.item_id_map[new_item_id] = r
        self.row_to_item_id = new_row_to_item_id
        
        # If this was the selected item, clear selection
        if self.currently_selected_item_id == item_id:
            self.currently_selected_item_id = None
            self.item_deselected.emit()
    
    def clear_all(self):
        """Clear all fits from the table"""
        self.fits_table.setRowCount(0)
        self.item_id_map.clear()
        self.row_to_item_id.clear()
        self.currently_selected_item_id = None
    
    def highlight_item(self, item_id):
        """Programmatically select a row corresponding to item_id"""
        if item_id not in self.item_id_map:
            return
        
        row = self.item_id_map[item_id]
        self.fits_table.selectRow(row)
    
    def on_selection_changed(self):
        """Handle row selection in the fit information table"""
        # Use selectedRanges to get full rows selected
        selected_ranges = self.fits_table.selectedRanges()
        
        if not selected_ranges:
            self.currently_selected_item_id = None
            self.item_deselected.emit()
            return
        
        # Get the first selected row
        row = selected_ranges[0].topRow()
        # Always get item_id from the Name column (column 0) since that's where it's stored
        name_item = self.fits_table.item(row, 0)
        
        if name_item:
            item_id = name_item.data(Qt.UserRole)
            self.currently_selected_item_id = item_id
            self.item_selected.emit(item_id)
