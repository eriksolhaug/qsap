"""
ListfitWindow - Multi-component spectrum fitting dialog
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon, QColor
import numpy as np
from scipy.optimize import curve_fit
from qsap.ui_utils import get_qsap_icon
from .redshift_line_selector import RedshiftLineSelector


class ComponentListTable(QtWidgets.QTableWidget):
    """Custom QTableWidget with right-click context menu for components"""
    
    request_set_guess = pyqtSignal(int, dict)  # component_row, component_dict
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)  # Add double-click handler
        self.listfit_window = parent  # Reference to parent ListfitWindow
    
    def _on_item_double_clicked(self, item):
        """Handle double-click on table items - open redshift dialog for Initial column"""
        row = self.row(item)
        col = self.column(item)
        
        # Check if this is the "Initial" column (column 3) and row is valid
        if col != 3 or row < 0 or row >= len(self.listfit_window.components):
            return
        
        component = self.listfit_window.components[row]
        comp_type = component.get('type', '').lower()
        
        # Only open dialog for redshift components
        if comp_type == 'redshift':
            self._create_redshift_guess(row, component)
    
    def _show_context_menu(self, position):
        """Show context menu at cursor position"""
        item = self.itemAt(position)
        if item is None:
            return
        
        row = self.row(item)
        if row < 0 or row >= len(self.listfit_window.components):
            return
        
        component = self.listfit_window.components[row]
        comp_type = component.get('type', '').lower()
        
        # Show context menu for Gaussian, Voigt, Polynomial, and Redshift
        if comp_type not in ['gaussian', 'voigt', 'polynomial', 'redshift', 'data_mask', 'polynomial_guess_mask']:
            return
        
        menu = QtWidgets.QMenu()
        
        # Add "Create Guess" action for non-mask types
        if comp_type == 'polynomial':
            action_set_guess = menu.addAction("Create Guess (Click & Drag)")
            action_set_guess.setToolTip("Click and drag to draw a straight line for polynomial guess")
            action_set_guess.triggered.connect(lambda: self.request_set_guess.emit(row, component))
        elif comp_type == 'redshift':
            action_set_guess = menu.addAction("Create Guess (Enter Redshift)")
            action_set_guess.setToolTip("Enter a redshift value as the initial guess")
            action_set_guess.triggered.connect(lambda: self._create_redshift_guess(row, component))
        elif comp_type in ['gaussian', 'voigt']:
            action_set_guess = menu.addAction("Create Guess (Click & Drag)")
            action_set_guess.setToolTip("Click and drag to draw a Gaussian/Voigt guess profile")
            action_set_guess.triggered.connect(lambda: self.request_set_guess.emit(row, component))
        
        # Check if guess exists
        guess = component.get('guess', {})
        has_guess = guess and any(v is not None for v in guess.values())
        
        # Show "Remove Guess" only if a guess exists
        if has_guess:
            menu.addSeparator()
            component_id = component.get('id')
            action_remove_guess = menu.addAction("Remove Guess")
            action_remove_guess.triggered.connect(lambda: self._remove_guess_and_update(row, component_id))
        
        # Add "Remove" option for redshift (removes redshift and all tied lines)
        if comp_type == 'redshift':
            menu.addSeparator()
            action_remove = menu.addAction("Remove Redshift & Tied Lines")
            action_remove.triggered.connect(lambda: self._remove_redshift_and_lines(row))
        
        menu.exec_(self.mapToGlobal(position))
    
    def _create_redshift_guess(self, row, component):
        """Create a guess for redshift parameter via dialog"""
        z, ok = QtWidgets.QInputDialog.getDouble(
            self.listfit_window,
            "Create Redshift Guess",
            "Enter redshift value (z):",
            value=0.0,
            minValue=-1.0,
            maxValue=10.0,
            decimals=6
        )
        
        if ok:
            component['guess'] = {'z': z}
            self._update_guess_indicator(row)
            print(f"[Guess] Redshift {component.get('label', '')}: z = {z}")
    
    def _remove_redshift_and_lines(self, redshift_row):
        """Remove redshift component and all lines tied to it"""
        if redshift_row < 0 or redshift_row >= len(self.listfit_window.components):
            return
        
        redshift_comp = self.listfit_window.components[redshift_row]
        if redshift_comp.get('type') != 'redshift':
            return
        
        # Get list of tied line indices
        tied_lines = redshift_comp.get('tied_lines', [])
        
        # Remove components in reverse order to maintain indices
        for line_idx in sorted(tied_lines, reverse=True):
            if 0 <= line_idx < len(self.listfit_window.components):
                self.listfit_window.components.pop(line_idx)
                self.listfit_window.component_list.removeRow(line_idx)
        
        # Remove the redshift component itself
        self.listfit_window.components.pop(redshift_row)
        self.listfit_window.component_list.removeRow(redshift_row)
        self.listfit_window.redshift_count = max(0, self.listfit_window.redshift_count - 1)
    
    def _remove_guess_and_update(self, row, component_id):
        """Remove guess from component and notify spectrum plotter"""
        if row < 0 or row >= len(self.listfit_window.components):
            return
        
        component = self.listfit_window.components[row]
        comp_type = component.get('type', '').lower()
        
        # Clear the guess dictionary based on component type
        if comp_type == 'polynomial':
            component['guess'] = {}
        elif comp_type == 'gaussian':
            component['guess'] = {'center': None, 'amp': None, 'stddev': None}
        elif comp_type == 'voigt':
            component['guess'] = {'center': None, 'amp': None, 'sigma': None, 'gamma': None}
        elif comp_type == 'redshift':
            component['guess'] = {'z': None}
        
        # Update visual indicator
        self._update_guess_indicator(row)
        
        # Update visual indicator
        self._update_guess_indicator(row)
        
        # Emit signal to spectrum plotter to remove the guess line
        self.listfit_window.request_remove_guess.emit(component_id)
    
    def _update_guess_indicator(self, row):
        """Update the visual indicator showing guess status in the table"""
        if row < 0 or row >= self.rowCount():
            return
        
        component = self.listfit_window.components[row]
        item = self.item(row, 0)
        if item is None:
            return
        
        label = component.get('label', '')
        
        # Simply display the label without checkmark or styling
        # (The Initial column now shows guess status clearly)
        item.setText(label)
        item.setForeground(QtGui.QColor("black"))
        
        # Update the Initial column display
        self.listfit_window._update_initial_display(row, component)


class ConstraintEditorDialog(QtWidgets.QDialog):
    """Dialog for editing constraints with Q key handling"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.reject()
            return
        super().keyPressEvent(event)


class ConstraintEditor(QtWidgets.QWidget):
    """Widget to edit constraints for a component"""
    
    def __init__(self, component, parent=None, all_components=None, set_constraint_callback=None):
        super().__init__(parent)
        self.component = component
        self.all_components = all_components or []
        self.linked_constraints = []  # List of {parameter: str, expression: str}
        self.set_constraint_callback = set_constraint_callback  # Callback for when Set button clicked
        self.init_ui()
    
    def init_ui(self):
        """Initialize constraint UI"""
        main_layout = QtWidgets.QHBoxLayout()
        
        # Left side: Constraint controls and linked constraints list
        left_layout = QtWidgets.QVBoxLayout()
        
        # Add header showing which component is being edited
        component_label = self.component.get('label', 'Component')
        header = QtWidgets.QLabel(f"Editing Constraints for: <b>{component_label}</b>")
        header.setStyleSheet("font-size: 11px; font-weight: bold; color: black; padding: 5px;")
        left_layout.addWidget(header)
        
        comp_type = self.component.get('type')
        
        if comp_type == 'gaussian':
            left_layout.addWidget(self._create_gaussian_constraints())
        elif comp_type == 'voigt':
            left_layout.addWidget(self._create_voigt_constraints())
        elif comp_type == 'polynomial':
            left_layout.addWidget(self._create_polynomial_constraints())
        
        # Constraint expression input
        expr_group = QtWidgets.QGroupBox("Link Parameters")
        expr_layout = QtWidgets.QVBoxLayout()
        
        # Examples
        examples_label = QtWidgets.QLabel(
            "Examples:\n"
            "  • g0_mu = g1_mu * (5008/4960)  (tie Gaussian centers to wavelength ratio)\n"
            "  • v1_mu = v0_mu * (5008/4960)  (tie Voigt centers to wavelength ratio)\n"
            "  • g0_sigma = g1_sigma  (equal Gaussian widths)\n"
            "  • v0_sigma = v1_sigma * (5008/4960)  (tie Voigt widths to wavelength ratio)\n"
            "  • g1_amp = g0_amp / 2  (amplitude scaling)\n"
            "  • g0_mean = g1_mean + 0.5  (offset)"
        )
        examples_label.setStyleSheet("font-size: 7px; color: #666666; font-style: italic;")
        expr_layout.addWidget(examples_label)
        
        expr_label = QtWidgets.QLabel("Enter constraint expression:")
        expr_label.setStyleSheet("font-size: 8px; color: gray;")
        expr_layout.addWidget(expr_label)
        
        self.constraint_expr_input = QtWidgets.QLineEdit()
        self.constraint_expr_input.setPlaceholderText("Example: g0_mu = g1_mu * (5008/4960)  or  v1_mu = v0_mu * (5008/4960)")
        self.constraint_expr_input.setMaximumHeight(30)
        self._track_focus_change(self.constraint_expr_input)
        expr_layout.addWidget(self.constraint_expr_input)
        
        expr_group.setLayout(expr_layout)
        left_layout.addWidget(expr_group)
        
        # Linked constraints section
        linked_group = QtWidgets.QGroupBox("Linked Constraints")
        linked_layout = QtWidgets.QVBoxLayout()
        
        linked_label = QtWidgets.QLabel("Active linked constraints:")
        linked_label.setStyleSheet("font-size: 9px; color: gray;")
        linked_layout.addWidget(linked_label)
        
        self.linked_constraints_list = QtWidgets.QListWidget()
        self.linked_constraints_list.setMaximumHeight(100)
        linked_layout.addWidget(self.linked_constraints_list)
        
        # + and - buttons
        button_layout = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.clicked.connect(self._add_linked_constraint)
        remove_btn = QtWidgets.QPushButton("-")
        remove_btn.setMaximumWidth(40)
        remove_btn.clicked.connect(self._remove_linked_constraint)
        button_layout.addWidget(add_btn)
        button_layout.addWidget(remove_btn)
        button_layout.addStretch()
        linked_layout.addLayout(button_layout)
        
        linked_group.setLayout(linked_layout)
        left_layout.addWidget(linked_group)
        left_layout.addStretch()
        
        # Right side: Component and parameter reference lists
        right_layout = QtWidgets.QVBoxLayout()
        
        # Components list
        right_layout.addWidget(QtWidgets.QLabel("Components:"))
        self.component_ref_list = QtWidgets.QListWidget()
        self.component_ref_list.itemClicked.connect(self._on_component_ref_clicked)
        self._populate_component_list()
        right_layout.addWidget(self.component_ref_list)
        
        # Parameters list (populated when component is selected)
        right_layout.addWidget(QtWidgets.QLabel("Parameters:"))
        self.parameter_ref_list = QtWidgets.QListWidget()
        self.parameter_ref_list.itemClicked.connect(self._on_parameter_selected)
        right_layout.addWidget(self.parameter_ref_list)
        
        # Active field indicator
        self.active_field_label = QtWidgets.QLabel("No field selected")
        self.active_field_label.setStyleSheet(
            "background-color: #ffffcc; color: #333333; padding: 4px; "
            "border: 1px solid #cccc00; border-radius: 3px; font-weight: bold; font-size: 9px;"
        )
        right_layout.addWidget(self.active_field_label)
        
        # Help text
        help_text = QtWidgets.QLabel(
            "Supported constraint expressions (equality only):\n"
            "Gaussian:  g0_mu, g0_amp, g0_sigma\n"
            "Voigt:  v0_mu, v0_amp, v0_sigma, v0_gamma\n"
            "\n"
            "Common use cases:\n"
            "  • g0_mu = g1_mu * (5008/4960)  — tie line centers to rest wavelength ratio\n"
            "  • g0_sigma = g1_sigma  — same line width\n"
            "  • v0_sigma = v1_sigma * (5008/4960)  — Voigt widths with wavelength scaling\n"
            "  • g0_sigma = 2 * g1_sigma  — width scaling\n"
            "\n"
            "For parameter bounds (min/max), use fields above.\n"
            "Inequality operators use bounds, not constraint expressions."
        )
        help_text.setStyleSheet("font-size: 7px; color: gray; font-style: italic;")
        right_layout.addWidget(help_text)
        
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        
        self.setLayout(main_layout)
        
        # Store selected component info for parameter insertion
        self.selected_component = None
    
    def _populate_component_list(self):
        """Populate component reference list with all components (including self)"""
        self.component_ref_list.clear()
        
        # Group components by type with their indices (including self for comparisons)
        for i, comp in enumerate(self.all_components):
            comp_type = comp.get('type')
            
            # Skip mask types
            if comp_type in ['polynomial_guess_mask', 'data_mask']:
                continue
            
            # Get display label - check if component has a label stored
            if comp.get('label'):
                display_label = comp.get('label')
            else:
                # Fallback to constructing label
                if comp_type == 'gaussian':
                    display_label = f"Gaussian {i}"
                elif comp_type == 'voigt':
                    display_label = f"Voigt {i}"
                elif comp_type == 'polynomial':
                    display_label = f"Polynomial {i}"
                else:
                    display_label = f"{comp_type.capitalize()} {i}"
            
            # Mark self component with asterisk
            if comp is self.component:
                display_label = f"{display_label} (this)"
            
            item = QtWidgets.QListWidgetItem(display_label)
            item.setData(QtCore.Qt.UserRole, (comp_type, i))
            self.component_ref_list.addItem(item)
    
    def _on_component_ref_clicked(self, item):
        """Handle component selection - populate parameter list"""
        comp_type, comp_idx = item.data(QtCore.Qt.UserRole)
        
        # Store selected component info
        self.selected_component = {
            'type': comp_type,
            'idx': comp_idx,
            'type_map': {'gaussian': 'g', 'voigt': 'v', 'polynomial': 'p'}
        }
        
        # Populate parameter list based on component type
        self.parameter_ref_list.clear()
        
        if comp_type == 'gaussian':
            params = ['mean', 'amp', 'stddev']
            param_names = ['Center (mean)', 'Amplitude (amp)', 'Width (stddev)']
        elif comp_type == 'voigt':
            params = ['center', 'amp', 'sigma', 'gamma']
            param_names = ['Center', 'Amplitude (amp)', 'Width (sigma)', 'Gamma']
        elif comp_type == 'polynomial':
            # Get the polynomial order from the component
            poly_comp = None
            for comp in self.all_components:
                if comp.get('type') == 'polynomial' and comp.get('id') == comp_idx:
                    poly_comp = comp
                    break
            
            order = poly_comp.get('order', 1) if poly_comp else 1
            params = [f'c{i}' for i in range(order + 1)]
            param_names = [f'Coefficient {i}' for i in range(order + 1)]
        else:
            return
        
        # Add parameters to list
        for param, param_name in zip(params, param_names):
            item = QtWidgets.QListWidgetItem(param_name)
            item.setData(QtCore.Qt.UserRole, param)
            self.parameter_ref_list.addItem(item)
    
    def _on_parameter_selected(self, item):
        """Insert selected parameter at cursor position in constraint expression field"""
        if self.selected_component is None:
            return
        
        parameter = item.data(QtCore.Qt.UserRole)
        comp_type = self.selected_component['type']
        comp_idx = self.selected_component['idx']
        type_map = self.selected_component['type_map']
        
        prefix = type_map.get(comp_type, '')
        
        # Build the full reference: e.g., "g0_mean" or "v1_sigma"
        component_id = f"{prefix}{comp_idx}_{parameter}"
        
        # Insert at cursor position in the constraint expression field
        text_field = self.constraint_expr_input
        cursor_pos = text_field.cursorPosition()
        current_text = text_field.text()
        
        new_text = current_text[:cursor_pos] + component_id + current_text[cursor_pos:]
        text_field.setText(new_text)
        
        # Move cursor after inserted text
        text_field.setCursorPosition(cursor_pos + len(component_id))
        text_field.setFocus()
    
    def _track_focus_change(self, text_field):
        """Track when constraint expression field gets focus"""
        text_field.focusInEvent = lambda e: self._on_expr_focus_in(text_field, e)
    
    def _on_expr_focus_in(self, text_field, event):
        """Called when constraint expression field gets focus"""
        self.active_field_label.setText("Ready to enter constraint expression")
        
        # Call original focusInEvent if it exists
        if hasattr(super(type(text_field), text_field), 'focusInEvent'):
            super(type(text_field), text_field).focusInEvent(event)
    
    def _parse_constraint_expression(self, expr_text):
        """
        Parse constraint expression to extract left-side parameter.
        
        Supports equality constraints only (lmfit limitation):
        - "g0_stddev = g1_stddev" → ("stddev", "g1_stddev")
        - "g1_stddev = 2 * g0_stddev" → ("stddev", "2 * g0_stddev")
        - "g0_mean = g1_mean + 0.5" → ("mean", "g1_mean + 0.5")
        
        Invalid examples (will raise ValueError):
        - "g0_amp > g1_amp" → Inequality operators not supported for expressions
        - "g1_stddev = ..." when editing g0 → Component mismatch
        
        Returns:
            tuple: (parameter_name, constraint_expression)
        
        Raises:
            ValueError: If expression is malformed or parameter doesn't match component
        """
        expr_text = expr_text.strip()
        
        # Only allow equality operator (=)
        # Inequality operators can't be used with lmfit's expr parameter
        if '=' not in expr_text:
            raise ValueError("Expression must contain operator (=). Inequality operators (>, <, >=, <=) are not supported for parameter linking.")
        
        # Check for inequality operators and give helpful error
        if any(op in expr_text for op in ['>', '<', '>=', '<=']):
            raise ValueError("Inequality operators (>, <, >=, <=) are not supported for parameter linking. Use equality (=) instead.")
        
        # Split on equals sign
        parts = expr_text.split('=', 1)
        if len(parts) != 2:
            raise ValueError("Expression must have exactly one '=' operator")
        
        left_part = parts[0].strip()
        right_part = parts[1].strip()
        
        # Get component info
        comp_type = self.component.get('type')
        
        # Build expected component prefix from component type
        type_map = {'gaussian': 'g', 'voigt': 'v', 'polynomial': 'p'}
        prefix = type_map.get(comp_type, '')
        
        # Find index of this component
        comp_idx = None
        for i, comp in enumerate(self.all_components):
            if comp is self.component:
                comp_idx = i
                break
        
        if comp_idx is None:
            raise ValueError("Component not found in component list")
        
        # Expected component ID (e.g., "g0")
        expected_comp_id = f"{prefix}{comp_idx}"
        
        # Extract parameter from left side
        # Expected format: g0_stddev, g0_mean, g0_amp, etc.
        if '_' not in left_part:
            raise ValueError(f"Invalid parameter format: '{left_part}'. Expected format: '{expected_comp_id}_<param>'")
        
        comp_id_part, param_part = left_part.split('_', 1)
        
        # Validate that this refers to the current component
        if comp_id_part != expected_comp_id:
            raise ValueError(f"Left side refers to {comp_id_part}, but editing {expected_comp_id}")
        
        # Validate parameter name matches component type
        if comp_type == 'gaussian':
            valid_params = ['mean', 'amp', 'stddev']
        elif comp_type == 'voigt':
            valid_params = ['center', 'amp', 'sigma', 'gamma']
        elif comp_type == 'polynomial':
            # Allow c0, c1, c2, etc.
            valid_params = [f'c{i}' for i in range(10)]  # Support up to c9
        else:
            raise ValueError(f"Unknown component type: {comp_type}")
        
        if param_part not in valid_params:
            raise ValueError(f"Invalid parameter '{param_part}' for {comp_type}. Valid: {', '.join(valid_params)}")
        
        # Return parameter name and the constraint expression (right side)
        return param_part, right_part
    
    def _add_linked_constraint(self):
        """Add constraint expression from the constraint expression field"""
        if not hasattr(self, 'constraint_expr_input'):
            QtWidgets.QMessageBox.warning(self, "Error", "Constraint expression field not found.")
            return
        
        expr_text = self.constraint_expr_input.text().strip()
        if not expr_text:
            QtWidgets.QMessageBox.warning(self, "Empty Expression", "Enter a constraint expression first.\n\nExamples:\n  g0_mean = g1_mean * (5008/4960)\n  v1_center = v0_center * (5008/4960)")
            return
        
        # Try to parse the expression
        try:
            parameter, constraint_expr = self._parse_constraint_expression(expr_text)
        except ValueError as e:
            QtWidgets.QMessageBox.critical(self, "Invalid Expression", f"Error in expression:\n\n{str(e)}")
            return
        
        # Create confirmation message
        component_label = self.component.get('label', 'Component')
        param_display = {
            'mean': 'Center (mean)',
            'center': 'Center',
            'amp': 'Amplitude',
            'stddev': 'Width (stddev)',
            'sigma': 'Width (sigma)',
            'gamma': 'Gamma',
        }
        
        if parameter.startswith('c'):
            param_label = f"Coefficient {parameter}"
        else:
            param_label = param_display.get(parameter, parameter.upper())
        
        confirmation = f"Add constraint:\n\n{component_label} → {param_label} = {constraint_expr}"
        
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Confirm Constraint")
        msg_box.setText(confirmation)
        msg_box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        msg_box.setDefaultButton(QtWidgets.QMessageBox.Yes)  # Set Yes as default
        
        reply = msg_box.exec()
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # Store the FULL constraint expression (including left-hand side parameter)
        # This is what the user entered in the expression field
        full_expression = expr_text
        
        # Add to linked constraints list
        self.linked_constraints.append({'parameter': parameter, 'expression': full_expression})
        
        # Update display list
        display_text = f"{param_label} = {constraint_expr}"
        self.linked_constraints_list.addItem(display_text)
        
        # Clear the input field
        self.constraint_expr_input.clear()
        self.constraint_expr_input.setFocus()

    
    def _remove_linked_constraint(self):
        """Remove the most recently added linked constraint"""
        if not self.linked_constraints:
            return
        
        self.linked_constraints.pop()
        if self.linked_constraints_list.count() > 0:
            self.linked_constraints_list.takeItem(self.linked_constraints_list.count() - 1)
    
    def _create_gaussian_constraints(self):
        group = QtWidgets.QGroupBox("Gaussian Constraints")
        layout = QtWidgets.QVBoxLayout()
        
        # Load existing constraints from component
        constraints = self.component.get('constraints', {})
        
        # Check if this is a redshift-tied component
        is_redshift_tied = self.component.get('tied_to_redshift', False)
        
        # Amplitude constraints
        amp_layout = QtWidgets.QHBoxLayout()
        amp_layout.addWidget(QtWidgets.QLabel("Amplitude (amp):"))
        self.amp_min = QtWidgets.QLineEdit()
        self.amp_min.setPlaceholderText("min")
        self.amp_min.setMaximumWidth(70)
        self.amp_min.setValidator(QDoubleValidator())
        self.amp_min.setText(constraints.get('amplitude_bounds', ('', ''))[0])
        amp_layout.addWidget(QtWidgets.QLabel("min:"))
        amp_layout.addWidget(self.amp_min)
        self.amp_max = QtWidgets.QLineEdit()
        self.amp_max.setPlaceholderText("max")
        self.amp_max.setMaximumWidth(70)
        self.amp_max.setValidator(QDoubleValidator())
        self.amp_max.setText(constraints.get('amplitude_bounds', ('', ''))[1])
        amp_layout.addWidget(QtWidgets.QLabel("max:"))
        amp_layout.addWidget(self.amp_max)
        self.amp_set_btn = QtWidgets.QPushButton("Set")
        self.amp_set_btn.setMaximumWidth(50)
        self.amp_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('amp'))
        amp_layout.addWidget(self.amp_set_btn)
        self.amp_fixed = QtWidgets.QCheckBox("Fixed")
        self.amp_fixed.setChecked(constraints.get('amplitude_fixed', False))
        self.amp_fixed.stateChanged.connect(lambda: self._update_amp_field_states())
        amp_layout.addWidget(self.amp_fixed)
        amp_layout.addStretch()
        layout.addLayout(amp_layout)
        
        # Center (mu) constraints - show redshift tie info if redshift-tied
        if is_redshift_tied:
            # Read-only display for redshift-tied center
            rest_wavelength = self.component.get('rest_wavelength')
            redshift_index = self.component.get('redshift_index', 0)
            
            mean_layout = QtWidgets.QHBoxLayout()
            mean_layout.addWidget(QtWidgets.QLabel("Center (mu):"))
            
            tie_info_label = QtWidgets.QLabel(f"Tied to Redshift z{redshift_index} — center = (1+z)*{rest_wavelength:.2f}")
            tie_info_label.setStyleSheet("font-style: italic; color: #0066cc;")
            mean_layout.addWidget(tie_info_label)
            mean_layout.addStretch()
            layout.addLayout(mean_layout)
            
            # Store the fields for consistency with non-tied version
            self.mean_min = None
            self.mean_max = None
            self.mean_fixed = None
        else:
            mean_layout = QtWidgets.QHBoxLayout()
            mean_layout.addWidget(QtWidgets.QLabel("Center (mu):"))
            self.mean_min = QtWidgets.QLineEdit()
            self.mean_min.setPlaceholderText("min")
            self.mean_min.setMaximumWidth(70)
            self.mean_min.setValidator(QDoubleValidator())
            self.mean_min.setText(constraints.get('mean_bounds', ('', ''))[0])
            mean_layout.addWidget(QtWidgets.QLabel("min:"))
            mean_layout.addWidget(self.mean_min)
            self.mean_max = QtWidgets.QLineEdit()
            self.mean_max.setPlaceholderText("max")
            self.mean_max.setMaximumWidth(70)
            self.mean_max.setValidator(QDoubleValidator())
            self.mean_max.setText(constraints.get('mean_bounds', ('', ''))[1])
            mean_layout.addWidget(QtWidgets.QLabel("max:"))
            mean_layout.addWidget(self.mean_max)
            self.mean_set_btn = QtWidgets.QPushButton("Set")
            self.mean_set_btn.setMaximumWidth(50)
            self.mean_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('mu'))
            mean_layout.addWidget(self.mean_set_btn)
            self.mean_fixed = QtWidgets.QCheckBox("Fixed")
            self.mean_fixed.setChecked(constraints.get('mean_fixed', False))
            self.mean_fixed.stateChanged.connect(lambda: self._update_mean_field_states())
            mean_layout.addWidget(self.mean_fixed)
            mean_layout.addStretch()
            layout.addLayout(mean_layout)
        
        # Width (sigma) constraints
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Width (sigma):"))
        self.sigma_min = QtWidgets.QLineEdit()
        self.sigma_min.setPlaceholderText("min")
        self.sigma_min.setMaximumWidth(70)
        self.sigma_min.setValidator(QDoubleValidator())
        self.sigma_min.setText(constraints.get('sigma_bounds', ('', ''))[0])
        sigma_layout.addWidget(QtWidgets.QLabel("min:"))
        sigma_layout.addWidget(self.sigma_min)
        self.sigma_max = QtWidgets.QLineEdit()
        self.sigma_max.setPlaceholderText("max")
        self.sigma_max.setMaximumWidth(70)
        self.sigma_max.setValidator(QDoubleValidator())
        self.sigma_max.setText(constraints.get('sigma_bounds', ('', ''))[1])
        sigma_layout.addWidget(QtWidgets.QLabel("max:"))
        sigma_layout.addWidget(self.sigma_max)
        self.sigma_set_btn = QtWidgets.QPushButton("Set")
        self.sigma_set_btn.setMaximumWidth(50)
        self.sigma_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('sigma'))
        sigma_layout.addWidget(self.sigma_set_btn)
        self.sigma_fixed = QtWidgets.QCheckBox("Fixed")
        self.sigma_fixed.setChecked(constraints.get('sigma_fixed', False))
        self.sigma_fixed.stateChanged.connect(lambda: self._update_sigma_field_states())
        sigma_layout.addWidget(self.sigma_fixed)
        sigma_layout.addStretch()
        layout.addLayout(sigma_layout)
        
        # Update field states based on Fixed checkboxes
        self._update_amp_field_states()
        self._update_mean_field_states()
        self._update_sigma_field_states()
        
        group.setLayout(layout)
        return group
    
    def _create_voigt_constraints(self):
        """Create constraint panel for Voigt"""
        group = QtWidgets.QGroupBox("Voigt Constraints")
        layout = QtWidgets.QVBoxLayout()
        
        # Load existing constraints from component
        constraints = self.component.get('constraints', {})
        
        # Check if this is a redshift-tied component
        is_redshift_tied = self.component.get('tied_to_redshift', False)
        
        # Amplitude
        amp_layout = QtWidgets.QHBoxLayout()
        amp_layout.addWidget(QtWidgets.QLabel("Amplitude (amp):"))
        self.amp_min = QtWidgets.QLineEdit()
        self.amp_min.setMaximumWidth(70)
        self.amp_min.setValidator(QDoubleValidator())
        self.amp_min.setText(constraints.get('amplitude_bounds', ('', ''))[0])
        amp_layout.addWidget(QtWidgets.QLabel("min:"))
        amp_layout.addWidget(self.amp_min)
        self.amp_max = QtWidgets.QLineEdit()
        self.amp_max.setMaximumWidth(70)
        self.amp_max.setValidator(QDoubleValidator())
        self.amp_max.setText(constraints.get('amplitude_bounds', ('', ''))[1])
        amp_layout.addWidget(QtWidgets.QLabel("max:"))
        amp_layout.addWidget(self.amp_max)
        self.amp_set_btn = QtWidgets.QPushButton("Set")
        self.amp_set_btn.setMaximumWidth(50)
        self.amp_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('amp'))
        amp_layout.addWidget(self.amp_set_btn)
        self.amp_fixed = QtWidgets.QCheckBox("Fixed")
        self.amp_fixed.setChecked(constraints.get('amplitude_fixed', False))
        self.amp_fixed.stateChanged.connect(lambda: self._update_amp_field_states())
        amp_layout.addWidget(self.amp_fixed)
        amp_layout.addStretch()
        layout.addLayout(amp_layout)
        
        # Center (mu) - show redshift tie info if redshift-tied
        if is_redshift_tied:
            # Read-only display for redshift-tied center
            rest_wavelength = self.component.get('rest_wavelength')
            redshift_index = self.component.get('redshift_index', 0)
            
            center_layout = QtWidgets.QHBoxLayout()
            center_layout.addWidget(QtWidgets.QLabel("Center (mu):"))
            
            tie_info_label = QtWidgets.QLabel(f"Tied to Redshift z{redshift_index} — center = (1+z)*{rest_wavelength:.2f}")
            tie_info_label.setStyleSheet("font-style: italic; color: #0066cc;")
            center_layout.addWidget(tie_info_label)
            center_layout.addStretch()
            layout.addLayout(center_layout)
            
            # Store the fields for consistency with non-tied version
            self.center_min = None
            self.center_max = None
            self.center_fixed = None
        else:
            center_layout = QtWidgets.QHBoxLayout()
            center_layout.addWidget(QtWidgets.QLabel("Center (mu):"))
            self.center_min = QtWidgets.QLineEdit()
            self.center_min.setMaximumWidth(70)
            self.center_min.setValidator(QDoubleValidator())
            self.center_min.setText(constraints.get('center_bounds', ('', ''))[0])
            center_layout.addWidget(QtWidgets.QLabel("min:"))
            center_layout.addWidget(self.center_min)
            self.center_max = QtWidgets.QLineEdit()
            self.center_max.setMaximumWidth(70)
            self.center_max.setValidator(QDoubleValidator())
            self.center_max.setText(constraints.get('center_bounds', ('', ''))[1])
            center_layout.addWidget(QtWidgets.QLabel("max:"))
            center_layout.addWidget(self.center_max)
            self.center_set_btn = QtWidgets.QPushButton("Set")
            self.center_set_btn.setMaximumWidth(50)
            self.center_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('mu'))
            center_layout.addWidget(self.center_set_btn)
            self.center_fixed = QtWidgets.QCheckBox("Fixed")
            self.center_fixed.setChecked(constraints.get('center_fixed', False))
            self.center_fixed.stateChanged.connect(lambda: self._update_center_field_states())
            center_layout.addWidget(self.center_fixed)
            center_layout.addStretch()
            layout.addLayout(center_layout)
        
        # Width (sigma)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Width (sigma):"))
        self.sigma_min = QtWidgets.QLineEdit()
        self.sigma_min.setMaximumWidth(70)
        self.sigma_min.setValidator(QDoubleValidator())
        self.sigma_min.setText(constraints.get('sigma_bounds', ('', ''))[0])
        sigma_layout.addWidget(QtWidgets.QLabel("min:"))
        sigma_layout.addWidget(self.sigma_min)
        self.sigma_max = QtWidgets.QLineEdit()
        self.sigma_max.setMaximumWidth(70)
        self.sigma_max.setValidator(QDoubleValidator())
        self.sigma_max.setText(constraints.get('sigma_bounds', ('', ''))[1])
        sigma_layout.addWidget(QtWidgets.QLabel("max:"))
        sigma_layout.addWidget(self.sigma_max)
        self.sigma_set_btn = QtWidgets.QPushButton("Set")
        self.sigma_set_btn.setMaximumWidth(50)
        self.sigma_set_btn.clicked.connect(lambda: self._request_set_constraint_bounds('sigma'))
        sigma_layout.addWidget(self.sigma_set_btn)
        self.sigma_fixed = QtWidgets.QCheckBox("Fixed")
        self.sigma_fixed.setChecked(constraints.get('sigma_fixed', False))
        self.sigma_fixed.stateChanged.connect(lambda: self._update_sigma_field_states())
        sigma_layout.addWidget(self.sigma_fixed)
        sigma_layout.addStretch()
        layout.addLayout(sigma_layout)
        
        # Damping (gamma)
        gamma_layout = QtWidgets.QHBoxLayout()
        gamma_layout.addWidget(QtWidgets.QLabel("Damping (gamma):"))
        self.gamma_min = QtWidgets.QLineEdit()
        self.gamma_min.setMaximumWidth(70)
        self.gamma_min.setValidator(QDoubleValidator())
        self.gamma_min.setText(constraints.get('gamma_bounds', ('', ''))[0])
        gamma_layout.addWidget(QtWidgets.QLabel("min:"))
        gamma_layout.addWidget(self.gamma_min)
        self.gamma_max = QtWidgets.QLineEdit()
        self.gamma_max.setMaximumWidth(70)
        self.gamma_max.setValidator(QDoubleValidator())
        self.gamma_max.setText(constraints.get('gamma_bounds', ('', ''))[1])
        gamma_layout.addWidget(QtWidgets.QLabel("max:"))
        gamma_layout.addWidget(self.gamma_max)
        self.gamma_fixed = QtWidgets.QCheckBox("Fixed")
        self.gamma_fixed.setChecked(constraints.get('gamma_fixed', False))
        gamma_layout.addWidget(self.gamma_fixed)
        gamma_layout.addStretch()
        layout.addLayout(gamma_layout)
        
        # Update field states based on Fixed checkboxes
        self._update_amp_field_states()
        self._update_center_field_states()
        self._update_sigma_field_states()
        
        group.setLayout(layout)
        return group
    
    def _create_polynomial_constraints(self):
        """Create constraint panel for Polynomial"""
        group = QtWidgets.QGroupBox("Polynomial Constraints")
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Polynomial coefficients are unconstrained by default."))
        layout.addWidget(QtWidgets.QLabel("Add constraints below if needed:"))
        
        group.setLayout(layout)
        return group
    
    def get_constraints(self):
        """Get constraint data from UI"""
        constraints = {}
        
        comp_type = self.component.get('type')
        is_redshift_tied = self.component.get('tied_to_redshift', False)
        
        if comp_type == 'gaussian':
            constraints['amplitude_bounds'] = (self.amp_min.text(), self.amp_max.text())
            # Skip center/mu constraints if redshift-tied
            if not is_redshift_tied:
                constraints['mean_bounds'] = (self.mean_min.text(), self.mean_max.text())
                constraints['mean_fixed'] = self.mean_fixed.isChecked()
                constraints['mean_fixed_value'] = self._get_fixed_value(self.mean_min, self.mean_max) if self.mean_fixed.isChecked() else None
            else:
                constraints['mean_bounds'] = ('', '')
                constraints['mean_fixed'] = False
                constraints['mean_fixed_value'] = None
            constraints['sigma_bounds'] = (self.sigma_min.text(), self.sigma_max.text())
            constraints['amplitude_fixed'] = self.amp_fixed.isChecked()
            constraints['amplitude_fixed_value'] = self._get_fixed_value(self.amp_min, self.amp_max) if self.amp_fixed.isChecked() else None
            constraints['sigma_fixed'] = self.sigma_fixed.isChecked()
            constraints['sigma_fixed_value'] = self._get_fixed_value(self.sigma_min, self.sigma_max) if self.sigma_fixed.isChecked() else None
            constraints['linked_constraints'] = self.linked_constraints
        
        elif comp_type == 'voigt':
            constraints['amplitude_bounds'] = (self.amp_min.text(), self.amp_max.text())
            # Skip center constraints if redshift-tied
            if not is_redshift_tied:
                constraints['center_bounds'] = (self.center_min.text(), self.center_max.text())
                constraints['center_fixed'] = self.center_fixed.isChecked()
                constraints['center_fixed_value'] = self._get_fixed_value(self.center_min, self.center_max) if self.center_fixed.isChecked() else None
            else:
                constraints['center_bounds'] = ('', '')
                constraints['center_fixed'] = False
                constraints['center_fixed_value'] = None
            constraints['sigma_bounds'] = (self.sigma_min.text(), self.sigma_max.text())
            constraints['gamma_bounds'] = (self.gamma_min.text(), self.gamma_max.text())
            constraints['amplitude_fixed'] = self.amp_fixed.isChecked()
            constraints['amplitude_fixed_value'] = self._get_fixed_value(self.amp_min, self.amp_max) if self.amp_fixed.isChecked() else None
            constraints['sigma_fixed'] = self.sigma_fixed.isChecked()
            constraints['sigma_fixed_value'] = self._get_fixed_value(self.sigma_min, self.sigma_max) if self.sigma_fixed.isChecked() else None
            constraints['gamma_fixed'] = self.gamma_fixed.isChecked()
            constraints['gamma_fixed_value'] = self._get_fixed_value(self.gamma_min, self.gamma_max) if self.gamma_fixed.isChecked() else None
            constraints['linked_constraints'] = self.linked_constraints
        
        elif comp_type == 'polynomial':
            constraints['linked_constraints'] = self.linked_constraints
        
        return constraints
    
    def _get_fixed_value(self, min_field, max_field):
        """Extract fixed value from min or max field (min takes priority)"""
        min_text = min_field.text().strip()
        max_text = max_field.text().strip()
        
        if min_text:
            return min_text
        elif max_text:
            return max_text
        return None
    
    def _update_amp_field_states(self):
        """Enable/disable amplitude min/max fields based on Fixed checkbox"""
        is_fixed = self.amp_fixed.isChecked()
        self.amp_min.setEnabled(not is_fixed)
        self.amp_max.setEnabled(not is_fixed)
        self.amp_set_btn.setEnabled(not is_fixed)
        if is_fixed:
            self.amp_min.setStyleSheet("background-color: #cccccc;")
            self.amp_max.setStyleSheet("background-color: #cccccc;")
        else:
            self.amp_min.setStyleSheet("")
            self.amp_max.setStyleSheet("")
    
    def _update_mean_field_states(self):
        """Enable/disable mean min/max fields based on Fixed checkbox"""
        if self.mean_fixed is None:
            return  # Redshift-tied component, no fields to update
        is_fixed = self.mean_fixed.isChecked()
        self.mean_min.setEnabled(not is_fixed)
        self.mean_max.setEnabled(not is_fixed)
        self.mean_set_btn.setEnabled(not is_fixed)
        if is_fixed:
            self.mean_min.setStyleSheet("background-color: #cccccc;")
            self.mean_max.setStyleSheet("background-color: #cccccc;")
        else:
            self.mean_min.setStyleSheet("")
            self.mean_max.setStyleSheet("")
    
    def _update_center_field_states(self):
        """Enable/disable center min/max fields based on Fixed checkbox"""
        if self.center_fixed is None:
            return  # Redshift-tied component, no fields to update
        is_fixed = self.center_fixed.isChecked()
        self.center_min.setEnabled(not is_fixed)
        self.center_max.setEnabled(not is_fixed)
        self.center_set_btn.setEnabled(not is_fixed)
        if is_fixed:
            self.center_min.setStyleSheet("background-color: #cccccc;")
            self.center_max.setStyleSheet("background-color: #cccccc;")
        else:
            self.center_min.setStyleSheet("")
            self.center_max.setStyleSheet("")
    
    def _update_sigma_field_states(self):
        """Enable/disable sigma min/max fields based on Fixed checkbox"""
        is_fixed = self.sigma_fixed.isChecked()
        self.sigma_min.setEnabled(not is_fixed)
        self.sigma_max.setEnabled(not is_fixed)
        self.sigma_set_btn.setEnabled(not is_fixed)
        if is_fixed:
            self.sigma_min.setStyleSheet("background-color: #cccccc;")
            self.sigma_max.setStyleSheet("background-color: #cccccc;")
        else:
            self.sigma_min.setStyleSheet("")
            self.sigma_max.setStyleSheet("")
    
    def _request_set_constraint_bounds(self, parameter):
        """Request to set constraint bounds interactively in spectrum plotter"""
        if self.set_constraint_callback:
            self.set_constraint_callback(parameter)
        else:
            print(f"[Constraints] Set button clicked for parameter: {parameter}")


class ListfitWindow(QtWidgets.QWidget):
    """Dialog for defining and fitting multiple spectrum components"""
    
    fit_requested = pyqtSignal(dict)  # Emits dict with 'components' and 'tied_parameters'
    bounds_cleared = pyqtSignal()  # Emits when window is cancelled to clear bounds
    cleanup_guesses = pyqtSignal()  # Emits when window is closed/cancelled to remove all drawn guesses
    components_changed = pyqtSignal(list)  # Emits when components are added/removed to update plot
    request_set_guess = pyqtSignal(int, dict)  # Emits (component_row, component_dict) when user requests to set guess
    request_remove_guess = pyqtSignal(int)  # Emits component_id when user requests to remove guess
    request_draw_data_mask_regions = pyqtSignal()  # Emits when user requests to draw data mask regions
    request_draw_polynomial_mask_regions = pyqtSignal()  # Emits when user requests to draw polynomial mask regions
    request_set_constraint_bounds = pyqtSignal(str)  # Emits parameter name (amp, mu, sigma, etc.) when user clicks "Set" button
    
    def __init__(self, bounds, resources_dir=None):
        super().__init__()
        self.bounds = bounds
        self.resources_dir = resources_dir
        self.components = []  # List of {'type': 'gaussian'|'voigt'|'polynomial'|'polynomial_guess_mask'|'data_mask'|'redshift', ...}
        self.gaussian_count = 0
        self.voigt_count = 0
        self.polynomial_count = 0
        self.polynomial_guess_mask_count = 0
        self.data_mask_count = 0
        self.redshift_count = 0
        self.redshift_line_selector = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QSAP - List Fit")
        # Load and set window icon
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(500, 100, 700, 500)
        
        layout = QtWidgets.QHBoxLayout()
        
        # Left side: Component controls
        left_layout = QtWidgets.QVBoxLayout()
        left_layout.addWidget(QtWidgets.QLabel("Add Components:"))
        
        # Gaussian controls
        gaussian_layout = QtWidgets.QHBoxLayout()
        self.gaussian_label = QtWidgets.QLabel("Gaussian:")
        self.btn_gaussian_add = QtWidgets.QPushButton("+")
        self.btn_gaussian_add.setMaximumWidth(40)
        self.btn_gaussian_add.clicked.connect(lambda: self.add_component('gaussian'))
        self.btn_gaussian_remove = QtWidgets.QPushButton("-")
        self.btn_gaussian_remove.setMaximumWidth(40)
        self.btn_gaussian_remove.clicked.connect(lambda: self.remove_component('gaussian'))
        gaussian_layout.addWidget(self.gaussian_label)
        gaussian_layout.addWidget(self.btn_gaussian_add)
        gaussian_layout.addWidget(self.btn_gaussian_remove)
        gaussian_layout.addStretch()
        left_layout.addLayout(gaussian_layout)
        
        # Voigt controls
        voigt_layout = QtWidgets.QHBoxLayout()
        self.voigt_label = QtWidgets.QLabel("Voigt:")
        self.btn_voigt_add = QtWidgets.QPushButton("+")
        self.btn_voigt_add.setMaximumWidth(40)
        self.btn_voigt_add.clicked.connect(lambda: self.add_component('voigt'))
        self.btn_voigt_remove = QtWidgets.QPushButton("-")
        self.btn_voigt_remove.setMaximumWidth(40)
        self.btn_voigt_remove.clicked.connect(lambda: self.remove_component('voigt'))
        voigt_layout.addWidget(self.voigt_label)
        voigt_layout.addWidget(self.btn_voigt_add)
        voigt_layout.addWidget(self.btn_voigt_remove)
        voigt_layout.addStretch()
        left_layout.addLayout(voigt_layout)
        
        # Polynomial controls
        poly_layout = QtWidgets.QVBoxLayout()
        poly_header = QtWidgets.QHBoxLayout()
        self.poly_label = QtWidgets.QLabel("Polynomial:")
        poly_header.addWidget(self.poly_label)
        poly_header.addStretch()
        poly_layout.addLayout(poly_header)
        
        poly_order_layout = QtWidgets.QHBoxLayout()
        self.poly_order_label = QtWidgets.QLabel("Order:")
        self.poly_order_input = QtWidgets.QLineEdit("1")
        self.poly_order_input.setMaximumWidth(80)
        self.poly_order_input.setValidator(QIntValidator(0, 10))
        poly_order_layout.addWidget(self.poly_order_label)
        poly_order_layout.addWidget(self.poly_order_input)
        poly_order_layout.addStretch()
        poly_layout.addLayout(poly_order_layout)
        
        poly_button_layout = QtWidgets.QHBoxLayout()
        self.btn_poly_add = QtWidgets.QPushButton("+")
        self.btn_poly_add.setMaximumWidth(40)
        self.btn_poly_add.clicked.connect(self.add_polynomial)
        self.btn_poly_remove = QtWidgets.QPushButton("-")
        self.btn_poly_remove.setMaximumWidth(40)
        self.btn_poly_remove.clicked.connect(lambda: self.remove_component('polynomial'))
        poly_button_layout.addWidget(QtWidgets.QLabel(""))
        poly_button_layout.addWidget(self.btn_poly_add)
        poly_button_layout.addWidget(self.btn_poly_remove)
        poly_button_layout.addStretch()
        poly_layout.addLayout(poly_button_layout)
        
        left_layout.addLayout(poly_layout)
        
        # Polynomial Guess Mask controls (for masking regions in polynomial initial guess)
        poly_mask_layout = QtWidgets.QVBoxLayout()
        poly_mask_header = QtWidgets.QHBoxLayout()
        self.poly_mask_label = QtWidgets.QLabel("Polynomial Guess Mask:")
        poly_mask_header.addWidget(self.poly_mask_label)
        poly_mask_header.addStretch()
        poly_mask_layout.addLayout(poly_mask_header)
        
        poly_mask_description = QtWidgets.QLabel("Mask wavelength ranges to exclude from Polynomial initial guess")
        poly_mask_description.setStyleSheet("font-size: 9px; color: gray; font-style: italic;")
        poly_mask_layout.addWidget(poly_mask_description)
        
        poly_mask_range_layout = QtWidgets.QHBoxLayout()
        self.poly_mask_min_label = QtWidgets.QLabel("Min λ:")
        self.poly_mask_min_input = QtWidgets.QLineEdit()
        self.poly_mask_min_input.setMaximumWidth(100)
        self.poly_mask_min_input.setPlaceholderText("e.g., 5500")
        poly_mask_range_layout.addWidget(self.poly_mask_min_label)
        poly_mask_range_layout.addWidget(self.poly_mask_min_input)
        
        self.poly_mask_max_label = QtWidgets.QLabel("Max λ:")
        self.poly_mask_max_input = QtWidgets.QLineEdit()
        self.poly_mask_max_input.setMaximumWidth(100)
        self.poly_mask_max_input.setPlaceholderText("e.g., 5550")
        poly_mask_range_layout.addWidget(self.poly_mask_max_label)
        poly_mask_range_layout.addWidget(self.poly_mask_max_input)
        poly_mask_range_layout.addStretch()
        poly_mask_layout.addLayout(poly_mask_range_layout)
        
        poly_mask_button_layout = QtWidgets.QHBoxLayout()
        self.btn_poly_mask_draw = QtWidgets.QPushButton("Draw")
        self.btn_poly_mask_draw.setMaximumWidth(75)
        self.btn_poly_mask_draw.clicked.connect(self.on_draw_polynomial_mask_regions)
        self.btn_poly_mask_add = QtWidgets.QPushButton("+")
        self.btn_poly_mask_add.setMaximumWidth(40)
        self.btn_poly_mask_add.clicked.connect(self.add_polynomial_guess_mask)
        self.btn_poly_mask_remove = QtWidgets.QPushButton("-")
        self.btn_poly_mask_remove.setMaximumWidth(40)
        self.btn_poly_mask_remove.clicked.connect(lambda: self.remove_component('polynomial_guess_mask'))
        poly_mask_button_layout.addWidget(self.btn_poly_mask_draw)
        poly_mask_button_layout.addWidget(self.btn_poly_mask_add)
        poly_mask_button_layout.addWidget(self.btn_poly_mask_remove)
        poly_mask_button_layout.addStretch()
        poly_mask_layout.addLayout(poly_mask_button_layout)
        
        left_layout.addLayout(poly_mask_layout)
        
        # Data Mask controls (for excluding regions from the fit)
        data_mask_layout = QtWidgets.QVBoxLayout()
        data_mask_header = QtWidgets.QHBoxLayout()
        self.data_mask_label = QtWidgets.QLabel("Data Mask:")
        data_mask_header.addWidget(self.data_mask_label)
        data_mask_header.addStretch()
        data_mask_layout.addLayout(data_mask_header)
        
        data_mask_description = QtWidgets.QLabel("Exclude wavelength ranges from the fit")
        data_mask_description.setStyleSheet("font-size: 9px; color: gray; font-style: italic;")
        data_mask_layout.addWidget(data_mask_description)
        
        data_mask_range_layout = QtWidgets.QHBoxLayout()
        self.data_mask_min_label = QtWidgets.QLabel("Min λ:")
        self.data_mask_min_input = QtWidgets.QLineEdit()
        self.data_mask_min_input.setMaximumWidth(100)
        self.data_mask_min_input.setPlaceholderText("e.g., 5500")
        data_mask_range_layout.addWidget(self.data_mask_min_label)
        data_mask_range_layout.addWidget(self.data_mask_min_input)
        
        self.data_mask_max_label = QtWidgets.QLabel("Max λ:")
        self.data_mask_max_input = QtWidgets.QLineEdit()
        self.data_mask_max_input.setMaximumWidth(100)
        self.data_mask_max_input.setPlaceholderText("e.g., 5550")
        data_mask_range_layout.addWidget(self.data_mask_max_label)
        data_mask_range_layout.addWidget(self.data_mask_max_input)
        data_mask_range_layout.addStretch()
        data_mask_layout.addLayout(data_mask_range_layout)
        
        data_mask_button_layout = QtWidgets.QHBoxLayout()
        self.btn_data_mask_draw = QtWidgets.QPushButton("Draw")
        self.btn_data_mask_draw.setMaximumWidth(75)
        self.btn_data_mask_draw.clicked.connect(self.on_draw_data_mask_regions)
        self.btn_data_mask_add = QtWidgets.QPushButton("+")
        self.btn_data_mask_add.setMaximumWidth(40)
        self.btn_data_mask_add.clicked.connect(self.add_data_mask)
        self.btn_data_mask_remove = QtWidgets.QPushButton("-")
        self.btn_data_mask_remove.setMaximumWidth(40)
        self.btn_data_mask_remove.clicked.connect(lambda: self.remove_component('data_mask'))
        data_mask_button_layout.addWidget(self.btn_data_mask_draw)
        data_mask_button_layout.addWidget(self.btn_data_mask_add)
        data_mask_button_layout.addWidget(self.btn_data_mask_remove)
        data_mask_button_layout.addStretch()
        data_mask_layout.addLayout(data_mask_button_layout)
        
        left_layout.addLayout(data_mask_layout)
        
        # Redshift fitting controls
        redshift_layout = QtWidgets.QVBoxLayout()
        redshift_header = QtWidgets.QHBoxLayout()
        self.redshift_label = QtWidgets.QLabel("Fit Redshift for Lines:")
        redshift_header.addWidget(self.redshift_label)
        redshift_header.addStretch()
        redshift_layout.addLayout(redshift_header)
        
        redshift_description = QtWidgets.QLabel("Fit a set of emission lines with a common redshift parameter")
        redshift_description.setStyleSheet("font-size: 9px; color: gray; font-style: italic;")
        redshift_layout.addWidget(redshift_description)
        
        redshift_button_layout = QtWidgets.QHBoxLayout()
        self.btn_fit_redshift = QtWidgets.QPushButton("Select Lines")
        self.btn_fit_redshift.setMaximumWidth(150)
        self.btn_fit_redshift.clicked.connect(self.on_fit_redshift_clicked)
        redshift_button_layout.addWidget(self.btn_fit_redshift)
        redshift_button_layout.addStretch()
        redshift_layout.addLayout(redshift_button_layout)
        
        left_layout.addLayout(redshift_layout)
        left_layout.addStretch()
        
        # Right side: Component list
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel("Components to Fit:"))
        self.component_list = ComponentListTable(self)
        self.component_list.setColumnCount(4)
        self.component_list.setHorizontalHeaderLabels(['Component', 'Symbol', 'Constraints', 'Initial'])
        self.component_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.component_list.setColumnWidth(0, 180)
        self.component_list.setColumnWidth(1, 50)
        self.component_list.setColumnWidth(2, 120)
        self.component_list.setColumnWidth(3, 140)
        self.component_list.itemClicked.connect(self.on_component_selected)
        self.component_list.request_set_guess.connect(self.on_request_set_guess)
        self.component_list.cellClicked.connect(self.on_table_cell_clicked)
        right_layout.addWidget(self.component_list)
        
        # Hint text
        hint_label = QtWidgets.QLabel("Click component for constraints | Click 'Initial' cell to set guess | Right-click for more options")
        hint_label.setStyleSheet("font-size: 9px; color: gray; font-style: italic; margin-top: 5px;")
        right_layout.addWidget(hint_label)
        
        # Tied Parameters section
        right_layout.addSpacing(15)
        right_layout.addWidget(QtWidgets.QLabel("Tied Parameters:"))
        self.tied_params_table = QtWidgets.QTableWidget(self)
        self.tied_params_table.setColumnCount(2)
        self.tied_params_table.setHorizontalHeaderLabels(['Parameter 1', 'Parameter 2 (Expression)'])
        self.tied_params_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tied_params_table.setMaximumHeight(120)
        self.tied_params_table.setColumnWidth(0, 140)
        self.tied_params_table.setColumnWidth(1, 210)
        self.tied_params_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tied_params_table.customContextMenuRequested.connect(self._show_tied_params_context_menu)
        right_layout.addWidget(self.tied_params_table)
        
        # Tied parameters hint
        tied_hint_label = QtWidgets.QLabel("Examples: g0_mu = g1_mu * (5008/4960)  |  g0_sigma = g1_sigma  |  Right-click to manage")
        tied_hint_label.setStyleSheet("font-size: 8px; color: gray; font-style: italic; margin-top: 2px;")
        right_layout.addWidget(tied_hint_label)
        
        # Buttons at bottom
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_add_tie = QtWidgets.QPushButton("+ Add Tie")
        self.btn_add_tie.setMaximumWidth(100)
        self.btn_add_tie.clicked.connect(self.on_add_tie)
        button_layout.addWidget(self.btn_add_tie)
        
        self.btn_equalize_sigma = QtWidgets.QPushButton("Equalize Sigma")
        self.btn_equalize_sigma.setMaximumWidth(130)
        self.btn_equalize_sigma.clicked.connect(self.on_equalize_sigma)
        button_layout.addWidget(self.btn_equalize_sigma)
        
        self.btn_equalize_sigma_redshift = QtWidgets.QPushButton("Equalize sigma (λᵢ/λ_ref)")
        self.btn_equalize_sigma_redshift.setMaximumWidth(150)
        self.btn_equalize_sigma_redshift.clicked.connect(self.on_equalize_sigma_redshift)
        self.btn_equalize_sigma_redshift.setEnabled(False)  # Disabled until redshift is added
        button_layout.addWidget(self.btn_equalize_sigma_redshift)
        
        button_layout.addStretch()
        self.btn_fit = QtWidgets.QPushButton("Calculate Fit")
        self.btn_fit.setMinimumHeight(40)
        self.btn_fit.clicked.connect(self.on_fit_requested)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setMinimumHeight(40)
        self.btn_cancel.clicked.connect(self.on_cancel_clicked)
        button_layout.addWidget(self.btn_fit)
        button_layout.addWidget(self.btn_cancel)
        right_layout.addLayout(button_layout)
        
        layout.addLayout(left_layout, 1)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)
    
    def add_component(self, comp_type):
        """Add a component of given type"""
        if comp_type == 'gaussian':
            label = f"Gaussian #{self.gaussian_count + 1}"
            component = {
                'type': 'gaussian',
                'id': len(self.components),
                'label': label,
                'guess': {'center': None, 'amp': None, 'stddev': None}
            }
            self.gaussian_count += 1
        elif comp_type == 'voigt':
            label = f"Voigt #{self.voigt_count + 1}"
            component = {
                'type': 'voigt',
                'id': len(self.components),
                'label': label,
                'guess': {'center': None, 'amp': None, 'sigma': None, 'gamma': None}
            }
            self.voigt_count += 1
        else:
            return
        
        self.components.append(component)
        self._add_component_to_table(label, component)
    
    def add_polynomial(self):
        """Add polynomial with specified order"""
        try:
            order = int(self.poly_order_input.text())
            if order < 0 or order > 10:
                order = 1
        except ValueError:
            order = 1
        
        label = f"Polynomial (order={order}) #{self.polynomial_count + 1}"
        component = {'type': 'polynomial', 'order': order, 'id': len(self.components), 'index': self.polynomial_count, 'label': label}
        self.components.append(component)
        self.polynomial_count += 1
        self._add_component_to_table(label, component)
    
    def add_polynomial_guess_mask(self):
        """Add polynomial guess mask with specified wavelength range"""
        try:
            min_lambda = float(self.poly_mask_min_input.text())
            max_lambda = float(self.poly_mask_max_input.text())
            
            if min_lambda >= max_lambda:
                QtWidgets.QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max wavelength")
                return
            
            label = f"Polynomial Guess Mask ({min_lambda:.2f}-{max_lambda:.2f} Å) #{self.polynomial_guess_mask_count + 1}"
            component = {'type': 'polynomial_guess_mask', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components), 'label': label}
            self.components.append(component)
            self.polynomial_guess_mask_count += 1
            self._add_component_to_table(label, component)
            
            # Clear input fields for next mask
            self.poly_mask_min_input.clear()
            self.poly_mask_max_input.clear()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid wavelength values")
    
    def add_data_mask(self):
        """Add data mask with specified wavelength range to exclude from fit"""
        try:
            min_lambda = float(self.data_mask_min_input.text())
            max_lambda = float(self.data_mask_max_input.text())
            
            if min_lambda >= max_lambda:
                QtWidgets.QMessageBox.warning(self, "Invalid Range", "Min wavelength must be less than Max wavelength")
                return
            
            label = f"Data Mask #{self.data_mask_count + 1}"
            component = {'type': 'data_mask', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components), 'label': label}
            self.components.append(component)
            self.data_mask_count += 1
            self._add_component_to_table(label, component)
            
            # Clear input fields for next mask
            self.data_mask_min_input.clear()
            self.data_mask_max_input.clear()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid wavelength values")
    
    def on_draw_data_mask_regions(self):
        """Handle 'Draw' button click for data masks - emit signal to spectrum plotter"""
        self.request_draw_data_mask_regions.emit()
    
    def add_drawn_data_mask_regions(self, regions):
        """Add data mask regions that were drawn by the user"""
        """regions is a list of (min_lambda, max_lambda) tuples"""
        for min_lambda, max_lambda in regions:
            if min_lambda >= max_lambda:
                continue
            
            label = f"Data Mask ({min_lambda:.2f}-{max_lambda:.2f} Å) #{self.data_mask_count + 1}"
            component = {'type': 'data_mask', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components), 'label': label}
            self.components.append(component)
            self.data_mask_count += 1
            self._add_component_to_table(label, component)
    
    def on_draw_polynomial_mask_regions(self):
        """Handle 'Draw' button click for polynomial masks - emit signal to spectrum plotter"""
        self.request_draw_polynomial_mask_regions.emit()
    
    def add_drawn_polynomial_mask_regions(self, regions):
        """Add polynomial mask regions that were drawn by the user"""
        """regions is a list of (min_lambda, max_lambda) tuples"""
        for min_lambda, max_lambda in regions:
            if min_lambda >= max_lambda:
                continue
            
            label = f"Polynomial Guess Mask #{self.polynomial_guess_mask_count + 1}"
            component = {'type': 'polynomial_guess_mask', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components), 'label': label}
            self.components.append(component)
            self.polynomial_guess_mask_count += 1
            self._add_component_to_table(label, component)
    
    def on_fit_redshift_clicked(self):
        """Handle 'Select Lines' button click for redshift fitting"""
        if self.resources_dir is None:
            QtWidgets.QMessageBox.warning(self, "Resources Error", "Resources directory not available")
            return
        
        self.redshift_line_selector = RedshiftLineSelector(self.resources_dir, self)
        self.redshift_line_selector.lines_selected.connect(self.add_redshift_lines)
        self.redshift_line_selector.exec()
    
    def add_redshift_lines(self, selected_lines):
        """Add selected emission lines tied to a new redshift parameter
        
        Args:
            selected_lines: List of {'line': Line, 'profile': 'gaussian'|'voigt'}
        """
        if not selected_lines:
            return
        
        # First, add the redshift parameter itself
        redshift_label = f"Redshift #{self.redshift_count + 1}"
        redshift_component = {
            'type': 'redshift',
            'id': len(self.components),
            'label': redshift_label,
            'redshift_number': self.redshift_count + 1,  # Track which redshift this is (1, 2, 3, etc.)
            'guess': {'z': None},
            'tied_lines': []  # List of indices of lines tied to this redshift
        }
        
        redshift_index = len(self.components)
        self.components.append(redshift_component)
        self._add_component_to_table(redshift_label, redshift_component)
        self.redshift_count += 1
        
        # Add each selected line as a component
        for line_data in selected_lines:
            line = line_data['line']
            profile = line_data['profile']
            
            # Create component based on profile type
            if profile == 'gaussian':
                # Create component label with symbol at front
                line_label = f"Gaussian #{self.gaussian_count + 1} {line.name} [z-tied]"
                component = {
                    'type': 'gaussian',
                    'id': len(self.components),
                    'label': line_label,
                    'guess': {'center': None, 'amp': None, 'stddev': None},
                    'rest_wavelength': line.wave,
                    'redshift_index': redshift_index,
                    'tied_to_redshift': True
                }
                self.gaussian_count += 1
            else:  # voigt
                # Create component label with symbol at front
                line_label = f"Voigt #{self.voigt_count + 1} {line.name} [z-tied]"
                component = {
                    'type': 'voigt',
                    'id': len(self.components),
                    'label': line_label,
                    'guess': {'center': None, 'amp': None, 'sigma': None, 'gamma': None},
                    'rest_wavelength': line.wave,
                    'redshift_index': redshift_index,
                    'tied_to_redshift': True
                }
                self.voigt_count += 1
            
            self.components.append(component)
            self._add_component_to_table(line_label, component)
            
            # Track this line in the redshift component
            redshift_component['tied_lines'].append(len(self.components) - 1)
            
            # Add a tied parameter: center_mu = (1+z)*rest_wavelength
            # We'll format it as: gi_mu = (1+z_j)*wavelength
            z_var = f"z{self.redshift_count}"  # e.g., z1, z2
            mu_var = f"g{self.gaussian_count - 1}_mu" if profile == 'gaussian' else f"v{self.voigt_count - 1}_mu"
            expression = f"({1 + 0})*{line.wave}"  # Placeholder, will be evaluated during fit
            
            # For now, we'll add this as a tied parameter in a format that can be processed
            # The actual evaluation will happen in the fitting code
            self._add_tied_parameter(mu_var, expression)
    
    def _add_tied_parameter(self, param1, param2_expr):
        """Add a tied parameter to the tied parameters table"""
        row = self.tied_params_table.rowCount()
        self.tied_params_table.insertRow(row)
        
        item1 = QtWidgets.QTableWidgetItem(param1)
        item1.setFlags(item1.flags() & ~Qt.ItemIsEditable)
        
        item2 = QtWidgets.QTableWidgetItem(param2_expr)
        item2.setFlags(item2.flags() & ~Qt.ItemIsEditable)
        
        self.tied_params_table.setItem(row, 0, item1)
        self.tied_params_table.setItem(row, 1, item2)
    
    def remove_component(self, comp_type):
        """Remove last component of given type"""
        try:
            for i in range(len(self.components) - 1, -1, -1):
                if self.components[i]['type'] == comp_type:
                    self.components.pop(i)
                    # Remove the row at the correct index (table rows match component list indices)
                    # Safety check: ensure row index is valid
                    if i >= 0 and i < self.component_list.rowCount():
                        self.component_list.removeRow(i)
                    
                    if comp_type == 'gaussian':
                        self.gaussian_count = max(0, self.gaussian_count - 1)
                    elif comp_type == 'voigt':
                        self.voigt_count = max(0, self.voigt_count - 1)
                    elif comp_type == 'polynomial':
                        self.polynomial_count = max(0, self.polynomial_count - 1)
                    elif comp_type == 'polynomial_guess_mask':
                        self.polynomial_guess_mask_count = max(0, self.polynomial_guess_mask_count - 1)
                    elif comp_type == 'data_mask':
                        self.data_mask_count = max(0, self.data_mask_count - 1)
                    # Emit signal to update the plot
                    self.components_changed.emit(self.components)
                    break
        except Exception as e:
            print(f"Error removing {comp_type} component: {e}")
            import traceback
            traceback.print_exc()
    
    def _add_component_to_table(self, label, component):
        """Add a component row to the table"""
        row = self.component_list.rowCount()
        self.component_list.insertRow(row)
        
        # Component name column
        name_item = QtWidgets.QTableWidgetItem(label)
        name_item.setData(Qt.UserRole, row)
        self.component_list.setItem(row, 0, name_item)
        
        # Symbol column (e.g., "g0", "g1", "v0", "p0", "z1", "z2", "z3")
        comp_type = component.get('type')
        if comp_type == 'gaussian':
            symbol = f"g{self.gaussian_count - 1}"
        elif comp_type == 'voigt':
            symbol = f"v{self.voigt_count - 1}"
        elif comp_type == 'polynomial':
            symbol = f"p{self.polynomial_count - 1}"
        elif comp_type == 'redshift':
            redshift_number = component.get('redshift_number', 1)
            symbol = f"z{redshift_number}"
        else:
            symbol = "?"
        
        symbol_item = QtWidgets.QTableWidgetItem(symbol)
        symbol_item.setFont(QtGui.QFont("Courier", 10, QtGui.QFont.Bold))
        symbol_item.setForeground(QtGui.QColor("#0066cc"))
        symbol_item.setTextAlignment(Qt.AlignCenter)
        self.component_list.setItem(row, 1, symbol_item)
        
        # Update button states based on component types
        self._update_equalize_buttons_state()
        
        # Constraints column (initially empty)
        constraints_item = QtWidgets.QTableWidgetItem("None")
        constraints_item.setForeground(QtGui.QColor("gray"))
        self.component_list.setItem(row, 2, constraints_item)
        
        # Initial parameters column
        initial_item = QtWidgets.QTableWidgetItem("(click to set)")
        initial_item.setForeground(QtGui.QColor("gray"))
        initial_item.setFont(QtGui.QFont("Courier", 9))
        self.component_list.setItem(row, 3, initial_item)
        self._update_initial_display(row, component)
    
    def _update_constraints_display(self, row, component):
        """Update the constraints column for a component"""
        constraints = component.get('constraints', {})
        if not constraints or all(not v for v in constraints.values()):
            display_text = "None"
            color = QtGui.QColor("gray")
        else:
            # Create a summary of constraints
            parts = []
            
            # Check for fixed parameters
            if constraints.get('amplitude_fixed'):
                parts.append("Amp: Fixed to Initial")
            if constraints.get('mean_fixed') or constraints.get('center_fixed'):
                parts.append("Mu: Fixed to Initial")
            if constraints.get('sigma_fixed'):
                parts.append("Sigma: Fixed to Initial")
            
            # Check for bounds
            if not constraints.get('amplitude_fixed') and (constraints.get('amplitude_bounds', ('', ''))[0] or constraints.get('amplitude_bounds', ('', ''))[1]):
                amp_bounds = constraints.get('amplitude_bounds', ('', ''))
                parts.append(f"Amp: [{amp_bounds[0]}, {amp_bounds[1]}]")
            if not (constraints.get('mean_fixed') or constraints.get('center_fixed')) and (constraints.get('mean_bounds', ('', ''))[0] or constraints.get('mean_bounds', ('', ''))[1]):
                mean_bounds = constraints.get('mean_bounds', ('', ''))
                parts.append(f"Mu: [{mean_bounds[0]}, {mean_bounds[1]}]")
            if not (constraints.get('mean_fixed') or constraints.get('center_fixed')) and (constraints.get('center_bounds', ('', ''))[0] or constraints.get('center_bounds', ('', ''))[1]):
                center_bounds = constraints.get('center_bounds', ('', ''))
                parts.append(f"Mu: [{center_bounds[0]}, {center_bounds[1]}]")
            if not constraints.get('sigma_fixed') and (constraints.get('sigma_bounds', ('', ''))[0] or constraints.get('sigma_bounds', ('', ''))[1]):
                sigma_bounds = constraints.get('sigma_bounds', ('', ''))
                parts.append(f"Sigma: [{sigma_bounds[0]}, {sigma_bounds[1]}]")
            
            if constraints.get('expression'):
                parts.append("Linked")
            
            display_text = ", ".join(parts) if parts else "None"
            color = QtGui.QColor("darkgreen") if parts else QtGui.QColor("gray")
        
        constraints_item = QtWidgets.QTableWidgetItem(display_text)
        constraints_item.setForeground(color)
        self.component_list.setItem(row, 2, constraints_item)
    
    def on_component_selected(self, item):
        """Handle component name cell click - show constraint editor only for column 0"""
        # Get the column of the clicked item
        col = self.component_list.column(item)
        
        # Only open constraints editor for Component name column (column 0)
        if col != 0:
            return
        
        # Find the component that was clicked
        row = self.component_list.row(item)
        if row < 0 or row >= len(self.components):
            return
        
        component = self.components[row]
        
        # Skip masks - they don't have constraints
        if component.get('type') in ['polynomial_guess_mask', 'data_mask']:
            return
        
        # Create a constraint editor dialog
        dialog = ConstraintEditorDialog(self)
        dialog.setWindowTitle(f"Edit Constraints - {self.component_list.item(row, 0).text()}")
        dialog.setGeometry(self.x() + self.width(), self.y(), 600, 600)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Define callback for Set button clicks
        def on_set_constraint(parameter):
            """Callback when user clicks Set button for a parameter"""
            # Store reference to editor so spectrum_plotter can update it
            self.current_constraint_editor = editor
            self.current_constraint_editor_dialog = dialog
            # Emit signal to spectrum_plotter to enter constraint bounds setting mode
            self.request_set_constraint_bounds.emit(parameter)
            # Don't close dialog - user will do that after dragging bounds on the plot
        
        # Create constraint editor widget and pass all components and callback
        editor = ConstraintEditor(component, dialog, all_components=self.components, 
                                 set_constraint_callback=on_set_constraint)
        layout.addWidget(editor)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("Apply")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Connect buttons
        def apply_constraints():
            constraints = editor.get_constraints()
            component['constraints'] = constraints
            self._update_constraints_display(row, component)
            dialog.accept()
        
        ok_btn.clicked.connect(apply_constraints)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Make dialog non-modal so Listfit and Spectrum Plotter windows remain active
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()
    
    def _format_guess_display(self, component):
        """Format the Initial column display for guess parameters or mask bounds"""
        comp_type = component.get('type', '').lower()
        
        # Handle mask types first (show bounds instead of guesses)
        if comp_type == 'data_mask' or comp_type == 'polynomial_guess_mask':
            min_lambda = component.get('min_lambda')
            max_lambda = component.get('max_lambda')
            if min_lambda is not None and max_lambda is not None:
                return f"λ: {min_lambda:.2f}–{max_lambda:.2f} Å"
            return "(no bounds set)"
        
        # Handle regular component guesses
        guess = component.get('guess', {})
        
        if not guess or all(v is None for v in guess.values()):
            return "(click to set)"
        
        try:
            if comp_type == 'gaussian':
                # For Gaussian: A, mu, sigma
                amp = guess.get('amp')
                center = guess.get('center')
                stddev = guess.get('stddev')
                
                if amp is None or center is None or stddev is None:
                    return "(incomplete)"
                
                return f"A={amp:.1f} mu={center:.1f} sigma={stddev:.1f}"
            
            elif comp_type == 'voigt':
                # For Voigt: A, mu, sigma, gamma
                amp = guess.get('amp')
                center = guess.get('center')
                sigma = guess.get('sigma')
                gamma = guess.get('gamma')
                
                if amp is None or center is None or sigma is None:
                    return "(incomplete)"
                
                if gamma is None:
                    return f"A={amp:.1f} mu={center:.1f} sigma={sigma:.1f}"
                else:
                    return f"A={amp:.1f} mu={center:.1f} σ={sigma:.2f} γ={gamma:.2f}"
            
            elif comp_type == 'polynomial':
                # For polynomial: display coefficients as c3, c2, c1, c0
                coeffs = guess.get('coefficients')
                order = guess.get('order')
                x_points = guess.get('x_points', [])
                
                if not coeffs:
                    return "(click to set)"
                
                order = order or len(coeffs) - 1
                
                # Format coefficients as c0, c1, c2, ... (highest order first)
                coeff_display = []
                for i, coeff in enumerate(coeffs):
                    power = len(coeffs) - 1 - i
                    # Determine parameter name: c0, c1, c2, c3, etc.
                    param_name = f"c{power}"
                    coeff_display.append(f"{param_name}={coeff:.3g}")
                
                return " ".join(coeff_display)
            
            elif comp_type == 'redshift':
                # For redshift: display z value
                z = guess.get('z')
                if z is None:
                    return "(click to set)"
                return f"z = {z:.6f}"
        except Exception as e:
            print(f"Error formatting guess display: {e}")
            return "(error)"
        
        return "(unknown)"
    
    def _update_initial_display(self, row, component):
        """Update the Initial column display for a component"""
        if row < 0 or row >= self.component_list.rowCount():
            return
        
        initial_text = self._format_guess_display(component)
        initial_item = self.component_list.item(row, 3)
        
        if initial_item is None:
            initial_item = QtWidgets.QTableWidgetItem(initial_text)
            self.component_list.setItem(row, 3, initial_item)
        else:
            initial_item.setText(initial_text)
        
        # Update color based on component type
        comp_type = component.get('type', '').lower()
        if comp_type in ['data_mask', 'polynomial_guess_mask']:
            # For masks, always show bounds in a neutral color (not editable in Initial column)
            initial_item.setForeground(QtGui.QColor("darkgreen"))
            initial_item.setFont(QtGui.QFont("Courier", 9))
        else:
            # For guesses, show in blue if complete, gray if incomplete
            guess = component.get('guess', {})
            if guess and any(v is not None for v in guess.values()):
                initial_item.setForeground(QtGui.QColor("darkblue"))
                initial_item.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
            else:
                initial_item.setForeground(QtGui.QColor("gray"))
                initial_item.setFont(QtGui.QFont("Courier", 9))
    
    def on_table_cell_clicked(self, row, col):
        """Handle cell clicks - trigger guess drawing on Initial column clicks, or open redshift dialog for redshifts"""
        if col != 3:  # Initial column is column 3 (0-indexed)
            return
        
        if row < 0 or row >= len(self.components):
            return
        
        component = self.components[row]
        comp_type = component.get('type')
        
        # For redshift components, open a dialog to set the redshift guess value
        if comp_type == 'redshift':
            self._open_redshift_guess_dialog(row, component)
        else:
            # For other components, trigger guess drawing mode
            self.component_list.request_set_guess.emit(row, component)
    
    def on_request_set_guess(self, row, component):
        """Handle request to set guess for a component - emit signal to spectrum plotter"""
        self.request_set_guess.emit(row, component)
    
    def _open_redshift_guess_dialog(self, row, component):
        """Open a dialog to set the redshift guess value"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Set Redshift Guess - {component.get('label', 'Redshift')}")
        dialog.setMinimumWidth(350)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Label
        label = QtWidgets.QLabel(f"Enter an initial guess for the redshift (z):")
        layout.addWidget(label)
        
        # Input field
        input_field = QtWidgets.QLineEdit()
        current_z = component.get('guess', {}).get('z')
        if current_z is not None:
            input_field.setText(str(current_z))
        else:
            input_field.setText("0.0")
        input_field.setPlaceholderText("e.g., 0.5 or 1.234")
        layout.addWidget(input_field)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        # Connect buttons
        def on_ok():
            try:
                z_value = float(input_field.text().strip())
                component['guess']['z'] = z_value
                self._update_initial_display(row, component)
                dialog.accept()
            except ValueError:
                QtWidgets.QMessageBox.warning(dialog, "Invalid Input", "Please enter a valid number for redshift.")
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        # Focus on input field and select all
        input_field.setFocus()
        input_field.selectAll()
        
        dialog.exec_()
    
    def on_add_tie(self):
        """Handle 'Add Tie' button - open dialog to add a new parameter tie"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Add Parameter Tie")
        dialog.setMinimumWidth(500)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Description label
        desc_label = QtWidgets.QLabel("Link two parameters together. Example: g0_mu = g1_mu * (5008/4960)")
        desc_label.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(desc_label)
        
        # Parameter 1 selection
        param1_layout = QtWidgets.QHBoxLayout()
        param1_layout.addWidget(QtWidgets.QLabel("Parameter 1:"))
        param1_combo = QtWidgets.QComboBox()
        param1_items = self._get_all_parameter_symbols()
        param1_combo.addItems(param1_items)
        param1_layout.addWidget(param1_combo)
        layout.addLayout(param1_layout)
        
        # Equals sign
        layout.addWidget(QtWidgets.QLabel("="))
        
        # Parameter 2 / Expression
        param2_layout = QtWidgets.QHBoxLayout()
        param2_layout.addWidget(QtWidgets.QLabel("Expression:"))
        param2_input = QtWidgets.QLineEdit()
        param2_input.setPlaceholderText("e.g., g1_mu * (5008/4960)  or  g1_sigma  or  5.5 * v0_gamma")
        param2_input.setMinimumWidth(300)
        param2_layout.addWidget(param2_input)
        layout.addLayout(param2_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("Add Tie")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        def add_tie_to_table():
            param1 = param1_combo.currentText()
            param2_expr = param2_input.text().strip()
            
            if not param1 or not param2_expr:
                QtWidgets.QMessageBox.warning(dialog, "Invalid Input", "Please fill in both fields")
                return
            
            self._add_tied_param_row(param1, param2_expr)
            dialog.accept()
        
        ok_btn.clicked.connect(add_tie_to_table)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    def _update_equalize_buttons_state(self):
        """Enable/disable equalize buttons based on component types"""
        # Check if there are any redshifts in the components
        has_redshifts = any(c.get('type') == 'redshift' for c in self.components)
        self.btn_equalize_sigma_redshift.setEnabled(has_redshifts)
    
    def on_equalize_sigma(self):
        """Set sigma to be the same for all profiles (Gaussian and Voigt)"""
        # Find all Gaussian and Voigt components
        profiles = [c for c in self.components if c.get('type') in ('gaussian', 'voigt')]
        
        if not profiles:
            QtWidgets.QMessageBox.information(self, "No Profiles", "Add at least one Gaussian or Voigt profile first")
            return
        
        if len(profiles) == 1:
            QtWidgets.QMessageBox.information(self, "Only One Profile", "Add multiple profiles to equalize sigma")
            return
        
        # Get the first profile and compute its symbol based on position
        first_comp = profiles[0]
        first_type = first_comp.get('type')
        
        # Compute symbol by counting how many profiles of this type come before it
        first_index_in_components = self.components.index(first_comp)
        gaussian_count_before = sum(1 for c in self.components[:first_index_in_components] if c.get('type') == 'gaussian')
        voigt_count_before = sum(1 for c in self.components[:first_index_in_components] if c.get('type') == 'voigt')
        
        if first_type == 'gaussian':
            first_symbol = f"g{gaussian_count_before}"
        elif first_type == 'voigt':
            first_symbol = f"v{voigt_count_before}"
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "First profile has unknown type")
            return
        
        first_sigma_param = f"{first_symbol}_sigma"
        
        # Tie all other profiles' sigmas to the first one
        ties_added = 0
        for profile in profiles[1:]:
            comp_type = profile.get('type')
            
            # Compute symbol by counting profiles of this type before it
            profile_index_in_components = self.components.index(profile)
            gaussian_count_before = sum(1 for c in self.components[:profile_index_in_components] if c.get('type') == 'gaussian')
            voigt_count_before = sum(1 for c in self.components[:profile_index_in_components] if c.get('type') == 'voigt')
            
            if comp_type == 'gaussian':
                symbol = f"g{gaussian_count_before}"
            elif comp_type == 'voigt':
                symbol = f"v{voigt_count_before}"
            else:
                continue
            
            sigma_param = f"{symbol}_sigma"
            
            # Check if this tie already exists
            tie_exists = False
            for row in range(self.tied_params_table.rowCount()):
                param1_item = self.tied_params_table.item(row, 0)
                param2_item = self.tied_params_table.item(row, 1)
                if param1_item and param2_item:
                    if param1_item.text() == sigma_param and param2_item.text() == first_sigma_param:
                        tie_exists = True
                        break
            
            if not tie_exists:
                self._add_tied_param_row(sigma_param, first_sigma_param)
                ties_added += 1
        
        if ties_added > 0:
            QtWidgets.QMessageBox.information(self, "Sigma Equalized", 
                f"Added {ties_added} tie(s) to equalize all profile sigmas to {first_sigma_param}")
        else:
            QtWidgets.QMessageBox.information(self, "No Changes", 
                f"All profile sigmas are already tied to {first_sigma_param}")
    
    def on_equalize_sigma_redshift(self):
        """Equalize sigma scaled by wavelength ratio (lambda_B/lambda_A) for a selected redshift"""
        # Find all redshifts
        redshifts = [c for c in self.components if c.get('type') == 'redshift']
        
        if not redshifts:
            QtWidgets.QMessageBox.information(self, "No Redshifts", "Add at least one redshift first")
            return
        
        # Find all Gaussian and Voigt components
        profiles = [c for c in self.components if c.get('type') in ('gaussian', 'voigt')]
        
        if not profiles:
            QtWidgets.QMessageBox.information(self, "No Profiles", "Add at least one Gaussian or Voigt profile first")
            return
        
        if len(profiles) == 1:
            QtWidgets.QMessageBox.information(self, "Only One Profile", "Add multiple profiles to equalize sigma with wavelength scaling")
            return
        
        # Show dialog to select redshift
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Select Redshift for Wavelength-Scaled Sigma Equalization")
        dialog.setMinimumWidth(400)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Instructions
        instr_label = QtWidgets.QLabel("Select which redshift's line list to scale sigmas for:\n\n" +
                                       "Profiles will have their widths tied as: σᵢ = σ_ref × (λᵢ/λ_ref)\n" +
                                       "This accounts for wavelength-dependent width changes.")
        layout.addWidget(instr_label)
        
        # Redshift selection
        redshift_layout = QtWidgets.QHBoxLayout()
        redshift_layout.addWidget(QtWidgets.QLabel("Redshift:")); 
        redshift_combo = QtWidgets.QComboBox()
        
        redshift_options = []
        for redshift_comp in redshifts:
            redshift_number = redshift_comp.get('redshift_number', 1)
            redshift_label = redshift_comp.get('label', f'Redshift #{redshift_number}')
            z_symbol = f"z{redshift_number}"
            redshift_options.append((z_symbol, redshift_label))
        
        for z_symbol, redshift_label in redshift_options:
            redshift_combo.addItem(redshift_label, z_symbol)
        
        redshift_layout.addWidget(redshift_combo)
        layout.addLayout(redshift_layout)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)
        
        dialog.setLayout(layout)
        
        def on_ok():
            selected_z_symbol = redshift_combo.currentData()
            dialog.accept()
            self._apply_sigma_redshift_equalization(selected_z_symbol)
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    def _apply_sigma_redshift_equalization(self, z_symbol):
        """Apply sigma*lambda_B/lambda_A equalization for a specific redshift (wavelength-dependent width scaling)"""
        # Find all Gaussian and Voigt components
        profiles = [c for c in self.components if c.get('type') in ('gaussian', 'voigt')]
        
        if len(profiles) < 2:
            QtWidgets.QMessageBox.information(self, "Not Enough Profiles", "Need at least 2 profiles")
            return
        
        # Get the first profile and its wavelength (this is lambda_A, the reference)
        first_comp = profiles[0]
        first_type = first_comp.get('type')
        first_wavelength = first_comp.get('rest_wavelength')
        
        if first_wavelength is None:
            QtWidgets.QMessageBox.warning(self, "Error", "First profile has no rest wavelength defined")
            return
        
        # Compute symbol by counting how many profiles of this type come before it
        first_index_in_components = self.components.index(first_comp)
        gaussian_count_before = sum(1 for c in self.components[:first_index_in_components] if c.get('type') == 'gaussian')
        voigt_count_before = sum(1 for c in self.components[:first_index_in_components] if c.get('type') == 'voigt')
        
        if first_type == 'gaussian':
            first_symbol = f"g{gaussian_count_before}"
        elif first_type == 'voigt':
            first_symbol = f"v{voigt_count_before}"
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "First profile has unknown type")
            return
        
        first_sigma_param = f"{first_symbol}_sigma"
        
        # Tie all other profiles' sigmas to the first one scaled by wavelength ratio (lambda_B / lambda_A)
        ties_added = 0
        for profile in profiles[1:]:
            comp_type = profile.get('type')
            profile_wavelength = profile.get('rest_wavelength')
            
            if profile_wavelength is None:
                # Skip profiles without wavelength info
                continue
            
            # Compute symbol by counting profiles of this type before it
            profile_index_in_components = self.components.index(profile)
            gaussian_count_before = sum(1 for c in self.components[:profile_index_in_components] if c.get('type') == 'gaussian')
            voigt_count_before = sum(1 for c in self.components[:profile_index_in_components] if c.get('type') == 'voigt')
            
            if comp_type == 'gaussian':
                symbol = f"g{gaussian_count_before}"
            elif comp_type == 'voigt':
                symbol = f"v{voigt_count_before}"
            else:
                continue
            
            sigma_param = f"{symbol}_sigma"
            
            # Create the expression: sigma_i = sigma_0 * (lambda_i / lambda_0)
            # This scales the line width proportionally to the rest wavelength
            expression = f"{first_sigma_param} * ({profile_wavelength} / {first_wavelength})"
            
            # Check if this tie already exists
            tie_exists = False
            for row in range(self.tied_params_table.rowCount()):
                param1_item = self.tied_params_table.item(row, 0)
                param2_item = self.tied_params_table.item(row, 1)
                if param1_item and param2_item:
                    if param1_item.text() == sigma_param and param2_item.text() == expression:
                        tie_exists = True
                        break
            
            if not tie_exists:
                self._add_tied_param_row(sigma_param, expression)
                ties_added += 1
        
        if ties_added > 0:
            QtWidgets.QMessageBox.information(self, "Sigma Equalized", 
                f"Added {ties_added} tie(s) to equalize all profile sigmas scaled by wavelength ratio (λ_i/λ_ref)")
        else:
            QtWidgets.QMessageBox.information(self, "No Changes", 
                f"All profile sigmas are already tied by wavelength ratio to {first_sigma_param}")
    
    def _get_all_parameter_symbols(self):
        """Get all available parameter symbols from components"""
        symbols = []
        for component in self.components:
            comp_type = component.get('type', '').lower()
            
            if comp_type == 'gaussian':
                symbol = component.get('symbol', '')
                if not symbol:
                    # Derive symbol from label
                    label = component.get('label', '')
                    for i in range(10):
                        if f"#{i}" in label:
                            symbol = f"g{i-1}"
                            break
                
                symbols.extend([f"{symbol}_amp", f"{symbol}_mu", f"{symbol}_sigma"])
            
            elif comp_type == 'voigt':
                symbol = component.get('symbol', '')
                if not symbol:
                    label = component.get('label', '')
                    for i in range(10):
                        if f"#{i}" in label:
                            symbol = f"v{i-1}"
                            break
                
                symbols.extend([f"{symbol}_amp", f"{symbol}_mu", f"{symbol}_sigma", f"{symbol}_gamma"])
        
        return sorted(symbols)
    
    def _add_tied_param_row(self, param1, param2_expr):
        """Add a row to the tied parameters table"""
        row = self.tied_params_table.rowCount()
        self.tied_params_table.insertRow(row)
        
        # Parameter 1 column
        param1_item = QtWidgets.QTableWidgetItem(param1)
        param1_item.setFont(QtGui.QFont("Courier", 9, QtGui.QFont.Bold))
        param1_item.setForeground(QtGui.QColor("#0066cc"))
        self.tied_params_table.setItem(row, 0, param1_item)
        
        # Parameter 2 (Expression) column
        param2_item = QtWidgets.QTableWidgetItem(param2_expr)
        param2_item.setFont(QtGui.QFont("Courier", 9))
        self.tied_params_table.setItem(row, 1, param2_item)
        
        print(f"[Tie] Added: {param1} = {param2_expr}")
    
    def _show_tied_params_context_menu(self, position):
        """Show context menu for tied parameters table"""
        item = self.tied_params_table.itemAt(position)
        if item is None:
            return
        
        row = self.tied_params_table.row(item)
        
        menu = QtWidgets.QMenu()
        action_remove = menu.addAction("Remove Tie")
        action_remove.triggered.connect(lambda: self._remove_tied_param(row))
        
        menu.exec_(self.tied_params_table.mapToGlobal(position))
    
    def _remove_tied_param(self, row):
        """Remove a tied parameter from the table"""
        if row < 0 or row >= self.tied_params_table.rowCount():
            return
        
        param1_item = self.tied_params_table.item(row, 0)
        param2_item = self.tied_params_table.item(row, 1)
        
        if param1_item and param2_item:
            print(f"[Tie] Removed: {param1_item.text()} = {param2_item.text()}")
        
        self.tied_params_table.removeRow(row)
    
    def on_fit_requested(self):
        """Emit signal to perform fitting"""
        if not self.components:
            QtWidgets.QMessageBox.warning(self, "No Components", "Please add at least one component to fit")
            return
        
        # Collect tied parameters from the table
        tied_params = []
        for row in range(self.tied_params_table.rowCount()):
            param1_item = self.tied_params_table.item(row, 0)
            param2_item = self.tied_params_table.item(row, 1)
            if param1_item and param2_item:
                tie = {
                    'param1': param1_item.text(),
                    'param2': param2_item.text()
                }
                tied_params.append(tie)
        
        # Store tied parameters in components dict
        fit_data = {
            'components': self.components,
            'tied_parameters': tied_params
        }
        
        # Print summary
        if tied_params:
            print(f"\n[Fit] {len(tied_params)} parameter tie(s) will be applied during fitting")
            for tie in tied_params:
                print(f"  • {tie['param1']} = {tie['param2']}")
        
        self.fit_requested.emit(fit_data)
        self.close()
    
    def keyPressEvent(self, event):
        """Handle key press events - forward Q to parent window"""
        from PyQt5.QtCore import Qt
        if event.key() == Qt.Key_Q:
            # Close this window and let Q propagate to parent
            self.close()
            return
        super().keyPressEvent(event)    
    def on_cancel_clicked(self):
        """Handle cancel button - clear bounds, cleanup guesses, and close"""
        self.bounds_cleared.emit()
        self.cleanup_guesses.emit()  # Signal to remove all drawn guesses
        self.close()