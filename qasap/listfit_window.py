"""
ListfitWindow - Multi-component spectrum fitting dialog
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIntValidator, QDoubleValidator, QIcon, QColor
import numpy as np
from scipy.optimize import curve_fit
from qasap.ui_utils import get_qasap_icon


class ConstraintEditorDialog(QtWidgets.QDialog):
    """Dialog for editing constraints with Q key handling"""
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.reject()
            return
        super().keyPressEvent(event)


class ConstraintEditor(QtWidgets.QWidget):
    """Widget to edit constraints for a component"""
    
    def __init__(self, component, parent=None, all_components=None):
        super().__init__(parent)
        self.component = component
        self.all_components = all_components or []
        self.linked_constraints = []  # List of {parameter: str, expression: str}
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
            "  • g0_mean = g1_mean  (set equal)\n"
            "  • g0_stddev = 2 * g1_stddev  (scale)\n"
            "  • g0_center = g1_center + 0.5  (offset)"
        )
        examples_label.setStyleSheet("font-size: 7px; color: #666666; font-style: italic;")
        expr_layout.addWidget(examples_label)
        
        expr_label = QtWidgets.QLabel("Enter constraint expression:")
        expr_label.setStyleSheet("font-size: 8px; color: gray;")
        expr_layout.addWidget(expr_label)
        
        self.constraint_expr_input = QtWidgets.QLineEdit()
        self.constraint_expr_input.setPlaceholderText("Example: g0_mean = g1_mean  or  g0_stddev = 2 * g1_stddev")
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
            "  • g0_mean = g1_mean  (set equal)\n"
            "  • g0_stddev = 2 * g1_stddev  (scale)\n"
            "  • g0_center = g1_center + 0.5  (offset)\n"
            "  • v0_amp = g0_amp / 2  (combine with arithmetic)\n"
            "\n"
            "For parameter bounds (min/max), use fields above.\n"
            "Inequality operators (>, <, >=, <=) use bounds instead."
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
            QtWidgets.QMessageBox.warning(self, "Empty Expression", "Enter a constraint expression first.\n\nExample: g0_stddev = g1_stddev")
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
        
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Confirm Constraint", 
            confirmation,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel
        )
        
        if reply != QtWidgets.QMessageBox.Yes:
            return
        
        # Add to linked constraints list
        self.linked_constraints.append({'parameter': parameter, 'expression': constraint_expr})
        
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
        
        # Amplitude constraints
        amp_layout = QtWidgets.QHBoxLayout()
        amp_layout.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.amp_min = QtWidgets.QLineEdit()
        self.amp_min.setPlaceholderText("min")
        self.amp_min.setMaximumWidth(80)
        self.amp_min.setValidator(QDoubleValidator())
        amp_layout.addWidget(QtWidgets.QLabel("min:"))
        amp_layout.addWidget(self.amp_min)
        self.amp_max = QtWidgets.QLineEdit()
        self.amp_max.setPlaceholderText("max")
        self.amp_max.setMaximumWidth(80)
        self.amp_max.setValidator(QDoubleValidator())
        amp_layout.addWidget(QtWidgets.QLabel("max:"))
        amp_layout.addWidget(self.amp_max)
        self.amp_fixed = QtWidgets.QCheckBox("Fixed")
        amp_layout.addWidget(self.amp_fixed)
        amp_layout.addStretch()
        layout.addLayout(amp_layout)
        
        # Mean constraints
        mean_layout = QtWidgets.QHBoxLayout()
        mean_layout.addWidget(QtWidgets.QLabel("Center (λ):"))
        self.mean_min = QtWidgets.QLineEdit()
        self.mean_min.setPlaceholderText("min")
        self.mean_min.setMaximumWidth(80)
        self.mean_min.setValidator(QDoubleValidator())
        mean_layout.addWidget(QtWidgets.QLabel("min:"))
        mean_layout.addWidget(self.mean_min)
        self.mean_max = QtWidgets.QLineEdit()
        self.mean_max.setPlaceholderText("max")
        self.mean_max.setMaximumWidth(80)
        self.mean_max.setValidator(QDoubleValidator())
        mean_layout.addWidget(QtWidgets.QLabel("max:"))
        mean_layout.addWidget(self.mean_max)
        self.mean_fixed = QtWidgets.QCheckBox("Fixed")
        mean_layout.addWidget(self.mean_fixed)
        mean_layout.addStretch()
        layout.addLayout(mean_layout)
        
        # Stddev constraints
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Width (σ):"))
        self.sigma_min = QtWidgets.QLineEdit()
        self.sigma_min.setPlaceholderText("min")
        self.sigma_min.setMaximumWidth(80)
        self.sigma_min.setValidator(QDoubleValidator())
        sigma_layout.addWidget(QtWidgets.QLabel("min:"))
        sigma_layout.addWidget(self.sigma_min)
        self.sigma_max = QtWidgets.QLineEdit()
        self.sigma_max.setPlaceholderText("max")
        self.sigma_max.setMaximumWidth(80)
        self.sigma_max.setValidator(QDoubleValidator())
        sigma_layout.addWidget(QtWidgets.QLabel("max:"))
        sigma_layout.addWidget(self.sigma_max)
        self.sigma_fixed = QtWidgets.QCheckBox("Fixed")
        sigma_layout.addWidget(self.sigma_fixed)
        sigma_layout.addStretch()
        layout.addLayout(sigma_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_voigt_constraints(self):
        """Create constraint panel for Voigt"""
        group = QtWidgets.QGroupBox("Voigt Constraints")
        layout = QtWidgets.QVBoxLayout()
        
        # Amplitude
        amp_layout = QtWidgets.QHBoxLayout()
        amp_layout.addWidget(QtWidgets.QLabel("Amplitude:"))
        self.amp_min = QtWidgets.QLineEdit()
        self.amp_min.setMaximumWidth(80)
        self.amp_min.setValidator(QDoubleValidator())
        amp_layout.addWidget(QtWidgets.QLabel("min:"))
        amp_layout.addWidget(self.amp_min)
        self.amp_max = QtWidgets.QLineEdit()
        self.amp_max.setMaximumWidth(80)
        self.amp_max.setValidator(QDoubleValidator())
        amp_layout.addWidget(QtWidgets.QLabel("max:"))
        amp_layout.addWidget(self.amp_max)
        self.amp_fixed = QtWidgets.QCheckBox("Fixed")
        amp_layout.addWidget(self.amp_fixed)
        amp_layout.addStretch()
        layout.addLayout(amp_layout)
        
        # Center
        center_layout = QtWidgets.QHBoxLayout()
        center_layout.addWidget(QtWidgets.QLabel("Center (λ):"))
        self.center_min = QtWidgets.QLineEdit()
        self.center_min.setMaximumWidth(80)
        self.center_min.setValidator(QDoubleValidator())
        center_layout.addWidget(QtWidgets.QLabel("min:"))
        center_layout.addWidget(self.center_min)
        self.center_max = QtWidgets.QLineEdit()
        self.center_max.setMaximumWidth(80)
        self.center_max.setValidator(QDoubleValidator())
        center_layout.addWidget(QtWidgets.QLabel("max:"))
        center_layout.addWidget(self.center_max)
        self.center_fixed = QtWidgets.QCheckBox("Fixed")
        center_layout.addWidget(self.center_fixed)
        center_layout.addStretch()
        layout.addLayout(center_layout)
        
        # Sigma
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Width (σ):"))
        self.sigma_min = QtWidgets.QLineEdit()
        self.sigma_min.setMaximumWidth(80)
        self.sigma_min.setValidator(QDoubleValidator())
        sigma_layout.addWidget(QtWidgets.QLabel("min:"))
        sigma_layout.addWidget(self.sigma_min)
        self.sigma_max = QtWidgets.QLineEdit()
        self.sigma_max.setMaximumWidth(80)
        self.sigma_max.setValidator(QDoubleValidator())
        sigma_layout.addWidget(QtWidgets.QLabel("max:"))
        sigma_layout.addWidget(self.sigma_max)
        self.sigma_fixed = QtWidgets.QCheckBox("Fixed")
        sigma_layout.addWidget(self.sigma_fixed)
        sigma_layout.addStretch()
        layout.addLayout(sigma_layout)
        
        # Gamma
        gamma_layout = QtWidgets.QHBoxLayout()
        gamma_layout.addWidget(QtWidgets.QLabel("Gamma (γ):"))
        self.gamma_min = QtWidgets.QLineEdit()
        self.gamma_min.setMaximumWidth(80)
        self.gamma_min.setValidator(QDoubleValidator())
        gamma_layout.addWidget(QtWidgets.QLabel("min:"))
        gamma_layout.addWidget(self.gamma_min)
        self.gamma_max = QtWidgets.QLineEdit()
        self.gamma_max.setMaximumWidth(80)
        self.gamma_max.setValidator(QDoubleValidator())
        gamma_layout.addWidget(QtWidgets.QLabel("max:"))
        gamma_layout.addWidget(self.gamma_max)
        self.gamma_fixed = QtWidgets.QCheckBox("Fixed")
        gamma_layout.addWidget(self.gamma_fixed)
        gamma_layout.addStretch()
        layout.addLayout(gamma_layout)
        
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
        
        if comp_type == 'gaussian':
            constraints['amplitude_bounds'] = (self.amp_min.text(), self.amp_max.text())
            constraints['mean_bounds'] = (self.mean_min.text(), self.mean_max.text())
            constraints['sigma_bounds'] = (self.sigma_min.text(), self.sigma_max.text())
            constraints['amplitude_fixed'] = self.amp_fixed.isChecked()
            constraints['amplitude_fixed_value'] = self._get_fixed_value(self.amp_min, self.amp_max) if self.amp_fixed.isChecked() else None
            constraints['mean_fixed'] = self.mean_fixed.isChecked()
            constraints['mean_fixed_value'] = self._get_fixed_value(self.mean_min, self.mean_max) if self.mean_fixed.isChecked() else None
            constraints['sigma_fixed'] = self.sigma_fixed.isChecked()
            constraints['sigma_fixed_value'] = self._get_fixed_value(self.sigma_min, self.sigma_max) if self.sigma_fixed.isChecked() else None
            constraints['linked_constraints'] = self.linked_constraints
        
        elif comp_type == 'voigt':
            constraints['amplitude_bounds'] = (self.amp_min.text(), self.amp_max.text())
            constraints['center_bounds'] = (self.center_min.text(), self.center_max.text())
            constraints['sigma_bounds'] = (self.sigma_min.text(), self.sigma_max.text())
            constraints['gamma_bounds'] = (self.gamma_min.text(), self.gamma_max.text())
            constraints['amplitude_fixed'] = self.amp_fixed.isChecked()
            constraints['amplitude_fixed_value'] = self._get_fixed_value(self.amp_min, self.amp_max) if self.amp_fixed.isChecked() else None
            constraints['center_fixed'] = self.center_fixed.isChecked()
            constraints['center_fixed_value'] = self._get_fixed_value(self.center_min, self.center_max) if self.center_fixed.isChecked() else None
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


class ListfitWindow(QtWidgets.QWidget):
    """Dialog for defining and fitting multiple spectrum components"""
    
    fit_requested = pyqtSignal(list)  # Emits list of components to fit
    bounds_cleared = pyqtSignal()  # Emits when window is cancelled to clear bounds
    components_changed = pyqtSignal(list)  # Emits when components are added/removed to update plot
    
    def __init__(self, bounds):
        super().__init__()
        self.bounds = bounds
        self.components = []  # List of {'type': 'gaussian'|'voigt'|'polynomial'|'polynomial_guess_mask'|'data_mask', ...}
        self.gaussian_count = 0
        self.voigt_count = 0
        self.polynomial_count = 0
        self.polynomial_guess_mask_count = 0
        self.data_mask_count = 0
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QASAP - List Fit")
        # Load and set window icon
        self.setWindowIcon(get_qasap_icon())
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
        self.btn_poly_mask_add = QtWidgets.QPushButton("+")
        self.btn_poly_mask_add.setMaximumWidth(40)
        self.btn_poly_mask_add.clicked.connect(self.add_polynomial_guess_mask)
        self.btn_poly_mask_remove = QtWidgets.QPushButton("-")
        self.btn_poly_mask_remove.setMaximumWidth(40)
        self.btn_poly_mask_remove.clicked.connect(lambda: self.remove_component('polynomial_guess_mask'))
        poly_mask_button_layout.addWidget(QtWidgets.QLabel(""))
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
        self.btn_data_mask_add = QtWidgets.QPushButton("+")
        self.btn_data_mask_add.setMaximumWidth(40)
        self.btn_data_mask_add.clicked.connect(self.add_data_mask)
        self.btn_data_mask_remove = QtWidgets.QPushButton("-")
        self.btn_data_mask_remove.setMaximumWidth(40)
        self.btn_data_mask_remove.clicked.connect(lambda: self.remove_component('data_mask'))
        data_mask_button_layout.addWidget(QtWidgets.QLabel(""))
        data_mask_button_layout.addWidget(self.btn_data_mask_add)
        data_mask_button_layout.addWidget(self.btn_data_mask_remove)
        data_mask_button_layout.addStretch()
        data_mask_layout.addLayout(data_mask_button_layout)
        
        left_layout.addLayout(data_mask_layout)
        left_layout.addStretch()
        
        # Right side: Component list
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(QtWidgets.QLabel("Components to Fit:"))
        self.component_list = QtWidgets.QTableWidget()
        self.component_list.setColumnCount(2)
        self.component_list.setHorizontalHeaderLabels(['Component', 'Constraints'])
        self.component_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.component_list.setColumnWidth(0, 250)
        self.component_list.setColumnWidth(1, 200)
        self.component_list.itemClicked.connect(self.on_component_selected)
        right_layout.addWidget(self.component_list)
        
        # Hint text
        hint_label = QtWidgets.QLabel("Click on a component to add constraints")
        hint_label.setStyleSheet("font-size: 9px; color: gray; font-style: italic; margin-top: 5px;")
        right_layout.addWidget(hint_label)
        
        # Buttons at bottom
        button_layout = QtWidgets.QHBoxLayout()
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
            component = {'type': 'gaussian', 'id': len(self.components), 'label': label}
            self.gaussian_count += 1
        elif comp_type == 'voigt':
            label = f"Voigt #{self.voigt_count + 1}"
            component = {'type': 'voigt', 'id': len(self.components), 'label': label}
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
        component = {'type': 'polynomial', 'order': order, 'id': len(self.components), 'label': label}
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
            
            label = f"Data Mask ({min_lambda:.2f}-{max_lambda:.2f} Å) #{self.data_mask_count + 1}"
            component = {'type': 'data_mask', 'min_lambda': min_lambda, 'max_lambda': max_lambda, 'id': len(self.components), 'label': label}
            self.components.append(component)
            self.data_mask_count += 1
            self._add_component_to_table(label, component)
            
            # Clear input fields for next mask
            self.data_mask_min_input.clear()
            self.data_mask_max_input.clear()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid wavelength values")
    
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
        
        # Constraints column (initially empty)
        constraints_item = QtWidgets.QTableWidgetItem("None")
        constraints_item.setForeground(QtGui.QColor("gray"))
        self.component_list.setItem(row, 1, constraints_item)
    
    def _update_constraints_display(self, row, component):
        """Update the constraints column for a component"""
        constraints = component.get('constraints', {})
        if not constraints or all(not v for v in constraints.values()):
            display_text = "None"
            color = QtGui.QColor("gray")
        else:
            # Create a summary of constraints
            parts = []
            if constraints.get('amplitude_fixed'):
                parts.append("Amp fixed")
            if constraints.get('mean_fixed') or constraints.get('center_fixed'):
                parts.append("Center fixed")
            if constraints.get('sigma_fixed'):
                parts.append("Width fixed")
            if constraints.get('expression'):
                parts.append("Linked")
            if constraints.get('amplitude_bounds')[0] or constraints.get('amplitude_bounds')[1]:
                parts.append("Amp bounds")
            if constraints.get('mean_bounds', ('', ''))[0] or constraints.get('mean_bounds', ('', ''))[1]:
                parts.append("Center bounds")
            
            display_text = ", ".join(parts) if parts else "None"
            color = QtGui.QColor("darkgreen") if parts else QtGui.QColor("gray")
        
        constraints_item = QtWidgets.QTableWidgetItem(display_text)
        constraints_item.setForeground(color)
        self.component_list.setItem(row, 1, constraints_item)
    
    def on_component_selected(self, item):
        """Handle component selection - show constraint editor"""
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
        
        # Create constraint editor widget and pass all components
        editor = ConstraintEditor(component, dialog, all_components=self.components)
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
        
        dialog.exec_()
    
    def on_fit_requested(self):
        """Emit signal to perform fitting"""
        if not self.components:
            QtWidgets.QMessageBox.warning(self, "No Components", "Please add at least one component to fit")
            return
        self.fit_requested.emit(self.components)
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
        """Handle cancel button - clear bounds and close"""
        self.bounds_cleared.emit()
        self.close()