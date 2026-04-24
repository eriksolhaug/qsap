"""
Line List Selector Window for QSAP
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon
from .linelist import LineList, get_available_line_lists
from qsap.ui_utils import get_qsap_icon


class LineListSelector(QtWidgets.QWidget):
    """Window for selecting and managing line lists to display"""
    
    line_lists_changed = pyqtSignal(list)  # Emits list of selected LineList objects with colors
    
    def __init__(self, resources_dir: str):
        super().__init__()
        self.resources_dir = resources_dir
        self.available_line_lists = get_available_line_lists(resources_dir)
        self.selected_line_lists = {}  # {linelist_name: {'linelist': LineList, 'color': str}}
        # Default annotation offsets (normalized 0-1 relative to plotting window)
        self.linelist_x_offset = 0.01  # 1% offset in x direction (positive = right)
        self.linelist_y_offset = 0.02  # 2% offset in y direction (positive = up)
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("QSAP - Line Lists")
        # Load and set window icon
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(1100, 100, 500, 600)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Available Line Lists:"))
        
        # List widget for available line lists
        self.linelist_table = QtWidgets.QTableWidget()
        self.linelist_table.setColumnCount(3)
        self.linelist_table.setHorizontalHeaderLabels(['Name', 'Lines', 'Color'])
        self.linelist_table.horizontalHeader().setStretchLastSection(False)
        self.linelist_table.setColumnWidth(0, 180)
        self.linelist_table.setColumnWidth(1, 80)
        self.linelist_table.setColumnWidth(2, 120)
        
        # Populate table
        self.populate_table()
        
        layout.addWidget(self.linelist_table)
        
        # Connect double-click to change color
        self.linelist_table.doubleClicked.connect(self.on_row_double_clicked)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton("Toggle Display")
        self.btn_toggle.clicked.connect(self.toggle_linelist)
        self.btn_change_color = QtWidgets.QPushButton("Change Color")
        self.btn_change_color.clicked.connect(self.change_color)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.close_window)
        
        button_layout.addWidget(self.btn_toggle)
        button_layout.addWidget(self.btn_change_color)
        button_layout.addWidget(self.btn_close)
        layout.addLayout(button_layout)
        
        # Annotation offset controls
        offset_group = QtWidgets.QGroupBox("Annotation Offsets")
        offset_layout = QtWidgets.QGridLayout()
        
        # X offset
        offset_layout.addWidget(QtWidgets.QLabel("X Offset (0-1):"), 0, 0)
        self.x_offset_input = QtWidgets.QDoubleSpinBox()
        self.x_offset_input.setRange(0.0, 1.0)
        self.x_offset_input.setSingleStep(0.01)
        self.x_offset_input.setValue(self.linelist_x_offset)
        self.x_offset_input.valueChanged.connect(self.on_offset_changed)
        offset_layout.addWidget(self.x_offset_input, 0, 1)
        
        # Y offset
        offset_layout.addWidget(QtWidgets.QLabel("Y Offset (0-1):"), 1, 0)
        self.y_offset_input = QtWidgets.QDoubleSpinBox()
        self.y_offset_input.setRange(0.0, 1.0)
        self.y_offset_input.setSingleStep(0.01)
        self.y_offset_input.setValue(self.linelist_y_offset)
        self.y_offset_input.valueChanged.connect(self.on_offset_changed)
        offset_layout.addWidget(self.y_offset_input, 1, 1)
        
        offset_group.setLayout(offset_layout)
        layout.addWidget(offset_group)
        
        self.setLayout(layout)
    
    def populate_table(self):
        """Populate the table with available line lists"""
        self.linelist_table.setRowCount(0)
        
        for linelist in self.available_line_lists:
            row = self.linelist_table.rowCount()
            self.linelist_table.insertRow(row)
            
            # Name column (checkbox)
            name_widget = QtWidgets.QCheckBox(linelist.name)
            if linelist.name in self.selected_line_lists:
                name_widget.setChecked(True)
            # Connect checkbox state changes directly to update display
            name_widget.stateChanged.connect(self.on_checkbox_state_changed)
            self.linelist_table.setCellWidget(row, 0, name_widget)
            
            # Lines count column
            lines_item = QtWidgets.QTableWidgetItem(str(len(linelist.lines)))
            lines_item.setFlags(lines_item.flags() & ~Qt.ItemIsEditable)
            self.linelist_table.setItem(row, 1, lines_item)
            
            # Color column
            color_button = QtWidgets.QPushButton()
            color = self.selected_line_lists[linelist.name]['color'] if linelist.name in self.selected_line_lists else linelist.color
            color_button.setStyleSheet(f"background-color: {color};")
            color_button.setText("")
            color_button.setFixedSize(40, 25)
            self.linelist_table.setCellWidget(row, 2, color_button)
    
    def toggle_linelist(self):
        """Toggle display of selected line list"""
        for row in range(self.linelist_table.rowCount()):
            checkbox = self.linelist_table.cellWidget(row, 0)
            if checkbox and isinstance(checkbox, QtWidgets.QCheckBox):
                linelist_name = checkbox.text()
                linelist = next((ll for ll in self.available_line_lists if ll.name == linelist_name), None)
                
                if checkbox.isChecked():
                    # Add to selected
                    if linelist and linelist_name not in self.selected_line_lists:
                        # Check if there's a saved color for this linelist
                        color_button = self.linelist_table.cellWidget(row, 2)
                        if color_button and isinstance(color_button, QtWidgets.QPushButton):
                            # Extract color from button's stylesheet if it exists
                            style = color_button.styleSheet()
                            if "background-color" in style:
                                # Parse the color from stylesheet
                                import re
                                match = re.search(r'background-color: ([^;]+);', style)
                                if match:
                                    color = match.group(1).strip()
                                else:
                                    color = linelist.color
                            else:
                                color = linelist.color
                        else:
                            color = linelist.color
                        
                        self.selected_line_lists[linelist_name] = {
                            'linelist': linelist,
                            'color': color
                        }
                else:
                    # Remove from selected
                    if linelist_name in self.selected_line_lists:
                        del self.selected_line_lists[linelist_name]
        
        self.emit_changes()
    
    def on_checkbox_state_changed(self, state):
        """Handle checkbox state changes - auto-toggle display"""
        self.toggle_linelist()
    
    def on_row_double_clicked(self, index):
        """Handle double-click on a row to change color"""
        row = index.row()
        col = index.column()
        
        # Allow double-click on any column to change color
        # Set current row and call change_color
        self.linelist_table.setCurrentCell(row, col)
        self.change_color()
    
    def change_color(self):
        """Change color of selected line list"""
        selected_row = self.linelist_table.currentRow()
        if selected_row < 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a line list first")
            return
        
        checkbox = self.linelist_table.cellWidget(selected_row, 0)
        if not checkbox or not isinstance(checkbox, QtWidgets.QCheckBox):
            return
        
        linelist_name = checkbox.text()
        current_color = self.selected_line_lists[linelist_name]['color'] if linelist_name in self.selected_line_lists else 'blue'
        
        # Open color picker
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current_color), self, "Select Color")
        
        if color.isValid():
            color_str = color.name()  # Returns #RRGGBB format
            
            # Update selected list color
            if linelist_name in self.selected_line_lists:
                self.selected_line_lists[linelist_name]['color'] = color_str
            
            # Update button color
            color_button = self.linelist_table.cellWidget(selected_row, 2)
            if color_button and isinstance(color_button, QtWidgets.QPushButton):
                color_button.setStyleSheet(f"background-color: {color_str};")
            
            self.emit_changes()
    
    def emit_changes(self):
        """Emit signal with current selected line lists"""
        line_lists_with_colors = list(self.selected_line_lists.values())
        self.line_lists_changed.emit(line_lists_with_colors)
    
    def on_offset_changed(self):
        """Handle offset value changes"""
        self.linelist_x_offset = self.x_offset_input.value()
        self.linelist_y_offset = self.y_offset_input.value()
        # Redisplay with new offsets
        self.emit_changes()
    
    def close_window(self):
        """Close the window"""
        self.close()
    
    def get_selected_line_lists(self):
        """Get currently selected line lists with their colors"""
        return list(self.selected_line_lists.values())
