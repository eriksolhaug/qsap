"""
Line List Selector Window for QASAP
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon
from .linelist import LineList, get_available_line_lists
from qasap.ui_utils import get_qasap_icon


class LineListSelector(QtWidgets.QWidget):
    """Window for selecting and managing line lists to display"""
    
    line_lists_changed = pyqtSignal(list)  # Emits list of selected LineList objects with colors
    
    def __init__(self, resources_dir: str):
        super().__init__()
        self.resources_dir = resources_dir
        self.available_line_lists = get_available_line_lists(resources_dir)
        self.selected_line_lists = {}  # {linelist_name: {'linelist': LineList, 'color': str}}
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("QASAP - Line Lists")
        # Load and set window icon
        self.setWindowIcon(get_qasap_icon())
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
        
        # Connect checkbox changes to auto-update
        self.linelist_table.itemChanged.connect(self.on_checkbox_changed)
        
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
    
    def on_checkbox_changed(self, item):
        """Handle checkbox state changes - auto-toggle display"""
        self.toggle_linelist()
    
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
    
    def close_window(self):
        """Close the window"""
        self.close()
    
    def get_selected_line_lists(self):
        """Get currently selected line lists with their colors"""
        return list(self.selected_line_lists.values())
