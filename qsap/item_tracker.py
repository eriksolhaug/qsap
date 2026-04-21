"""
ItemTracker - Window to track and manage plotted spectrum features
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from qsap.ui_utils import get_qsap_icon


class ColorBoxDelegate(QtWidgets.QStyledItemDelegate):
    """Custom delegate to draw color boxes in table"""
    def paint(self, painter, option, index):
        color_text = index.data()
        if color_text:
            color = QtGui.QColor(color_text)
            painter.fillRect(option.rect, color)
            painter.drawRect(option.rect)


class ItemTracker(QtWidgets.QWidget):
    """Window for tracking and managing plotted spectrum features"""
    
    item_deleted = pyqtSignal(str)  # Emits item_id when deleted
    item_selected = pyqtSignal(str)  # Emits item_id when selected
    item_individually_deselected = pyqtSignal(str)  # Emits item_id when individually deselected from multi-selection
    item_deselected = pyqtSignal()   # Emits when no items are selected
    estimate_redshift = pyqtSignal(str)  # Emits item_id when estimate redshift is selected
    items_changed = pyqtSignal()  # Emits when items list is updated (added or removed)
    
    def __init__(self):
        super().__init__()
        self.items = {}  # {item_id: {'type': 'gaussian', 'name': 'Gaussian 1', 'position': 'bounds or value', 'color': 'red', ...}}
        self.item_table = None
        self.previously_selected_ids = set()  # Track previously selected items to detect changes
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QSAP - Item Tracker")
        # Load and set window icon
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(100, 550, 600, 300)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Plotted Features:"))
        
        # Table widget with columns
        self.item_table = QtWidgets.QTableWidget()
        self.item_table.setColumnCount(4)
        self.item_table.setHorizontalHeaderLabels(['Name', 'Type', 'Color', 'Position'])
        self.item_table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.item_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.item_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.item_table.customContextMenuRequested.connect(self.show_context_menu)
        self.item_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.item_table.horizontalHeader().setStretchLastSection(True)
        self.item_table.setColumnWidth(0, 150)
        self.item_table.setColumnWidth(1, 100)
        self.item_table.setColumnWidth(2, 100)
        self.item_table.setColumnWidth(3, 150)
        layout.addWidget(self.item_table)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.btn_delete = QtWidgets.QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_clear_all = QtWidgets.QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all)
        button_layout.addWidget(self.btn_delete)
        button_layout.addWidget(self.btn_clear_all)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_item(self, item_id, item_type, name, position='', color='gray', line_obj=None):
        """Add an item to tracker"""
        self.items[item_id] = {
            'type': item_type,
            'name': name,
            'position': position,
            'color': color,
            'line_obj': line_obj
        }
        self.refresh_table()
        self.items_changed.emit()
    
    def remove_item(self, item_id):
        """Remove item from tracker"""
        if item_id in self.items:
            del self.items[item_id]
            self.refresh_table()
            self.items_changed.emit()
    
    def refresh_table(self):
        """Refresh the displayed table"""
        self.item_table.setRowCount(0)
        for item_id, item_info in self.items.items():
            row = self.item_table.rowCount()
            self.item_table.insertRow(row)
            
            # Name column
            name_item = QtWidgets.QTableWidgetItem(item_info['name'])
            name_item.setData(Qt.UserRole, item_id)
            self.item_table.setItem(row, 0, name_item)
            
            # Type column
            type_item = QtWidgets.QTableWidgetItem(item_info['type'])
            self.item_table.setItem(row, 1, type_item)
            
            # Color column
            color_item = QtWidgets.QTableWidgetItem(item_info['color'])
            color_item.setBackground(QtGui.QColor(item_info['color']))
            self.item_table.setItem(row, 2, color_item)
            
            # Position column
            pos_item = QtWidgets.QTableWidgetItem(str(item_info['position']))
            self.item_table.setItem(row, 3, pos_item)
    
    def show_context_menu(self, position):
        """Show right-click context menu"""
        # Get the clicked row
        row = self.item_table.rowAt(position.y())
        if row < 0:
            return
        
        # Get the item being right-clicked
        item = self.item_table.item(row, 0)
        if not item:
            return
        
        item_id = item.data(Qt.UserRole)
        item_type = self.items[item_id]['type'] if item_id in self.items else None
        
        menu = QtWidgets.QMenu()
        
        # Add estimate redshift option for Gaussians and Voigts
        estimate_redshift_action = None
        if item_type in ['gaussian', 'voigt']:
            estimate_redshift_action = menu.addAction("Estimate Redshift")
            menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.item_table.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected()
        elif estimate_redshift_action and action == estimate_redshift_action:
            self.estimate_redshift.emit(item_id)
    
    def on_selection_changed(self):
        """Handle item selection in the table - only emit for actual changes"""
        # Get currently selected row indices
        selected_rows = set(index.row() for index in self.item_table.selectedIndexes())
        
        # Get the item IDs for currently selected rows
        current_selected_ids = set()
        for row in selected_rows:
            item = self.item_table.item(row, 0)
            if item:
                item_id = item.data(Qt.UserRole)
                current_selected_ids.add(item_id)
        
        # Find newly selected items (not in previous selection)
        newly_selected = current_selected_ids - self.previously_selected_ids
        
        # Emit signals only for newly selected items
        for item_id in newly_selected:
            self.item_selected.emit(item_id)
        
        # Find individually deselected items (were selected but not anymore)
        individually_deselected = self.previously_selected_ids - current_selected_ids
        
        # Emit signals for individually deselected items
        for item_id in individually_deselected:
            self.item_individually_deselected.emit(item_id)
        
        # If nothing is selected now, but something was selected before, emit full deselection
        if not current_selected_ids and self.previously_selected_ids:
            self.item_deselected.emit()
        
        # Update tracking for next call
        self.previously_selected_ids = current_selected_ids
    
    def delete_selected(self):
        """Delete selected items"""
        selected_rows = set(index.row() for index in self.item_table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            item = self.item_table.item(row, 0)
            item_id = item.data(Qt.UserRole)
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
    
    def clear_all(self):
        """Clear all items"""
        item_ids = list(self.items.keys())
        for item_id in item_ids:
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
    
    def highlight_item(self, item_id):
        """Programmatically select a row corresponding to item_id"""
        if item_id not in self.items:
            return
        
        # Find the row for this item_id
        for row in range(self.item_table.rowCount()):
            item = self.item_table.item(row, 0)
            if item and item.data(Qt.UserRole) == item_id:
                self.item_table.selectRow(row)
                return
