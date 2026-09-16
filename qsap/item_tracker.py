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
    calculate_ew = pyqtSignal(str)  # Emits item_id when calculate equivalent width is selected
    items_changed = pyqtSignal()  # Emits when items list is updated (added or removed)
    item_display_toggled = pyqtSignal(str, bool)  # Emits (item_id, display_state) when Display checkbox is toggled
    
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
        
        # Table widget with columns (Fit Color strip added as column 0)
        self.item_table = QtWidgets.QTableWidget()
        self.item_table.setColumnCount(6)
        self.item_table.setHorizontalHeaderLabels(['Fit', 'Display', 'Name', 'Type', 'Color', 'Position'])
        self.item_table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.item_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.item_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.item_table.customContextMenuRequested.connect(self.show_context_menu)
        self.item_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.item_table.itemChanged.connect(self.on_item_changed)  # Connect to handle checkbox changes
        self.item_table.horizontalHeader().setStretchLastSection(True)
        self.item_table.setColumnWidth(0, 20)   # Fit color strip (narrow)
        self.item_table.setColumnWidth(1, 70)   # Display checkbox
        self.item_table.setColumnWidth(2, 150)  # Name
        self.item_table.setColumnWidth(3, 100)  # Type
        self.item_table.setColumnWidth(4, 100)  # Color
        self.item_table.setColumnWidth(5, 150)  # Position
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
    
    def add_item(self, item_id, item_type, name, position='', color='gray', line_obj=None, fit_id=None, fit_color='#999999'):
        """Add an item to tracker
        
        Args:
            item_id: Unique identifier for this item
            item_type: Type of item ('gaussian', 'voigt', 'polynomial', 'marker', etc.)
            name: Display name
            position: Position/bounds description
            color: Color for the item line/marker
            line_obj: Line object reference for visibility toggling
            fit_id: Fit ID this item belongs to (for grouping)
            fit_color: Color of the fit (for color strip)
        """
        self.items[item_id] = {
            'type': item_type,
            'name': name,
            'position': position,
            'color': color,
            'line_obj': line_obj,
            'displayed': True,  # Default to visible
            'zorder': None,  # Store original zorder to preserve it when toggling visibility
            'fit_id': fit_id,  # Which fit this belongs to
            'fit_color': fit_color  # Color strip for the fit
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
        self.item_table.blockSignals(True)  # Block signals while updating to avoid triggering changes
        self.item_table.setRowCount(0)
        for item_id, item_info in self.items.items():
            row = self.item_table.rowCount()
            self.item_table.insertRow(row)
            
            # Fit color strip column (column 0) - narrow color bar with fit ID
            fit_id = item_info.get('fit_id')
            fit_id_text = f"{fit_id}" if fit_id is not None else ""
            fit_color_item = QtWidgets.QTableWidgetItem(fit_id_text)
            fit_color_item.setBackground(QtGui.QColor(item_info.get('fit_color', '#999999')))
            fit_color_item.setForeground(QtGui.QColor('white'))  # White text for contrast
            fit_color_item.setTextAlignment(Qt.AlignCenter)  # Center the text
            font = fit_color_item.font()
            font.setPointSize(10)
            font.setBold(True)
            fit_color_item.setFont(font)  # Make text bold and larger
            fit_color_item.setData(Qt.UserRole, item_id)
            fit_color_item.setFlags(fit_color_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.item_table.setItem(row, 0, fit_color_item)
            
            # Display checkbox column (column 1)
            display_checkbox = QtWidgets.QTableWidgetItem()
            display_checkbox.setCheckState(Qt.Checked if item_info.get('displayed', True) else Qt.Unchecked)
            display_checkbox.setData(Qt.UserRole, item_id)
            display_checkbox.setFlags(display_checkbox.flags() | Qt.ItemIsUserCheckable)
            self.item_table.setItem(row, 1, display_checkbox)
            
            # Name column (column 2)
            name_item = QtWidgets.QTableWidgetItem(item_info['name'])
            name_item.setData(Qt.UserRole, item_id)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.item_table.setItem(row, 2, name_item)
            
            # Type column (column 3)
            type_item = QtWidgets.QTableWidgetItem(item_info['type'])
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.item_table.setItem(row, 3, type_item)
            
            # Color column (column 4)
            color_item = QtWidgets.QTableWidgetItem('')  # Empty text, just show the color
            color_item.setBackground(QtGui.QColor(item_info['color']))
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.item_table.setItem(row, 4, color_item)
            
            # Position column (column 5)
            pos_item = QtWidgets.QTableWidgetItem(str(item_info['position']))
            pos_item.setFlags(pos_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
            self.item_table.setItem(row, 5, pos_item)
        
        self.item_table.blockSignals(False)  # Re-enable signals
    
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
        
        # Add estimate redshift and calculate EW options for Gaussians and Voigts
        estimate_redshift_action = None
        calculate_ew_action = None
        if item_type in ['gaussian', 'voigt']:
            estimate_redshift_action = menu.addAction("Estimate Redshift")
            calculate_ew_action = menu.addAction("Calculate Equivalent Width")
            menu.addSeparator()
        
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.item_table.mapToGlobal(position))
        if action == delete_action:
            self.delete_selected()
        elif estimate_redshift_action and action == estimate_redshift_action:
            self.estimate_redshift.emit(item_id)
        elif calculate_ew_action and action == calculate_ew_action:
            self.calculate_ew.emit(item_id)
    
    def on_selection_changed(self):
        """Handle item selection in the table - only emit for actual changes"""
        # Get currently selected row indices
        selected_rows = set(index.row() for index in self.item_table.selectedIndexes())
        
        # Get the item IDs for currently selected rows
        current_selected_ids = set()
        for row in selected_rows:
            item = self.item_table.item(row, 1)  # Display checkbox is now in column 1
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
    
    def on_item_changed(self, item):
        """Handle Display checkbox state changes"""
        # Only process if this is the Display checkbox column (now column 1, not 0)
        if self.item_table.column(item) != 1:
            return
        
        row = self.item_table.row(item)
        if row < 0:
            return
        
        # Get the item_id from the checkbox
        checkbox_item = self.item_table.item(row, 1)
        if not checkbox_item:
            return
        
        item_id = checkbox_item.data(Qt.UserRole)
        if not item_id or item_id not in self.items:
            return
        
        # Update the displayed state
        is_checked = checkbox_item.checkState() == Qt.Checked
        self.items[item_id]['displayed'] = is_checked
        
        # Emit signal so spectrum plotter can handle visibility toggle
        self.item_display_toggled.emit(item_id, is_checked)
    
    def delete_selected(self):
        """Delete selected items"""
        selected_rows = set(index.row() for index in self.item_table.selectedIndexes())
        for row in sorted(selected_rows, reverse=True):
            item = self.item_table.item(row, 1)  # Get item_id from Display checkbox column
            item_id = item.data(Qt.UserRole)
            self.item_deleted.emit(item_id)
            self.remove_item(item_id)
    
    def clear_all(self):
        """Clear all items"""
        # Collect item IDs first since handlers will modify self.items
        item_ids = list(self.items.keys())
        # Only emit signals - don't call remove_item() here because
        # the signal handlers (on_item_deleted_from_tracker) will call unregister_item()
        # which calls remove_item(), so we'd be removing twice and crashing
        for item_id in item_ids:
            self.item_deleted.emit(item_id)
    
    def highlight_item(self, item_id):
        """Programmatically select a row corresponding to item_id"""
        if item_id not in self.items:
            return
        
        # Find the row for this item_id
        for row in range(self.item_table.rowCount()):
            item = self.item_table.item(row, 1)  # Get item_id from Display checkbox column
            if item and item.data(Qt.UserRole) == item_id:
                self.item_table.selectRow(row)
                return
