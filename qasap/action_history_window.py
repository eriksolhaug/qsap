"""
ActionHistoryWindow - UI for viewing and navigating action history
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon
from qasap.ui_utils import get_qasap_icon


class ActionHistoryWindow(QtWidgets.QWidget):
    """Window for displaying action history and navigating through undo/redo"""
    
    action_selected = pyqtSignal(int)  # Emits index when user clicks an action
    
    def __init__(self):
        super().__init__()
        self.action_history = None
        self.history_list = []
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("QASAP - Action History")
        self.setWindowIcon(get_qasap_icon())
        self.setGeometry(100, 400, 500, 300)
        
        layout = QtWidgets.QVBoxLayout()
        
        # Title
        layout.addWidget(QtWidgets.QLabel("Action History:"))
        
        # List widget to show history
        self.history_list_widget = QtWidgets.QListWidget()
        self.history_list_widget.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.history_list_widget)
        
        # Control buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.undo_button = QtWidgets.QPushButton("← Undo")
        self.undo_button.clicked.connect(self.on_undo)
        button_layout.addWidget(self.undo_button)
        
        self.redo_button = QtWidgets.QPushButton("Redo →")
        self.redo_button.clicked.connect(self.on_redo)
        button_layout.addWidget(self.redo_button)
        
        self.clear_button = QtWidgets.QPushButton("Clear History")
        self.clear_button.clicked.connect(self.on_clear)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def set_action_history(self, action_history):
        """Set reference to the ActionHistory object"""
        self.action_history = action_history
        self.refresh_display()
    
    def refresh_display(self):
        """Update the list display based on current history"""
        if not self.action_history:
            return
        
        self.history_list_widget.clear()
        self.history_list = self.action_history.get_history_list()
        current_pos = self.action_history.get_current_position()
        
        for i, action in enumerate(self.history_list):
            # Format: timestamp - description
            time_str = action.timestamp.strftime("%H:%M:%S")
            item_text = f"{time_str} - {action.description}"
            
            item = QtWidgets.QListWidgetItem(item_text)
            
            # Highlight current position
            if i == current_pos:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setBackground(QtGui.QColor(100, 150, 255, 100))
            
            self.history_list_widget.addItem(item)
        
        # Scroll to current position if history exists
        if 0 <= current_pos < len(self.history_list):
            self.history_list_widget.setCurrentRow(current_pos)
        
        # Update button states
        self.undo_button.setEnabled(self.action_history.can_undo())
        self.redo_button.setEnabled(self.action_history.can_redo())
    
    def on_item_clicked(self, item):
        """Handle clicking on a history item"""
        row = self.history_list_widget.row(item)
        self.action_selected.emit(row)
    
    def on_undo(self):
        """Handle undo button click"""
        if self.action_history and self.action_history.can_undo():
            self.action_history.undo()
            self.refresh_display()
            self.action_selected.emit(self.action_history.get_current_position())
    
    def on_redo(self):
        """Handle redo button click"""
        if self.action_history and self.action_history.can_redo():
            self.action_history.redo()
            self.refresh_display()
            self.action_selected.emit(self.action_history.get_current_position())
    
    def on_clear(self):
        """Handle clear history button click"""
        if self.action_history:
            reply = QtWidgets.QMessageBox.question(
                self, 'Clear History?', 
                'Are you sure you want to clear all action history?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
            )
            if reply == QtWidgets.QMessageBox.Yes:
                self.action_history.clear_history()
                self.refresh_display()
