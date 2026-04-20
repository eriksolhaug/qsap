"""
MainWindow - QMainWindow wrapper for QSAP that provides menu bar integration
"""

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from qsap.ui_utils import get_qsap_icon


class QSAPMainWindow(QtWidgets.QMainWindow):
    """Main window for QSAP with integrated menu bar"""
    
    def __init__(self, spectrum_plotter):
        super().__init__()
        self.spectrum_plotter = spectrum_plotter
        
        # Dictionary to track window menu items and their associated windows
        # Initialize BEFORE create_menu_bar() since it needs these
        self.window_menu_items = {}
        self.tracked_windows = {}
        
        # Set window properties
        self.setWindowTitle("QSAP - Menu Bar")
        self.setWindowIcon(get_qsap_icon())
        self.setGeometry(50, 50, 200, 100)
        
        # Create a simple widget for the main window (just to have something in it)
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Connect to window show/hide signals to update menu state
        # This will be called after windows are created
    
    def create_menu_bar(self):
        """Create the menu bar with File and View menus"""
        menubar = self.menuBar()
        
        # Create File menu
        self.file_menu = menubar.addMenu("File")
        
        # Add Quit action to File menu
        quit_action = self.file_menu.addAction("Quit")
        quit_action.setShortcut("Cmd+Q")  # Standard macOS shortcut
        quit_action.triggered.connect(self.quit_application)
        
        # Create Edit menu
        self.edit_menu = menubar.addMenu("Edit")
        
        # Add Undo action
        undo_action = self.edit_menu.addAction("Undo")
        undo_action.setShortcut("Cmd+Z")
        undo_action.triggered.connect(self.on_undo)
        
        # Add Redo action
        redo_action = self.edit_menu.addAction("Redo")
        redo_action.setShortcut("Cmd+Shift+Z")
        redo_action.triggered.connect(self.on_redo)
        
        # Create View menu
        self.view_menu = menubar.addMenu("View")
        
        # Add menu items - these will be populated after windows are created
        self.update_view_menu()
    
    def update_view_menu(self):
        """Update the View menu with all available windows"""
        # Clear existing items
        self.view_menu.clear()
        self.window_menu_items.clear()
        
        # Get windows from the spectrum plotter
        windows = self.get_available_windows()
        
        # Add each window as a checkable menu item
        for window_name, window_obj in windows.items():
            action = self.view_menu.addAction(window_name)
            action.setCheckable(True)
            action.setChecked(window_obj.isVisible() if hasattr(window_obj, 'isVisible') else False)
            action.triggered.connect(lambda checked, name=window_name, obj=window_obj: self.toggle_window(name, obj))
            
            self.window_menu_items[window_name] = action
            self.tracked_windows[window_name] = window_obj
    
    def get_available_windows(self):
        """Get dictionary of available windows from the spectrum plotter"""
        windows = {}
        
        # Control Panel (the plotter itself has control panel elements)
        windows["Control Panel"] = self.spectrum_plotter
        
        # Spectrum Plotter (the matplotlib figure window)
        if hasattr(self.spectrum_plotter, 'fig') and self.spectrum_plotter.fig:
            if hasattr(self.spectrum_plotter.fig, 'canvas'):
                if hasattr(self.spectrum_plotter.fig.canvas, 'manager'):
                    if hasattr(self.spectrum_plotter.fig.canvas.manager, 'window'):
                        windows["Spectrum Plotter"] = self.spectrum_plotter.fig.canvas.manager.window
        
        # Item Tracker
        if hasattr(self.spectrum_plotter, 'item_tracker'):
            windows["Item Tracker"] = self.spectrum_plotter.item_tracker
        
        # Fit Information Window
        if hasattr(self.spectrum_plotter, 'fit_information_window'):
            windows["Fit Information"] = self.spectrum_plotter.fit_information_window
        
        # Help Window
        if hasattr(self.spectrum_plotter, 'help_window') and self.spectrum_plotter.help_window:
            windows["Help"] = self.spectrum_plotter.help_window
        
        # Line List Window
        if hasattr(self.spectrum_plotter, 'linelist_window') and self.spectrum_plotter.linelist_window:
            windows["Line List"] = self.spectrum_plotter.linelist_window
        
        # Listfit Window
        if hasattr(self.spectrum_plotter, 'listfit_window') and self.spectrum_plotter.listfit_window:
            windows["Listfit"] = self.spectrum_plotter.listfit_window
        
        # Action History Window
        if hasattr(self.spectrum_plotter, 'action_history_window'):
            windows["Action History"] = self.spectrum_plotter.action_history_window
        
        return windows
    
    def toggle_window(self, window_name, window_obj):
        """Toggle the visibility of a window"""
        if hasattr(window_obj, 'isVisible'):
            if window_obj.isVisible():
                window_obj.hide()
            else:
                window_obj.show()
        
        # Update the menu item checkmark
        if window_name in self.window_menu_items:
            action = self.window_menu_items[window_name]
            action.setChecked(window_obj.isVisible() if hasattr(window_obj, 'isVisible') else False)
    
    def refresh_view_menu(self):
        """Refresh the View menu - useful after new windows are created"""
        # Disconnect old signals and update
        self.update_view_menu()
        
        # Re-connect to window visibility changes
        self.connect_window_signals()
    
    def quit_application(self):
        """Quit the application"""
        QtWidgets.QApplication.quit()
    
    def on_undo(self):
        """Handle undo action"""
        if hasattr(self.spectrum_plotter, 'on_undo'):
            self.spectrum_plotter.on_undo()
    
    def on_redo(self):
        """Handle redo action"""
        if hasattr(self.spectrum_plotter, 'on_redo'):
            self.spectrum_plotter.on_redo()
    
    def connect_window_signals(self):
        """Connect to window show/hide signals to update menu state"""
        for window_name, window_obj in self.tracked_windows.items():
            if hasattr(window_obj, 'shown') or hasattr(window_obj, 'hidden'):
                # Some windows may have these signals
                if hasattr(window_obj, 'shown'):
                    try:
                        window_obj.shown.connect(lambda name=window_name: self.on_window_shown(name))
                    except:
                        pass
                if hasattr(window_obj, 'hidden'):
                    try:
                        window_obj.hidden.connect(lambda name=window_name: self.on_window_hidden(name))
                    except:
                        pass
    
    def on_window_shown(self, window_name):
        """Update menu when window is shown"""
        if window_name in self.window_menu_items:
            self.window_menu_items[window_name].setChecked(True)
    
    def on_window_hidden(self, window_name):
        """Update menu when window is hidden"""
        if window_name in self.window_menu_items:
            self.window_menu_items[window_name].setChecked(False)
    
    def closeEvent(self, event):
        """Handle main window close event"""
        # Close all associated windows
        for window_name, window_obj in self.tracked_windows.items():
            if hasattr(window_obj, 'close'):
                try:
                    window_obj.close()
                except:
                    pass
        
        event.accept()
