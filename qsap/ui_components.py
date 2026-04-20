"""
UI Components module - imports and re-exports all UI classes

This module provides clean imports for the PyQt5-based GUI components:
- SpectrumPlotter: Main interactive spectrum plotting widget
- SpectrumPlotterApp: Application wrapper
- LineListWindow: Line list dialog window

All actual implementations are in separate modules for code organization.
"""

from .spectrum_plotter import SpectrumPlotter
from .spectrum_plotter_app import SpectrumPlotterApp
from .linelist_window import LineListWindow

__all__ = ['SpectrumPlotter', 'SpectrumPlotterApp', 'LineListWindow']
