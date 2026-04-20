"""
SpectrumPlotterApp - Application wrapper for SpectrumPlotter
"""

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QIcon
from .spectrum_plotter import SpectrumPlotter
from qasap.ui_utils import get_qasap_icon

class SpectrumPlotterApp(QWidget):
    def __init__(self, plotter):
        super().__init__()
        self.plotter = plotter
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("QASAP - Quick Analysis of Spectra and Profiles")
        # Load and set window icon
        self.setWindowIcon(get_qasap_icon())
        layout = QVBoxLayout()

        # File input
        self.file_label = QLabel("FITS File Path:")
        self.file_input = QLineEdit()
        self.file_input.setText(self.plotter.fits_file)
        layout.addWidget(self.file_label)
        layout.addWidget(self.file_input)

        # Redshift input
        self.redshift_label = QLabel("Redshift:")
        self.redshift_input = QLineEdit(str(self.plotter.redshift))
        layout.addWidget(self.redshift_label)
        layout.addWidget(self.redshift_input)

        # Zoom factor input
        self.zoom_label = QLabel("Zoom Factor:")
        self.zoom_input = QLineEdit(str(self.plotter.zoom_factor))
        layout.addWidget(self.zoom_label)
        layout.addWidget(self.zoom_input)

        # File flag input
        self.flag_label = QLabel("File Flag (0, 1, or 2):")
        self.flag_input = QLineEdit(str(self.plotter.file_flag))
        layout.addWidget(self.flag_label)
        layout.addWidget(self.flag_input)

        # Plot button
        self.plot_button = QPushButton("Plot Spectrum")
        self.plot_button.clicked.connect(self.on_plot)
        layout.addWidget(self.plot_button)

        self.setLayout(layout)

    def on_plot(self):
        # Update plotter attributes from GUI inputs
        self.plotter.fits_file = self.file_input.text()
        self.plotter.redshift = float(self.redshift_input.text())
        self.plotter.zoom_factor = float(self.zoom_input.text())
        self.plotter.file_flag = int(self.flag_input.text())

        # Call the plotting function
        self.plotter.plot_spectrum()

