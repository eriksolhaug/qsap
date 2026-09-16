"""
FitDiagnosticsPanel - Display and manage fit diagnostic information
"""

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QColor
from qsap.ui_utils import get_qsap_icon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class FitDiagnosticsPanel(QtWidgets.QWidget):
    """Panel for displaying diagnostic information about fits"""
    
    # Signal emitted when a fit is selected in the panel
    fit_selected = pyqtSignal(int)  # Emits fit_id
    
    def __init__(self, parent=None):
        """Initialize the Fit Diagnostics Panel"""
        super().__init__(parent)
        self.setWindowTitle("Fit Diagnostics")
        
        # Store fit metadata (will be updated by parent)
        self.fits_data = {}  # fit_id → {type, r_squared, chi2_reduced, akaike, bayesian, n_params, n_data, condition_num, flag, timestamp}
        self.fit_colors = {}  # fit_id → color hex
        
        # Create main layout
        main_layout = QtWidgets.QVBoxLayout()
        
        # Create description label
        description = QtWidgets.QLabel("Diagnostic metrics for all fits:")
        main_layout.addWidget(description)
        
        # Create diagnostics table with scroll area
        self.diagnostics_table = QtWidgets.QTableWidget()
        self.diagnostics_table.setColumnCount(9)
        self.diagnostics_table.setHorizontalHeaderLabels([
            "Fit", "Flag", "R²", "χ²_red", "AIC", "BIC", "Type", "Params", "Cond."
        ])
        # Make columns resizable by dragging header dividers
        self.diagnostics_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.diagnostics_table.horizontalHeader().setStretchLastSection(False)
        self.diagnostics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.diagnostics_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.diagnostics_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.diagnostics_table.customContextMenuRequested.connect(self.show_context_menu)
        
        # Add table to scroll area
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.diagnostics_table)
        scroll_area.setWidgetResizable(True)
        main_layout.addWidget(scroll_area)
        
        # Create button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        self.clear_button = QtWidgets.QPushButton("Clear All")
        self.clear_button.clicked.connect(self.clear_diagnostics)
        button_layout.addWidget(self.clear_button)
        
        button_layout.addStretch()
        
        self.export_button = QtWidgets.QPushButton("Export CSV")
        self.export_button.clicked.connect(self.export_diagnostics)
        button_layout.addWidget(self.export_button)
        
        main_layout.addLayout(button_layout)
        
        self.setLayout(main_layout)
    
    def update_fit_diagnostics(self, fit_id, fit_type, diagnostics_dict, fit_color='#999999', 
                               covariance=None, parameter_names=None, fit_data=None, 
                               wavelength=None, flux=None, model=None, residuals=None):
        """Update or add a fit's diagnostic information
        
        Args:
            fit_id: Unique fit identifier
            fit_type: Type of fit ('Gaussian', 'Voigt', 'Continuum', 'Listfit', etc.)
            diagnostics_dict: Dict with keys: r_squared, chi2_reduced, akaike, bayesian, n_params, n_data, condition_num, flag
            fit_color: Hex color for this fit
            covariance: Optional covariance matrix (n_params x n_params)
            parameter_names: Optional list of parameter names
            fit_data: Optional fit data dictionary
            wavelength: Optional wavelength array for residuals plot
            flux: Optional flux/data array for residuals plot
            model: Optional model array for residuals plot
            residuals: Optional residuals array
        """
        print(f"[DEBUG] fit_diagnostics_panel.update_fit_diagnostics called: fit_id={fit_id}, fit_type={fit_type}")
        
        self.fits_data[fit_id] = {
            'type': fit_type,
            'r_squared': diagnostics_dict.get('r_squared'),
            'chi2_reduced': diagnostics_dict.get('chi2_reduced'),
            'akaike': diagnostics_dict.get('akaike'),
            'bayesian': diagnostics_dict.get('bayesian'),
            'n_params': diagnostics_dict.get('n_params'),
            'n_data': diagnostics_dict.get('n_data'),
            'condition_num': diagnostics_dict.get('condition_num'),
            'flag': diagnostics_dict.get('flag', '🟡'),
            'covariance': covariance,
            'parameter_names': parameter_names,
            'fit_data': fit_data,
            'wavelength': wavelength,
            'flux': flux,
            'model': model,
            'residuals': residuals,
        }
        self.fit_colors[fit_id] = fit_color
        
        print(f"[DEBUG] About to call refresh_table(), fits_data keys: {list(self.fits_data.keys())}")
        self.refresh_table()
        print(f"[DEBUG] refresh_table() completed")
    
    def remove_fit(self, fit_id):
        """Remove a fit from the diagnostics display"""
        if fit_id in self.fits_data:
            del self.fits_data[fit_id]
        if fit_id in self.fit_colors:
            del self.fit_colors[fit_id]
        self.refresh_table()
    
    def clear_diagnostics(self):
        """Clear all diagnostics"""
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All", 
            "Clear all fit diagnostics?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.fits_data.clear()
            self.fit_colors.clear()
            self.refresh_table()
    
    def export_diagnostics(self):
        """Export diagnostics to CSV file"""
        if not self.fits_data:
            QtWidgets.QMessageBox.information(self, "Export", "No diagnostics to export")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Fit Diagnostics", "", "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w') as f:
                # Write header
                f.write("Fit ID,Type,Flag,R²,χ²_red,AIC,BIC,N Params,N Data,Condition Number\n")
                
                # Write data rows
                for fit_id in sorted(self.fits_data.keys()):
                    data = self.fits_data[fit_id]
                    flag = data.get('flag', '?')
                    r2 = data.get('r_squared', '')
                    chi2_red = data.get('chi2_reduced', '')
                    aic = data.get('akaike', '')
                    bic = data.get('bayesian', '')
                    n_params = data.get('n_params', '')
                    n_data = data.get('n_data', '')
                    cond = data.get('condition_num', '')
                    
                    f.write(f"{fit_id},{data['type']},{flag},{r2},{chi2_red},{aic},{bic},{n_params},{n_data},{cond}\n")
            
            QtWidgets.QMessageBox.information(self, "Export", f"Diagnostics exported to:\n{file_path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
    
    def refresh_table(self):
        """Refresh the diagnostics table display"""
        self.diagnostics_table.blockSignals(True)
        self.diagnostics_table.setRowCount(0)
        
        for fit_id in sorted(self.fits_data.keys()):
            data = self.fits_data[fit_id]
            row = self.diagnostics_table.rowCount()
            self.diagnostics_table.insertRow(row)
            
            # Fit ID column
            fit_id_item = QtWidgets.QTableWidgetItem(str(fit_id))
            fit_id_item.setBackground(QColor(self.fit_colors.get(fit_id, '#999999')))
            fit_id_item.setForeground(QtGui.QColor('white'))
            fit_id_item.setTextAlignment(Qt.AlignCenter)
            fit_id_item.setData(Qt.UserRole, fit_id)
            fit_id_item.setFlags(fit_id_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 0, fit_id_item)
            
            # Flag column (visual indicator)
            flag = data.get('flag', '🟡')
            flag_item = QtWidgets.QTableWidgetItem(str(flag))
            flag_item.setTextAlignment(Qt.AlignCenter)
            flag_item.setFlags(flag_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 1, flag_item)
            
            # R² column
            r2 = data.get('r_squared')
            r2_text = f"{r2:.4f}" if r2 is not None else "N/A"
            r2_item = QtWidgets.QTableWidgetItem(r2_text)
            r2_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            r2_item.setFlags(r2_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 2, r2_item)
            
            # χ²_red column
            chi2_red = data.get('chi2_reduced')
            chi2_text = f"{chi2_red:.6f}" if chi2_red is not None else "N/A"
            chi2_item = QtWidgets.QTableWidgetItem(chi2_text)
            chi2_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            chi2_item.setFlags(chi2_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 3, chi2_item)
            
            # AIC column
            aic = data.get('akaike')
            aic_text = f"{aic:.2f}" if aic is not None else "N/A"
            aic_item = QtWidgets.QTableWidgetItem(aic_text)
            aic_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            aic_item.setFlags(aic_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 4, aic_item)
            
            # BIC column
            bic = data.get('bayesian')
            bic_text = f"{bic:.2f}" if bic is not None else "N/A"
            bic_item = QtWidgets.QTableWidgetItem(bic_text)
            bic_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            bic_item.setFlags(bic_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 5, bic_item)
            
            # Type column
            type_item = QtWidgets.QTableWidgetItem(data.get('type', 'Unknown'))
            type_item.setTextAlignment(Qt.AlignCenter)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 6, type_item)
            
            # N Params column
            n_params = data.get('n_params')
            params_text = str(n_params) if n_params is not None else "N/A"
            params_item = QtWidgets.QTableWidgetItem(params_text)
            params_item.setTextAlignment(Qt.AlignCenter)
            params_item.setFlags(params_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 7, params_item)
            
            # Condition Number column
            cond = data.get('condition_num')
            cond_text = f"{cond:.2e}" if cond is not None and cond != 'N/A' else "N/A"
            cond_item = QtWidgets.QTableWidgetItem(cond_text)
            cond_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            cond_item.setFlags(cond_item.flags() & ~Qt.ItemIsEditable)
            self.diagnostics_table.setItem(row, 8, cond_item)
        
        self.diagnostics_table.blockSignals(False)
    
    def show_context_menu(self, position):
        """Show right-click context menu with visualization options"""
        row = self.diagnostics_table.rowAt(position.y())
        if row < 0:
            return
        
        item = self.diagnostics_table.item(row, 0)
        if not item:
            return
        
        fit_id = item.data(Qt.UserRole)
        
        menu = QtWidgets.QMenu()
        
        # Add context menu actions
        cov_action = menu.addAction("Plot Covariance Matrix")
        res_action = menu.addAction("Plot Residuals")
        corr_action = menu.addAction("Plot Correlation Matrix")
        report_action = menu.addAction("Show Fit Report")
        
        action = menu.exec_(self.diagnostics_table.mapToGlobal(position))
        
        if action == cov_action:
            self.show_covariance_matrix(fit_id)
        elif action == res_action:
            self.show_residuals(fit_id)
        elif action == corr_action:
            self.show_correlation_matrix(fit_id)
        elif action == report_action:
            self.show_fit_report(fit_id)
    
    def show_covariance_matrix(self, fit_id):
        """Display covariance matrix heatmap for a fit"""
        if fit_id not in self.fits_data:
            QtWidgets.QMessageBox.warning(self, "Error", f"Fit #{fit_id} not found")
            return
        
        data = self.fits_data[fit_id]
        covariance = data.get('covariance')
        parameter_names = data.get('parameter_names')
        
        if covariance is None:
            QtWidgets.QMessageBox.information(
                self, "Covariance Matrix",
                f"Covariance matrix data not available for Fit #{fit_id}\n\n"
                "Covariance is typically available for Listfit and lmfit-based fits.")
            return
        
        # Convert to numpy array if needed
        if not isinstance(covariance, np.ndarray):
            try:
                covariance = np.array(covariance)
            except:
                QtWidgets.QMessageBox.warning(self, "Error", "Could not convert covariance matrix to array")
                return
        
        # Ensure parameter_names matches covariance size
        # If not provided or mismatched, generate generic labels
        if parameter_names is None or len(parameter_names) != covariance.shape[0]:
            parameter_names = [f'param_{i}' for i in range(covariance.shape[0])]
        
        # Create dialog with matplotlib figure
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Covariance Matrix - Fit #{fit_id}")
        dialog.setGeometry(100, 100, 700, 600)
        
        # Create matplotlib figure
        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot covariance matrix as heatmap
        # Use symmetric normalization around zero for diverging colormap
        abs_max = np.max(np.abs(covariance))
        norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
        cmap = plt.cm.RdBu_r  # Red-Blue reversed (red=positive, blue=negative)
        
        if HAS_SEABORN:
            # Use seaborn for nicer heatmap with text color based on background darkness
            im = sns.heatmap(covariance, annot=True, fmt='.3e', cmap='RdBu_r', 
                           xticklabels=parameter_names, yticklabels=parameter_names,
                           cbar_kws={'label': 'Covariance'}, ax=ax, square=True,
                           cbar=True, norm=norm, vmin=-abs_max, vmax=abs_max)
            
            # Set text color to white on dark backgrounds, black on light backgrounds
            # Calculate luminance for each cell to determine text color
            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[1]):
                    val = covariance[i, j]
                    normalized_val = norm(val)
                    rgb = cmap(normalized_val)[:3]
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    text_color = 'white' if luminance < 0.5 else 'black'
                    # Find and update the text at this position
                    if i * covariance.shape[1] + j < len(ax.texts):
                        ax.texts[i * covariance.shape[1] + j].set_color(text_color)
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(covariance, cmap='RdBu_r', aspect='auto', norm=norm)
            ax.set_xticks(np.arange(len(parameter_names or [])))
            ax.set_yticks(np.arange(len(parameter_names or [])))
            if parameter_names:
                ax.set_xticklabels(parameter_names, rotation=45, ha='right')
                ax.set_yticklabels(parameter_names)
            fig.colorbar(im, ax=ax, label='Covariance')
            
            # Annotate with values and proper text color
            for i in range(covariance.shape[0]):
                for j in range(covariance.shape[1]):
                    val = covariance[i, j]
                    normalized_val = norm(val)
                    rgb = cmap(normalized_val)[:3]
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    text_color = 'white' if luminance < 0.5 else 'black'
                    text = ax.text(j, i, f'{val:.2e}',
                                  ha="center", va="center", color=text_color, fontsize=8)
        
        ax.set_title(f"Covariance Matrix - Fit #{fit_id}")
        fig.tight_layout()
        
        # Embed in PyQt5 canvas
        canvas = FigureCanvas(fig)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(canvas)
        
        # Add button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        save_button = QtWidgets.QPushButton("Save as PNG")
        save_button.clicked.connect(lambda: self._save_figure(fig, f"covariance_fit{fit_id}.png"))
        button_layout.addWidget(save_button)
        
        button_layout.addStretch()
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def show_residuals(self, fit_id):
        """Display residuals plot (data, model, residuals) for a fit"""
        if fit_id not in self.fits_data:
            QtWidgets.QMessageBox.warning(self, "Error", f"Fit #{fit_id} not found")
            return
        
        data = self.fits_data[fit_id]
        wavelength = data.get('wavelength')
        flux = data.get('flux')
        model = data.get('model')
        residuals_arr = data.get('residuals')
        
        if wavelength is None or flux is None or model is None:
            QtWidgets.QMessageBox.information(
                self, "Residuals Plot",
                f"Residuals data not available for Fit #{fit_id}\n\n"
                "Residuals plot requires wavelength, flux, and model data.")
            return
        
        # Create dialog with matplotlib figure
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Residuals Plot - Fit #{fit_id}")
        dialog.setGeometry(100, 100, 900, 700)
        
        # Create matplotlib figure with subplots
        fig = Figure(figsize=(9, 7), dpi=100)
        
        # Top subplot: Data and Model overlay
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(wavelength, flux, 'o-', color='blue', alpha=0.6, label='Data', linewidth=1, markersize=3)
        ax1.plot(wavelength, model, '-', color='red', linewidth=2, label='Model')
        ax1.set_ylabel('Flux')
        ax1.set_title(f"Data and Model Fit - Fit #{fit_id}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle subplot: Residuals
        if residuals_arr is None:
            residuals_arr = flux - model
        
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.scatter(wavelength, residuals_arr, color='green', alpha=0.6, s=10)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Bottom subplot: Residuals histogram
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.hist(residuals_arr, bins=30, color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Wavelength')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        # Embed in PyQt5 canvas
        canvas = FigureCanvas(fig)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(canvas)
        
        # Add button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        save_button = QtWidgets.QPushButton("Save as PNG")
        save_button.clicked.connect(lambda: self._save_figure(fig, f"residuals_fit{fit_id}.png"))
        button_layout.addWidget(save_button)
        
        button_layout.addStretch()
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def show_correlation_matrix(self, fit_id):
        """Display correlation matrix heatmap for a fit"""
        if fit_id not in self.fits_data:
            QtWidgets.QMessageBox.warning(self, "Error", f"Fit #{fit_id} not found")
            return
        
        data = self.fits_data[fit_id]
        covariance = data.get('covariance')
        parameter_names = data.get('parameter_names')
        
        if covariance is None:
            QtWidgets.QMessageBox.information(
                self, "Correlation Matrix",
                f"Correlation matrix data not available for Fit #{fit_id}\n\n"
                "Correlation is typically available for Listfit and lmfit-based fits.")
            return
        
        # Convert to numpy array if needed
        if not isinstance(covariance, np.ndarray):
            try:
                covariance = np.array(covariance)
            except:
                QtWidgets.QMessageBox.warning(self, "Error", "Could not convert covariance matrix to array")
                return
        
        # Calculate correlation matrix from covariance
        try:
            diag_std = np.sqrt(np.diag(covariance))
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                correlation = covariance / np.outer(diag_std, diag_std)
                correlation = np.nan_to_num(correlation)
        except:
            QtWidgets.QMessageBox.warning(self, "Error", "Could not calculate correlation from covariance")
            return
        
        # Create dialog with matplotlib figure
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Correlation Matrix - Fit #{fit_id}")
        dialog.setGeometry(100, 100, 700, 600)
        
        # Create matplotlib figure
        fig = Figure(figsize=(7, 6), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot correlation matrix as heatmap
        if HAS_SEABORN:
            # Use seaborn for nicer heatmap
            sns.heatmap(correlation, annot=True, fmt='.3f', cmap='RdBu_r', 
                       xticklabels=parameter_names, yticklabels=parameter_names,
                       cbar_kws={'label': 'Correlation'}, ax=ax, square=True, 
                       vmin=-1, vmax=1)
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(correlation, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax.set_xticks(np.arange(len(parameter_names or [])))
            ax.set_yticks(np.arange(len(parameter_names or [])))
            if parameter_names:
                ax.set_xticklabels(parameter_names, rotation=45, ha='right')
                ax.set_yticklabels(parameter_names)
            fig.colorbar(im, ax=ax, label='Correlation')
            
            # Annotate with values
            for i in range(correlation.shape[0]):
                for j in range(correlation.shape[1]):
                    text = ax.text(j, i, f'{correlation[i, j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(f"Correlation Matrix - Fit #{fit_id}")
        fig.tight_layout()
        
        # Embed in PyQt5 canvas
        canvas = FigureCanvas(fig)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(canvas)
        
        # Add button layout
        button_layout = QtWidgets.QHBoxLayout()
        
        save_button = QtWidgets.QPushButton("Save as PNG")
        save_button.clicked.connect(lambda: self._save_figure(fig, f"correlation_fit{fit_id}.png"))
        button_layout.addWidget(save_button)
        
        button_layout.addStretch()
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
        dialog.exec_()
    
    def _save_figure(self, fig, filename):
        """Save a matplotlib figure to file"""
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Figure", filename, "PNG Files (*.png);;PDF Files (*.pdf)"
        )
        
        if file_path:
            try:
                fig.savefig(file_path, dpi=150, bbox_inches='tight')
                QtWidgets.QMessageBox.information(self, "Success", f"Figure saved to:\n{file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save figure:\n{e}")

    
    def show_fit_report(self, fit_id):
        """Display detailed fit report"""
        # Get fit data
        if fit_id not in self.fits_data:
            QtWidgets.QMessageBox.warning(self, "Error", f"Fit #{fit_id} not found")
            return
        
        data = self.fits_data[fit_id]
        
        # Create report text
        report = f"""
╔══════════════════════════════════════╗
║         FIT DIAGNOSTIC REPORT        ║
╚══════════════════════════════════════╝

Fit ID:           {fit_id}
Fit Type:         {data.get('type', 'Unknown')}
Status:           {data.get('flag', '?')}

─────────────────────────────────────
  GOODNESS OF FIT
─────────────────────────────────────
R²:               {data.get('r_squared', 'N/A')}
Reduced χ²:       {data.get('chi2_reduced', 'N/A')}

─────────────────────────────────────
  MODEL SELECTION CRITERIA
─────────────────────────────────────
Akaike IC:        {data.get('akaike', 'N/A')}
Bayesian IC:      {data.get('bayesian', 'N/A')}

─────────────────────────────────────
  PARAMETER INFORMATION
─────────────────────────────────────
N Parameters:     {data.get('n_params', 'N/A')}
N Data Points:    {data.get('n_data', 'N/A')}
Condition Num:    {data.get('condition_num', 'N/A')}
"""
        
        # Show in a dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Fit Report - Fit #{fit_id}")
        dialog.setGeometry(100, 100, 500, 400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Text display
        text_display = QtWidgets.QTextEdit()
        text_display.setText(report)
        text_display.setReadOnly(True)
        text_display.setFont(QtGui.QFont('Courier', 10))
        layout.addWidget(text_display)
        
        # Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec_()
