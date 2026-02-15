#!/usr/bin/env python
"""
QASAP - Quick & Advanced Spectrum Analysis Package
Version 0.10

Interactive spectral analysis tool with intelligent file format detection.
Modular architecture with spectrum I/O and analysis functions.

Author: Erik Solhaug
License: MIT
"""

import argparse
import sys
import numpy as np
from PyQt5 import QtWidgets

# Import modular components from qasap package ONLY (no v0.5 dependencies)
from qasap.spectrum_io import SpectrumIO
from qasap.spectrum_analysis import SpectrumAnalysis
from qasap.ui_components import SpectrumPlotter, SpectrumPlotterApp
from qasap.format_picker_dialog import FormatPickerDialog
from qasap.main_window import QASAPMainWindow


def main():
    """Main entry point for QASAP"""
    
    parser = argparse.ArgumentParser(
        description='QASAP v0.8 - Spectrum Analysis Package',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qasap.py spectrum.fits                    # Auto-detect format
  python qasap.py spectrum.fits --detect          # Show detected formats
  python qasap.py spectrum.fits --fmt fits:image1d # Force format
  python qasap.py 2d_alfosc.fits --alfosc --bin 10 --output 1d.fits
  python qasap.py spectrum.fits --lsf 15.5        # Apply LSF (15.5 km/s)
        """
    )
    
    parser.add_argument('fits_file', type=str, nargs='?',
                        help='Path to spectrum file (FITS or ASCII)')
    parser.add_argument('--fmt', type=str, default=None,
                        help='Force format (auto-detects if omitted)')
    parser.add_argument('--detect', action='store_true',
                        help='Show detected formats and exit')
    parser.add_argument('--redshift', type=float, default=0.0,
                        help='Redshift value')
    
    # LSF handling
    parser.add_argument('--lsf', type=str, default=None,
                        help='Line Spread Function: FWHM in km/s or path to LSF file')
    
    parser.add_argument('--version', action='version',
                        version='QASAP v0.10')
    
    args = parser.parse_args()
    
    # Handle format detection mode
    if args.detect:
        if not args.fits_file:
            print("Error: --detect requires a spectrum file")
            sys.exit(1)
        candidates = SpectrumIO.detect_spectrum_format(args.fits_file)
        print(f"\nFormat detection for: {args.fits_file}\n")
        for i, c in enumerate(candidates, 1):
            print(f"{i}. {c['key']:<25} Score: {c['score']:>3}  {c['notes']}")
        print()
        sys.exit(0)
    
    # Initialize PyQt5 application first (needed for dialogs)
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    
    # Set application icon (shows in Dock on macOS)
    from qasap.ui_utils import get_qasap_icon
    app.setWindowIcon(get_qasap_icon())
    
    # Load spectrum if provided
    wav = None
    spec = None
    err = None
    meta = {}
    file_flag = 0
    
    if args.fits_file:
        try:
            # Determine format and options
            if args.fmt:
                fmt = args.fmt
                options = {}
                print(f"Using forced format: {fmt}")
            else:
                # Auto-detect and show picker dialog
                candidates = SpectrumIO.detect_spectrum_format(args.fits_file)
                if not candidates:
                    print("Error: Could not auto-detect format")
                    sys.exit(1)
                
                # If multiple candidates, show dialog; always show for clarity
                print(f"\nShowing format selection dialog...\n")
                dialog = FormatPickerDialog(args.fits_file, candidates, parent=None)
                print("DEBUG: Dialog created successfully")
                result = dialog.exec_()
                print(f"DEBUG: Dialog result: {result}")
                
                if result != QtWidgets.QDialog.Accepted:
                    print("Format selection cancelled")
                    sys.exit(0)
                
                selection = dialog.get_selection()
                if not selection:
                    print("No format selected")
                    sys.exit(0)
                
                fmt, options = selection
                print(f"Selected format: {fmt}\n")
            
            # Load the spectrum with selected format
            wav, spec, err, meta = SpectrumIO.read_spectrum(args.fits_file, fmt=fmt, options=options)
            
            print(f"Loaded {len(wav)} wavelength points")
            print(f"Wavelength: {wav[0]:.2f} - {wav[-1]:.2f} Å")
            print(f"Flux range: {np.min(spec):.2e} - {np.max(spec):.2e}")
            print(f"Source: {meta.get('source', 'unknown')}")
            
            # Apply LSF if requested
            if args.lsf:
                try:
                    lsf_fwhm = SpectrumAnalysis.parse_lsf_spec(args.lsf)
                    print(f"Applying LSF (FWHM: {lsf_fwhm:.2f} km/s)...")
                    spec = SpectrumAnalysis.apply_lsf(wav, spec, lsf_fwhm)
                    print("LSF applied successfully")
                except Exception as e:
                    print(f"Error applying LSF: {e}")
                    sys.exit(1)
            
            # Apply redshift if requested
            if args.redshift != 0.0:
                z = args.redshift
                wav_rest = wav / (1 + z)
                print(f"Shifted to rest-frame (z={z}): {wav_rest[0]:.2f} - {wav_rest[-1]:.2f} Å")
                wav = wav_rest
            
            # Determine file_flag from format
            file_flag_map = {
                'ascii:2col': 10,
                'ascii:3col': 3,
                'ascii:flex': 3,
                'fits:image1d': 2,
                'fits:table:vector': 5,
                'fits:table:columns': 9,
                'fits:ext:spectrum': 7,
            }
            file_flag = file_flag_map.get(fmt, 0)
            
            print("\nLaunching interactive plotter...\n")
            
        except Exception as e:
            print(f"Error loading spectrum: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("No spectrum file specified. Starting QASAP with empty plotter.")
        print("Use the 'Open' button in the control panel to load a spectrum.\n")
    
    # Launch interactive GUI plotter
    plotter = SpectrumPlotter(
        fits_file=args.fits_file,
        redshift=args.redshift,
        zoom_factor=0.1,
        file_flag=file_flag,
        lsf=args.lsf or "10"
    )
    
    # Load spectrum data if available
    if wav is not None and spec is not None:
        plotter.load_spectrum_data(wav, spec, err, meta, args.fits_file)
    
    # Create main window with menu bar (wraps the plotter)
    main_window = QASAPMainWindow(plotter)
    
    # Show control panel and create the spectrum plot window FIRST
    plotter.show()
    plotter.plot_spectrum()  # Always create the plot window, even if empty
    plotter.raise_()
    plotter.activateWindow()
    
    # Refresh the View menu after windows are created
    main_window.refresh_view_menu()
    # Keep main window hidden but ensure it's created for menu bar access on macOS
    # The menu bar will appear in the system menu bar on macOS even though the window is hidden
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

