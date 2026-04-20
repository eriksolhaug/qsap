"""
QASAP - Quick & Advanced Spectrum Analysis Package
Version is read from version.txt for centralized version management.
"""

from pathlib import Path

# Read version from version.txt
def _get_version():
    version_file = Path(__file__).parent.parent / 'version.txt'
    try:
        return version_file.read_text().strip()
    except Exception:
        return "0.11"  # fallback

__version__ = _get_version()
__author__ = "Erik Solhaug"

from .spectrum_io import SpectrumIO
from .spectrum_analysis import SpectrumAnalysis
from .ui_components import SpectrumPlotter, SpectrumPlotterApp, LineListWindow

# Import main function from root qasap.py for entry point
import sys
import importlib.util
from pathlib import Path

_qasap_py = Path(__file__).parent.parent / 'qasap.py'
spec = importlib.util.spec_from_file_location("qasap_main", _qasap_py)
qasap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qasap_module)
main = qasap_module.main

__all__ = [
    'SpectrumIO',
    'SpectrumAnalysis',
    'SpectrumPlotter',
    'SpectrumPlotterApp',
    'LineListWindow',
    'main',
]
