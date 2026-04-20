"""
QSAP - Quick Spectrum Analysis Program
Version is read from version.txt for centralized version management.
"""

from pathlib import Path

# Read version from version.txt
def _get_version():
    version_file = Path(__file__).parent.parent / 'version.txt'
    try:
        return version_file.read_text().strip()
    except Exception:
        return "0.12"  # fallback

__version__ = _get_version()
__author__ = "Erik Solhaug"

from .spectrum_io import SpectrumIO
from .spectrum_analysis import SpectrumAnalysis
from .ui_components import SpectrumPlotter, SpectrumPlotterApp, LineListWindow

# Import main function from root qsap.py for entry point
import sys
import importlib.util
from pathlib import Path

_qsap_py = Path(__file__).parent.parent / 'qsap.py'
spec = importlib.util.spec_from_file_location("qsap_main", _qsap_py)
qsap_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qsap_module)
main = qsap_module.main

__all__ = [
    'SpectrumIO',
    'SpectrumAnalysis',
    'SpectrumPlotter',
    'SpectrumPlotterApp',
    'LineListWindow',
    'main',
]
