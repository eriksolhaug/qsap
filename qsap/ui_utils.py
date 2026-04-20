"""
UI utilities for QSAP windows
"""

from pathlib import Path
from PyQt5.QtGui import QIcon


def get_qsap_icon():
    """Load and return the QSAP logo as a QIcon for use in all windows."""
    # Try multiple possible paths to find the logo
    possible_paths = [
        # When running from package installation
        Path(__file__).parent.parent / 'logo' / 'qsap_logo.png',
        # Alternative path
        Path(__file__).parent / '..' / 'logo' / 'qsap_logo.png',
    ]
    
    for logo_path in possible_paths:
        if logo_path.exists():
            return QIcon(str(logo_path.resolve()))
    
    # If logo not found, return empty icon
    return QIcon()
