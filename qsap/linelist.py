"""
Line list management for QASAP
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Line:
    """Represents a single spectral line"""
    wave: float   # rest wavelength (Å)
    name: str


@dataclass
class LineList:
    """Represents a collection of spectral lines"""
    name: str
    lines: List[Line]
    color: str = "#1f77b4"  # Default blue
    path: Optional[Path] = None


def load_line_list(path: str) -> LineList:
    """
    Load a line list from a file.
    
    Format: wavelength,name (CSV/TSV format)
    Lines starting with # are treated as comments.
    
    Args:
        path: Path to the line list file
        
    Returns:
        LineList object with loaded lines
    """
    p = Path(path)
    lines: List[Line] = []
    list_name = p.stem  # Filename without extension

    # CSV/TSV: wavelength,name   (third column, if any, is ignored)
    with open(p, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            parts = [t.strip() for t in s.replace("\t", ",").split(",")]
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
            except (ValueError, IndexError):
                continue
            name = parts[1]
            lines.append(Line(wave=w, name=name))

    return LineList(name=list_name, lines=lines, path=p)


def get_available_line_lists(resources_dir: str) -> List[LineList]:
    """
    Get all available line lists from the resources directory.
    
    Args:
        resources_dir: Path to the resources directory (contains linelist/ subdirectory)
        
    Returns:
        List of LineList objects sorted with SDSS lists first
    """
    resource_path = Path(resources_dir) / "linelist"
    sdss_lists = []
    other_lists = []
    
    if not resource_path.exists():
        return []
    
    # Load all .txt files from the linelist directory
    for file_path in sorted(resource_path.glob("*.txt")):
        try:
            line_list = load_line_list(str(file_path))
            # Prioritize SDSS line lists
            if "sdss" in line_list.name.lower():
                sdss_lists.append(line_list)
            else:
                other_lists.append(line_list)
        except Exception as e:
            print(f"Error loading line list {file_path}: {e}")
    
    # Return SDSS lists first, then others
    return sdss_lists + other_lists
