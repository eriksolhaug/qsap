"""
Spectrum I/O module - handles reading various spectrum file formats

Features:
- Auto-detect common ASCII and FITS spectrum layouts
- Intelligent format detection with confidence scoring
- Support for 11+ file format variations
- Optional manual format specification via fmt parameter
"""

import numpy as np
from pathlib import Path
from astropy.io import fits
from typing import Tuple, Optional, Dict, Any, List


class SpectrumIO:
    """Handle reading spectrum data from various file formats with auto-detection"""
    
    @staticmethod
    def detect_spectrum_format(filepath: str) -> List[Dict[str, Any]]:
        """
        Inspect file and return candidate format keys with confidence scores.
        
        Returns:
            List of dicts with keys: 
            {"key": "ascii:3col", "score": 90, "notes": "...", "options": {...}}
        """
        path = Path(filepath)
        out: List[Dict[str, Any]] = []
        ext = path.suffix.lower()
        
        def add(key: str, score: int, notes: str, options: Optional[Dict[str, Any]] = None):
            out.append({"key": key, "score": score, "notes": notes, "options": options or {}})
        
        # ASCII files (including no extension)
        if ext in (".txt", ".dat", ".csv", ".tsv", ".sed", ".ascii") or ext == "":
            try:
                delim, ncol = SpectrumIO._peek_ascii_layout(path)
                # If delimiter is None (whitespace) and we detected 3+ columns, 
                # also offer 2-column as an option (most common for spectrum files)
                if delim is None and ncol >= 2:
                    # Try to read with whitespace and check actual column consistency
                    try:
                        test_data = np.genfromtxt(path, comments="#", delimiter=None, max_rows=50)
                        if test_data.ndim == 2:
                            actual_cols = test_data.shape[1]
                            # If actual data has 2 columns, prefer 2-column format
                            if actual_cols == 2:
                                add("ascii:2col", 90, f"2 columns (whitespace-delimited)",
                                    {"delimiter": None})
                            elif actual_cols >= 3:
                                add("ascii:3col", 90, f"{actual_cols} columns (whitespace-delimited)",
                                    {"delimiter": None})
                        else:
                            add("ascii:2col", 85, f"2 columns (detected)",
                                {"delimiter": None})
                    except:
                        add("ascii:2col", 85, f"2 columns (likely whitespace-delimited)",
                            {"delimiter": None})
                elif ncol >= 3:
                    add("ascii:3col", 90, f"{ncol} columns (delimiter={repr(delim)})",
                        {"delimiter": delim})
                elif ncol == 2:
                    add("ascii:2col", 85, f"2 columns (delimiter={repr(delim)})",
                        {"delimiter": delim})
                else:
                    add("ascii:flex", 60, f"{ncol} columns (needs mapping)",
                        {"delimiter": delim})
            except Exception as e:
                add("ascii:flex", 40, f"ASCII read error: {e!s} (manual mapping likely)")
        
        # FITS files
        if ext in (".fits", ".fit", ".fts"):
            try:
                with fits.open(path, memmap=True) as hdul:
                    # Primary 1D image?
                    if len(hdul) > 0 and getattr(hdul[0], "data", None) is not None:
                        data = hdul[0].data
                        if isinstance(data, np.ndarray) and data.ndim == 1:
                            add("fits:image1d", 95, "Primary HDU contains 1D image", {"hdu": 0})
                    
                    # Check extensions
                    for i, h in enumerate(hdul):
                        if not hasattr(h, "data") or h.data is None:
                            continue
                        
                        if isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                            cols = list(h.columns.names or [])
                            lc = [c.lower() for c in cols]
                            
                            # Named SPECTRUM extension?
                            extname = (h.header.get("EXTNAME") or "").upper()
                            if extname == "SPECTRUM":
                                add("fits:ext:spectrum", 96, "HDU 'SPECTRUM'",
                                    {"extname": "SPECTRUM"})
                            
                            # Have wave/flux?
                            has_wave = any(k in lc for k in ("wave", "lambda", "wavelength"))
                            has_flux = any(k in lc for k in ("flux",))
                            has_errivar = any(k in lc for k in ("err", "ivar"))
                            
                            if has_wave and has_flux and has_errivar:
                                try:
                                    rec0 = h.data[0]
                                    wname = SpectrumIO._pick_name(["WAVE", "wave", "lambda"], cols)
                                    fname = SpectrumIO._pick_name(["FLUX", "flux"], cols)
                                    arrish = (hasattr(rec0[wname], "__len__") and 
                                             hasattr(rec0[fname], "__len__"))
                                except Exception:
                                    arrish = False
                                
                                if arrish:
                                    add("fits:table:vector", 92,
                                        f"Table HDU[{i}] with vector row arrays", {"hdu": i})
                                else:
                                    add("fits:table:columns", 88,
                                        f"Table HDU[{i}] with per-pixel columns", {"hdu": i})
            except Exception as e:
                add("fits:image1d", 30, f"FITS open error: {e!s}")
        
        return out
    
    @staticmethod
    def read_spectrum(filepath: str, fmt: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
        """
        Load a 1-D spectrum from ASCII/FITS layouts.
        
        Parameters
        ----------
        filepath : str
            Path to spectrum file
        fmt : str, optional
            Forced format key. If None, auto-detects
        options : dict, optional
            Reader options (delimiter, colmap, hdu, etc.)
            
        Returns
        -------
        wav : np.ndarray
            Wavelength array
        spec : np.ndarray
            Flux array
        err : np.ndarray
            Error/uncertainty array
        meta : dict
            Metadata including format, source, units
        """
        
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(str(filepath))
        
        options = options or {}
        
        # Auto-detect if fmt not provided
        if fmt is None:
            candidates = SpectrumIO.detect_spectrum_format(str(filepath))
            if not candidates:
                raise ValueError(f"Could not determine format for: {filepath.name}")
            # Take highest-confidence candidate
            candidates.sort(key=lambda c: c["score"], reverse=True)
            best = candidates[0]
            fmt = best["key"]
            # Merge suggested options
            options = {**best.get("options", {}), **options}
        
        # Dispatch to readers
        if fmt == "ascii:2col":
            return SpectrumIO._read_ascii(filepath, delimiter=options.get("delimiter"),
                                         colmap={"wave": 0, "flux": 1, "err": None},
                                         units=options.get("units"),
                                         wave_unit=options.get("wave_unit", 1.0))
        
        elif fmt == "ascii:3col":
            return SpectrumIO._read_ascii(filepath, delimiter=options.get("delimiter"),
                                         colmap={"wave": 0, "flux": 1, "err": 2},
                                         units=options.get("units"),
                                         wave_unit=options.get("wave_unit", 1.0))
        
        elif fmt == "ascii:flex":
            cm = options.get("colmap", {"wave": 0, "flux": 1, "err": None})
            # If no delimiter specified in options, auto-detect it
            delim = options.get("delimiter")
            if delim is None:
                delim, _ = SpectrumIO._peek_ascii_layout(filepath)
            return SpectrumIO._read_ascii(filepath, delimiter=delim,
                                         colmap=cm, units=options.get("units"),
                                         wave_unit=options.get("wave_unit", 1.0))
        
        elif fmt == "fits:image1d":
            return SpectrumIO._read_fits_image1d(filepath, hdu=options.get("hdu", 0))
        
        elif fmt == "fits:table:vector":
            return SpectrumIO._read_fits_table_vector(
                filepath,
                hdu=options.get("hdu", 1),
                wave_cols=options.get("wave_cols", ["WAVE", "wave", "lambda"]),
                flux_cols=options.get("flux_cols", ["FLUX", "flux"]),
                err_cols=options.get("err_cols", ["ERR", "err"]),
                ivar_cols=options.get("ivar_cols", ["IVAR", "ivar"]),
                row=options.get("row", 0)
            )
        
        elif fmt == "fits:table:columns":
            return SpectrumIO._read_fits_table_columns(
                filepath,
                hdu=options.get("hdu", 1),
                wave_cols=options.get("wave_cols", ["WAVE", "wave", "lambda"]),
                flux_cols=options.get("flux_cols", ["FLUX", "flux"]),
                err_cols=options.get("err_cols", ["ERR", "err"]),
                ivar_cols=options.get("ivar_cols", ["IVAR", "ivar"])
            )
        
        elif fmt == "fits:ext:spectrum":
            return SpectrumIO._read_fits_named_spectrum(filepath, 
                                                       extname=options.get("extname", "SPECTRUM"))
        
        else:
            raise ValueError(f"Unsupported format key: {fmt}")
    
    # ===== ASCII readers =====
    
    @staticmethod
    def _peek_ascii_layout(path: Path, max_lines: int = 200) -> Tuple[str, int]:
        """Return (delimiter, ncol) by scanning first few non-comment lines."""
        trials = [("\t", "tab"), (None, "whitespace"), (",", "comma")]
        rows: List[str] = []
        
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                rows.append(s)
                if len(rows) >= 5:
                    break
        
        if not rows:
            return (None, 0)  # Default to whitespace if no rows
        
        best_delim = None
        best_ncol = 0
        best_score = -1
        
        for delim, _name in trials:
            if delim is None:
                # Use None for whitespace splitting (handles multiple spaces/tabs)
                counts = [len(r.split()) for r in rows]
            else:
                counts = [len(r.split(delim)) for r in rows]
            ncol = max(counts) if counts else 0
            
            if ncol == 0:
                continue
            
            # Score based on: consistent column count + reasonable number of columns
            # Consistency: all rows have same column count
            consistency = 1.0 if len(set(counts)) == 1 else 0.5
            # Reasonableness: prefer 1-5 columns for spectrum data
            if 1 <= ncol <= 5:
                reasonableness = 1.0
            elif ncol > 5:
                reasonableness = 0.1  # Very unlikely
            else:
                reasonableness = 0.5
            
            score = consistency * reasonableness * ncol
            
            if score > best_score:
                best_score = score
                best_ncol = ncol
                best_delim = delim
        
        return (best_delim, best_ncol)
    
    @staticmethod
    def _read_ascii(path: Path, delimiter: Optional[str],
                   colmap: Dict[str, Optional[int]],
                   units: Optional[str] = None,
                   wave_unit: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Generic ASCII reader (2 or 3 columns typical).
        
        Parameters
        ----------
        wave_unit : float
            Wavelength unit conversion factor relative to Angstroms.
            Default 1.0 (wavelength already in Angstroms).
            Examples: 10 for nm, 10000 for microns
        """
        # Use filling_values=np.nan for explicit delimiters (tab, comma)
        # but NOT for None (whitespace) which has different behavior
        if delimiter is None:
            # For whitespace-delimited files, don't use filling_values
            data = np.genfromtxt(path, comments="#", delimiter=None)
        else:
            # For explicit delimiters, use filling_values to handle missing columns
            data = np.genfromtxt(path, comments="#", delimiter=delimiter, filling_values=np.nan)
        
        if data.ndim == 1:
            data = data[None, :]
        ncol = data.shape[1]
        
        def pick(idx: Optional[int], default=None):
            if idx is None:
                return default
            if idx < 0 or idx >= ncol:
                raise ValueError(f"Requested column {idx} but file has {ncol} columns.")
            return data[:, idx]
        
        wav = np.asarray(pick(colmap.get("wave")))
        flux = np.asarray(pick(colmap.get("flux")))
        err = pick(colmap.get("err"))
        
        if wav is None or flux is None:
            raise ValueError("ASCII reader requires at least wave & flux columns.")
        
        # Apply wavelength unit conversion
        if wave_unit != 1.0:
            wav = wav * wave_unit
        
        # Legacy unit string conversion (backward compatibility)
        if units:
            u = str(units).lower()
            if u in ("nm", "nanometer", "nanometers"):
                wav = wav * 10.0  # nm -> Å
            elif u in ("um", "micron", "microns", "µm", "μm"):
                wav = wav * 1e4  # µm -> Å
        
        if err is not None:
            err = np.asarray(err)
        # else: err stays None - no automatic error creation
        
        wave_unit_str = {"a": "Å", "angstrom": "Å", "nm": "nm", "um": "µm", "μm": "µm"}.get(
            (str(units).lower() if units else "a"), "Å")
        
        meta = {
            "source": "ascii",
            "path": str(path),
            "delimiter": delimiter,
            "colmap": colmap,
            "units": units,
            "wave_unit": wave_unit_str,
            "flux_unit": "adu",
            "ncol": ncol,
        }
        return wav, flux, err, meta
    
    # ===== FITS readers =====
    
    @staticmethod
    def _pick_name(preferred: List[str], available: List[str]) -> str:
        """Pick first name in preferred that exists in available (case-insensitive)."""
        aset = {a: a for a in available}
        lower = {a.lower(): a for a in available}
        for p in preferred:
            if p in aset:
                return p
            pl = p.lower()
            if pl in lower:
                return lower[pl]
        raise KeyError(f"None of {preferred} found in: {available}")
    
    @staticmethod
    def _read_fits_image1d(path: Path, hdu: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Read 1D image from FITS HDU."""
        with fits.open(path, memmap=True) as hdul:
            h = hdul[hdu]
            spec = np.asarray(h.data, dtype=float).flatten()
            
            # Get wavelength from header
            header = h.header
            crpix1 = header.get('CRPIX1', 1)
            crval1 = header.get('CRVAL1', 0)
            cdelt1 = header.get('CDELT1', 1)
            wav = crval1 + (np.arange(len(spec)) - (crpix1 - 1)) * cdelt1
            
            # No automatic error creation - leave as None
            err = None
            
            meta = {
                "source": "fits:image1d",
                "path": str(path),
                "hdu": hdu,
                "wave_unit": "Å",
                "flux_unit": "adu",
            }
            return wav, spec, err, meta
    
    @staticmethod
    def _read_fits_table_vector(path: Path, hdu: int = 1,
                               wave_cols: List[str] = None,
                               flux_cols: List[str] = None,
                               err_cols: List[str] = None,
                               ivar_cols: List[str] = None,
                               row: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Read FITS table where wave/flux are vector arrays in a single row."""
        if wave_cols is None:
            wave_cols = ["WAVE", "wave", "lambda"]
        if flux_cols is None:
            flux_cols = ["FLUX", "flux"]
        if err_cols is None:
            err_cols = ["ERR", "err"]
        if ivar_cols is None:
            ivar_cols = ["IVAR", "ivar"]
        
        with fits.open(path, memmap=True) as hdul:
            h = hdul[hdu]
            data = h.data
            cols = list(h.columns.names or [])
            
            wname = SpectrumIO._pick_name(wave_cols, cols)
            fname = SpectrumIO._pick_name(flux_cols, cols)
            
            rec = data[row]
            wav = np.asarray(rec[wname], dtype=float)
            flux = np.asarray(rec[fname], dtype=float)
            
            # Try error or ivar
            try:
                ename = SpectrumIO._pick_name(err_cols, cols)
                err = np.asarray(rec[ename], dtype=float)
            except Exception:
                try:
                    iname = SpectrumIO._pick_name(ivar_cols, cols)
                    ivar = np.asarray(rec[iname], dtype=float)
                    err = np.sqrt(1.0 / np.where(ivar > 0, ivar, 1e-10))
                except Exception:
                    err = None  # No error or ivar available
            
            meta = {
                "source": "fits:table:vector",
                "path": str(path),
                "hdu": hdu,
                "wave_unit": "Å",
                "flux_unit": "adu",
            }
            return wav, flux, err, meta
    
    @staticmethod
    def _read_fits_table_columns(path: Path, hdu: int = 1,
                                wave_cols: List[str] = None,
                                flux_cols: List[str] = None,
                                err_cols: List[str] = None,
                                ivar_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Read FITS table where each spectrum pixel is a separate row."""
        if wave_cols is None:
            wave_cols = ["WAVE", "wave", "lambda"]
        if flux_cols is None:
            flux_cols = ["FLUX", "flux"]
        if err_cols is None:
            err_cols = ["ERR", "err"]
        if ivar_cols is None:
            ivar_cols = ["IVAR", "ivar"]
        
        with fits.open(path, memmap=True) as hdul:
            h = hdul[hdu]
            data = h.data
            cols = list(h.columns.names or [])
            
            wname = SpectrumIO._pick_name(wave_cols, cols)
            fname = SpectrumIO._pick_name(flux_cols, cols)
            
            wav = np.asarray(data[wname], dtype=float)
            flux = np.asarray(data[fname], dtype=float)
            
            # Try error or ivar
            try:
                ename = SpectrumIO._pick_name(err_cols, cols)
                err = np.asarray(data[ename], dtype=float)
            except Exception:
                try:
                    iname = SpectrumIO._pick_name(ivar_cols, cols)
                    ivar = np.asarray(data[iname], dtype=float)
                    err = np.sqrt(1.0 / np.where(ivar > 0, ivar, 1e-10))
                except Exception:
                    err = None  # No error or ivar available
            
            meta = {
                "source": "fits:table:columns",
                "path": str(path),
                "hdu": hdu,
                "wave_unit": "Å",
                "flux_unit": "adu",
            }
            return wav, flux, err, meta
    
    @staticmethod
    def _read_fits_named_spectrum(path: Path, extname: str = "SPECTRUM") -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Read FITS SPECTRUM extension."""
        with fits.open(path, memmap=True) as hdul:
            h = hdul[extname]
            data = h.data
            if data is None:
                raise ValueError(f"HDU '{extname}' has no data.")
            
            cols = list(data.columns.names or [])
            
            try:
                wname = SpectrumIO._pick_name(["WAVE", "wave", "lambda"], cols)
                fname = SpectrumIO._pick_name(["FLUX", "flux"], cols)
                
                rec0 = data[0]
                wav = np.asarray(rec0[wname], dtype=float)
                flux = np.asarray(rec0[fname], dtype=float)
                
                # Try error or ivar
                try:
                    ename = SpectrumIO._pick_name(["ERR", "err"], cols)
                    err = np.asarray(rec0[ename], dtype=float)
                except Exception:
                    try:
                        iname = SpectrumIO._pick_name(["IVAR", "ivar"], cols)
                        ivar = np.asarray(rec0[iname], dtype=float)
                        err = np.sqrt(1.0 / np.where(ivar > 0, ivar, 1e-10))
                    except Exception:
                        err = None  # No error or ivar available
                
                meta = {
                    "source": "fits:ext:spectrum",
                    "path": str(path),
                    "extname": extname,
                    "wave_unit": "Å",
                    "flux_unit": "adu",
                }
                return wav, flux, err, meta
            
            except Exception as e:
                raise ValueError(f"Could not read SPECTRUM extension: {e}")
    
    # ===== Utility readers =====
    
    @staticmethod
    def read_lines(filename='emlines.txt') -> Tuple[np.ndarray, np.ndarray]:
        """Read emission line catalog."""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            return data[:, 1].astype(float), data[:, 0]
        except Exception as e:
            print(f"Error reading line catalog: {e}")
            return np.array([]), np.array([])
    
    @staticmethod
    def read_oscillator_strengths(filename='emlines_osc.txt') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read emission lines with oscillator strengths."""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            return data[:, 1].astype(float), data[:, 0], data[:, 2].astype(float)
        except Exception as e:
            print(f"Error reading oscillator strengths: {e}")
            return np.array([]), np.array([]), np.array([])
    
    @staticmethod
    def read_instrument_bands(filename='instrument_bands.txt') -> Tuple[List[Tuple], np.ndarray]:
        """Read instrument filter definitions."""
        try:
            data = np.genfromtxt(filename, dtype=str, delimiter=',')
            band_ranges = list(zip(data[:, 1].astype(float), data[:, 2].astype(float)))
            return band_ranges, data[:, 0]
        except Exception as e:
            print(f"Error reading instrument bands: {e}")
            return [], []
    
    # ===== ALFOSC 2D → 1D Extraction =====
    
    @staticmethod
    def extract_1d_from_2d_alfosc(
        fits_file: str, 
        bin_width: int = 10,
        left_bound: Optional[int] = None,
        right_bound: Optional[int] = None,
        output_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Extract 1D spectrum from 2D ALFOSC spectroscopic data.
        
        Parameters:
        -----------
        fits_file : str
            Path to 2D FITS file
        bin_width : int
            Bin width in pixels for wavelength binning (default: 10)
        left_bound : int, optional
            Left x-pixel bound for extraction
        right_bound : int, optional
            Right x-pixel bound for extraction
        output_file : str, optional
            If provided, save extracted spectrum to this file
            
        Returns:
        --------
        wavelength : np.ndarray
            1D wavelength array
        flux : np.ndarray
            1D flux array (spatially summed)
        output_path : str
            Path where spectrum was saved (or empty string if not saved)
        """
        try:
            with fits.open(fits_file) as hdul:
                # Get primary HDU data
                data_2d = hdul[0].data
                header = hdul[0].header
                
                if data_2d is None:
                    raise ValueError("No data in primary HDU")
                
                # Get wavelength calibration from header
                naxis1 = header.get('NAXIS1', data_2d.shape[1])
                crpix1 = header.get('CRPIX1', 1)
                crval1 = header.get('CRVAL1', 0)
                cdelt1 = header.get('CDELT1', 1)
                
                # Create wavelength array
                wav = crval1 + (np.arange(naxis1) - crpix1 + 1) * cdelt1
                
                # Apply bounds
                if left_bound is not None or right_bound is not None:
                    left = left_bound or 0
                    right = right_bound or naxis1
                    data_2d = data_2d[:, left:right]
                    wav = wav[left:right]
                
                # Sum spatially and bin
                flux_1d = np.sum(data_2d, axis=0)
                
                if bin_width > 1:
                    # Rebin wavelength and flux
                    n_bins = len(wav) // bin_width
                    wav_binned = np.zeros(n_bins)
                    flux_binned = np.zeros(n_bins)
                    
                    for i in range(n_bins):
                        start = i * bin_width
                        end = (i + 1) * bin_width
                        wav_binned[i] = np.mean(wav[start:end])
                        flux_binned[i] = np.mean(flux_1d[start:end])
                    
                    wav, flux_1d = wav_binned, flux_binned
                
                # Save to file if requested
                output_path = ""
                if output_file:
                    # Create FITS file
                    hdu = fits.PrimaryHDU()
                    hdu.data = flux_1d
                    hdu.header['CTYPE1'] = 'WAVE'
                    hdu.header['CRPIX1'] = 1
                    hdu.header['CRVAL1'] = wav[0]
                    hdu.header['CDELT1'] = wav[1] - wav[0] if len(wav) > 1 else 1
                    hdu.header['CUNIT1'] = 'Angstrom'
                    hdu.header['EXTRACTED_FROM'] = fits_file
                    hdu.header['BIN_WIDTH'] = bin_width
                    
                    hdu.writeto(output_file, overwrite=True)
                    output_path = output_file
                    print(f"Saved extracted 1D spectrum to: {output_file}")
                
                return wav, flux_1d, output_path
                
        except Exception as e:
            raise ValueError(f"Error extracting 1D from ALFOSC 2D: {e}")
