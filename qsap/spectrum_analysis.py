"""
Spectrum analysis module - line fitting, continuum fitting, equivalent width calculations
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz
from scipy.interpolate import interp1d
import lmfit
from lmfit import Model, Parameters, conf_interval, minimize


class SpectrumAnalysis:
    """Handle spectral analysis including fitting and EW calculations"""
    
    @staticmethod
    def gaussian(x, amp, mean, stddev):
        """Simple Gaussian profile"""
        return amp * np.exp(-0.5 * ((x - mean) / stddev) ** 2)
    
    @staticmethod
    def voigt(x, amp, mean, fwhm_g, fwhm_l):
        """Voigt profile"""
        sigma = fwhm_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm_l / 2
        z = ((x - mean) + 1j * gamma) / (sigma * np.sqrt(2))
        return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    
    @staticmethod
    def multi_gaussian(x, *params):
        """Sum of multiple Gaussians"""
        y = np.zeros_like(x, dtype=float)
        for i in range(0, len(params), 3):
            amp, mean, stddev = params[i], params[i+1], params[i+2]
            y += SpectrumAnalysis.gaussian(x, amp, mean, stddev)
        return y
    
    @staticmethod
    def poly_continuum(x, *params):
        """Polynomial continuum"""
        return np.polyval(params, x)
    
    @staticmethod
    def power_law_continuum(x, amp, index):
        """Power-law continuum"""
        return amp * x ** index
    
    @staticmethod
    def fit_gaussian(x, y, guess_params=None, weights=None):
        """
        Fit Gaussian profile to data
        
        Parameters
        ----------
        x : np.ndarray
            Wavelength/position array
        y : np.ndarray
            Flux/intensity array
        guess_params : tuple, optional
            Initial guess (amp, mean, stddev)
        weights : np.ndarray, optional
            Weights for fitting
            
        Returns
        -------
        params : np.ndarray
            Fitted parameters (amp, mean, stddev)
        covariance : np.ndarray
            Covariance matrix
        """
        
        if guess_params is None:
            amp_guess = np.max(y) - np.min(y)
            mean_guess = x[np.argmax(y)]
            stddev_guess = (x[-1] - x[0]) / 10
            guess_params = [amp_guess, mean_guess, stddev_guess]
        
        try:
            params, covariance = curve_fit(
                SpectrumAnalysis.gaussian, x, y,
                p0=guess_params,
                sigma=weights if weights is not None else None,
                absolute_sigma=True,
                maxfev=10000
            )
            return params, covariance
        except Exception as e:
            print(f"Gaussian fitting failed: {e}")
            return None, None
    
    @staticmethod
    def fit_voigt(x, y, guess_params=None, weights=None):
        """Fit Voigt profile to data"""
        
        if guess_params is None:
            amp_guess = np.max(y) - np.min(y)
            mean_guess = x[np.argmax(y)]
            fwhm_g_guess = (x[-1] - x[0]) / 10
            fwhm_l_guess = (x[-1] - x[0]) / 20
            guess_params = [amp_guess, mean_guess, fwhm_g_guess, fwhm_l_guess]
        
        try:
            params, covariance = curve_fit(
                SpectrumAnalysis.voigt, x, y,
                p0=guess_params,
                sigma=weights if weights is not None else None,
                absolute_sigma=True,
                maxfev=10000
            )
            return params, covariance
        except Exception as e:
            print(f"Voigt fitting failed: {e}")
            return None, None
    
    @staticmethod
    def fit_continuum(x, y, order=1, method='poly'):
        """
        Fit continuum to data
        
        Parameters
        ----------
        x : np.ndarray
            Wavelength array
        y : np.ndarray
            Flux array
        order : int
            Polynomial order (default: 1 for linear)
        method : str
            'poly' for polynomial, 'powerlaw' for power-law
            
        Returns
        -------
        params : np.ndarray
            Continuum parameters
        continuum : np.ndarray
            Fitted continuum
        """
        
        if method == 'poly':
            try:
                params = np.polyfit(x, y, order)
                continuum = np.polyval(params, x)
                return params, continuum
            except Exception as e:
                print(f"Continuum fitting failed: {e}")
                return None, None
        
        elif method == 'powerlaw':
            try:
                params, _ = curve_fit(
                    SpectrumAnalysis.power_law_continuum, x, y,
                    p0=[1, -1],
                    maxfev=10000
                )
                continuum = SpectrumAnalysis.power_law_continuum(x, *params)
                return params, continuum
            except Exception as e:
                print(f"Power-law continuum fitting failed: {e}")
                return None, None
    
    @staticmethod
    def calculate_equivalent_width(x, y, continuum, x_bounds):
        """
        Calculate equivalent width
        
        Parameters
        ----------
        x : np.ndarray
            Wavelength array
        y : np.ndarray
            Flux array
        continuum : np.ndarray or float
            Continuum level(s)
        x_bounds : tuple
            (x_min, x_max) for EW region
            
        Returns
        -------
        ew : float
            Equivalent width in Angstroms
        ew_err : float
            Equivalent width uncertainty
        """
        
        mask = (x >= x_bounds[0]) & (x <= x_bounds[1])
        x_region = x[mask]
        y_region = y[mask]
        
        if isinstance(continuum, np.ndarray):
            cont_region = continuum[mask]
        else:
            cont_region = np.ones_like(y_region) * continuum
        
        if len(x_region) < 2:
            return 0.0, 0.0
        
        # EW = integral of (1 - flux/continuum) dx
        normalized_flux = 1 - y_region / cont_region
        ew = np.trapz(normalized_flux, x_region)
        
        # Simple error estimate
        ew_err = np.sqrt(np.sum((0.1 * y_region / cont_region) ** 2)) * (x_region[-1] - x_region[0])
        
        return ew, ew_err
    
    @staticmethod
    def calculate_residuals(x, y, model):
        """Calculate fit residuals"""
        residuals = y - model
        chi_squared = np.sum((residuals ** 2))
        reduced_chi_squared = chi_squared / (len(y) - 3)
        return residuals, chi_squared, reduced_chi_squared
    
    @staticmethod
    def smooth_spectrum(x, y, kernel_width):
        """
        Smooth spectrum with kernel
        
        Parameters
        ----------
        x : np.ndarray
            Wavelength array
        y : np.ndarray
            Flux array
        kernel_width : float
            Kernel width in wavelength units
            
        Returns
        -------
        y_smooth : np.ndarray
            Smoothed flux array
        """
        
        dx = np.median(np.diff(x))
        kernel_size = int(kernel_width / dx)
        
        if kernel_size < 2:
            kernel_size = 2
        
        kernel = np.hanning(kernel_size) / np.sum(np.hanning(kernel_size))
        y_smooth = np.convolve(y, kernel, mode='same')
        
        return y_smooth
    
    # ===== LSF (Line Spread Function) Handling =====
    
    @staticmethod
    def parse_lsf_spec(lsf_spec: str) -> float:
        """
        Parse LSF specification string.
        
        Parameters
        ----------
        lsf_spec : str
            LSF specification. Can be:
            - "10" (FWHM in km/s)
            - "10.5" (float FWHM in km/s)
            - "path/to/lsf_file.txt" (path to LSF data)
            
        Returns
        -------
        fwhm_kms : float
            FWHM in km/s
        """
        try:
            # Try to parse as float (km/s)
            return float(lsf_spec)
        except ValueError:
            # Try to read from file
            try:
                data = np.loadtxt(lsf_spec)
                # Assume first column is FWHM or entire array is FWHM values
                return float(np.mean(data)) if data.ndim > 0 else float(data)
            except Exception as e:
                raise ValueError(f"Could not parse LSF spec '{lsf_spec}': {e}")
    
    @staticmethod
    def apply_lsf(wav, spec, lsf_fwhm_kms: float) -> np.ndarray:
        """
        Apply Line Spread Function (LSF) convolution to spectrum.
        
        Parameters
        ----------
        wav : np.ndarray
            Wavelength array (Angstroms)
        spec : np.ndarray
            Flux array
        lsf_fwhm_kms : float
            LSF FWHM in km/s
            
        Returns
        -------
        spec_lsf : np.ndarray
            LSF-convolved spectrum
        """
        from scipy.constants import c as speed_of_light
        
        # Convert LSF from km/s to wavelength units
        c_ang_s = speed_of_light * 1e10  # Angstroms/s
        fwhm_ang = lsf_fwhm_kms * 1e3 * wav / c_ang_s  # Gaussian FWHM per wavelength
        
        # Average FWHM
        avg_fwhm = np.mean(fwhm_ang)
        
        # Get wavelength spacing
        dw = np.median(np.diff(wav))
        
        # Convert FWHM to sigma
        sigma_pix = (avg_fwhm / dw) / (2 * np.sqrt(2 * np.log(2)))
        
        if sigma_pix < 1:
            return spec  # No smoothing needed
        
        # Gaussian kernel
        kernel_size = int(4 * sigma_pix) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
        kernel = np.exp(-0.5 * (x / sigma_pix) ** 2)
        kernel /= np.sum(kernel)
        
        # Convolve
        spec_lsf = np.convolve(spec, kernel, mode='same')
        
        return spec_lsf
