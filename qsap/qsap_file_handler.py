"""
QSAP File Handler - Unified .qsap file format for all fit types

Handles creation, parsing, and management of QSAP files which store fit results
in a human-readable, structured text format.
"""

import os
from datetime import datetime
import json
import numpy as np


class QSAPFileHandler:
    """Manages .qsap file creation and parsing for unified fit storage"""
    
    FILE_FORMAT_VERSION = "1.3"
    
    def __init__(self, save_directory=None):
        self.save_directory = save_directory or os.path.expanduser("~/QSAP_fits")
        os.makedirs(self.save_directory, exist_ok=True)
    
    def generate_filename(self, fit_type, fit_mode, spectrum_filename):
        """Generate standardized .qsap filename
        
        Args:
            fit_type: 'Gaussian', 'Voigt', 'Continuum', 'Listfit', or 'Redshift'
            fit_mode: 'Single', 'Multi-Gaussian', 'Listfit', etc.
            spectrum_filename: Base name of the spectrum file
            
        Returns:
            Full path to the .qsap file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spectrum_base = os.path.splitext(os.path.basename(spectrum_filename))[0]
        
        fit_type_lower = fit_type.lower()
        fit_mode_lower = fit_mode.lower().replace(' ', '-').replace('multi-gaussian', 'multi')
        
        filename = f"fit_{timestamp}_{fit_type_lower}_{fit_mode_lower}_{spectrum_base}.qsap"
        return os.path.join(self.save_directory, filename)
    
    def create_gaussian_qsap(self, fit_dict, spectrum_filename, fit_mode='Single',
                             spectrum_info=None):
        """Create a .qsap file for Gaussian fit(s)
        
        Args:
            fit_dict: Single dict or list of dicts with gaussian parameters
            spectrum_filename: Path to spectrum file
            fit_mode: 'Single' or 'Multi-Gaussian'
            spectrum_info: Dict with spectrum metadata
            
        Returns:
            Path to created file, content as string
        """
        filepath = self.generate_filename('Gaussian', fit_mode, spectrum_filename)
        
        # Ensure fit_dict is a list
        if not isinstance(fit_dict, list):
            fit_dict = [fit_dict]
        
        content = self._build_header('Gaussian', fit_mode, spectrum_filename, spectrum_info)
        
        for idx, fit in enumerate(fit_dict, 1):
            content += self._build_gaussian_component(fit, idx)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def create_voigt_qsap(self, fit_dict, spectrum_filename, fit_mode='Single',
                          spectrum_info=None):
        """Create a .qsap file for Voigt fit(s)"""
        filepath = self.generate_filename('Voigt', fit_mode, spectrum_filename)
        
        # Ensure fit_dict is a list
        if not isinstance(fit_dict, list):
            fit_dict = [fit_dict]
        
        content = self._build_header('Voigt', fit_mode, spectrum_filename, spectrum_info)
        
        for idx, fit in enumerate(fit_dict, 1):
            content += self._build_voigt_component(fit, idx)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def create_continuum_qsap(self, fit_dict, spectrum_filename, spectrum_info=None):
        """Create a .qsap file for continuum fit(s)"""
        filepath = self.generate_filename('Continuum', 'Single', spectrum_filename)
        
        # Ensure fit_dict is a list
        if not isinstance(fit_dict, list):
            fit_dict = [fit_dict]
        
        content = self._build_header('Continuum', 'Single', spectrum_filename, spectrum_info)
        
        for idx, fit in enumerate(fit_dict, 1):
            content += self._build_continuum_component(fit, idx)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def create_listfit_qsap(self, fit_list, spectrum_filename, spectrum_info=None, lmfit_result=None):
        """Create a .qsap file for listfit (contains multiple profiles and masks)
        
        Args:
            fit_list: List of fit dicts (gaussians, voigts, polynomials, polynomial_guess_mask, data_mask, fit_diagnostics)
            spectrum_filename: Path to spectrum file
            spectrum_info: Dict with spectrum metadata
            lmfit_result: Optional lmfit result object for extracting covariance and tie info
            
        Returns:
            Path to created file, content as string
        """
        filepath = self.generate_filename('Listfit', 'Listfit', spectrum_filename)
        
        content = self._build_header('Listfit', 'Listfit', spectrum_filename, spectrum_info)
        
        # Track type-specific counters for symbols (g0, g1, v0, p0, c0, Z1, Z2, etc.)
        type_counters = {'gaussian': 0, 'voigt': 0, 'polynomial': 0, 'chebyshev': 0, 'redshift': 0}
        
        for idx, fit in enumerate(fit_list, 1):
            fit_type = fit.get('type', 'gaussian').lower()
            
            if fit_type == 'fit_diagnostics':
                # Add fit diagnostics section
                content += "[FIT_DIAGNOSTICS]\n"
                content += f"SSR={self._format_value(fit.get('ssr'))}\n"
                content += f"SSR_NU={self._format_value(fit.get('ssr_nu'))}\n"
                if fit.get('chi2') is not None:
                    content += f"CHI2={self._format_value(fit.get('chi2'))}\n"
                    content += f"CHI2_REDUCED={self._format_value(fit.get('chi2_reduced'))}\n"
                content += f"AKAIKE_INFO_CRITERION={self._format_value(fit.get('akaike_info_criterion'))}\n"
                content += f"BAYESIAN_INFO_CRITERION={self._format_value(fit.get('bayesian_info_criterion'))}\n"
                if fit.get('r_squared') is not None:
                    content += f"R_SQUARED={self._format_value(fit.get('r_squared'))}\n"
                content += f"N_DATA_POINTS={fit.get('n_data_points')}\n"
                content += f"N_PARAMETERS={fit.get('n_parameters')}\n"
                content += f"N_DEGREES_FREEDOM={fit.get('n_degrees_freedom')}\n"
                content += f"FIT_SUCCESS={fit.get('fit_success')}\n"
                content += "\n"
            elif fit_type == 'gaussian':
                symbol = f"g{type_counters['gaussian']}"
                content += self._build_gaussian_component(fit, idx, symbol)
                type_counters['gaussian'] += 1
            elif fit_type == 'voigt':
                symbol = f"v{type_counters['voigt']}"
                content += self._build_voigt_component(fit, idx, symbol)
                type_counters['voigt'] += 1
            elif fit_type == 'polynomial':
                symbol = f"p{type_counters['polynomial']}"
                content += self._build_polynomial_component(fit, idx, symbol)
                type_counters['polynomial'] += 1
            elif fit_type == 'chebyshev':
                symbol = f"c{type_counters['chebyshev']}"
                content += self._build_chebyshev_component(fit, idx, symbol)
                type_counters['chebyshev'] += 1
            elif fit_type == 'redshift':
                symbol = f"Z{type_counters['redshift'] + 1}"
                content += self._build_redshift_component(fit, idx, symbol)
                type_counters['redshift'] += 1
            elif fit_type == 'polynomial_guess_mask':
                content += self._build_polynomial_guess_mask_component(fit, idx)
            elif fit_type == 'data_mask':
                content += self._build_data_mask_component(fit, idx)
            elif fit_type == 'constraints':
                content += self._build_constraints_block(fit)
        
        # Add covariance matrix and tie information if lmfit result is available
        if lmfit_result is not None:
            content += self._build_covariance_block(lmfit_result)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def create_redshift_qsap(self, redshift_data, spectrum_filename, parent_fit_id=None,
                             parent_component_id=None):
        """Create a .qsap file for redshift estimation
        
        Args:
            redshift_data: Dict with redshift parameters
            spectrum_filename: Path to spectrum file
            parent_fit_id: ID of parent gaussian fit
            parent_component_id: Component ID of parent fit
            
        Returns:
            Path to created file, content as string
        """
        filepath = self.generate_filename('Redshift', 'Single', spectrum_filename)
        
        content = "[METADATA]\n"
        content += f"FILE_FORMAT_VERSION={self.FILE_FORMAT_VERSION}\n"
        content += f"TYPE=Redshift\n"
        content += f"SPECTRUM_FILE={os.path.basename(spectrum_filename)}\n"
        if parent_fit_id:
            content += f"PARENT_FIT_ID={parent_fit_id}\n"
        if parent_component_id:
            content += f"PARENT_COMPONENT_ID={parent_component_id}\n"
        content += f"DATE_TIME={datetime.now().isoformat()}\n"
        content += "\n" + "[REDSHIFT_DATA]\n"
        
        # Define the order of keys for consistent output
        # Basic parameters first
        basic_keys = ['REDSHIFT', 'LINE_ID', 'LINE_WAVELENGTH_REST', 'LINE_WAVELENGTH_OBSERVED', 
                      'LINE_WAVELENGTH_OBSERVED_ERR', 'RADIAL_VELOCITY', 'HELIOCENTRIC_VELOCITY', 
                      'SYSTEMIC_VELOCITY', 'ERROR_REDSHIFT', 'ERROR_VELOCITY', 'METHOD']
        # MC parameters (will be present if MC method was used)
        mc_keys = ['REDSHIFT_BEST', 'REDSHIFT_MEDIAN', 'REDSHIFT_MEAN', 
                   'REDSHIFT_1SIGMA', 'REDSHIFT_2SIGMA', 'REDSHIFT_3SIGMA']
        
        # Write basic parameters first
        for key in basic_keys:
            if key.lower() in redshift_data or key in redshift_data:
                # Handle both lowercase and uppercase keys
                actual_key = key.lower() if key.lower() in redshift_data else key
                value = redshift_data[actual_key]
                content += f"{key}={self._format_value(value)}\n"
        
        # Then write MC parameters if present
        for key in mc_keys:
            if key.lower() in redshift_data or key in redshift_data:
                actual_key = key.lower() if key.lower() in redshift_data else key
                value = redshift_data[actual_key]
                content += f"{key}={self._format_value(value)}\n"
        
        # Write any remaining keys not in the predefined lists
        for key, value in redshift_data.items():
            if key.upper() not in basic_keys and key.upper() not in mc_keys:
                if key != 'type':  # Skip type indicator
                    content += f"{key.upper()}={self._format_value(value)}\n"
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def create_equivalent_width_qsap(self, ew_result, fit_dict, fit_type, spectrum_filename, spectrum_info=None):
        """Create a .qsap file for Equivalent Width calculation results
        
        Args:
            ew_result: Dict with EW calculation results from _calculate_equivalent_width_monte_carlo
            fit_dict: Dict with fitted profile parameters
            fit_type: 'Gaussian' or 'Voigt'
            spectrum_filename: Path to spectrum file
            spectrum_info: Dict with spectrum metadata
            
        Returns:
            Path to created file, content as string
        """
        filepath = self.generate_filename('EquivalentWidth', 'Single', spectrum_filename)
        
        content = "[METADATA]\n"
        content += f"FILE_FORMAT_VERSION={self.FILE_FORMAT_VERSION}\n"
        content += f"TYPE=EquivalentWidth\n"
        content += f"PROFILE_TYPE={fit_type}\n"
        content += f"SPECTRUM_FILE={os.path.basename(spectrum_filename)}\n"
        content += f"DATE_TIME={datetime.now().isoformat()}\n"
        
        if spectrum_info:
            if 'wavelength_unit' in spectrum_info:
                content += f"WAVELENGTH_UNIT={spectrum_info['wavelength_unit']}\n"
            if 'wavelength_range' in spectrum_info:
                wav_range = spectrum_info['wavelength_range']
                content += f"WAVELENGTH_RANGE={wav_range[0]:.4f}-{wav_range[1]:.4f}\n"
        
        content += "\n[EQUIVALENT_WIDTH]\n"
        
        # EW results in CAPS with separate lines for credible intervals
        if 'ew_best' in ew_result:
            content += f"EQUIVALENT_WIDTH_BEST={ew_result['ew_best']:.6f}\n"
        if 'ew_median' in ew_result:
            content += f"EQUIVALENT_WIDTH_MEDIAN={ew_result['ew_median']:.6f}\n"
        if 'ew_mean' in ew_result:
            content += f"EQUIVALENT_WIDTH_MEAN={ew_result['ew_mean']:.6f}\n"
        
        # Credible intervals on separate lines
        if 'ew_1sigma_lower' in ew_result and 'ew_1sigma_upper' in ew_result:
            lower = ew_result['ew_1sigma_lower']
            upper = ew_result['ew_1sigma_upper']
            content += f"EQUIVALENT_WIDTH_1SIGMA=-{abs(lower):.6f},+{upper:.6f}\n"
        
        if 'ew_2sigma_lower' in ew_result and 'ew_2sigma_upper' in ew_result:
            lower = ew_result['ew_2sigma_lower']
            upper = ew_result['ew_2sigma_upper']
            content += f"EQUIVALENT_WIDTH_2SIGMA=-{abs(lower):.6f},+{upper:.6f}\n"
        
        if 'ew_3sigma_lower' in ew_result and 'ew_3sigma_upper' in ew_result:
            lower = ew_result['ew_3sigma_lower']
            upper = ew_result['ew_3sigma_upper']
            content += f"EQUIVALENT_WIDTH_3SIGMA=-{abs(lower):.6f},+{upper:.6f}\n"
        
        # Profile parameters that were used for calculation
        content += "\n[PROFILE_PARAMETERS]\n"
        fit_type_lower = fit_type.lower()
        
        if fit_type_lower == 'gaussian':
            if 'amp' in fit_dict:
                content += f"AMPLITUDE={fit_dict['amp']:.6f}\n"
            if 'mean' in fit_dict:
                content += f"MEAN={fit_dict['mean']:.6f}\n"
            if 'stddev' in fit_dict:
                content += f"STDDEV={fit_dict['stddev']:.6f}\n"
            if 'bounds' in fit_dict:
                content += f"BOUNDS={fit_dict['bounds'][0]:.6f}-{fit_dict['bounds'][1]:.6f}\n"
        elif fit_type_lower == 'voigt':
            if 'amplitude' in fit_dict:
                content += f"AMPLITUDE={fit_dict['amplitude']:.6f}\n"
            if 'center' in fit_dict:
                content += f"CENTER={fit_dict['center']:.6f}\n"
            elif 'mean' in fit_dict:
                content += f"CENTER={fit_dict['mean']:.6f}\n"
            if 'sigma' in fit_dict:
                content += f"SIGMA={fit_dict['sigma']:.6f}\n"
            if 'gamma' in fit_dict:
                content += f"GAMMA={fit_dict['gamma']:.6f}\n"
            if 'bounds' in fit_dict:
                content += f"BOUNDS={fit_dict['bounds'][0]:.6f}-{fit_dict['bounds'][1]:.6f}\n"
        
        # Quality metrics if available
        if fit_dict.get('chi2') is not None:
            content += f"\n[FIT_QUALITY]\n"
            content += f"CHI_SQUARED={fit_dict.get('chi2'):.6f}\n"
            if fit_dict.get('chi2_nu'):
                content += f"CHI_SQUARED_NU={fit_dict.get('chi2_nu'):.6f}\n"
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        return filepath, content
    
    def _build_header(self, fit_type, fit_mode, spectrum_filename, spectrum_info=None):
        """Build QSAP file header with metadata"""
        content = "[METADATA]\n"
        content += f"FILE_FORMAT_VERSION={self.FILE_FORMAT_VERSION}\n"
        content += f"TYPE={fit_type}\n"
        content += f"MODE={fit_mode}\n"
        content += f"SPECTRUM_FILE={os.path.basename(spectrum_filename)}\n"
        content += f"DATE_TIME={datetime.now().isoformat()}\n"
        
        if spectrum_info:
            if 'wavelength_unit' in spectrum_info:
                content += f"WAVELENGTH_UNIT={spectrum_info['wavelength_unit']}\n"
            if 'wavelength_range' in spectrum_info:
                wav_range = spectrum_info['wavelength_range']
                content += f"WAVELENGTH_RANGE={wav_range[0]:.4f}-{wav_range[1]:.4f}\n"
            if 'rest_wavelength' in spectrum_info and spectrum_info['rest_wavelength']:
                content += f"REST_WAVELENGTH={spectrum_info['rest_wavelength']}\n"
            if 'velocity_mode' in spectrum_info:
                content += f"VELOCITY_MODE={spectrum_info['velocity_mode']}\n"
            if 'scale_factor' in spectrum_info:
                content += f"SCALE_FACTOR={spectrum_info['scale_factor']}\n"
        
        content += "\n"
        return content
    
    def _build_gaussian_component(self, fit, component_num, symbol=None):
        """Build a single Gaussian component section
        
        Args:
            fit: Fit dictionary
            component_num: Component number in overall list
            symbol: Component symbol (e.g., 'g0', 'g1') for constraint references
        """
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Gaussian\n"
        if symbol:
            content += f"SYMBOL={symbol}\n"
        
        # Core parameters
        if 'fit_id' in fit:
            content += f"FIT_ID={fit['fit_id']}\n"
        if 'component_id' in fit:
            content += f"COMPONENT_ID={fit['component_id']}\n"
        
        # Line information
        if 'line_id' in fit and fit['line_id']:
            content += f"LINE_ID={fit['line_id']}\n"
        if 'line_wavelength' in fit and fit['line_wavelength']:
            content += f"LINE_WAVELENGTH={fit['line_wavelength']}\n"
        if 'rest_wavelength' in fit and fit['rest_wavelength']:
            content += f"REST_WAVELENGTH={fit['rest_wavelength']}\n"
        
        # Initial guesses (if provided from listfit or other sources)
        # Note: In v1.4+, use MU (μ) and SIGMA (σ) instead of MEAN and STD_DEV
        # The parse routine will support both old (v1.3) and new (v1.4) formats for backward compatibility
        if 'amp_initial' in fit and fit['amp_initial'] is not None:
            content += f"AMPLITUDE_INITIAL={fit['amp_initial']}\n"
        if 'mean_initial' in fit and fit['mean_initial'] is not None:
            content += f"MU_INITIAL={fit['mean_initial']}\n"
        if 'stddev_initial' in fit and fit['stddev_initial'] is not None:
            content += f"SIGMA_INITIAL={fit['stddev_initial']}\n"
        
        # Gaussian parameters with errors (best fit)
        if 'amp' in fit:
            content += f"AMPLITUDE={self._format_param(fit.get('amp'), fit.get('amp_err'))}\n"
        if 'mean' in fit:
            content += f"MU={self._format_param(fit.get('mean'), fit.get('mean_err'))}\n"
        if 'stddev' in fit:
            content += f"SIGMA={self._format_param(fit.get('stddev'), fit.get('stddev_err'))}\n"
        
        # Bounds
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
            print(f"[QSAP_WRITE] Gaussian component {component_num}: wrote bounds={bounds}", flush=True)
        else:
            print(f"[QSAP_WRITE] WARNING: Gaussian component {component_num} has NO bounds!", flush=True)
        
        # Quality metrics
        if 'chi2' in fit:
            # Check if errors were available during fitting
            if fit.get('has_errors', False):
                content += f"CHI_SQUARED={fit['chi2']}\n"
            else:
                content += f"SSR={fit['chi2']}\n"  # Sum of squared residuals (no errors)
        if 'chi2_nu' in fit:
            if fit.get('has_errors', False):
                content += f"CHI_SQUARED_NU={fit['chi2_nu']}\n"
            else:
                content += f"SSR_NU={fit['chi2_nu']}\n"  # SSR per degree of freedom
        
        # Equivalent width (if calculated) - Gaussian component
        if 'equivalent_width' in fit:
            ew = fit.get('equivalent_width')
            # Check if we have Monte Carlo credible intervals with best/median/mean
            if ('ew_best' in fit and 'ew_median' in fit and 'ew_mean' in fit and
                'equivalent_width_1sigma_lower' in fit and 
                'equivalent_width_1sigma_upper' in fit):
                # New format with best, median, mean, and credible intervals on separate lines
                ew_best = fit.get('ew_best')
                ew_median = fit.get('ew_median')
                ew_mean = fit.get('ew_mean')
                ew_1s_lower = fit.get('equivalent_width_1sigma_lower')
                ew_1s_upper = fit.get('equivalent_width_1sigma_upper')
                ew_2s_lower = fit.get('equivalent_width_2sigma_lower')
                ew_2s_upper = fit.get('equivalent_width_2sigma_upper')
                ew_3s_lower = fit.get('equivalent_width_3sigma_lower')
                ew_3s_upper = fit.get('equivalent_width_3sigma_upper')
                content += f"EQUIVALENT_WIDTH_BEST={ew_best:.6f}\n"
                content += f"EQUIVALENT_WIDTH_MEDIAN={ew_median:.6f}\n"
                content += f"EQUIVALENT_WIDTH_MEAN={ew_mean:.6f}\n"
                content += f"EQUIVALENT_WIDTH_1SIGMA=-{abs(ew_1s_lower):.6f},+{ew_1s_upper:.6f}\n"
                content += f"EQUIVALENT_WIDTH_2SIGMA=-{abs(ew_2s_lower):.6f},+{ew_2s_upper:.6f}\n"
                content += f"EQUIVALENT_WIDTH_3SIGMA=-{abs(ew_3s_lower):.6f},+{ew_3s_upper:.6f}\n"
            elif ('equivalent_width_1sigma_lower' in fit and 
                'equivalent_width_1sigma_upper' in fit):
                # Old format with only credible intervals (backward compatible)
                ew_1s_lower = fit.get('equivalent_width_1sigma_lower')
                ew_1s_upper = fit.get('equivalent_width_1sigma_upper')
                ew_2s_lower = fit.get('equivalent_width_2sigma_lower')
                ew_2s_upper = fit.get('equivalent_width_2sigma_upper')
                ew_3s_lower = fit.get('equivalent_width_3sigma_lower')
                ew_3s_upper = fit.get('equivalent_width_3sigma_upper')
                content += f"EQUIVALENT_WIDTH={ew:.4f} 1sigma: -{ew_1s_lower:.4f}/+{ew_1s_upper:.4f} 2sigma: -{ew_2s_lower:.4f}/+{ew_2s_upper:.4f} 3sigma: -{ew_3s_lower:.4f}/+{ew_3s_upper:.4f}\n"
            else:
                # Fallback to old format if no credible intervals
                content += f"EQUIVALENT_WIDTH={self._format_param(ew, fit.get('equivalent_width_err'))}\n"
        
        # Mode information
        if 'is_velocity_mode' in fit:
            content += f"VELOCITY_MODE={fit['is_velocity_mode']}\n"
        
        # System redshift
        if 'z_sys' in fit and fit['z_sys']:
            content += f"SYSTEM_REDSHIFT={fit['z_sys']}\n"
        
        # Redshift tying (for line list fitting with redshift constraints)
        if 'tied_redshift' in fit and fit['tied_redshift']:
            content += f"TIED_REDSHIFT={fit['tied_redshift']}\n"
        
        # Covariance matrix (3x3 for Gaussian: amp, mean, stddev)
        if 'covariance' in fit and fit['covariance'] is not None:
            cov = fit['covariance']
            if isinstance(cov, list):
                cov = np.array(cov)
            # Store as flattened 3x3 matrix
            for i in range(3):
                for j in range(3):
                    content += f"COV_{i}_{j}={cov[i][j]}\n"
        
        content += "\n"
        return content
    
    def _build_voigt_component(self, fit, component_num, symbol=None):
        """Build a single Voigt component section
        
        Args:
            fit: Fit dictionary
            component_num: Component number in overall list
            symbol: Component symbol (e.g., 'v0', 'v1') for constraint references
        """
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Voigt\n"
        if symbol:
            content += f"SYMBOL={symbol}\n"
        
        # Core parameters
        if 'fit_id' in fit:
            content += f"FIT_ID={fit['fit_id']}\n"
        if 'component_id' in fit:
            content += f"COMPONENT_ID={fit['component_id']}\n"
        
        # Line information
        if 'line_id' in fit and fit['line_id']:
            content += f"LINE_ID={fit['line_id']}\n"
        if 'line_wavelength' in fit and fit['line_wavelength']:
            content += f"LINE_WAVELENGTH={fit['line_wavelength']}\n"
        if 'rest_wavelength' in fit and fit['rest_wavelength']:
            content += f"REST_WAVELENGTH={fit['rest_wavelength']}\n"
        
        # Initial guesses (if provided from listfit)
        # Note: In v1.4+, use MU (μ) and SIGMA (σ) instead of MEAN
        # The parse routine will support both old (v1.3) and new (v1.4) formats for backward compatibility
        if 'amplitude_initial' in fit and fit['amplitude_initial'] is not None:
            content += f"AMPLITUDE_INITIAL={fit['amplitude_initial']}\n"
        if 'mean_initial' in fit and fit['mean_initial'] is not None:
            content += f"MU_INITIAL={fit['mean_initial']}\n"
        if 'sigma_initial' in fit and fit['sigma_initial'] is not None:
            content += f"SIGMA_INITIAL={fit['sigma_initial']}\n"
        if 'gamma_initial' in fit and fit['gamma_initial'] is not None:
            content += f"GAMMA_INITIAL={fit['gamma_initial']}\n"
        
        # Voigt parameters with errors (best fit)
        if 'amplitude' in fit:
            content += f"AMPLITUDE={self._format_param(fit.get('amplitude'), fit.get('amplitude_err'))}\n"
        if 'mean' in fit:
            content += f"MU={self._format_param(fit.get('mean'), fit.get('mean_err'))}\n"
        elif 'center' in fit:
            content += f"MU={self._format_param(fit.get('center'), fit.get('center_err'))}\n"
        if 'sigma' in fit:
            content += f"SIGMA={self._format_param(fit.get('sigma'), fit.get('sigma_err'))}\n"
        if 'gamma' in fit:
            content += f"GAMMA={self._format_param(fit.get('gamma'), fit.get('gamma_err'))}\n"
        
        # Doppler parameter
        if 'b' in fit:
            content += f"B_DOPPLER={fit['b']}\n"
        if 'logT_eff' in fit:
            content += f"LOG_T_EFF={fit['logT_eff']}\n"
        
        # Bounds
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
            print(f"[QSAP_WRITE] Voigt component {component_num}: wrote bounds={bounds}", flush=True)
        else:
            print(f"[QSAP_WRITE] WARNING: Voigt component {component_num} has NO bounds!", flush=True)
        
        # Quality metrics
        if 'chi2' in fit:
            # Check if errors were available during fitting
            if fit.get('has_errors', False):
                content += f"CHI_SQUARED={fit['chi2']}\n"
            else:
                content += f"SSR={fit['chi2']}\n"  # Sum of squared residuals (no errors)
        if 'chi2_nu' in fit:
            if fit.get('has_errors', False):
                content += f"CHI_SQUARED_NU={fit['chi2_nu']}\n"
            else:
                content += f"SSR_NU={fit['chi2_nu']}\n"  # SSR per degree of freedom
        
        # Equivalent width (if calculated) - Voigt component
        if 'equivalent_width' in fit:
            ew = fit.get('equivalent_width')
            # Check if we have Monte Carlo credible intervals with best/median/mean
            if ('ew_best' in fit and 'ew_median' in fit and 'ew_mean' in fit and
                'equivalent_width_1sigma_lower' in fit and 
                'equivalent_width_1sigma_upper' in fit):
                # New format with best, median, mean, and credible intervals on separate lines
                ew_best = fit.get('ew_best')
                ew_median = fit.get('ew_median')
                ew_mean = fit.get('ew_mean')
                ew_1s_lower = fit.get('equivalent_width_1sigma_lower')
                ew_1s_upper = fit.get('equivalent_width_1sigma_upper')
                ew_2s_lower = fit.get('equivalent_width_2sigma_lower')
                ew_2s_upper = fit.get('equivalent_width_2sigma_upper')
                ew_3s_lower = fit.get('equivalent_width_3sigma_lower')
                ew_3s_upper = fit.get('equivalent_width_3sigma_upper')
                content += f"EQUIVALENT_WIDTH_BEST={ew_best:.6f}\n"
                content += f"EQUIVALENT_WIDTH_MEDIAN={ew_median:.6f}\n"
                content += f"EQUIVALENT_WIDTH_MEAN={ew_mean:.6f}\n"
                content += f"EQUIVALENT_WIDTH_1SIGMA=-{abs(ew_1s_lower):.6f},+{ew_1s_upper:.6f}\n"
                content += f"EQUIVALENT_WIDTH_2SIGMA=-{abs(ew_2s_lower):.6f},+{ew_2s_upper:.6f}\n"
                content += f"EQUIVALENT_WIDTH_3SIGMA=-{abs(ew_3s_lower):.6f},+{ew_3s_upper:.6f}\n"
            elif ('equivalent_width_1sigma_lower' in fit and 
                'equivalent_width_1sigma_upper' in fit):
                # Old format with only credible intervals (backward compatible)
                ew_1s_lower = fit.get('equivalent_width_1sigma_lower')
                ew_1s_upper = fit.get('equivalent_width_1sigma_upper')
                ew_2s_lower = fit.get('equivalent_width_2sigma_lower')
                ew_2s_upper = fit.get('equivalent_width_2sigma_upper')
                ew_3s_lower = fit.get('equivalent_width_3sigma_lower')
                ew_3s_upper = fit.get('equivalent_width_3sigma_upper')
                content += f"EQUIVALENT_WIDTH={ew:.4f} 1sigma: -{ew_1s_lower:.4f}/+{ew_1s_upper:.4f} 2sigma: -{ew_2s_lower:.4f}/+{ew_2s_upper:.4f} 3sigma: -{ew_3s_lower:.4f}/+{ew_3s_upper:.4f}\n"
            else:
                # Fallback to old format if no credible intervals
                content += f"EQUIVALENT_WIDTH={self._format_param(ew, fit.get('equivalent_width_err'))}\n"
        
        # Mode information
        if 'is_velocity_mode' in fit:
            content += f"VELOCITY_MODE={fit['is_velocity_mode']}\n"
        
        # System redshift
        if 'z_sys' in fit and fit['z_sys']:
            content += f"SYSTEM_REDSHIFT={fit['z_sys']}\n"
        
        # Redshift tying (for line list fitting with redshift constraints)
        if 'tied_redshift' in fit and fit['tied_redshift']:
            content += f"TIED_REDSHIFT={fit['tied_redshift']}\n"
        
        # Covariance matrix (for Voigt: amplitude, center, sigma, gamma)
        if 'covariance' in fit and fit['covariance'] is not None:
            cov = fit['covariance']
            if isinstance(cov, list):
                cov = np.array(cov)
            # Store as flattened matrix (handles both 3x3 and 4x4)
            nparams = cov.shape[0]
            for i in range(nparams):
                for j in range(nparams):
                    content += f"COV_{i}_{j}={cov[i][j]}\n"
        
        content += "\n"
        return content
    
    def _build_continuum_component(self, fit, component_num):
        """Build a single continuum component section"""
        content = f"[CONTINUUM_{component_num}]\n"
        content += "TYPE=Continuum\n"
        
        # Polynomial order
        if 'poly_order' in fit:
            content += f"POLY_ORDER={fit['poly_order']}\n"
        
        # Bounds (combined min-max for backward compatibility)
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
        
        # Individual regions (new format - preserves separate regions)
        if 'individual_regions' in fit:
            individual_regions = fit['individual_regions']
            content += f"NUM_REGIONS={len(individual_regions)}\n"
            for idx, region in enumerate(individual_regions):
                content += f"REGION_{idx}_LOWER={region[0]}\n"
                content += f"REGION_{idx}_UPPER={region[1]}\n"
        
        # Polynomial coefficients
        if 'coeffs' in fit:
            coeffs = fit['coeffs']
            for idx, coeff in enumerate(coeffs):
                coeff_err = fit.get('coeffs_err', [None] * len(coeffs))[idx]
                content += f"COEFF_{idx}={self._format_param(coeff, coeff_err)}\n"
        
        # Mode information
        if 'is_velocity_mode' in fit:
            content += f"VELOCITY_MODE={fit['is_velocity_mode']}\n"
        
        content += "\n"
        return content
    
    def _build_polynomial_component(self, fit, component_num, symbol=None):
        """Build a polynomial component (used in listfit)
        
        Args:
            fit: Fit dictionary
            component_num: Component number in overall list
            symbol: Component symbol (e.g., 'p0', 'p1') for constraint references
        """
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Polynomial\n"
        if symbol:
            content += f"SYMBOL={symbol}\n"
        
        if 'poly_order' in fit:
            content += f"POLY_ORDER={fit['poly_order']}\n"
        
        # Bounds
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
            print(f"[QSAP_WRITE] Polynomial component {component_num}: wrote bounds={bounds}", flush=True)
        else:
            print(f"[QSAP_WRITE] WARNING: Polynomial component {component_num} has NO bounds!", flush=True)
        
        # Initial guesses (if provided from listfit)
        if 'coeffs_initial' in fit:
            coeffs_initial = fit['coeffs_initial']
            for idx, coeff_init in enumerate(coeffs_initial):
                if coeff_init is not None:
                    content += f"COEFF_{idx}_INITIAL={coeff_init}\n"
        
        # Best fit coefficients with errors
        if 'coeffs' in fit:
            coeffs = fit['coeffs']
            for idx, coeff in enumerate(coeffs):
                coeff_err = fit.get('coeffs_err', [None] * len(coeffs))[idx]
                content += f"COEFF_{idx}={self._format_param(coeff, coeff_err)}\n"
        
        content += "\n"
        return content
    
    def _build_chebyshev_component(self, fit, component_num, symbol=None):
        """Build a Chebyshev polynomial component (used in listfit)
        
        Args:
            fit: Fit dictionary (must include 'coeffs', 'lam_min', 'lam_max', 'bounds')
            component_num: Component number in overall list
            symbol: Component symbol (e.g., 'c0', 'c1') for constraint references
        
        NOTE: Chebyshev coefficients are stored in the rescaled [-1, 1] frame.
        Domain bounds (lam_min, lam_max) must be stored to reconstruct the mapping.
        """
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Chebyshev\n"
        if symbol:
            content += f"SYMBOL={symbol}\n"
        
        # Chebyshev degree (inferred from coefficients)
        if 'coeffs' in fit:
            content += f"DEGREE={len(fit['coeffs']) - 1}\n"
        
        # Domain bounds (CRITICAL for rescaling)
        if 'lam_min' in fit and 'lam_max' in fit:
            content += f"DOMAIN_MIN={self._format_value(fit['lam_min'])}\n"
            content += f"DOMAIN_MAX={self._format_value(fit['lam_max'])}\n"
            print(f"[QSAP_WRITE] Chebyshev component {component_num}: wrote domain=[{fit['lam_min']}, {fit['lam_max']}]", flush=True)
        else:
            print(f"[QSAP_WRITE] WARNING: Chebyshev component {component_num} has NO domain bounds!", flush=True)
        
        # Fit bounds (wavelength range where fit was performed)
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
        
        # Initial guesses (if provided from listfit)
        if 'coeffs_initial' in fit:
            coeffs_initial = fit['coeffs_initial']
            for idx, coeff_init in enumerate(coeffs_initial):
                if coeff_init is not None:
                    content += f"COEFF_{idx}_INITIAL={coeff_init}\n"
        
        # Best fit coefficients with errors (stored in rescaled [-1, 1] frame)
        if 'coeffs' in fit:
            coeffs = fit['coeffs']
            for idx, coeff in enumerate(coeffs):
                coeff_err = fit.get('coeffs_err', [None] * len(coeffs))[idx]
                content += f"COEFF_{idx}={self._format_param(coeff, coeff_err)}\n"
        
        content += "\n"
        return content
    
    def _build_polynomial_guess_mask_component(self, fit, component_num):
        """Build a polynomial guess mask component (used in listfit)"""
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=PolynomialGuessMask\n"
        
        if 'min_lambda' in fit:
            content += f"MIN_LAMBDA={fit['min_lambda']}\n"
        if 'max_lambda' in fit:
            content += f"MAX_LAMBDA={fit['max_lambda']}\n"
        
        content += "\n"
        return content
    
    def _build_data_mask_component(self, fit, component_num):
        """Build a data mask component (used in listfit)"""
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=DataMask\n"
        
        if 'min_lambda' in fit:
            content += f"MIN_LAMBDA={fit['min_lambda']}\n"
        if 'max_lambda' in fit:
            content += f"MAX_LAMBDA={fit['max_lambda']}\n"
        
        content += "\n"
        return content
    
    def _build_redshift_component(self, fit, component_num, symbol=None):
        """Build a redshift component block for listfit
        
        Args:
            fit: Fit dictionary with redshift parameters
            component_num: Component number
            symbol: Component symbol (e.g., 'Z1', 'Z2')
        """
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Redshift\n"
        if symbol:
            content += f"SYMBOL={symbol}\n"
        
        # Redshift value with uncertainties
        if 'redshift' in fit or 'value' in fit:
            z_value = fit.get('redshift', fit.get('value'))
            z_err = fit.get('error_redshift', fit.get('error', fit.get('uncertainty')))
            content += f"REDSHIFT={self._format_param(z_value, z_err)}\n"
        
        # Initial guess
        if 'z_initial' in fit or 'initial' in fit:
            z_initial = fit.get('z_initial', fit.get('initial'))
            if z_initial is not None:
                content += f"REDSHIFT_INITIAL={z_initial}\n"
        
        # Redshift label/description (if provided)
        if 'label' in fit:
            content += f"LABEL={fit['label']}\n"
        
        # Redshift number (z1, z2, z3, etc.)
        if 'redshift_number' in fit:
            content += f"REDSHIFT_NUMBER={fit['redshift_number']}\n"
        
        # Number of profiles tied to this redshift
        if 'num_profiles' in fit:
            content += f"NUM_TIED_PROFILES={fit['num_profiles']}\n"
        
        content += "\n"
        return content
        return content
    
    def _build_constraints_block(self, fit):
        """Build the [CONSTRAINTS] block documenting all applied constraints
        
        Constraints include:
        - Tied parameters (expressions): e.g., g0_mean = g1_mean * (5008/4960)
        - Fixed values: e.g., g0_amp fixed at 150.0
        - Bounds: e.g., g0_mean between 4300 and 4350
        """
        content = "[CONSTRAINTS]\n"
        constraints_data = fit.get('constraints', {})
        
        if not constraints_data:
            content += "NONE=No constraints applied\n"
        else:
            for comp_key, comp_constraints in constraints_data.items():
                if not comp_constraints or all(not v for v in comp_constraints.values()):
                    continue
                
                has_constraints = False
                section_content = f"\n# Constraints for {comp_key}\n"
                
                # Tied/linked parameter expressions
                linked_constraints = comp_constraints.get('linked_constraints', [])
                if linked_constraints:
                    for constraint in linked_constraints:
                        if isinstance(constraint, dict):
                            # Format: {'parameter': 'name', 'expression': 'expr'}
                            expr = constraint.get('expression')
                            if expr:
                                section_content += f"  TIED_PARAMETER={expr}\n"
                                has_constraints = True
                        else:
                            # Might be a string directly
                            section_content += f"  TIED_PARAMETER={constraint}\n"
                            has_constraints = True
                
                # Fixed parameters
                if comp_constraints.get('amplitude_fixed'):
                    section_content += f"  FIXED=amplitude\n"
                    has_constraints = True
                if comp_constraints.get('mean_fixed') or comp_constraints.get('center_fixed'):
                    section_content += f"  FIXED=center/mean\n"
                    has_constraints = True
                if comp_constraints.get('sigma_fixed') or comp_constraints.get('stddev_fixed'):
                    section_content += f"  FIXED=width (sigma/stddev)\n"
                    has_constraints = True
                if comp_constraints.get('gamma_fixed'):
                    section_content += f"  FIXED=gamma (Voigt)\n"
                    has_constraints = True
                
                # Bounds for amplitude
                amp_bounds = comp_constraints.get('amplitude_bounds')
                if amp_bounds and (amp_bounds[0] or amp_bounds[1]):
                    min_val = amp_bounds[0] if amp_bounds[0] else "none"
                    max_val = amp_bounds[1] if amp_bounds[1] else "none"
                    section_content += f"  AMPLITUDE_BOUNDS={min_val}..{max_val}\n"
                    has_constraints = True
                
                # Bounds for mean/center
                mean_bounds = comp_constraints.get('mean_bounds')
                if mean_bounds and (mean_bounds[0] or mean_bounds[1]):
                    try:
                        min_val = f"{float(mean_bounds[0]):.2f}" if mean_bounds[0] else "none"
                    except (ValueError, TypeError):
                        min_val = str(mean_bounds[0]) if mean_bounds[0] else "none"
                    try:
                        max_val = f"{float(mean_bounds[1]):.2f}" if mean_bounds[1] else "none"
                    except (ValueError, TypeError):
                        max_val = str(mean_bounds[1]) if mean_bounds[1] else "none"
                    section_content += f"  CENTER_BOUNDS={min_val}..{max_val}\n"
                    has_constraints = True
                
                center_bounds = comp_constraints.get('center_bounds')
                if center_bounds and (center_bounds[0] or center_bounds[1]):
                    try:
                        min_val = f"{float(center_bounds[0]):.2f}" if center_bounds[0] else "none"
                    except (ValueError, TypeError):
                        min_val = str(center_bounds[0]) if center_bounds[0] else "none"
                    try:
                        max_val = f"{float(center_bounds[1]):.2f}" if center_bounds[1] else "none"
                    except (ValueError, TypeError):
                        max_val = str(center_bounds[1]) if center_bounds[1] else "none"
                    section_content += f"  CENTER_BOUNDS={min_val}..{max_val}\n"
                    has_constraints = True
                
                # Bounds for sigma/stddev
                sigma_bounds = comp_constraints.get('sigma_bounds')
                if sigma_bounds and (sigma_bounds[0] or sigma_bounds[1]):
                    min_val = sigma_bounds[0] if sigma_bounds[0] else "none"
                    max_val = sigma_bounds[1] if sigma_bounds[1] else "none"
                    section_content += f"  SIGMA_BOUNDS={min_val}..{max_val}\n"
                    has_constraints = True
                
                # Bounds for gamma
                gamma_bounds = comp_constraints.get('gamma_bounds')
                if gamma_bounds and (gamma_bounds[0] or gamma_bounds[1]):
                    min_val = gamma_bounds[0] if gamma_bounds[0] else "none"
                    max_val = gamma_bounds[1] if gamma_bounds[1] else "none"
                    section_content += f"  GAMMA_BOUNDS={min_val}..{max_val}\n"
                    has_constraints = True
                
                # Only add section if it has actual constraints
                if has_constraints:
                    content += section_content
        
        content += "\n"
        return content
    
    def _build_covariance_block(self, lmfit_result):
        """Build covariance matrix and tie expression blocks for post-processing MC calculations
        
        Enables loading fitted profiles later and recalculating EW or other quantities
        without re-running the fit.
        
        Args:
            lmfit_result: lmfit result object with var_names, covar, params
            
        Returns:
            String with [COVARIANCE_MATRIX], [FREE_PARAMETER_VALUES], [TIE_EXPRESSIONS] sections
        """
        import json
        import numpy as np
        
        content = ""
        
        try:
            # Extract free parameter names and values
            free_param_names = lmfit_result.var_names  # List of free parameter names
            if not free_param_names:
                # No free parameters - all are tied/fixed, can't do MC sampling
                return content
            
            free_param_values = []
            for pname in free_param_names:
                val = lmfit_result.params[pname].value
                free_param_values.append(float(val) if val is not None else 0.0)
            
            # Extract covariance matrix (only for free parameters)
            covar = lmfit_result.covar
            if covar is None:
                # No covariance available - singularor numerical issues
                return content
            
            covar_array = np.array(covar, dtype=float)
            
            # Build [COVARIANCE_MATRIX] section
            content += "[COVARIANCE_MATRIX]\n"
            content += f"FREE_PARAMETERS={','.join(free_param_names)}\n"
            content += f"MATRIX_SIZE={len(free_param_names)}\n"
            # Flatten covariance matrix to CSV (row-major)
            covar_flat = covar_array.flatten().tolist()
            covar_csv = ','.join(f"{v:.10e}" for v in covar_flat)
            content += f"COVARIANCE_FLAT={covar_csv}\n"
            content += "\n"
            
            # Build [FREE_PARAMETER_VALUES] section
            content += "[FREE_PARAMETER_VALUES]\n"
            for pname, pval in zip(free_param_names, free_param_values):
                content += f"{pname}={pval:.10e}\n"
            content += "\n"
            
            # Build [TIE_EXPRESSIONS] section (expressions for tied parameters)
            tie_expressions = {}
            for pname in lmfit_result.params:
                if pname not in free_param_names:  # This is a tied parameter
                    expr = lmfit_result.params[pname].expr
                    if expr:
                        tie_expressions[pname] = expr
            
            if tie_expressions:
                content += "[TIE_EXPRESSIONS]\n"
                for pname, expr in sorted(tie_expressions.items()):
                    content += f"{pname}={expr}\n"
                content += "\n"
            
            # Build [COMPONENT_REGISTRY] section (maps params to components)
            # Use prefix to determine component type and number
            component_registry = {}
            for pname in lmfit_result.params:
                # Extract prefix (g0, g1, v0, v1, p0, z1, etc.)
                prefix_match = None
                if pname.startswith('g') and pname[1].isdigit():
                    idx = 1
                    while idx < len(pname) and pname[idx].isdigit():
                        idx += 1
                    if idx < len(pname) and pname[idx] == '_':
                        prefix = pname[:idx]
                        param_part = pname[idx+1:]
                        comp_type = 'gaussian'
                        prefix_match = (prefix, comp_type, param_part)
                elif pname.startswith('v') and pname[1].isdigit():
                    idx = 1
                    while idx < len(pname) and pname[idx].isdigit():
                        idx += 1
                    if idx < len(pname) and pname[idx] == '_':
                        prefix = pname[:idx]
                        param_part = pname[idx+1:]
                        comp_type = 'voigt'
                        prefix_match = (prefix, comp_type, param_part)
                elif pname.startswith('p') and pname[1].isdigit():
                    idx = 1
                    while idx < len(pname) and pname[idx].isdigit():
                        idx += 1
                    if idx < len(pname) and pname[idx] == '_':
                        prefix = pname[:idx]
                        param_part = pname[idx+1:]
                        comp_type = 'polynomial'
                        prefix_match = (prefix, comp_type, param_part)
                elif pname.startswith('z') and pname[1].isdigit():
                    prefix = pname  # z1, z2, etc. are complete param names
                    comp_type = 'redshift'
                    prefix_match = (prefix, comp_type, '')
                
                if prefix_match:
                    prefix, comp_type, param_part = prefix_match
                    if prefix not in component_registry:
                        component_registry[prefix] = {'type': comp_type, 'params': []}
                    if param_part:
                        component_registry[prefix]['params'].append(pname)
                    else:
                        # Redshift component
                        component_registry[prefix]['params'].append(pname)
            
            if component_registry:
                content += "[COMPONENT_REGISTRY]\n"
                for comp_id in sorted(component_registry.keys()):
                    comp_info = component_registry[comp_id]
                    comp_type = comp_info['type']
                    params_str = ','.join(comp_info['params'])
                    content += f"{comp_id}={comp_type}|{params_str}\n"
                content += "\n"
        
        except Exception as e:
            # If anything fails, just skip covariance output
            print(f"[WARNING] Could not extract covariance matrix: {e}")
            import traceback
            traceback.print_exc()
        
        return content
    
    def _format_param(self, value, error=None):
        """Format parameter with error in value±error notation"""
        if value is None:
            return "None"
        if error is None or error != error:  # Check for NaN
            return f"{value}"
        return f"{value}±{error}"
    
    def _format_value(self, value):
        """Format any value for file storage"""
        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return str(value)
    
    def parse_qsap_file(self, filepath):
        """Parse a .qsap file and return structured data
        
        Returns:
            Dict with 'metadata' and 'components' keys
        """
        data = {'metadata': {}, 'components': []}
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        current_component = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Section headers
            if line.startswith('[') and line.endswith(']'):
                # Save previous component if exists
                if current_component:
                    if current_section == 'METADATA':
                        data['metadata'] = current_component
                    else:
                        data['components'].append(current_component)
                
                current_section = line[1:-1]  # Remove brackets
                current_component = {}
                continue
            
            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                parsed_value = self._parse_value(value)
                current_component[key] = parsed_value
        
        # Save last component
        if current_component:
            if current_section == 'METADATA':
                data['metadata'] = current_component
            else:
                data['components'].append(current_component)
        
        # Normalize parameter names for backward compatibility (v1.3 -> v1.4+)
        # Maps old parameter names to new Greek letter names
        data = self._normalize_parameter_names(data)
        
        return data
    
    def _normalize_parameter_names(self, data):
        """Normalize old parameter names (v1.3) to new Greek letter names (v1.4+)
        
        Backward compatibility mapping:
        - MEAN_INITIAL -> MU_INITIAL
        - STD_DEV_INITIAL -> SIGMA_INITIAL
        - MEAN -> MU
        - STD_DEV -> SIGMA
        - CENTER -> MU (Voigt)
        """
        for component in data.get('components', []):
            # Mapping of old names to new names
            rename_map = {
                'MEAN_INITIAL': 'MU_INITIAL',
                'STD_DEV_INITIAL': 'SIGMA_INITIAL',
                'MEAN': 'MU',
                'STD_DEV': 'SIGMA',
                'CENTER': 'MU',  # For Voigt components stored as CENTER
            }
            
            # Apply renaming: only rename if new name doesn't already exist
            for old_name, new_name in rename_map.items():
                if old_name in component and new_name not in component:
                    component[new_name] = component.pop(old_name)
        
        return data
    
    def _parse_value(self, value_str):
        """Parse value string to appropriate Python type"""
        if value_str == 'None':
            return None
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
        
        # Try parsing as number with error
        if '±' in value_str:
            parts = value_str.split('±')
            try:
                val = float(parts[0])
                err = float(parts[1])
                return (val, err)
            except (ValueError, IndexError):
                pass
        
        # Try parsing as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Try parsing as JSON (for lists/dicts)
        if value_str.startswith('[') or value_str.startswith('{'):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value_str
    
    def load_covariance_from_qsap(self, filepath):
        """Load covariance matrix and tie expressions from a .qsap file for post-processing MC calculations
        
        Returns:
            Dict with keys:
            - 'free_parameter_names': list of free parameter names
            - 'free_parameter_values': list of best-fit values
            - 'free_parameter_covariance': numpy array (NxN)
            - 'tie_expressions': dict mapping tied param names to expressions
            - 'component_registry': dict mapping component IDs to param lists
            or None if no covariance data found
        """
        import numpy as np
        
        covariance_data = {
            'free_parameter_names': [],
            'free_parameter_values': [],
            'free_parameter_covariance': None,
            'tie_expressions': {},
            'component_registry': {}
        }
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            current_section = None
            matrix_size = None
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Section headers
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1]  # Remove brackets
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if current_section == 'COVARIANCE_MATRIX':
                        if key == 'FREE_PARAMETERS':
                            covariance_data['free_parameter_names'] = [s.strip() for s in value.split(',')]
                        elif key == 'MATRIX_SIZE':
                            matrix_size = int(value)
                        elif key == 'COVARIANCE_FLAT':
                            # Parse flattened covariance matrix
                            flat_values = [float(v.strip()) for v in value.split(',')]
                            if matrix_size and len(flat_values) == matrix_size ** 2:
                                covariance_data['free_parameter_covariance'] = np.array(
                                    flat_values
                                ).reshape((matrix_size, matrix_size))
                    
                    elif current_section == 'FREE_PARAMETER_VALUES':
                        # Store as dict first, then convert to array in correct order
                        if '_free_param_dict' not in covariance_data:
                            covariance_data['_free_param_dict'] = {}
                        covariance_data['_free_param_dict'][key] = float(value)
                    
                    elif current_section == 'TIE_EXPRESSIONS':
                        covariance_data['tie_expressions'][key] = value
                    
                    elif current_section == 'COMPONENT_REGISTRY':
                        # Format: comp_id=type|param1,param2,...
                        comp_info = value.split('|')
                        comp_type = comp_info[0] if comp_info else 'unknown'
                        param_list = comp_info[1].split(',') if len(comp_info) > 1 else []
                        covariance_data['component_registry'][key] = {
                            'type': comp_type,
                            'params': [p.strip() for p in param_list]
                        }
            
            # Convert free parameter dict to ordered array
            if '_free_param_dict' in covariance_data:
                free_param_dict = covariance_data.pop('_free_param_dict')
                ordered_values = []
                for pname in covariance_data['free_parameter_names']:
                    ordered_values.append(free_param_dict.get(pname, 0.0))
                covariance_data['free_parameter_values'] = ordered_values
            
            # Check if we have complete data
            has_covariance = (
                covariance_data['free_parameter_names'] and
                covariance_data['free_parameter_values'] and
                covariance_data['free_parameter_covariance'] is not None
            )
            
            if has_covariance:
                return covariance_data
            else:
                return None
        
        except Exception as e:
            print(f"[MC] Error loading covariance from .qsap: {e}")
            import traceback
            traceback.print_exc()
            return None
