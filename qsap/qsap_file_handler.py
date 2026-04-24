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
    
    FILE_FORMAT_VERSION = "1.1"
    
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
    
    def create_listfit_qsap(self, fit_list, spectrum_filename, spectrum_info=None):
        """Create a .qsap file for listfit (contains multiple profiles and masks)
        
        Args:
            fit_list: List of fit dicts (gaussians, voigts, polynomials, polynomial_guess_mask, data_mask, fit_diagnostics)
            spectrum_filename: Path to spectrum file
            spectrum_info: Dict with spectrum metadata
            
        Returns:
            Path to created file, content as string
        """
        filepath = self.generate_filename('Listfit', 'Listfit', spectrum_filename)
        
        content = self._build_header('Listfit', 'Listfit', spectrum_filename, spectrum_info)
        
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
                content += self._build_gaussian_component(fit, idx)
            elif fit_type == 'voigt':
                content += self._build_voigt_component(fit, idx)
            elif fit_type == 'polynomial':
                content += self._build_polynomial_component(fit, idx)
            elif fit_type == 'polynomial_guess_mask':
                content += self._build_polynomial_guess_mask_component(fit, idx)
            elif fit_type == 'data_mask':
                content += self._build_data_mask_component(fit, idx)
        
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
        content += f"FILE_FORMAT_VERSION=1.1\n"
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
    
    def _build_gaussian_component(self, fit, component_num):
        """Build a single Gaussian component section"""
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Gaussian\n"
        
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
        if 'amp_initial' in fit and fit['amp_initial'] is not None:
            content += f"AMPLITUDE_INITIAL={fit['amp_initial']}\n"
        if 'mean_initial' in fit and fit['mean_initial'] is not None:
            content += f"MEAN_INITIAL={fit['mean_initial']}\n"
        if 'stddev_initial' in fit and fit['stddev_initial'] is not None:
            content += f"STD_DEV_INITIAL={fit['stddev_initial']}\n"
        
        # Gaussian parameters with errors (best fit)
        if 'amp' in fit:
            content += f"AMPLITUDE={self._format_param(fit.get('amp'), fit.get('amp_err'))}\n"
        if 'mean' in fit:
            content += f"MEAN={self._format_param(fit.get('mean'), fit.get('mean_err'))}\n"
        if 'stddev' in fit:
            content += f"STD_DEV={self._format_param(fit.get('stddev'), fit.get('stddev_err'))}\n"
        
        # Bounds
        if 'bounds' in fit:
            bounds = fit['bounds']
            content += f"BOUNDS_LOWER={bounds[0]}\n"
            content += f"BOUNDS_UPPER={bounds[1]}\n"
        
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
        
        # Covariance matrix (3x3 for Gaussian: amp, mean, stddev)
        if 'covariance' in fit and fit['covariance']:
            cov = fit['covariance']
            if isinstance(cov, list):
                cov = np.array(cov)
            # Store as flattened 3x3 matrix
            for i in range(3):
                for j in range(3):
                    content += f"COV_{i}_{j}={cov[i][j]}\n"
        
        content += "\n"
        return content
    
    def _build_voigt_component(self, fit, component_num):
        """Build a single Voigt component section"""
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Voigt\n"
        
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
        if 'amplitude_initial' in fit and fit['amplitude_initial'] is not None:
            content += f"AMPLITUDE_INITIAL={fit['amplitude_initial']}\n"
        if 'mean_initial' in fit and fit['mean_initial'] is not None:
            content += f"MEAN_INITIAL={fit['mean_initial']}\n"
        if 'sigma_initial' in fit and fit['sigma_initial'] is not None:
            content += f"SIGMA_INITIAL={fit['sigma_initial']}\n"
        if 'gamma_initial' in fit and fit['gamma_initial'] is not None:
            content += f"GAMMA_INITIAL={fit['gamma_initial']}\n"
        
        # Voigt parameters with errors (best fit)
        if 'amplitude' in fit:
            content += f"AMPLITUDE={self._format_param(fit.get('amplitude'), fit.get('amplitude_err'))}\n"
        if 'mean' in fit:
            content += f"MEAN={self._format_param(fit.get('mean'), fit.get('mean_err'))}\n"
        elif 'center' in fit:
            content += f"MEAN={self._format_param(fit.get('center'), fit.get('center_err'))}\n"
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
        
        # Covariance matrix (for Voigt: amplitude, center, sigma, gamma)
        if 'covariance' in fit and fit['covariance']:
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
    
    def _build_polynomial_component(self, fit, component_num):
        """Build a polynomial component (used in listfit)"""
        content = f"[COMPONENT_{component_num}]\n"
        content += "TYPE=Polynomial\n"
        
        if 'poly_order' in fit:
            content += f"POLY_ORDER={fit['poly_order']}\n"
        
        # Bounds
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
        
        # Best fit coefficients with errors
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
