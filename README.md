# <img src="logo/qsap_logo.png" alt="QSAP Logo" width="80">&nbsp;&nbsp;QSAP: Quick Spectrum Analysis Program

*An Analysis Tool for Astronomical Spectra*

## Version 1.1

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-blue?logo=pyqt&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-1.17%2B-yellow)
![scipy](https://img.shields.io/badge/scipy-1.5%2B-brightgreen?logo=python&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.1%2B-orange?logo=python&logoColor=white)
![astropy](https://img.shields.io/badge/astropy-4.0%2B-red?logo=python&logoColor=white)
![lmfit](https://img.shields.io/badge/lmfit-0.9%2B-purple?logo=python&logoColor=white)
![emcee](https://img.shields.io/badge/emcee-3.0%2B-yellowgreen?logo=python&logoColor=white)

Interactive Python tool for 1D spectral analysis. QSAP provides both quick-look functionality and some more advanced analysis features including multi-component line fitting, continuum modeling, and Bayesian MCMC fitting.

## Features

- **File Format Auto-Detection**: Automatically detects 7+ ASCII and FITS spectrum formats
- **Interactive Plotting**: 1D spectrum visualization with a variety of navigation controls
- **Multi-Component Fitting**: Gaussian and Voigt profile fitting (single and multi-component) with modes for simultaneous fitting using unconstrained and/or constrained optimization
- **Line Analysis**: Redshift estimation, dynamic line list display and velocity restframe toggling
- **Item and Action Trackers**: Centralized management of all components (Gaussians, Voigts, polynomials, continuum) with multi-select and deletion + undo/redo capabilities
- **Utilities**: Filter overlays, smoothing functionality, etc.

## Table of Contents

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Download](#download)
  - [Conda Environment Setup](#conda-environment-setup)
- [Usage](#usage)
  - [Auto-Detection](#auto-detection-recommended)
  - [Supported Formats](#supported-formats)
  - [Command-Line Options](#command-line-options)
  - [Making `qsap` Executable](#making-qsap-executable)
- [Upgrading Versions](#upgrading-versions)
- [Quick Start](#quick-start)
  - [Running QSAP](#running-qsap)
  - [Opening a Spectrum](#opening-a-spectrum)
  - [Spectrum Loading Options](#spectrum-loading-options)
    - [Format Auto-Detection](#format-auto-detection)
    - [Wavelength Unit Selection](#wavelength-unit-selection)
    - [Air-to-Vacuum Wavelength Conversion](#air-to-vacuum-wavelength-conversion)
    - [NaN and Infinity Value Handling](#nan-and-infinity-value-handling)
- [Package Structure](#package-structure)
- [Data Files](#data-files)
- [User Interface Windows](#user-interface-windows)
  - [1. Spectrum Plotter (Main Window)](#1-spectrum-plotter-main-window)
  - [2. Control Panel (Right-Side Dock, Tabbed)](#2-control-panel-right-side-dock-tabbed)
  - [3. Options Panel (Right-Side Dock)](#3-options-panel-right-side-dock)
  - [4. Terminal/Output Panel (Bottom Dock)](#4-terminaloutput-panel-bottom-dock)
  - [5. LineList Window](#5-linelist-window)
  - [6. Listfit Window (Multi-Component Fitting)](#6-listfit-window-multi-component-fitting)
  - [7. Item Tracker Window](#7-item-tracker-window)
  - [8. Action History Window](#8-action-history-window)
  - [9. Fit Information Window](#9-fit-information-window)
- [QSAP File Format (.qsap)](#qsap-file-format-qsap)
- [Versions](#versions)
- [Citation](#citation)
- [License](#license)
- [Author](#author)

## Installation

### Requirements
- Python 3.7+
- numpy, scipy, matplotlib, astropy, pandas, lmfit, PyQt5, emcee, corner

### Download

The newest available version of QSAP is **v1.1**.

To **download a specific tagged version** (e.g., v1.1), run:

```bash
# Clone only the tag (shallow clone, fastest)
git clone --depth 1 --branch v1.1 https://github.com/eriksolhaug/qsap.git

# Or clone the whole repo and checkout the tag
git clone https://github.com/eriksolhaug/qsap.git
cd qsap
git checkout v1.1
```

You can download main branch as it is. However, **this is not recommended** as the main branch may be actively undergoing development:

```bash
git clone https://github.com/eriksolhaug/qsap.git
```

### Conda Environment Setup

Then install the required dependencies in a conda environment:

```bash
# Create a new conda environment
conda create -n qsap python=3.8

# Activate the environment
conda activate qsap

# Enter qsap directory
cd qsap # This is the repo you cloned from github

# Install dependencies
pip install -r requirements.txt
# OR install requirements using conda
conda install numpy scipy matplotlib astropy pandas lmfit pyqt5 emcee corner

# Run QSAP
python qsap.py <spectrum.fits>

# An example spectrum is available. Run...
python qsap.py example/sample_spectrum.txt
# ...to get started!
```


## Usage

### Auto-Detection (Recommended)

```bash
# Automatic format detection
python qsap.py <spectrum.fits>

# Preview detected formats
python qsap.py <spectrum.fits> --detect

# Force specific format if needed
python qsap.py <spectrum.fits> --fmt fits:image1d
```

### Supported Formats

| Format | Description |
|--------|-------------|
| `ascii:2col` | 2-column ASCII (wavelength, flux/transmission with optional # comments) |
| `ascii:3col` | 3-column ASCII (wavelength, flux, error with optional # comments) |
| `ascii:flex` | Flexible ASCII with custom column mapping |
| `fits:image1d` | 1D FITS image with wavelength in header |
| `fits:table:vector` | FITS table with wave/flux as vector arrays in rows |
| `fits:table:columns` | FITS table with per-pixel columns |
| `fits:ext:spectrum` | FITS SPECTRUM extension |

### Command-Line Options

```
--fmt              Force format (auto-detects if omitted)
--detect           Show detected formats and exit
--redshift         Initial redshift for displaying line lists (default: 0.0)
--version          Show version and exit
```

### Making `qsap` Executable

To run QSAP from anywhere as a simple `qsap` command, use one of these options:

**Option 1: Package Installation (Recommended)**

Install QSAP as a Python package, which automatically creates the executable:

```bash
# From inside the qsap directory
pip install .

# Or in development mode (allows you to modify the code and run it with the updated changes using the qsap command)
pip install -e .
```

Now you can run QSAP from anywhere:

```bash
qsap <path/to/spectrum.fits>
```

This method uses the entry point defined in `setup.py` to create a proper command-line executable.

**Option 2: Manual Symlink**

1. **Make the script executable:**
   ```bash
   chmod +x qsap.py
   ```

2. **Create a symlink in a directory on your PATH:**
   ```bash
   # Find your qsap installation path
   QSAP_PATH=$(pwd)/qsap.py
   
   # Link to a bin directory in your PATH (example: /usr/local/bin)
   sudo ln -s $QSAP_PATH /usr/local/bin/qsap
   ```

## Upgrading Versions

If you already have QSAP installed and want to upgrade to a newer version, follow one of these methods:

### Method 1: Git Pull

If you cloned from git and want to update to the latest version:

```bash
# Navigate to your qsap directory
cd /path/to/qsap

# Pull the latest changes
git pull origin main

# Reinstall (if dependencies changed)
pip install -e .

# Verify
qsap --help
```

### Method 2: Update to a Specific Version Tag

To update to a specific version (e.g., v1.1):

```bash
# Navigate to your qsap directory
cd /path/to/qsap

# Fetch all available versions/tags from remote
git fetch origin

# Checkout the specific version you want
git checkout v1.1

# Reinstall (if dependencies changed)
pip install -e .

# Verify
qsap --help
```

### Method 3: Fresh Installation

If you prefer a clean installation:

```bash
# Clone the latest version
git clone https://github.com/eriksolhaug/qsap.git
# OR clone a specific version tag
git clone --depth 1 --branch v1.1 https://github.com/eriksolhaug/qsap.git

# Navigate to the new qsap directory
cd qsap

# Install
pip install -e .

# Verify
qsap --help
```

### Troubleshooting Version Updates

**"command not found: qsap"** after updating:
- Make sure your conda environment is activated: `conda activate qsap`
- Try reinstalling: `pip install -e .`

**Import errors or missing modules**:
- Update dependencies: `pip install -r requirements.txt`
- Or reinstall all requirements: `pip install --upgrade -r requirements.txt`

**Old version still running**:
- Check which qsap is being used: `which qsap`
- Verify it points to your new installation
- If it's a symlink (Method 2 of initial install), update it to point to the new location

## Quick Start

### Running QSAP

Once installed with `pip install -e .`, you have two ways to launch QSAP:

**Without a Spectrum File:**
```bash
qsap
```
This launches QSAP with an empty plotter. You can then load a spectrum using the Open button in the Control Panel.

**With a Spectrum File:**
```bash
qsap <path/to/spectrum.fits>
qsap example/sample_spectrum.txt
qsap data/my_spectrum.txt --redshift 0.1
```

### Opening a Spectrum

You can load a spectrum file in two ways:

1. **Command Line**: Provide the file path as an argument when launching:
   ```bash
   qsap /path/to/your/spectrum.fits
   ```

2. **GUI Open Button**: 
   - Launch QSAP by running `qsap` in the terminal (specifying a spectrum file is not required)
   - In the Control Panel (left side), click the **Open** button
   - A file browser will appear—navigate to and select your spectrum file
   - QSAP automatically detects the file format and loads the data

### Spectrum Loading Options

Once a spectrum file is selected, QSAP provides several configuration options before loading:

#### Format Auto-Detection

QSAP automatically detects spectrum file formats by inspecting the file structure:
- **ASCII formats**: Detects 2-column, 3-column, and flexible multi-column layouts
- **FITS formats**: Identifies 1D images, table extensions, vector arrays, and named SPECTRUM extensions
- **Confidence scoring**: Shows ranked list of likely formats with detection confidence

If auto-detection produces multiple candidates, a format selection dialog appears. You can:
- Review detection scores and metadata (number of columns, table structure, etc.)
- Select the correct format manually, or accept the highest-confidence match
- Force a specific format using `--fmt` on the command line

#### Wavelength Unit Selection

QSAP supports multiple wavelength unit conventions:

**Available Units:**
- **Ångström (Å)** — Default, 1 Å = 0.1 nm = 10⁻¹⁰ m
- **Nanometers (nm)** — 1 nm = 10 Å = 10⁻⁹ m
- **Micrometers (µm)** — 1 µm = 10,000 Å = 10⁻⁶ m

**How to Select:**
- Set during spectrum loading (dialog or Settings tab)
- Changes affect all wavelength displays (plot axes, line lists, fit results)
- Automatically applied to loaded spectrum and line identification catalogs

#### Air-to-Vacuum Wavelength Conversion

For optical/UV spectra taken through air (e.g., telescope observations), wavelengths should often be converted to vacuum wavelengths for precise line identification and redshift calculations.

**Conversion Method:**
- Uses the formula from Donald Morton (2000, ApJ. Suppl., 130, 403). See [VALD Air-to-Vacuum Conversion](https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion).

**How to Apply:**
- Available during spectrum loading or in Settings tab
- Select **"Air → Vacuum"** to convert observed air wavelengths to vacuum
- Select **"Vacuum → Air"** if your spectrum is already in vacuum but needs air wavelengths
- **Default**: No conversion (assumes vacuum wavelengths)

#### NaN and Infinity Value Handling

Some spectrum files contain invalid or missing values (NaN = "Not a Number", Inf = "Infinity"), which can cause fitting failures.

**How to Handle:**
- **Detect & Replace**: During loading, QSAP scans for NaN/Inf pixels
- **Replacement Options**:
  - Replace with **0.0** (zero flux)
  - Replace with **1.0** (e.g., assume unity continuum)
  - Replace with **local median** (interpolate from nearby pixels)
  - Replace with **custom value** (user-specified)
- Default: Replace with **0.0** (zero flux)

## Package Structure

```
.
├── qsap.py                  # Main entry point (root level)
├── setup.py                 # Package setup and installation
├── requirements.txt         # Python dependencies
├── LICENSE
├── README.md
├── __init__.py              # Package initialization
├── example/                 # Example data
│   └── sample_spectrum.txt  # Sample spectrum file to get started
├── qsap/                    # Main package directory
│   ├── __init__.py
│   ├── spectrum_io.py                  # File I/O with auto-detection
│   ├── spectrum_analysis.py            # Fitting and analysis functions
│   ├── spectrum_plotter.py             # Main visualization widget
│   ├── spectrum_plotter_app.py         # Application wrapper
│   ├── ui_components.py                # UI component exports
│   ├── ui_utils.py                     # UI utilities (icons, styling)
│   ├── linelist_window.py              # Line identification window
│   ├── linelist_selector_window.py     # Line list management UI
│   ├── linelist.py                     # Line list data structures
│   ├── listfit_window.py               # Multi-component fitting dialog
│   ├── item_tracker.py                 # Component tracking and management
│   ├── fit_information_window.py       # Fit information display window
│   ├── action_history_window.py        # Action history and undo/redo
│   ├── format_picker_dialog.py         # Format selection dialog
│   └── qsap_file_handler.py            # .qsap file creation and parsing
└── resources/
    ├── linelist/                # Line list catalogs
    │   ├── emlines.txt
    │   ├── emlines_osc.txt
    │   ├── sdss.txt
    │   ├── sdss_emission.txt
    │   ├── sdss_absorption.txt
    │   └── sdss_sky.txt
    └── bands/                   # Instrument band definitions
        └── instrument_bands.txt
```

## Data Files

**Example Data** (in `example/`):
- `sample_spectrum.txt`: Sample spectrum file to get started with QSAP

**Line Lists** (in `resources/linelist/`):
- `emlines.txt`: Emission line catalog
- `emlines_osc.txt`: Lines with oscillator strengths
- `sdss_emission.txt`: SDSS emission line catalog
- `sdss_absorption.txt`: SDSS stellar absorption features
- `sdss_sky.txt`: Telluric sky emission lines

**Instrument Bands** (in `resources/bands/`):
- `instrument_bands.txt`: Filter and instrument bandpass definitions


## User Interface Windows

### 1. Spectrum Plotter (Main Window)
The central interactive spectrum visualization with the following controls:

**Key Features:**
- Real-time spectrum display with wavelength calibration
- Zoom and pan capabilities
- Automated continuum detection and normalization
- Multi-component profile fitting visualization
- Redshift tracking with selected line highlighting

**Keyboard Shortcuts:**

**Navigation & View Controls:**
- `[` / `]` - Pan left/right through spectrum
- `\` (backslash) - Reset spectrum view to starting bounds
- `x` - Center on wavelength position under cursor
- `u` / `i` - Set lower/upper x-bounds (wavelength bounds)
- `T` / `t` - Zoom in/out horizontally (narrow/widen x-range)
- `Y` / `y` - Zoom in/out vertically (narrow/widen y-range)
- `O` / `P` - Set lower/upper y-bounds (flux bounds)
- `l` - Toggle log y-axis
- `L` - Toggle log x-axis
- `f` - Enter fullscreen mode
- `1`-`9` - Apply Gaussian smoothing with different kernel sizes
- `0` - Remove smoothing (restore original spectrum)
- `~` (tilde) - Toggle between step plot and line plot
- `` ` `` (backtick) - Save screenshot of plot

**Fitting Modes:**
- `m` - Enter continuum fitting mode (define regions with `SPACE`, then hit `ENTER` to perform continuum fit -- note that you can change the polynomial order in the Control Panel)
- `M` - Remove a continuum region
- `d` - Enter Single Mode Gaussian fit (SPACE to select bounds)
- `|` (pipe) - Enter Multi-Gaussian fit mode (fit multiple Gaussians simultaneously, define bounds with `SPACE` and hit `ENTER` to perform fit)
- `n` - Single mode Voigt profile fitting
- `H` - Enter Listfit window (a continuum does not need to have been fitted, escape with `ESC`)
- `r` - Toggle residual panel

Note for `d`, `|`, and `n`:  A continuum must have already been fitted. Define bounds with `SPACE`.

**Line List:**
- `e` - Open Line List window

Note: You will need to select the line list and Toggle Display to view lines. Set the redshift applied to the line list in the Control Panel.

**Line Profiles:**
- `w` - Remove fitted profile under cursor
- `,` (comma) - Add a line tag to fitted profile under cursor
- `<` (less than) - Remove tag from fitted profile under cursor
- `;` (semicolon) - Show/toggle total line for Single Mode fitted lines
- `v` - Calculate equivalent width of fitted line. In progress. Use with caution.

**Redshift & Velocity:**
- `z` - Enter redshift mode (select already fitted line under cursor with `SPACE`, escape with `ESC`)
- `b` - Activate velocity mode (set rest-frame wavelength). In progress.

**Instrument Filters & Bands:**
- `!` through `)` (Shift+1-0) - Toggle instrument bandpass overlays (press Shift+number)
- `-` / `_` / `=` / `+` - Show filter bandpasses (requires downloaded filter files)

**Item Management:**
- `j` - Toggle Item Tracker window visibility

**Help:**
- `?` - Show keyboard shortcuts help window

**File Storage:**
All saved screenshots, redshifts, and profile info are stored in the directory where QSAP was launched from.

### Fitting Engines by Mode

QSAP employs different fitting algorithms optimized for each analysis task:

| Mode | Key | Fitting Engine | Method | Use Case |
|------|-----|----------------|--------|----------|
| Single Gaussian (`d`) | `d` | **scipy.optimize.curve_fit** | Least-squares optimization | Quick individual profile fitting |
| Single Voigt (`n`) | `n` | **scipy.optimize.curve_fit** | Least-squares optimization | Quick individual profile fitting with natural broadening |
| Multi-Gaussian (`\|`) | `\|` | **scipy.optimize.curve_fit** | Least-squares optimization (sequential) | Multiple Gaussian fitting in same region |
| Listfit (`H`) | `H` | **lmfit (leastsq)** | Composite Model with simultaneous parameter fitting | Multi-component simultaneous fitting with full covariance estimation |
| Bayesian MCMC (`:`) | `:` | **emcee** | Posterior sampling (work in progress) | Posterior probability distributions for parameters |

**Key Differences:**

- **Single Mode (`d`/`n`) and Multi-Gaussian Mode (`|`)**: Uses `scipy.optimize.curve_fit` for individual component fitting. Fast but limited uncertainty estimation. Suitable for isolated, well-separated lines.

- **Listfit Mode (`H`)**: Uses `lmfit`'s composite Model system with `leastsq` minimization. Allows simultaneous fitting of multiple Gaussians, Voigts, and polynomial backgrounds with full parameter correlation tracking. Provides robust error estimation through covariance matrix analysis. **Recommended for crowded spectral regions or blended profiles.**

- **MCMC Mode (`:`)**: Uses `emcee` for Bayesian posterior sampling (currently in development). Provides posterior distributions and uncertainty quantification beyond frequentist approach. (Currently only works for a single line first estimated using Single Mode)


### 2. Control Panel (Right-Side Dock, Tabbed)
The Control Panel is now organized as a tabbed interface on the right side of the main window (v1.1+). It contains three main tabs:

#### Control Panel Tab
Displays real-time analysis information and fitting parameters:

- **Load Spectrum Button**: Click to open a spectrum file via GUI
- **Load Fit Button**: Click to open previously saved fit files (.qsap or .csv)
- **Redshift Controls**: 
  - Manual text input for redshift value
  - Arrow buttons for fine-tuning (±0.001, ±0.01, ±0.1)
  - Apply button to confirm changes
  - Updates line list display in real-time
- **Undo/Redo Buttons**: Navigate through action history
- **Help Button**: Opens keyboard shortcuts reference
- **Quit Button**: Close QSAP

#### Item Tracker Tab
Centralized management of all fitted components (Gaussians, Voigts, continua, etc.):
- **Real-time Item List**: Shows all active components with type, position, and color
- **Multi-select**: Select multiple items for batch operations
- **Delete Selected**: Remove highlighted components from plot
- **Fit Information**: Click items to highlight corresponding spectral features

#### Settings Tab
Configuration for file I/O directories:
- **Save Directory**: Set default location for saving fits and screenshots
- **Load Directory**: Set default location for opening spectrum and fit files
- **Browse Buttons**: GUI file browser for each directory setting

### 3. Options Panel (Right-Side Dock)
Advanced fitting and analysis controls accessible via dropdown menus and buttons (v1.1+).

#### Fit Section
Controls for different fitting modes:

**Continuum Fitting:**
- **Mode Dropdown**: Activate/deactivate continuum region fitting
- **Polynomial Order**: Adjust polynomial degree (+/- buttons or direct input)
- **Enter Button**: Confirm and start continuum fitting in the spectrum

**Line Profiles:**
- **Single Gaussian**: Fit individual Gaussian profiles
- **Multi Gaussian**: Fit multiple Gaussians simultaneously
- **Enter Button**: Confirm and start profile fitting

**Advanced Modes:**
- **Listfit**: Open the Listfit dialog for component-by-component fitting
- **Bayes Fit**: Bayesian MCMC posterior sampling (in development)

#### Calculate Section
Analysis tools and visualization options:

**Redshift & Velocity:**
- **Estimate Redshift**: Interactive redshift estimation mode from fitted lines
- **Velocity x-axis**: Toggle velocity vs wavelength display

**Plot Options:**
- **Plot Residual**: Toggle residual panel display below spectrum
- **Plot Total Line**: Show composite fit profile for Single Mode fitting

**Equivalent Width Calculation:**
- **Calculate EW Automatically**: Toggle auto-computation of line equivalent widths
- **Plot MC Profiles**: Show Monte Carlo error profiles for equivalent width estimates
- **Delete All MC Profiles**: Clear all error profile visualizations

#### Deactivate All Button
Master control to exit all active fitting modes and reset the interface to idle state

### 4. Terminal/Output Panel (Bottom Dock)
Real-time terminal output and analysis displaying:
- **Spectrum Loading Messages**: File format detection, data validation
- **Fitting Results**: Component parameters, χ² values, convergence status
- **Warnings/Errors**: Issues encountered during analysis

### 5. LineList Window
Interactive line identification and management interface for viewing and selecting spectral lines:

**Access:**
- Press `e` on the spectrum to open the LineList Window
- Provides access to multiple line list catalogs

**Line Lists:**
- **sdss_emission.txt**: SDSS emission line catalog
- **sdss_absorption.txt**: SDSS stellar absorption features
- **sdss_sky.txt**: Telluric sky emission lines
- **emlines.txt**: Common emission lines (Hα, Hβ, [OIII], [OII], Ly-α, etc.) from https://github.com/cconroy20/fsps
- **emlines_osc.txt**: Added oscillator strength data to emlines.txt


### 6. Listfit Window (Multi-Component Fitting)
Dialog for simultaneous fitting of multiple spectral components:

**Workflow:**
1. Press `H` on spectrum to activate listfit mode
2. Define fitting bounds by clicking two wavelengths (spacebar confirms each)
3. Listfit Window opens with component selection
4. Choose component types and quantities:
   - **Gaussian**: Simple Gaussian profiles
   - **Voigt**: Gaussian + Lorentzian (more realistic for emission lines)
   - **Polynomial**: Continuum (specify order)
5. Click "Calculate Fit" to perform simultaneous fitting. Toggle total line `;` to see the total of the component profiles.

**Parameters:**
- Each component shows fitted parameters (amplitude, center wavelength, width/sigma)
- Errors computed from covariance matrix
- Quality metrics: χ² and reduced χ²

### 7. Item Tracker Window
Centralized feature management panel is available as its own tab in the Control Panel widget:

**Display Columns:**
- **Name**: Feature identifier (e.g., "Gaussian 1", "Voigt 2")
- **Type**: Component type (gaussian, voigt, polynomial, continuum)
- **Color**: Visual indicator with color box showing plot color (HTML codes are used)
- **Position**: Wavelength bounds or center position

**Operations:**
- Multi-select items with Control/Shift+Click
- Right-click context menu: "Delete" to remove from plot, "Estimate Redshift" or "Calculate Equivalent Width" of the specified profile (for Gaussians and Voigts only)
- "Delete Selected" button: Remove multiple items at once
- "Clear All" button: Remove all features

### 8. Action History Window
Comprehensive undo/redo interface for tracking all modifications to the spectrum analysis:

- Press `Ctrl+Z` to undo or `Ctrl+Shift+Z` to redo
- Right-click on Item Tracker items for context menu access to history

**Under some development - might not work optimally.**

### 9. Fit Information Window
Comprehensive display of fitted components and saved `.qsap` file management:

**Access:**
- Automatically shows when fits are performed
- Can be toggled via View menu or keyboard shortcut

**Display Sections:**

#### Fitted Profiles Tab
Real-time table of all fitted components in the current session:
- **Name**: Component identifier
- **Type**: Component type (gaussian, voigt, continuum, listfit)
- **Parameters**: Full fit details including:
  - Center wavelength (λ) with uncertainty
  - Amplitude with uncertainty
  - FWHM/Sigma with uncertainty
  - χ² and reduced χ²
  - Bounds (wavelength range fitted)
  - Line identifications (if applicable)
  - Redshift information (z_sys, rest wavelength)
  - Additional type-specific parameters (gamma for Voigt, polynomial order for continuum, etc.)

#### Saved Files Tab
Overview and management of `.qsap` files created during the current session:
- **Files List**: Scrollable list showing all `.qsap` files generated in this QSAP session
  - Display format: `{timestamp}_{fit_type}_{fit_mode}_{spectrum_name}.qsap`
  - Organized chronologically (newest first)
  - Click to select and view file contents
- **File Contents**: Scrollable text display panel showing:
  - Selected `.qsap` file in full (human-readable format)
  - Metadata header (version, spectrum info, timestamp)
  - All fitted component parameters in structured format
  - Quality metrics and error information
- **File Actions**:
  - **Open**: Open file in external text editor
  - **Reload Dir**: Refresh list if files added externally
  - **Copy**: Copy file path to clipboard

**Session Management:**
- File list resets when:
  - A new spectrum is loaded
  - QSAP application is restarted
  - User selects "Clear Files" button
- All `.qsap` files are automatically saved to disk in the configured save directory whenever a fit or calculation is performed

**Workflow:**
1. Perform a fit (Gaussian, Voigt, or Listfit)
2. Fit Information Window automatically opens or updates
3. Select components to view their detailed parameters
4. Export or save fit results to file
5. Use parameters for further analysis (equivalent width, line properties, etc.)

## QSAP File Format (.qsap)

QSAP uses a unified `.qsap` file format to store all fitting results in human-readable, structured text files.

**File Naming Convention:**
```
fit_{TIMESTAMP}_{FIT_TYPE}_{FIT_MODE}_{SPECTRUM_NAME}.qsap
```
Example: `fit_20260424_143022_gaussian_single_sample_spectrum.qsap`

**File Contents:**
- **Metadata Header**: Version, spectrum filename, wavelength range, creation timestamp
- **Fit Summary**: Fit type (Gaussian, Voigt, Continuum, Listfit), number of components, quality metrics
- **Component Parameters**: For each fitted component:
  - Center wavelength with uncertainty
  - Amplitude/strength with uncertainty
  - Width parameters (sigma, FWHM, gamma) with uncertainties
  - Chi-squared and reduced chi-squared values
  - Wavelength bounds
  - Associated line identifications (if any)

**File Storage:**
- Default location: User-configured save directory (set in Settings tab)
- Automatically generated after each fit (Gaussian, Voigt, Listfit, or Continuum)
- Files persist on disk after QSAP closes
- Session list in Fit Information Window resets on new spectrum/app restart, but files remain saved

**File Access:**
- View in Fit Information Window → Saved Files tab
- Open in external text editor via "Open" button
- Load fit parameters back into QSAP via Control Panel → "Load Fit" button

## Monte Carlo Estimation of Uncertainties

QSAP uses Monte Carlo sampling to estimate uncertainties on derived quantities like equivalent width (EW) and redshift. This approach generates thousands of synthetic spectra by randomly perturbing fitted parameters within their uncertainties, providing robust confidence intervals.

### How It Works

**Monte Carlo Process:**
1. Start with fit results including covariance matrix (parameter correlations and uncertainties)
2. Generate N samples (**N=1000–10000**) of fitted parameters from a multivariate normal distribution defined by the covariance matrix (i.e., randomly varying each parameter around its best-fit value according to its fitted uncertainty, while respecting correlations between parameters)
3. For each sample, recalculate the quantity of interest (EW or redshift)
4. Build a distribution of computed values
5. Extract percentiles to determine confidence intervals

**Toggling Monte Carlo Calculations:**

Two checkboxes control MC behavior (in Options Panel → Calculate section):

- **"Calculate Equivalent Width Automatically"** (default: ON)
  - Automatically runs MC equivalent width estimation after Gaussian/Voigt fits
  - Results saved in `.qsap` file in the `[EQUIVALENT_WIDTH]` block
  - MC EW profiles plotted to spectrum as shaded confidence regions (if next option enabled)

- **"Plot MC Profiles Automatically"** (default: OFF)
  - Controls visualization of Monte Carlo profile samples on the spectrum
  - When enabled: shows 100 randomly selected realized profiles (from the MC samples) as semi-transparent lines
  - Provides visual sense of uncertainty in profile shape
  - Can be toggled at any time; redraws spectrum with/without MC profiles

### Confidence Interval Calculation

**Important Clarification:** Confidence intervals are derived from the **distribution of MC samples** using **symmetric percentiles around the median**, not the mean.

**Sigma Levels Definition (68%–95%–99.7% rule):**

| Sigma Level | Lower Percentile | Upper Percentile | Coverage | Interpretation |
|------------|------------------|------------------|----------|----------------|
| 1-sigma | 16th | 84th | 68% | 68% of samples lie between these bounds |
| 2-sigma | 2.28th | 97.72nd | 95.4% | 95.4% of samples lie between these bounds |
| 3-sigma | 0.135th | 99.865th | 99.73% | 99.73% of samples lie between these bounds |

**Calculation** (using equivalent width as example):
1. Compute EW for each of N MC samples → distribution of EW values
2. Find median: 50th percentile = best estimate
3. Calculate 1-sigma: 
   - Lower bound = median − 16th percentile
   - Upper bound = 84th percentile − median
4. Asymmetric error bars reported as: `±{lower},+{upper}`

### Reporting in .qsap Files

**Equivalent Width Block:**
```
[EQUIVALENT_WIDTH]
EQUIVALENT_WIDTH_BEST=0.945231      (from fitted parameters, no MC)
EQUIVALENT_WIDTH_MEDIAN=0.924156    (median of MC distribution)
EQUIVALENT_WIDTH_MEAN=0.923445      (mean of MC distribution)
EQUIVALENT_WIDTH_1SIGMA=-0.078234,+0.081245  (1σ credible interval)
EQUIVALENT_WIDTH_2SIGMA=-0.156123,+0.162445  (2σ credible interval)
EQUIVALENT_WIDTH_3SIGMA=-0.234456,+0.243567  (3σ credible interval)
```

**Redshift Block:**
```
[REDSHIFT_DATA]
REDSHIFT_BEST=0.125634             (from single fitted line center)
REDSHIFT_MEDIAN=0.125612           (median of MC distribution)
REDSHIFT_MEAN=0.125598             (mean of MC distribution)
REDSHIFT_1SIGMA=-0.000234,+0.000241  (1σ credible interval)
REDSHIFT_2SIGMA=-0.000468,+0.000482  (2σ credible interval)
REDSHIFT_3SIGMA=-0.000702,+0.000723  (3σ credible interval)
```

### Best Practices

- **EW Calculations**: Use MC uncertainty estimates for published results (more robust than parameter covariance)
- **Low S/N Spectra**: MC provides realistic uncertainty estimates even when parameter errors seem underestimated
- **Covariance Quality**: If continuum fit is poor, MC uncertainties may be inflated but generally more trustworthy than formal errors
- **Plot MC Profiles**: Enable for visual inspection of fit quality and realistic profile variations

## Versions

- **v1.1** (current): Complete UI redesign with tabbed Control Panel (Control Panel, Item Tracker, Settings tabs), new Options panel with fit and calculate controls, bottom-docked Terminal/Output panel, improved line list offset precision, support for air-to-vacuum wavelength conversion, wavelength unit selection (Ångström/nm/µm), configurable NaN/Inf replacement, graceful application shutdown.
- **v0.12** (stable): First implementation of more user-friendly GUI
- **v0.11** (stable): Action history (undo/redo actions), fit information window, output log (printing from terminal) in spectrum viewer, improved item tracker functionality.
- **v0.10** (stable): Added robust Listfit capabilities. Listfit now works for higher-order Polynomials (>2) and with multiple line profiles fitted simultaneously.
- **v0.9** (stable): Fixed Listfit parameter error extraction, PolynomialModel integration for proper coefficient variation, Item Tracker synchronization with internal storage removal
- **v0.8**: Listfit mode for simultaneous multi-component fitting, ItemTracker for centralized feature management with multi-select and deletion, auto-fit registration, redshift mode improvements
- **v0.7** (stable): Available as tagged release on GitHub. Refactored with intelligent format auto-detection, modular architecture, and comprehensive UI
- **v0.5** (legacy): Available in `qsap_v0.5/` directory with full Voigt/Gaussian fitting, MCMC, velocity mode

## Citation

```
Solhaug, E. (2025). QSAP: Quick Spectrum Analysis Program.
https://github.com/eriksolhaug/qsap
```

## License

MIT License - See LICENSE file

## Author

Erik Solhaug
