# <img src="logo/qasap_logo.png" alt="QASAP Logo" width="80">&nbsp;&nbsp;QASAP: Quick Analysis of Spectra and Profiles

*An Analysis Tool for Astronomical Spectra*

## Version 0.11

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15%2B-blue?logo=pyqt&logoColor=white)
![numpy](https://img.shields.io/badge/numpy-1.17%2B-yellow)
![scipy](https://img.shields.io/badge/scipy-1.5%2B-brightgreen?logo=python&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.1%2B-orange?logo=python&logoColor=white)
![astropy](https://img.shields.io/badge/astropy-4.0%2B-red?logo=python&logoColor=white)
![lmfit](https://img.shields.io/badge/lmfit-0.9%2B-purple?logo=python&logoColor=white)
![emcee](https://img.shields.io/badge/emcee-3.0%2B-yellowgreen?logo=python&logoColor=white)

Interactive Python tool for 1D spectral analysis with file format detection. QASAP provides both quick-look functionality and some more advanced analysis features including multi-component line fitting, continuum modeling, and Bayesian MCMC fitting.

## Features

- **File Format Auto-Detection**: Automatically detects 7+ ASCII and FITS spectrum formats
- **Interactive Plotting**: 1D spectrum visualization with a variety of navigation controls
- **Multi-Component Fitting**: Gaussian and Voigt profile fitting (single and multi-component) with modes for simultaneous fitting using unconstrained and/or constrained optimization
- **Line Analysis**: Redshift estimation, dynamic line list display and velocity restframe toggling
- **Item and Action Trackers**: Centralized management of all components (Gaussians, Voigts, polynomials, continuum) with multi-select and deletion + undo/redo capabilities
- **Utilities**: Filter overlays, smoothing functionality, etc.

## Installation

### Requirements
- Python 3.7+
- numpy, scipy, matplotlib, astropy, pandas, lmfit, PyQt5, emcee, corner

### Download

```bash
git clone https://github.com/eriksolhaug/qasap.git
```

To **download a specific tagged version** (e.g., v0.9):

```bash
# Clone only the tag (shallow clone, fastest)
git clone --depth 1 --branch v0.9 https://github.com/eriksolhaug/qasap.git

# Or clone the whole repo and checkout the tag
git clone https://github.com/eriksolhaug/qasap.git
cd qasap
git checkout v0.9
```

### Conda Environment Setup

Then install the required packages (regardless of version):

```bash
# Create a new conda environment
conda create -n qasap python=3.8

# Activate the environment
conda activate qasap

# Enter qasap directory
cd qasap # This is the repo you cloned from github

# Install dependencies
pip install -r requirements.txt
# OR install requirements using conda
conda install numpy scipy matplotlib astropy pandas lmfit pyqt emcee corner

# Run QASAP
python qasap.py <spectrum.fits>
```


## Usage

### Auto-Detection (Recommended)

```bash
# Automatic format detection
python qasap.py <spectrum.fits>

# Preview detected formats
python qasap.py <spectrum.fits> --detect

# Force specific format if needed
python qasap.py <spectrum.fits> --fmt fits:image1d
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
--zoom_factor      Y-axis zoom for `y` key (default: 0.1)
--lsf              LSF width in km/s or path to file (default: "10", in development)
--gui              Launch GUI for interactive input
```

### Making `qasap` Executable

To run QASAP from anywhere as a simple `qasap` command, use one of these options:

**Option 1: Package Installation (Recommended)**

Install QASAP as a Python package, which automatically creates the executable:

```bash
# From inside the qasap directory
pip install .

# Or in development mode (allows you to modify the code and run it with the updated changes using the qasap command)
pip install -e .
```

Now you can run QASAP from anywhere:

```bash
qasap <path/to/spectrum.fits>
```

This method uses the entry point defined in `setup.py` to create a proper command-line executable.

**Option 2: Manual Symlink**

1. **Make the script executable:**
   ```bash
   chmod +x qasap.py
   ```

2. **Create a symlink in a directory on your PATH:**
   ```bash
   # Find your qasap installation path
   QASAP_PATH=$(pwd)/qasap.py
   
   # Link to a bin directory in your PATH (example: /usr/local/bin)
   sudo ln -s $QASAP_PATH /usr/local/bin/qasap
   ```

## Upgrading Versions

If you already have QASAP installed (e.g., v0.9) and want to upgrade to a newer version (e.g., v0.10), follow these steps:

### Method 1: Clean Installation (Recommended)

This is the safest approach if you installed with `pip install -e .`:

```bash
# 1. Deactivate your conda environment (if using conda)
conda deactivate

# 2. Remove the old installation
pip uninstall qasap

# 3. Remove the old local directory (optional but recommended)
rm -rf /path/to/old/qasap

# 4. Clone the new version
git clone https://github.com/eriksolhaug/qasap.git
# OR clone a specific version tag
git clone --depth 1 --branch v0.10 https://github.com/eriksolhaug/qasap.git

# 5. Navigate to the new qasap directory
cd qasap

# 6. Reactivate your conda environment
conda activate qasap

# 7. Install the new version
pip install -e .

# 8. Verify installation
qasap --help
```

### Method 2: Update Existing Repository

If you cloned from git and want to update to a newer version:

**Option A: Update to a Specific Version Tag (e.g., v0.10)**

```bash
# 1. Navigate to your qasap directory
cd </path/to/qasap>

# 2. Fetch all available versions/tags from remote
git fetch origin

# 3. Checkout the specific version you want
git checkout v0.10

# 4. Reinstall (if dependencies changed)
pip install -e .

# 5. Verify
qasap --help
```

**Option B: Update to the Latest Development Version**

```bash
# 1. Navigate to your qasap directory
cd /path/to/qasap

# 2. Pull the latest changes
git pull origin main

# 3. Reinstall (if dependencies changed)
pip install -e .

# 4. Verify
qasap --help
```

# 4. Reinstall the package (if dependencies changed)
pip install -e .

# 5. Verify the update
qasap --help
```

### Method 3: Manual Update (Without Git)

If you don't have git or prefer not to use it:

```bash
# 1. Download the new version from GitHub (as a zip file)
# Visit: https://github.com/eriksolhaug/qasap/releases
# Or use curl to download a specific tag:
curl -L -o qasap-v0.10.zip https://github.com/eriksolhaug/qasap/archive/refs/tags/v0.10.zip

# 2. Unzip the downloaded file
unzip qasap-v0.10.zip

# 3. Remove the old qasap directory
rm -rf /path/to/old/qasap

# 4. Navigate to the new directory
cd qasap-v0.10

# 5. Reinstall
pip install -e .

# 6. Verify
qasap --help
```

### Troubleshooting Version Updates

**"command not found: qasap"** after updating:
- Make sure your conda environment is activated: `conda activate qasap`
- Try reinstalling: `pip install -e .`

**Import errors or missing modules**:
- Update dependencies: `pip install -r requirements.txt`
- Or reinstall all requirements: `pip install --upgrade -r requirements.txt`

**Old version still running**:
- Check which qasap is being used: `which qasap`
- Verify it points to your new installation
- If it's a symlink (Method 2 of initial install), update it to point to the new location

## Quick Start

### Running QASAP

Once installed with `pip install -e .`, you have two ways to launch QASAP:

**Without a Spectrum File:**
```bash
qasap
```
This launches QASAP with an empty plotter. You can then load a spectrum using the Open button in the Control Panel.

**With a Spectrum File:**
```bash
qasap <path/to/spectrum.fits>
qasap example/sample_spectrum.txt
qasap data/my_spectrum.txt --redshift 0.1
```

### Opening a Spectrum

You can load a spectrum file in two ways:

1. **Command Line**: Provide the file path as an argument when launching:
   ```bash
   qasap /path/to/your/spectrum.fits
   ```

2. **GUI Open Button**: 
   - Launch QASAP with `qasap` (no file required)
   - In the Control Panel (left side), click the **Open** button
   - A file browser will appear—navigate to and select your spectrum file
   - QASAP automatically detects the file format and loads the data

## Package Structure

```
.
├── qasap.py                 # Main entry point (root level)
├── setup.py                 # Package setup and installation
├── requirements.txt         # Python dependencies
├── LICENSE
├── README.md
├── __init__.py              # Package initialization
├── qasap/                   # Main package directory
│   ├── __init__.py
│   ├── spectrum_io.py           # File I/O with auto-detection
│   ├── spectrum_analysis.py     # Fitting and analysis functions
│   ├── spectrum_plotter.py      # Main visualization widget
│   ├── spectrum_plotter_app.py  # Application wrapper
│   ├── ui_components.py         # UI component exports
│   ├── linelist_window.py       # Line identification window
│   ├── linelist_selector_window.py # Line list management UI
│   ├── linelist.py              # Line list data structures
│   ├── listfit_window.py        # Multi-component fitting dialog
│   ├── item_tracker.py          # Component tracking and management
│   └── format_picker_dialog.py  # Format selection dialog
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
- `a` - Save Gaussian fit info to file
- `A` - Save Voigt fit info to file
- `S` - Save continuum fit info to file
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
All saved screenshots, redshifts, and profile info are stored in the directory where QASAP was launched from.

### Fitting Engines by Mode

QASAP employs different fitting algorithms optimized for each analysis task:

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


### 2. Control Panel
Displays real-time analysis information and fitting parameters:

**Sections:**
- **Quick Stats**: Line center, FWHM, equivalent width, signal-to-noise
- **Redshift**: Set redshift for displayed emission lines (toggled with key `e`)
- **Polynomial Order**: Set the order for the polynomial for Continuum mode (with key `m`)

### 3. LineList Window
Interactive line identification and management interface for viewing and selecting spectral lines:

**Access:**
- Press `e` on the spectrum to open the LineList Window
- Provides access to multiple line list catalogs

**Line Lists:**
- **emlines.txt**: Common emission lines (Hα, Hβ, [OIII], [OII], Ly-α, etc.)
- **emlines_osc.txt**: Emission lines with oscillator strength data
- **instrument_bands.txt**: Instrument and filter bandpass definitions
- **sdss_emission.txt**: SDSS emission line catalog
- **sdss_absorption.txt**: SDSS stellar absorption features
- **sdss_sky.txt**: Telluric sky emission lines

**Interface:**
The LineList Window uses a dual-panel design:

**Left Panl - Line List Selection:**
- Displays all available line list catalogs with line counts
- Format: `{ListName} ({count} lines)`
- Scrollable
- Click to select a line list and view its contents

**Right Panel - Lines Display:**
- Dynamically populates when you select a line list from the left panel
- Shows all lines in the format: `{LineName}: {Wavelength} Å`
- Scrollable
- Double-click a line to select it for redshift estimation

**Workflow:**
1. Press `e` to open LineList Window
2. Select a line list from the left panel (e.g., "sdss_emission.txt")
3. Right panel populates with all lines from that list
4. Double-click a line (e.g., "H alpha: 6564.61 Å")
5. Window closes and redshift is estimated from the selected line (see terminal for output)

**Redshift Integration:**
Lines in the spectrum automatically update when redshift is changed via:
- Manual entry in Control Panel
- Arrow buttons in the LineList Window
- Automatic redshift estimation from fitted components (`z` key)

### 4. Listfit Window (Multi-Component Fitting)
Dialog for simultaneous fitting of multiple spectral components:

**Workflow:**
1. Press `H` on spectrum to activate listfit mode
2. Define fitting bounds by clicking two wavelengths (spacebar confirms each)
3. Listfit Window opens with component selection
4. Choose component types and quantities:
   - **Gaussian**: Simple Gaussian profiles
   - **Voigt**: Gaussian + Lorentzian (more realistic for emission lines)
   - **Polynomial**: Continuum background (specify order)
5. Click "Calculate Fit" to perform simultaneous fitting
6. Fitted components displayed with individual colors:
   - Red: Gaussian components
   - Orange: Voigt components
   - Magenta: Polynomial continuum
   - Dark Blue: Combined total fit

**Parameters:**
- Each component shows fitted parameters (amplitude, center wavelength, width/sigma)
- Errors computed from covariance matrix
- Quality metrics: χ² and reduced χ²

### 5. Item Tracker Window
Centralized feature management panel accessed with `*` key:

**Display Columns:**
- **Name**: Feature identifier (e.g., "Gaussian 1", "Voigt 2")
- **Type**: Component type (gaussian, voigt, polynomial, continuum)
- **Position**: Wavelength bounds or center position
- **Color**: Visual indicator with color box showing plot color

**Operations:**
- Multi-select items with Control/Shift+Click
- Right-click context menu: "Delete" to remove from plot
- "Delete Selected" button: Remove multiple items at once
- "Clear All" button: Remove all features

## Versions

- **v0.11** (current): Action history (undo/redo actions), fit information window, output log (printing from terminal) in spectrum viewer, improved item tracker functionality.
- **v0.10** (stable): Added robust Listfit capabilities. Listfit now works for higher-order Polynomials (>2) and with multiple line profiles fitted simultaneously.
- **v0.9** (stable): Fixed Listfit parameter error extraction, PolynomialModel integration for proper coefficient variation, Item Tracker synchronization with internal storage removal
- **v0.8**: Listfit mode for simultaneous multi-component fitting, ItemTracker for centralized feature management with multi-select and deletion, auto-fit registration, redshift mode improvements
- **v0.7** (stable): Available as tagged release on GitHub. Refactored with intelligent format auto-detection, modular architecture, and comprehensive UI
- **v0.5** (legacy): Available in `qasap_v0.5/` directory with full Voigt/Gaussian fitting, MCMC, velocity mode

## Citation

```
Solhaug, E. (2025). QASAP: Quick Analysis of Spectra and Profiles.
https://github.com/eriksolhaug/qasap
```

## License

MIT License - See LICENSE file

## Author

Erik Solhaug
