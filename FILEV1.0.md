# QSAP File Format v1.0 Specification

## Overview

The `.qsap` file format is a unified, human-readable plain-text format for storing spectral fit results from QSAP (Quick Spectrum Analysis Program). It supports multiple fit types including Gaussian, Voigt, Continuum, Listfit, and Redshift analyses, all while maintaining a consistent structural organization.

**File Extension**: `.qsap`  
**Format**: Plain text (UTF-8)  
**Encoding**: Can be opened and edited in any text editor  
**Version**: 1.0  

---

## Terminology

### Block
A "block" refers to a labeled section within a `.qsap` file that groups related information. Each block is demarcated by a header in square brackets (e.g., `[METADATA]`, `[COMPONENT_1]`). Blocks are separated by blank lines for readability.

### Block Types
- **METADATA**: Contains global file-level information and spectrum details
- **COMPONENT_N**: Contains individual fit component parameters (Gaussian, Voigt, Polynomial in Listfit)
- **CONTINUUM_N**: Contains continuum polynomial parameters
- **REDSHIFT_DATA**: Contains redshift determination results

---

## File Naming Convention

Filenames follow the pattern:
```
fit_YYYYMMDD_HHMMSS_<TYPE>_<MODE>_<SPECTRUM>.qsap
```

Where:
- `YYYYMMDD_HHMMSS`: ISO timestamp (year, month, day, hour, minute, second)
- `<TYPE>`: One of `gaussian`, `voigt`, `continuum`, `listfit`, `redshift`
- `<MODE>`: One of `single`, `multi`, `listfit` (or fit-specific designation)
- `<SPECTRUM>`: Base filename of the spectrum file (without path)

**Example**: `fit_20260421_093045_gaussian_multi_sample_spectrum.qsap`

---

## METADATA Block

The METADATA block appears at the beginning of every `.qsap` file and contains universal information about the fit.

### Required Fields
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=<Gaussian|Voigt|Continuum|Listfit|Redshift>
SPECTRUM_FILE=<basename>
DATE_TIME=<ISO timestamp>
```

### Optional Fields
```
MODE=<Single|Multi-Gaussian|Listfit|...>
WAVELENGTH_UNIT=<Angstrom|nm|...>
WAVELENGTH_RANGE=<min>-<max>
REST_WAVELENGTH=<value>
VELOCITY_MODE=<True|False>
ZOOM_FACTOR=<float>
PARENT_FIT_ID=<id>                    # For child fits
PARENT_COMPONENT_ID=<id>              # For child components
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `FILE_FORMAT_VERSION` | Always "1.0" for this specification |
| `TYPE` | The type of fit stored (see Fit Types below) |
| `MODE` | Specific fitting mode (Multi-Gaussian, Listfit, etc.) |
| `SPECTRUM_FILE` | Base name of the source spectrum file |
| `DATE_TIME` | ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SS.ffffff) |
| `WAVELENGTH_UNIT` | Physical unit of wavelength values (typically Angstrom) |
| `WAVELENGTH_RANGE` | Range of wavelengths analyzed (min-max format) |
| `REST_WAVELENGTH` | Rest-frame wavelength for velocity conversions |
| `VELOCITY_MODE` | Whether spectrum is in velocity or wavelength space |
| `ZOOM_FACTOR` | Zoom level used during analysis (for reproducibility) |
| `PARENT_FIT_ID` | For derived fits, ID of parent fit |
| `PARENT_COMPONENT_ID` | For derived fits, ID of parent component |

**Example METADATA block:**
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Gaussian
MODE=Multi-Gaussian
SPECTRUM_FILE=sample_spectrum.txt
DATE_TIME=2026-04-21T09:30:45.123456
WAVELENGTH_UNIT=Angstrom
WAVELENGTH_RANGE=1205.00-1220.00
REST_WAVELENGTH=1215.67
VELOCITY_MODE=False
ZOOM_FACTOR=1.2

```

---

## Fit Types and Their Blocks

### 1. Gaussian Fits

**File Type**: `TYPE=Gaussian`  
**Modes**: `Single`, `Multi-Gaussian`  
**Block Type**: `[COMPONENT_N]`

Stores individual Gaussian line profile fits. Each component is a separate `[COMPONENT_N]` block.

#### COMPONENT Block Structure
```
[COMPONENT_N]
TYPE=Gaussian
FIT_ID=<identifier>
COMPONENT_ID=<identifier>
LINE_ID=<line_name>
LINE_WAVELENGTH=<wavelength>
REST_WAVELENGTH=<rest_wavelength>
AMPLITUDE=<value>±<error>
MEAN=<value>±<error>
STD_DEV=<value>±<error>
BOUNDS_LOWER=<wavelength>
BOUNDS_UPPER=<wavelength>
CHI_SQUARED=<value>
CHI_SQUARED_NU=<reduced_chi_squared>
VELOCITY_MODE=<True|False>
SYSTEM_REDSHIFT=<redshift_value>

```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `TYPE` | Always "Gaussian" |
| `FIT_ID` | Unique identifier for the fit session |
| `COMPONENT_ID` | Unique identifier for this specific component |
| `LINE_ID` | Astronomical line designation (e.g., "Ly-alpha", "CIV") |
| `LINE_WAVELENGTH` | Observed wavelength of the line |
| `REST_WAVELENGTH` | Rest-frame wavelength of the line |
| `AMPLITUDE` | Gaussian peak intensity with error |
| `MEAN` | Gaussian center wavelength with error |
| `STD_DEV` | Gaussian width (σ) with error |
| `BOUNDS_LOWER` | Fitting region lower bound |
| `BOUNDS_UPPER` | Fitting region upper bound |
| `CHI_SQUARED` | χ² statistic |
| `CHI_SQUARED_NU` | Reduced χ² (χ²/DOF) |
| `EQUIVALENT_WIDTH` | Equivalent width with error (when calculable) |
| `VELOCITY_MODE` | Whether fit was in velocity space |
| `SYSTEM_REDSHIFT` | System redshift if applicable |

#### Example: Single Gaussian
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Gaussian
MODE=Single
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T09:30:45.123456
WAVELENGTH_UNIT=Angstrom

[COMPONENT_1]
TYPE=Gaussian
FIT_ID=fit_001
COMPONENT_ID=comp_001
LINE_ID=Ly-alpha
LINE_WAVELENGTH=1215.89
REST_WAVELENGTH=1215.67
AMPLITUDE=2.345e+02±1.2e+01
MEAN=1215.89±0.05
STD_DEV=0.85±0.03
BOUNDS_LOWER=1214.50
BOUNDS_UPPER=1217.20
CHI_SQUARED=45.67
CHI_SQUARED_NU=1.23
EQUIVALENT_WIDTH=0.412±0.008
VELOCITY_MODE=False

```

#### Example: Multi-Gaussian
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Gaussian
MODE=Multi-Gaussian
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T09:30:45.123456

[COMPONENT_1]
TYPE=Gaussian
AMPLITUDE=2.345e+02±1.2e+01
MEAN=1215.89±0.05
STD_DEV=0.85±0.03
...

[COMPONENT_2]
TYPE=Gaussian
AMPLITUDE=1.234e+02±0.8e+01
MEAN=1218.45±0.03
STD_DEV=1.23±0.05
...

```

---

### 2. Voigt Fits

**File Type**: `TYPE=Voigt`  
**Modes**: `Single`, `Multi-Voigt`  
**Block Type**: `[COMPONENT_N]`

Stores Voigt profile fits, which combine Gaussian and Lorentzian components for absorption/emission lines.

#### COMPONENT Block Structure
```
[COMPONENT_N]
TYPE=Voigt
FIT_ID=<identifier>
COMPONENT_ID=<identifier>
LINE_ID=<line_name>
LINE_WAVELENGTH=<wavelength>
REST_WAVELENGTH=<rest_wavelength>
AMPLITUDE=<value>±<error>
MEAN=<value>±<error>
SIGMA=<value>±<error>
GAMMA=<value>±<error>
B_DOPPLER=<value>
LOG_T_EFF=<value>
BOUNDS_LOWER=<wavelength>
BOUNDS_UPPER=<wavelength>
CHI_SQUARED=<value>
CHI_SQUARED_NU=<reduced_chi_squared>
VELOCITY_MODE=<True|False>
SYSTEM_REDSHIFT=<redshift_value>

```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `TYPE` | Always "Voigt" |
| `AMPLITUDE` | Voigt profile peak intensity with error |
| `MEAN` | Voigt profile center with error |
| `SIGMA` | Gaussian width component (σ) with error |
| `GAMMA` | Lorentzian width component (damping) with error |
| `B_DOPPLER` | Doppler b-parameter (km/s) |
| `LOG_T_EFF` | Effective temperature (log scale) |

#### Example: Voigt Fit
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Voigt
MODE=Single
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T10:15:30.654321

[COMPONENT_1]
TYPE=Voigt
LINE_ID=CIV
AMPLITUDE=3.456e+02±2.1e+01
MEAN=1548.76±0.08
SIGMA=1.23±0.05
GAMMA=0.45±0.02
B_DOPPLER=45.2
LOG_T_EFF=4.5
BOUNDS_LOWER=1546.00
BOUNDS_UPPER=1551.00
CHI_SQUARED=52.34
CHI_SQUARED_NU=1.31
EQUIVALENT_WIDTH=0.567±0.012

```

---

### 3. Continuum Fits

**File Type**: `TYPE=Continuum`  
**Modes**: `Single`  
**Block Type**: `[CONTINUUM_N]`

Stores polynomial continuum fits, useful for modeling smooth background variations in spectra.

#### CONTINUUM Block Structure
```
[CONTINUUM_N]
TYPE=Continuum
POLY_ORDER=<order>
BOUNDS_LOWER=<wavelength>
BOUNDS_UPPER=<wavelength>
COEFF_0=<value>±<error>
COEFF_1=<value>±<error>
COEFF_2=<value>±<error>
...
VELOCITY_MODE=<True|False>

```

#### Field Descriptions

| Field | Description |
|-------|-------------|
| `TYPE` | Always "Continuum" |
| `POLY_ORDER` | Polynomial degree (e.g., 1 for linear, 2 for quadratic) |
| `BOUNDS_LOWER` | Wavelength range lower bound |
| `BOUNDS_UPPER` | Wavelength range upper bound |
| `COEFF_N` | Polynomial coefficient and its error (from `np.polyfit`) |

The polynomial coefficients are stored in order from highest degree to lowest, matching the `np.polyfit` convention. For example, a 2nd-order polynomial uses `COEFF_0` (x²), `COEFF_1` (x), and `COEFF_2` (constant).

#### Example: Continuum Fit
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Continuum
MODE=Single
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T10:45:12.123456
WAVELENGTH_RANGE=1200.00-1250.00

[CONTINUUM_1]
TYPE=Continuum
POLY_ORDER=2
BOUNDS_LOWER=1200.50
BOUNDS_UPPER=1249.75
COEFF_0=-1.234e-05±5.6e-07
COEFF_1=3.456e-02±1.2e-03
COEFF_2=1.234e+02±2.3e+00
VELOCITY_MODE=False

```

---

### 4. Listfit Fits

**File Type**: `TYPE=Listfit`  
**Modes**: `Listfit`  
**Block Types**: `[METADATA]`, `[COMPONENT_N]`, `[FIT_DIAGNOSTICS]`

Stores multiple fit components of potentially mixed types (Gaussian, Voigt, Polynomial) in a single file. Each component is a separate block with its own TYPE specification.

#### Structure
- One `[METADATA]` block containing global fit information
- Multiple `[COMPONENT_N]` blocks, each with its own TYPE field

#### COMPONENT Block Structure

Each component block begins with `TYPE=Gaussian`, `TYPE=Voigt`, or `TYPE=Polynomial` to specify its nature, then includes the relevant fit parameters.

**Gaussian Component:**
```
[COMPONENT_N]
TYPE=Gaussian
AMPLITUDE=<value>±<error>
MEAN=<value>±<error>
STD_DEV=<value>±<error>
...
```

**Voigt Component:**
```
[COMPONENT_N]
TYPE=Voigt
AMPLITUDE=<value>±<error>
MEAN=<value>±<error>
SIGMA=<value>±<error>
GAMMA=<value>±<error>
...
```

**Polynomial Component:**
```
[COMPONENT_N]
TYPE=Polynomial
POLY_ORDER=<order>
COEFF_0=<value>±<error>
COEFF_1=<value>±<error>
...
```

#### FIT_DIAGNOSTICS Block Structure

The `[FIT_DIAGNOSTICS]` block contains comprehensive diagnostic measures from the lmfit fitting engine. This block appears **after all component blocks** and provides quality metrics for the overall fit.

**Fields:**
- `SSR`: Sum of squared residuals (always present)
- `SSR_NU`: SSR divided by degrees of freedom = reduced chi-squared value (always present)
- `CHI2`: Chi-squared statistic (only present if error spectrum provided; otherwise `None`)
- `CHI2_REDUCED`: Reduced chi-squared = chi-squared / degrees of freedom (only if error spectrum; otherwise `None`)
- `AKAIKE_INFO_CRITERION`: AIC model selection criterion (always present)
- `BAYESIAN_INFO_CRITERION`: BIC model selection criterion (always present)
- `R_SQUARED`: Coefficient of determination R² (present if calculable from residuals; may be `None` for poor fits)
- `N_DATA_POINTS`: Number of data points used in fitting
- `N_PARAMETERS`: Number of free parameters in the fit
- `N_DEGREES_FREEDOM`: Degrees of freedom (data points - parameters)
- `FIT_SUCCESS`: Boolean indicating successful fit convergence

**Example:**
```
[FIT_DIAGNOSTICS]
SSR=45.231
SSR_NU=0.8923
CHI2=45.231
CHI2_REDUCED=0.8923
AKAIKE_INFO_CRITERION=234.567
BAYESIAN_INFO_CRITERION=251.234
R_SQUARED=0.9876
N_DATA_POINTS=56
N_PARAMETERS=8
N_DEGREES_FREEDOM=48
FIT_SUCCESS=True
```

#### Example: Listfit with Mixed Components
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Listfit
MODE=Listfit
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T11:20:33.987654
WAVELENGTH_RANGE=1200.00-1300.00

[COMPONENT_1]
TYPE=Gaussian
LINE_ID=Ly-alpha
AMPLITUDE=2.345e+02±1.2e+01
MEAN=1215.89±0.05
STD_DEV=0.85±0.03
BOUNDS_LOWER=1214.50
BOUNDS_UPPER=1217.20

[COMPONENT_2]
TYPE=Voigt
LINE_ID=CIV
AMPLITUDE=1.567e+02±0.9e+01
MEAN=1548.76±0.08
SIGMA=1.23±0.05
GAMMA=0.45±0.02
BOUNDS_LOWER=1546.00
BOUNDS_UPPER=1551.00

[COMPONENT_3]
TYPE=Polynomial
POLY_ORDER=1
BOUNDS_LOWER=1200.00
BOUNDS_UPPER=1214.50
COEFF_0=0.01234±0.00012
COEFF_1=98.765±1.23

[FIT_DIAGNOSTICS]
SSR=128.456
SSR_NU=2.301
CHI2=128.456
CHI2_REDUCED=2.301
AKAIKE_INFO_CRITERION=567.123
BAYESIAN_INFO_CRITERION=589.456
R_SQUARED=0.9234
N_DATA_POINTS=70
N_PARAMETERS=11
N_DEGREES_FREEDOM=59
FIT_SUCCESS=True

```

---

### 5. Redshift Fits

**File Type**: `TYPE=Redshift`  
**Modes**: Single determination  
**Block Type**: `[REDSHIFT_DATA]`

Stores redshift determination results, typically derived from a parent Gaussian or Voigt fit.

#### Block Structure
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Redshift
SPECTRUM_FILE=<basename>
DATE_TIME=<ISO timestamp>
PARENT_FIT_ID=<id>
PARENT_COMPONENT_ID=<id>

[REDSHIFT_DATA]
<parameter_name>=<value>
<parameter_name>=<value>
...
```

#### Common Redshift Parameters

| Parameter | Description |
|-----------|-------------|
| `REDSHIFT` | Determined redshift value (z) |
| `REST_WAVELENGTH` | Rest-frame wavelength |
| `OBSERVED_WAVELENGTH` | Observed wavelength |
| `RADIAL_VELOCITY` | Radial velocity (km/s) |
| `HELIOCENTRIC_VELOCITY` | Heliocentric velocity (km/s) |
| `SYSTEMIC_VELOCITY` | Systemic velocity (km/s) |
| `ERROR_REDSHIFT` | Uncertainty in redshift |
| `ERROR_VELOCITY` | Uncertainty in velocity |
| `METHOD` | Method used for determination |

#### Example: Redshift Fit
```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Redshift
SPECTRUM_FILE=sample.txt
DATE_TIME=2026-04-21T11:45:22.555555
PARENT_FIT_ID=fit_001
PARENT_COMPONENT_ID=comp_001

[REDSHIFT_DATA]
REDSHIFT=0.00456
REST_WAVELENGTH=1215.67
OBSERVED_WAVELENGTH=1220.23
RADIAL_VELOCITY=1368.5
HELIOCENTRIC_VELOCITY=1362.1
ERROR_REDSHIFT=0.00012
ERROR_VELOCITY=36.0
METHOD=Gaussian_Line_Center

```

---

## Parameter Value Formatting

### Standard Format
Simple numeric values are written as plain numbers:
```
VALUE=1.234
VALUE=5.6e-07
```

### Values with Uncertainties
Parameters with measurement errors use the format:
```
PARAMETER=<value>±<error>
```

Examples:
```
AMPLITUDE=2.345e+02±1.2e+01
MEAN=1215.89±0.05
COEFF_0=-1.234e-05±5.6e-07
```

This notation clearly indicates both the measured value and its associated measurement uncertainty.

### Boolean Values
```
VELOCITY_MODE=True
VELOCITY_MODE=False
```

---

## Complete File Example: Multi-Component Listfit

```
[METADATA]
FILE_FORMAT_VERSION=1.0
TYPE=Listfit
MODE=Listfit
SPECTRUM_FILE=NGC6240_spectrum.txt
DATE_TIME=2026-04-21T12:30:45.123456
WAVELENGTH_UNIT=Angstrom
WAVELENGTH_RANGE=1205.00-1560.00
REST_WAVELENGTH=0.0
VELOCITY_MODE=False
ZOOM_FACTOR=1.5

[COMPONENT_1]
TYPE=Gaussian
FIT_ID=fit_ngc6240_001
COMPONENT_ID=comp_001
LINE_ID=Ly-alpha
LINE_WAVELENGTH=1215.98
REST_WAVELENGTH=1215.67
AMPLITUDE=3.456e+02±2.1e+01
MEAN=1215.98±0.06
STD_DEV=0.92±0.04
BOUNDS_LOWER=1214.00
BOUNDS_UPPER=1217.50
CHI_SQUARED=38.45
CHI_SQUARED_NU=1.15

[COMPONENT_2]
TYPE=Voigt
FIT_ID=fit_ngc6240_001
COMPONENT_ID=comp_002
LINE_ID=CIV_1548
LINE_WAVELENGTH=1549.01
REST_WAVELENGTH=1548.20
AMPLITUDE=2.123e+02±1.5e+01
MEAN=1549.01±0.08
SIGMA=1.34±0.06
GAMMA=0.58±0.03
B_DOPPLER=52.3
LOG_T_EFF=4.6
BOUNDS_LOWER=1546.50
BOUNDS_UPPER=1551.50
CHI_SQUARED=42.12
CHI_SQUARED_NU=1.28

[COMPONENT_3]
TYPE=Voigt
FIT_ID=fit_ngc6240_001
COMPONENT_ID=comp_003
LINE_ID=CIV_1550
LINE_WAVELENGTH=1550.65
REST_WAVELENGTH=1550.77
AMPLITUDE=1.856e+02±1.2e+01
MEAN=1550.65±0.09
SIGMA=1.41±0.07
GAMMA=0.52±0.02
B_DOPPLER=51.8
LOG_T_EFF=4.5
BOUNDS_LOWER=1548.00
BOUNDS_UPPER=1553.00
CHI_SQUARED=39.87
CHI_SQUARED_NU=1.21

```

---

## Best Practices

### File Organization
1. **One fit per file**: Each `.qsap` file should represent a single fitting session or analysis
2. **Consistent naming**: Use the standard naming convention for easy identification
3. **Metadata accuracy**: Ensure METADATA block is complete and accurate

### Data Preservation
1. **Error tracking**: Always include measurement uncertainties with values
2. **Bounds documentation**: Record the wavelength bounds used for each fit
3. **Reproducibility**: Store ZOOM_FACTOR and VELOCITY_MODE for session reproducibility

### Version Control
- `.qsap` files should be tracked in version control (commit and push regularly)
- Use meaningful commit messages referencing the fit types and sources
- Consider excluding large spectral data files but keeping `.qsap` summaries

---

## File Validation

A valid `.qsap` file must:
1. ✓ Start with a `[METADATA]` block
2. ✓ Contain `FILE_FORMAT_VERSION=1.0` in METADATA
3. ✓ Contain a `TYPE` field indicating the fit type
4. ✓ Include appropriate component blocks for the specified type
5. ✓ Use consistent parameter formatting throughout
6. ✓ Have all sections properly delimited with block headers

---

## Future Extensions

This specification is designed for extensibility. Future versions may include:
- Covariance matrix storage for correlated parameters
- Chi-squared distribution details
- Integration with astronomical databases (SIMBAD, NED references)
- Data quality flags and pipeline provenance
- Absorption/emission line system associations

---

## References

- QSAP GitHub: [link to repository]
- Spectral fitting best practices: [relevant papers/links]
- ISO 8601 timestamp format: https://en.wikipedia.org/wiki/ISO_8601

---

**Last Updated**: 2026-04-21  
**Specification Version**: 1.0  
**Format Authority**: QSAP Development Team
