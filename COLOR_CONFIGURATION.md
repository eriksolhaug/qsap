# QASAP Color Configuration System

QASAP now uses a centralized color configuration system that allows you to customize all colors and line styles used throughout the application.

## Configuration File

Colors are defined in `qasap/config_colors.json`. This file is loaded when QASAP starts, and all plotting calls reference this configuration instead of using hardcoded color values.

## Configuration Structure

The configuration file is organized into the following sections:

### `profiles` - Spectral Profile Colors and Styles

```json
"profiles": {
  "gaussian": {
    "color": "red",
    "linestyle": "--",
    "linewidth": 1.5
  },
  "voigt": {
    "color": "orange",
    "linestyle": "--",
    "linewidth": 1.5
  },
  "continuum_line": {
    "color": "magenta",
    "linestyle": "--",
    "linewidth": 1.5
  },
  "continuum_region": {
    "color": "magenta",
    "alpha": 0.3,
    "hatch": "//"
  },
  "total_line": {
    "color": "#003d7a",
    "linestyle": "-",
    "linewidth": 2
  }
}
```

- **gaussian**: Color and style for single Gaussian profile fits
- **voigt**: Color and style for Voigt profile fits
- **continuum_line**: Color and style for fitted continuum lines
- **continuum_region**: Color, transparency, and hatch pattern for continuum region selection areas
- **total_line**: Color and style for the total Listfit composite line

### `spectrum` - Data and Error Spectrum Colors

```json
"spectrum": {
  "data": {
    "color": "black",
    "linestyle": "-"
  },
  "error": {
    "color": "red",
    "linestyle": "--",
    "alpha": 0.4
  }
}
```

- **data**: Color and line style for the loaded spectrum data
- **error**: Color, line style, and transparency for error (uncertainty) spectrum

### `residual` - Residual Panel Colors

```json
"residual": {
  "color": "royalblue",
  "linestyle": "-"
}
```

- **residual**: Color and line style for the residual plot

### `preview` - Live Preview Colors

```json
"preview": {
  "color": "lime",
  "linestyle": "-",
  "linewidth": 2
}
```

- **preview**: Color, line style, and width for live preview of fits being adjusted (appears during redshift mode)

### `reference_lines` - Reference and Axis Lines

```json
"reference_lines": {
  "color": "gray",
  "linestyle": "--",
  "linewidth": 1
}
```

- **reference_lines**: Color, line style, and width for zero-level reference lines and axis markers

## Customizing Colors

To change any colors or line styles:

1. Open `qasap/config_colors.json` in your editor
2. Modify the color values using any of these formats:
   - Named colors: `"red"`, `"blue"`, `"magenta"`, etc.
   - Hex colors: `"#FF0000"` (red), `"#003d7a"` (dark blue)
   - RGB notation: `"(1, 0, 0)"` for red

3. Available line styles:
   - `"-"` or `"solid"`: Solid line
   - `"--"` or `"dashed"`: Dashed line
   - `"-."` or `"dashdot"`: Dash-dot line
   - `":"` or `"dotted"`: Dotted line

4. Save the file and restart QASAP

## Example Customizations

### High Contrast Theme

```json
{
  "profiles": {
    "gaussian": {"color": "red", "linestyle": "--", "linewidth": 2},
    "voigt": {"color": "blue", "linestyle": "--", "linewidth": 2},
    "continuum_line": {"color": "green", "linestyle": "--", "linewidth": 2},
    ...
  }
}
```

### Print-Friendly Theme (Grayscale)

```json
{
  "profiles": {
    "gaussian": {"color": "black", "linestyle": "--", "linewidth": 1.5},
    "voigt": {"color": "gray", "linestyle": "-.", "linewidth": 1.5},
    "continuum_line": {"color": "darkgray", "linestyle": "--", "linewidth": 1.5},
    ...
  }
}
```

## Implementation Details

The color configuration is loaded in the `SpectrumPlotter.__init__()` method via the `_load_color_config()` function. The function:

1. Attempts to load colors from `config_colors.json`
2. Falls back to hardcoded defaults if the file is not found
3. Stores the configuration in `self.colors` for access throughout the application

All color references throughout `spectrum_plotter.py` now use the centralized configuration dictionary instead of hardcoded strings.

### Color Access Pattern

Instead of hardcoded colors:
```python
# Old way (NOT used anymore)
self.ax.plot(x, y, color='red', linestyle='--')
```

The code now uses:
```python
# New way (using configuration)
gaussian_color = self.colors['profiles']['gaussian']
self.ax.plot(x, y, color=gaussian_color['color'], linestyle=gaussian_color['linestyle'])
```

This approach provides:
- **Centralized management**: All colors in one file
- **Consistency**: All Gaussians use the same color automatically
- **Flexibility**: Easy to customize without editing source code  
- **Maintainability**: Single source of truth for visual styling
