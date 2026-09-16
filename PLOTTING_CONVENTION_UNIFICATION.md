# Single/Multi Gaussian & Voigt Plotting Convention Unification

## Summary of Changes

Unified the plotting convention for Single Gaussian, Multi-Gaussian, and Single Voigt fit modes to match Listfit's behavior. Profiles now plot from y=0 instead of being offset upward by the continuum value.

## Code Changes

### Gaussian Profile Plotting (Lines 2532-2544)
**Before:**
```python
if fit.get('is_listfit_component', False):
    existing_continuum = np.zeros_like(x_plot)
else:
    _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
    if a is not None and b is not None:
        existing_continuum = self.continuum_model(x_plot, a, b)
    else:
        existing_continuum = np.zeros_like(x_plot)
y_plot = self.gaussian(x_plot, fit['amp'], fit['mean'], fit['stddev']) + existing_continuum
```

**After:**
```python
# Plot profile alone from y=0 (unified with Listfit convention)
# The continuum is displayed separately as its own line
y_plot = self.gaussian(x_plot, fit['amp'], fit['mean'], fit['stddev'])
```

### Voigt Profile Plotting (Lines 2574-2586)
**Before:**
```python
if fit.get('is_listfit_component', False):
    existing_continuum = np.zeros_like(x_plot)
else:
    _, a, b = self.get_existing_continuum(fit['bounds'][0], fit['bounds'][1])
    if a is not None and b is not None:
        existing_continuum = self.continuum_model(x_plot, a, b)
    else:
        existing_continuum = np.zeros_like(x_plot)
y_plot = self.voigt(x_plot, fit['amp'], fit['center'], fit['sigma'], fit['gamma']) + existing_continuum
```

**After:**
```python
# Plot profile alone from y=0 (unified with Listfit convention)
# The continuum is displayed separately as its own line
y_plot = self.voigt(x_plot, fit['amp'], fit['center'], fit['sigma'], fit['gamma'])
```

## Scope 1: Profile Curve Plotting Functions ✓ FIXED

**Files Modified:**
- Lines 2532-2544: Gaussian profile plotting - removed continuum offset
- Lines 2574-2586: Voigt profile plotting - removed continuum offset

**Impact:** Profiles now plot from y=0 like Listfit, with continuum displayed as a separate line.

## Scope 2: Total Line Calculation ✓ AUTOMATICALLY CORRECT

**Function:** `draw_total_line()` (lines 11009-11056)

**How it works:** 
1. Reads plotted line objects from item tracker
2. Interpolates each line to common x-grid
3. Sums all interpolated values: `total_y += interpolated`

**Before fix (with continuum offset in profiles):**
- Continuum line: y = continuum(x)
- Gaussian line: y = gaussian(x) + continuum(x) [was offset]
- Sum: y_total = continuum(x) + gaussian(x) + continuum(x) = 2×continuum(x) + gaussian(x) ✗ WRONG

**After fix (no offset):**
- Continuum line: y = continuum(x)
- Gaussian line: y = gaussian(x) [not offset]
- Sum: y_total = continuum(x) + gaussian(x) ✓ CORRECT

**Conclusion:** Total line automatically becomes correct after removing profile offsets. No code changes needed.

## Scope 3: EW & MC Profile Features ✓ NO CHANGES NEEDED

### EW Calculation (`_calculate_equivalent_width_monte_carlo`, lines 7251-7400+)

**How it works:**
1. Samples from fit parameter distributions (using covariance matrices)
2. For each sample: computes profile(x) from sampled parameters
3. Computes continuum(x) from sampled continuum coefficients
4. Calculates EW from sampled residual profiles and continua

**Key insight:** Uses RAW PARAMETERS, not plotted line data
```python
# Line 7318-7330: Extracts parameters from fit_dict
amp = fit_dict.get('amp')
mean = fit_dict.get('mean')
stddev = fit_dict.get('stddev')
covariance = fit_dict.get('covariance')
# Uses these to sample and calculate EW
```

**Conclusion:** EW calculations are unaffected because they compute directly from fitted parameters. No changes needed.

### MC Profile Overlay (`plot_mc_profiles`, lines 9035-9088)

**How it works:**
1. Gets pre-computed MC samples from ew_result (which contain sampled residual profiles + continua)
2. Plots continuum realizations: `line_cont, = self.ax.plot(x_grid, continuum, ...)`
3. Plots flux realizations: `flux = continuum + residual`

**Key insight:** Uses MC-sampled parameters via pre-computed samples
```python
# Line 9078: continuum samples and profile samples are pre-computed from parameters
continuum = continuum_samples[sample_idx]
residual = profile_samples[sample_idx]
flux = continuum + residual
```

**Conclusion:** MC profiles are unaffected because they use sampled parameters. No changes needed.

## Scope 4: Residual Panel Calculation ✓ ALREADY CORRECT

**Function:** `calculate_residuals()` (lines 5414-5540)

**How it works:**
1. Computes gaussian_sum from raw parameters: `gaussian_sum[mask] += self.gaussian(comp_x, amp, mean, stddev)`
2. Computes voigt_sum from raw parameters: `voigt_sum[mask] += self.voigt(comp_x, ...)`
3. Computes continuum_sum from raw polynomial/Chebyshev
4. Returns: `return self.spec - gaussian_sum - voigt_sum - continuum_sum - listfit_poly_sum`

**Residual formula:**
```
residual = data - (gaussian + voigt + continuum)
```

**Key insight:** This formula is INDEPENDENT of how profiles are plotted:
- Whether profile is drawn at y=gaussian(x) or y=gaussian(x)+continuum(x)
- The residual calculation still uses raw values
- The visual position of the profile line doesn't affect residuals

**Before fix:**
- Profile plotted at: gaussian(x) + continuum(x)
- Residual computed as: data - gaussian(x) - continuum(x) ✓ CORRECT
- Residual panel shows correct deviations

**After fix:**
- Profile plotted at: gaussian(x)
- Residual computed as: data - gaussian(x) - continuum(x) ✓ STILL CORRECT
- Residual panel shows same correct deviations

**Conclusion:** Residual panel is already correct and unchanged. Residual values before/after fix are identical because residuals use raw parameters, not plotted display values.

## Verification Results

### Scope Item 1: Profile Plotting
✅ COMPLETE: Gaussian and Voigt profiles now plot from y=0 without continuum offset, matching Listfit convention.

### Scope Item 2: Total Line
✅ NO CHANGES NEEDED: Total line automatically correct after removing offsets. Sums to continuum(x) + gaussian(x) + voigt(x) as expected.

### Scope Item 3: EW and MC Features
✅ NO CHANGES NEEDED: Both use raw fitted parameters via Monte Carlo sampling, not plotted line data. Unaffected by display change.

### Scope Item 4: Residual Panel
✅ VERIFIED CORRECT: Residuals computed from raw parameters, not plotted values. Residual values identical before/after fix because formula doesn't depend on profile display position.

## Expected Behavior Changes

### Visual Changes
1. **Single Gaussian on continuum:**
   - Before: Gaussian profile appears visually sitting on top of continuum
   - After: Gaussian curve starts from y=0, with separate continuum line below

2. **Total line visualization:**
   - Before: Would appear above data (double continuum)
   - After: Correctly overlaps data as continuum + profiles

3. **Residual panel:**
   - Before: Shows data - continuum - profiles
   - After: Shows same residuals (values unchanged)

### Functional Behaviors (Unchanged)
- Fitted parameters (amplitude, mean, stddev, etc.) - identical
- Chi-squared values - identical
- Equivalent width calculations - identical
- Monte Carlo profile uncertainties - identical
- Residual statistics - identical
- Redshift mode behavior - unchanged

## Testing Verification Points

1. **Single Gaussian + Continuum case:**
   - Load spectrum
   - Fit continuum
   - Fit Single Gaussian on top
   - Verify: Gaussian line now starts from y=0, not offset upward
   - Verify: Total line correctly overlays data

2. **Residual invariance:**
   - Load same spectrum
   - Compare residual values before/after fix
   - Should show no change (or ±numerical precision)

3. **Listfit behavior:**
   - Load existing Listfit .qsap file
   - Verify: Plot unchanged
   - Verify: Total line unchanged
   - Verify: Residuals unchanged

## Code Quality
- Python syntax validated ✓
- All changes compile ✓
- Logic verified by code inspection ✓
- No changes to fit parameters or algorithms ✓
- Only affects visual display layer ✓
