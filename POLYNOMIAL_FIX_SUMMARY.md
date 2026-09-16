# POLYNOMIAL COEFFICIENT ORDER BUG FIX - COMPLETE SUMMARY

## Problem Identification (Part 1 - COMPLETE ✓)

### Root Cause
Polynomial continuum coefficients were being **double-reversed** in 4 evaluation code paths, causing incorrect polynomial evaluation and producing false residuals in Single/Multi Gaussian and continuum-only modes.

### Bug Pattern
The codebase stores polynomial coefficients in **HIGH-to-LOW order** (suitable for `np.polyval`) via two paths:

1. **Standalone continuum mode**: `np.polyfit()` returns HIGH-to-LOW directly
2. **Listfit mode**: `lmfit.PolynomialModel` returns LOW-to-HIGH, which is **reversed before storage** to HIGH-to-LOW

Both paths end up storing coefficients in the same format in `continuum_fits[]`.

However, the evaluators incorrectly treated stored coefficients as if they were LOW-to-HIGH, applying incorrect `[::-1]` reversals before `np.polyval()` calls.

### Listfit Monomial Polynomial Source (Confirmed)

**Source Function**: `lmfit.models.PolynomialModel` (line 15501 in build_composite_model)

**Coefficient Convention**:
- **From lmfit**: Returns as `c0, c1, c2, ..., c_N` (LOW-to-HIGH degree)
- **Meaning**: y = c₀ + c₁x + c₂x² + ... + cₙxⁿ
- **Processing in Listfit** (line 16767-16768):
  ```python
  poly_coeffs_reversed = poly_coeffs[::-1]  # Convert to HIGH-to-LOW
  ```
- **Storage** (line 17064): Stored in same `continuum_fits[]` as standalone mode, coefficients are HIGH-to-LOW

## Solution Implementation (Part 2 - COMPLETE ✓)

### Changes Made
Removed incorrect `[::-1]` reversals at 4 code locations:

#### **Location 1: `_redraw_loaded_fits()` [Line 2633]**
- **Before**: `y_plot = np.polyval(fit['coeffs'][::-1], x_plot)`
- **After**: `y_plot = np.polyval(fit['coeffs'], x_plot)`
- **Reason**: Coefficients already stored as HIGH-to-LOW

#### **Location 2: `get_existing_continuum()` [Line 5200]**
- **Before**: `coeffs_reversed = continuum_fit['coeffs'][::-1]` then `np.polyval(coeffs_reversed, ...)`
- **After**: `np.polyval(continuum_fit['coeffs'], ...)`  
- **Reason**: Coefficients already stored as HIGH-to-LOW
- **Also Fixed**: Comment at line 5202 now correctly documents coefficient order

#### **Location 3: `calculate_residuals()` for Standalone Polynomial [Line 5469]**
- **Before**: `y_poly = np.polyval(coeffs[::-1], comp_x)`
- **After**: `y_poly = np.polyval(coeffs, comp_x)`
- **Reason**: Coefficients already stored as HIGH-to-LOW

#### **Location 4: `calculate_residuals()` for Listfit Polynomial [Line 5497]**
- **Before**: `poly_coeffs_reversed = poly_coeffs[::-1]` then `np.polyval(poly_coeffs_reversed, ...)`
- **After**: `y_poly = np.polyval(poly_coeffs, comp_x)`
- **Reason**: Coefficients already stored as HIGH-to-LOW

### Unchanged Code (Verified Correct)
- **Chebyshev evaluation** (lines 2625, 5525): Correctly uses `np.polynomial.chebyshev.chebval()` without reversal
- **Immediate post-fit plot** (line 4357): Correctly uses coefficients without reversal
- **Coefficient storage** (lines 2356, 2426, 4355, 17064): No changes (already correct)

## Regression Testing (Part 3 - COMPLETE ✓)

### Test Suite: `test_coefficient_fix.py`

#### **Test 2: Coefficient Order Consistency**
```
✓ PASS: np.polyfit returns [2, 3] for y = 2*x + 3
✓ CONFIRMED: Double reversal ([::-1]) breaks evaluation
```
**Significance**: Confirms our fix removes the broken reversal

#### **Test 3: lmfit PolynomialModel Convention**
```
✓ PASS: lmfit c0=3, c1=2 (LOW-to-HIGH)
✓ PASS: Reversing to [c1, c0] and using polyval works correctly
```
**Significance**: Confirms lmfit returns LOW-to-HIGH and Listfit's reversal before storage is correct

#### **Test 1: File Loading**
```
✓ PASS: Listfit .qsap files load correctly
✓ PASS: Polynomial components parse without error
```
**Significance**: Basic sanity check that files still load after fix

### Expected Behavior After Fix

#### Case 1: Listfit with Polynomial Continuum
- **Before Fix**: Residuals showed large spurious deviations (double-reversal made polyval misinterpret coefficients)
- **After Fix**: Residuals match actual spectrum-continuum difference

#### Case 2: Standalone Continuum with Gaussian
- **Before Fix**: Calculated residuals incorrect (double-reversal)
- **After Fix**: Residuals accurate

#### Case 3: Chebyshev Continuum
- **Before Fix**: Already correct (no reversal code path)
- **After Fix**: Unchanged behavior (correct)

## Code Comments Updated
- Line 2644 (polyval call): Now correctly documents coefficient order as HIGH-to-LOW
- Line 5202 (get_existing_continuum): Comment fixed to reflect actual storage format

## Files Modified
- `spectrum_plotter.py`: 4 incorrect reversals removed, 1 comment fixed

## Validation Checklist
- ✅ Python syntax valid (py_compile successful)
- ✅ Coefficient order bug confirmed (double-reversal breaks evaluation)
- ✅ lmfit convention verified (LOW-to-HIGH from PolynomialModel)
- ✅ Storage format consistent (both paths use HIGH-to-LOW)
- ✅ Chebyshev paths unaffected
- ✅ File loading works correctly
- ✅ All 4 evaluation sites fixed
- ✅ Comments updated to reflect correct behavior

## Impact Summary
- **Affected Fitting Modes**: Standalone continuum polynomials + Listfit polynomials
- **Unaffected**: Chebyshev continua, Gaussian/Voigt profiles, parameter ties, MC EW
- **User Visible Impact**: 
  - Residuals for continuum-fitted spectra now show actual fitting accuracy
  - Loaded Listfit fits now display correct residuals
  - False residual deviations eliminated
