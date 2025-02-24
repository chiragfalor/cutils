
import pandas as pd
import numpy as np
import xarray as xr


def cum_corr(s1: pd.Series, s2: pd.Series, wts: pd.Series|None=None, normalized: bool=True) -> pd.Series:
    """
    Calculate the cumulative un-demeaned correlation between two series.
    
    Parameters
    ----------
    s1 : pd.Series
        The first series.
    s2 : pd.Series
        The second series.
    wts : pd.Series, optional
        The weights to apply to the correlation calculation. If None, no weights are applied.
    normalized : bool, optional
        If True, the correlation at each time step is normalized by the size at that step. 
        If False, the correlation is normalized by the whole sample size. Default is True.
    
    Returns
    -------
    pd.Series
        The cumulative correlation between the two series.

    Example
    -------
    >>> s1 = pd.Series([2, 1, 0, 1, 2])
    >>> s2 = pd.Series([2, 1, 0, -1, -2])
    >>> cum_corr(s1, s2)
    0    1.000000
    1    1.000000
    2    1.000000
    3    0.666667
    4    0.000000
    dtype: float64
    >>> wts = pd.Series([0.1, 0.1, 0.1, 0.3, 0.3])
    >>> cum_corr(s1, s2, wts)
    0    1.00
    1    1.00
    2    1.00
    3    0.25
    4   -0.50
    dtype: float64
    """
    if wts is None:
        wts = np.ones_like(s1)
    else:
        wts = pd.Series(wts, index=s1.index) if isinstance(s1, pd.Series) else np.array(wts)

    assert len(s1) == len(s2) == len(wts), "All series must have the same length."
    
    s1, s2, wts = s1.astype(np.float64), s2.astype(np.float64), wts.astype(np.float64)
    cov = (s1 * s2 * wts).cumsum()
    norm = np.sqrt((s1**2 * wts).cumsum() * (s2**2 * wts).cumsum()) if normalized else np.sqrt((s1**2 * wts).sum() * (s2**2 * wts).sum())
    corr = cov / norm
    return corr


def cum_regress(feats: pd.DataFrame | pd.Series, resp: pd.Series, wts: pd.Series | None = None, grouper: pd.Grouper | None = None) -> pd.DataFrame | pd.Series:
    """
    Calculate the cumulative regression coefficients between a dataframe of features and a target series.

    Parameters
    ----------
    feats : pd.DataFrame | pd.Series
        The features to regress against the target series.
    resp : pd.Series
        The target series.
    wts : pd.Series, optional
        The weights to apply to the regression calculation. If None, no weights are applied.
    grouper : pd.Grouper, optional
        The groups over which to calculate the regression coefficients. If None, no grouping is applied.

    Returns
    -------
    pd.DataFrame | pd.Series
        The cumulative regression coefficients between the features and the target series.
    """
    raise NotImplementedError("This function has not been implemented yet. Should implement this function using xarray.")


# def cum_regress(feats: pd.DataFrame | pd.Series,
#                 resp: pd.Series,
#                 wts: pd.Series | None = None,
#                 grouper: pd.Grouper | None = None) -> pd.DataFrame | pd.Series:
#     """
#     Calculate the cumulative regression coefficients between a dataframe of features and a target series.

#     For each time t, the regression coefficients βₜ solve the weighted least-squares problem
#     minimizing ∑₍ᵢ₌₀₎ᵗ wᵢ (respᵢ - featsᵢ * β)². Equivalently, if we denote
#     Sₓₓ(t) = ∑₍ᵢ₌₀₎ᵗ wᵢ featsᵢ featsᵢᵀ and Sₓy(t) = ∑₍ᵢ₌₀₎ᵗ wᵢ featsᵢ respᵢ,
#     then βₜ = Sₓₓ(t)⁻¹ Sₓy(t). (For the univariate case, this is simply
#     βₜ = ∑₍ᵢ₌₀₎ᵗ wᵢ featsᵢ respᵢ / ∑₍ᵢ₌₀₎ᵗ wᵢ featsᵢ².)

#     Parameters
#     ----------
#     feats : pd.DataFrame | pd.Series
#         The features to regress against the target series.
#     resp : pd.Series
#         The target series.
#     wts : pd.Series, optional
#         The weights to apply to the regression calculation. If None, weights of 1 are used.
#     grouper : pd.Grouper, optional
#         The groups over which to calculate the regression coefficients. If provided, the cumulative sums 
#         will reset at each group boundary.

#     Returns
#     -------
#     pd.DataFrame | pd.Series
#         The cumulative regression coefficients between the features and the target series.
#     """
#     # Use unit weights if none are provided.
#     if wts is None:
#         wts = pd.Series(1, index=resp.index)
#     else:
#         wts = pd.Series(wts, index=resp.index)
    
#     # Check that lengths agree.
#     if isinstance(feats, pd.Series):
#         assert len(feats) == len(resp) == len(wts), "All series must have the same length."
#     else:
#         assert len(feats) == len(resp) == len(wts), "All series must have the same length."

#     # --- Grouping case ---
#     if grouper is not None:
#         # When a grouper is provided, split the data by group and then apply cum_regress to each group.
#         if isinstance(feats, pd.Series):
#             df = pd.DataFrame({
#                 "feat": feats,
#                 "resp": resp,
#                 "wts": wts
#             })
#             # For each group, use the univariate formula.
#             def group_uni(g):
#                 # Avoid division by zero; early observations may yield 0 in the denominator.
#                 num = (g["wts"] * g["feat"] * g["resp"]).cumsum()
#                 den = (g["wts"] * g["feat"]**2).cumsum()
#                 return num / den
#             result = df.groupby(grouper, group_keys=False).apply(group_uni)
#             result = result.sort_index()
#             return result
#         else:
#             # feats is a DataFrame.
#             df = feats.copy()
#             df["resp"] = resp
#             df["wts"] = wts
#             def group_multi(g):
#                 g_feats = g[feats.columns]
#                 g_resp = g["resp"]
#                 g_wts = g["wts"]
#                 return cum_regress(g_feats, g_resp, g_wts, grouper=None)
#             result = df.groupby(grouper, group_keys=False).apply(group_multi)
#             result = result.sort_index()
#             return result

#     # --- No grouping; use xarray for cumulative calculations ---
#     if isinstance(feats, pd.Series):
#         # Univariate case: compute cumulative sums of (wts * feat^2) and (wts * feat * resp)
#         X = xr.DataArray(feats.to_numpy(), dims=["time"], coords={"time": feats.index})
#         Y = xr.DataArray(resp.to_numpy(), dims=["time"], coords={"time": resp.index})
#         W = xr.DataArray(wts.to_numpy(), dims=["time"], coords={"time": wts.index})
#         Sxx = (W * X**2).cumsum(dim="time")
#         Sxy = (W * X * Y).cumsum(dim="time")
#         beta = Sxy / Sxx
#         # Return a pandas Series.
#         return pd.Series(beta.values, index=feats.index)
#     else:
#         # Multivariate case: feats is a DataFrame.
#         # Convert to xarray DataArrays.
#         X = xr.DataArray(feats.to_numpy(), dims=["time", "feature"],
#                          coords={"time": feats.index, "feature": feats.columns})
#         Y = xr.DataArray(resp.to_numpy(), dims=["time"], coords={"time": resp.index})
#         W = xr.DataArray(wts.to_numpy(), dims=["time"], coords={"time": wts.index})
#         # Compute, for each time, the weighted outer product X[i] X[i]^T.
#         # Use broadcasting: X[:, :, None] has dims (time, feature, new_feature)
#         X_outer = X[:, :, None] * X[:, None, :]
#         # Rename the new dimension to "feature2" for clarity.
#         X_outer = X_outer.rename({"feature_1": "feature2"})
#         # Cumulative weighted outer product:
#         Sxx = (W[:, None, None] * X_outer).cumsum(dim="time")
#         # Cumulative weighted cross-product between X and Y:
#         Sxy = (W * X * Y).cumsum(dim="time")  # dims: (time, feature)

#         # Define a function to solve the linear system at each time.
#         def solve_beta(Sxx_slice, Sxy_slice):
#             # Sxx_slice: 2D array (m x m), Sxy_slice: 1D array (m,)
#             try:
#                 return np.linalg.solve(Sxx_slice, Sxy_slice)
#             except np.linalg.LinAlgError:
#                 # If singular, return NaNs.
#                 return np.full(Sxy_slice.shape, np.nan)
        
#         # Apply the solver at each time step.
#         beta = xr.apply_ufunc(
#             solve_beta,
#             Sxx,
#             Sxy,
#             input_core_dims=[["feature", "feature2"], ["feature"]],
#             output_core_dims=[["feature"]],
#             vectorize=True,
#             dask="parallelized",
#             output_dtypes=[float]
#         )
#         # Convert the result to a pandas DataFrame.
#         beta_df = pd.DataFrame(beta.values, index=feats.index, columns=feats.columns)
#         return beta_df




if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    import doctest
    doctest.testmod(verbose=True)