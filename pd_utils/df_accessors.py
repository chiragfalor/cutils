import pandas as pd
import pandas_flavor as pf
from typing import Union, Sequence, Callable
import numpy as np
import warnings

if __package__ is None:
    from series_accessors import Series
else:
    from .series_accessors import Series

print("loading df_utils")

#########################################
# --- DATAFRAME ACCESSOR (generalized) ---
#########################################

@pf.register_dataframe_accessor("wtd")
class WtdDataFrameAccessor:
    """
    A pandas DataFrame accessor for computing weighted statistics.

    The accessor assumes that weights are provided per row—either via the
    optional argument to the methods or (if not provided) via a custom attribute
    ``df.wts``. Only numeric columns are processed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [2, 3, 4, 5],
    ...     'C': ['x', 'y', 'z', 'w']  # non-numeric column (ignored)
    ... })
    >>> df.wts = pd.Series([1, 1, 1, 1])
    >>> df.wtd.mean()
    A    2.5
    B    3.5
    dtype: float64
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_wts(self, wts=None) -> pd.Series:
        """
        Returns the row weights for the DataFrame.

        If no weights are provided and the DataFrame does not have a 'wts' attribute,
        equal weights (of 1) are assumed.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3]})
        >>> df.wts = pd.Series([0.5, 1.0, 1.5])
        >>> df.wtd.get_wts().tolist()
        [0.5, 1.0, 1.5]
        >>> df2 = pd.DataFrame({'A': [1, 2, 3]})
        >>> df2.wtd.get_wts().tolist()
        [1, 1, 1]
        """
        if wts is not None:
            wts = pd.Series(wts, index=self._df.index)
        elif hasattr(self._df, "wts"):
            wts = self._df.wts
        else:
            warnings.warn(
                "DataFrame has no `wts` attribute. Assuming equal weights for all rows.",
                UserWarning,
            )
            wts = pd.Series(1, index=self._df.index)
        if len(wts) != len(self._df):
            raise ValueError("Length of DataFrame and weights must be the same")
        return wts

    def _apply_across_cols(self, func: Callable, wts: pd.Series | None = None) -> pd.Series:
        """
        Applies a function across all numeric columns of the DataFrame, weighted by the provided weights.
        """
        wts = self.get_wts(wts)
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        result = {}
        for col in numeric_cols:
            result[col] = func(self._df[col], wts)
        sample_value = next(iter(result.values()))
        if isinstance(sample_value, pd.Series):
            return pd.DataFrame(result)
        elif isinstance(sample_value, (int, float)):
            return pd.Series(result)
        else:
            raise TypeError("Unexpected result type")

    def mean(self, wts: pd.Series | None = None) -> pd.Series:
        """
        Computes the weighted mean for each numeric column.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [2, 3, 4, 5],
        ...     'C': ['x', 'y', 'z', 'w']
        ... })
        >>> df.wts = pd.Series([1, 2, 0, 0])
        >>> df.wtd.mean()
        A    1.666667
        B    2.666667
        dtype: float64
        """
        mean_fn = lambda sr, wts: sr.wtd.mean(wts)
        return self._apply_across_cols(mean_fn, wts)

    def std(self, wts: pd.Series | None = None) -> pd.Series:
        """
        Computes the weighted standard deviation for each numeric column.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [2, 3, 4, 5]
        ... })
        >>> df.wts = pd.Series([1, 1, 1, 1])
        >>> df.wtd.std().round(2)
        A    1.12
        B    1.12
        dtype: float64
        """
        std_fn = lambda sr, wts: sr.wtd.std(wts)
        return self._apply_across_cols(std_fn, wts)

    def quantile(
        self,
        q: Union[float, Sequence[float]] = 0.5,
        wts: pd.Series | None = None,
        *,
        method: str = "inverted_cdf",
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes weighted quantile(s) for each numeric column.
        
        Parameters
        ----------
        q : float or Sequence[float], default 0.5
            The quantile or sequence of quantiles (values between 0 and 1) to compute.
        wts : pd.Series, optional
            The row weights. If not provided, equal weights are assumed.
        method : str, optional
            The method to use for percentile calculation (passed to np.percentile).
        
        Returns
        -------
        pd.Series or pd.DataFrame
            If q is a float, returns a Series of quantiles per column;
            if q is a sequence, returns a DataFrame (rows indexed by quantile levels).
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [2, 3, 4, 5],
        ... })
        >>> # Use unequal weights:
        >>> df.wts = pd.Series([1, 2, 3, 4])
        >>> df.wtd.quantile(q=0.55)
        A    3
        B    4
        dtype: int64
        >>> df.wtd.quantile(q=[0.25, 0.5, 0.75])
              A  B
        0.25  2  3
        0.50  3  4
        0.75  4  5
        """
        if not isinstance(q, (float, int, Sequence)):
            raise TypeError("q must be a float, int, or a sequence of floats")
        qtl_fn = lambda sr, wts: sr.wtd.quantile(wts, q=q, method=method)
        return self._apply_across_cols(qtl_fn, wts)

    def size(self, wts: pd.Series | None = None) -> pd.Series:
        """
        Computes the weighted 'size' (a weighted root-mean-square) for each numeric column.
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5]})
        >>> # Use unequal weights:
        >>> df.wts = pd.Series([1, 2, 3, 4])
        >>> np.round(df.wtd.size(), 2)
        A    3.16
        B    4.12
        dtype: float64
        """
        size_fn = lambda sr, wts: sr.wtd.size(wts)
        return self._apply_across_cols(size_fn, wts)

    def corr(self, wts: pd.Series | None = None) -> pd.DataFrame:
        """
        Computes the weighted (un-demeaned) correlation matrix among the numeric columns.
        
        The correlation between columns i and j is computed as:
        
            corr(i,j) = weighted_mean( i * j ) / ( size(i) * size(j) )
        
        where the 'size' is defined as sqrt(weighted_mean(s^2)).
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, np.nan], 'B': [4, 3, 2, 1, np.nan]})
        >>> # Use unequal weights:
        >>> df.wts = pd.Series([1, 2, 3, 4, 1])
        >>> np.round(df.wtd.corr(), 2)
              A     B
        A  1.00  0.71
        B  0.71  1.00
        """
        wts = self.get_wts(wts)
        numeric_df = self._df.select_dtypes(include=[np.number])
        mask = ~wts.isnull()
        X, wts = numeric_df[mask].fillna(0).values, wts[mask].to_numpy()
        wts = wts / wts.sum()
        weighted_cov = (wts[:, None] * X).T @ X
        wtd_std = weighted_cov.diagonal() ** 0.5
        wtd_corr_matrix = weighted_cov / wtd_std[:, None] / wtd_std

        return pd.DataFrame(
            wtd_corr_matrix,
            index=numeric_df.columns,
            columns=numeric_df.columns,
        )

    def beta(self, wts: pd.Series | None = None) -> pd.DataFrame:
        """
        Computes the weighted beta for each numeric column.
        """
        wts = self.get_wts(wts)
        numeric_df = self._df.select_dtypes(include=[np.number])
        mask = ~wts.isnull()
        X, wts = numeric_df[mask].fillna(0).values, wts[mask].to_numpy()
        wts = wts / wts.sum()
        weighted_cov = (wts[:, None] * X).T @ X
        betas = weighted_cov / weighted_cov.diagonal()

        return pd.DataFrame(
            betas,
            index=numeric_df.columns,
            columns=numeric_df.columns,
        )


    def describe(
        self, percentiles: Sequence[float] = [0.25, 0.5, 0.75], wts: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Generates weighted descriptive statistics for each numeric column.
        
        For each column, the following statistics are computed:
        
            count, mean, std, size, min, the specified percentiles, and max.
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [2, 3, 4, 5]
        ... })
        >>> # Use unequal weights:
        >>> df.wts = pd.Series([1, 2, 3, 4])
        >>> df.wtd.describe(percentiles=[0.25, 0.5, 0.75])
                      A         B
        count  4.000000  4.000000
        mean   3.000000  4.000000
        std    1.000000  1.000000
        size   3.162278  4.123106
        min    1.000000  2.000000
        25%    2.000000  3.000000
        50%    3.000000  4.000000
        75%    4.000000  5.000000
        max    4.000000  5.000000
        """
        desc_fn = lambda sr, wts: sr.wtd.describe(wts, percentiles=percentiles)
        return self._apply_across_cols(desc_fn, wts)

# for type-hinting purposes only
class DataFrame(pd.DataFrame):
    wtd: WtdDataFrameAccessor


if __name__ == "__main__":
    # Simple tests / examples
    warnings.simplefilter("ignore")
    
    import doctest
    doctest.testmod()
