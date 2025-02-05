import pandas as pd
import pandas_flavor as pf
from typing import Sequence
import numpy as np
import warnings
print("loading series_utils")


@pf.register_series_method
def qtl_clip(s: pd.Series, qtl: float=0.01, *, lower_qtl=None, upper_qtl=None):
    """
    Trims values at the input quantiles.

    Parameters
    ----------
    s : pd.Series
        The input series to be clipped.
    qtl : float, optional, default 0.01 (1 percentile clipping on both ends)
        The quantile threshold for clipping. If `lower_qtl` and `upper_qtl` are not specified, `qtl` is used for both lower and upper bounds.
    lower_qtl : float, optional
        The quantile threshold for the lower bound. If not specified, it defaults to `qtl`.
    upper_qtl : float, optional
        The quantile threshold for the upper bound. If not specified, it defaults to 1 - `qtl`.

    Returns
    -------
    pd.Series
        The clipped series with values outside the specified quantiles trimmed.

    Notes
    -----
    This function clips the input series at the specified quantiles. If `lower_qtl` and `upper_qtl` are not specified, the function uses `qtl` for both lower and upper bounds. 
    The clipping is done using the `clip` method of pandas Series, with the bounds determined by the quantiles of the series. 
    The `interpolation` parameter for quantile calculation is set to 'higher' for the lower bound and 'lower' for the upper bound to ensure that the bounds are inclusive.

    Examples
    >>> s = pd.Series(range(1, 11), dtype=float)
    >>> s.qtl_clip(qtl=0.1)
    0    2.0
    1    2.0
    2    3.0
    3    4.0
    4    5.0
    5    6.0
    6    7.0
    7    8.0
    8    9.0
    9    9.0
    dtype: float64
    >>> s = pd.Series(range(1, 11), dtype=float)
    >>> s.qtl_clip(lower_qtl=0.2, upper_qtl=0.5)
    0    3.0
    1    3.0
    2    3.0
    3    4.0
    4    5.0
    5    5.0
    6    5.0
    7    5.0
    8    5.0
    9    5.0
    dtype: float64
    """
    if (lower_qtl is None) and (upper_qtl is None):
        lower_qtl = qtl
        upper_qtl = 1 - qtl

    lower_bound = (
        s.quantile(lower_qtl, interpolation="higher") if lower_qtl is not None else None
    )
    upper_bound = (
        s.quantile(upper_qtl, interpolation="lower") if upper_qtl is not None else None
    )

    if (lower_bound is not None) and (upper_bound is not None):
        assert (
            lower_bound < upper_bound
        ), "Excessive clipping. Lower bound is more than upper bound"

    return s.clip(lower=lower_bound, upper=upper_bound)

# def _category_desc(s: pd.Series, topk: int=5) -> pd.Series:
    

@pf.register_series_method
def desc(s: pd.Series, percentiles: Sequence[float]=(0.25, 0.5, 0.75), topk: int=5) -> pd.Series:
    """
    Generates descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, proportion of NaN values. For categorical data, the function provides the value counts of top_k most frequent values, and the entropy of the distribution.

    Parameters
    ----------
    s : pd.Series
        The input series for which the descriptive statistics are to be calculated.

    Returns
    -------
    pd.Series
        The descriptive statistics for the input series.

    Examples
    --------
    >>> s = pd.Series(range(1, 11), dtype=float)
    >>> s.desc()
    len         10.00000
    nan_frac     0.00000
    mean         5.50000
    std          3.02765
    min          1.00000
    25%          3.25000
    50%          5.50000
    75%          7.75000
    max         10.00000
    dtype: float64
    >>> s = pd.Series(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', np.nan], dtype=object)
    >>> s.desc()
    len               10
    nan_frac         0.1
    unique             3
    entropy     1.584963
    frac_a      0.333333
    frac_b      0.333333
    frac_c      0.333333
    dtype: object
    """
    desc = pd.Series(
            {
                'len': int(len(s)),
                'nan_frac': s.isnull().mean(),
            },
            dtype=object,
        )
    if pd.api.types.is_numeric_dtype(s):
        desc: pd.Series = pd.concat([desc, s.describe(percentiles=percentiles)])
        # drop count
        desc.drop("count", inplace=True)
        # set dtype to float
        desc = desc.astype(float)
    elif pd.api.types.is_object_dtype(s):
        s = s.dropna()
        norm_val_counts = s.value_counts(normalize=True)
        desc['unique'] = int(len(norm_val_counts))
        entropy = -(norm_val_counts*np.log2(norm_val_counts)).sum()
        desc["entropy"] = entropy
        topk_val_counts = norm_val_counts.head(topk).rename(lambda x: f"frac_{x}")
        desc = pd.concat([desc, topk_val_counts])
        desc = desc.astype(object)
    else:
        raise TypeError("Series must be of numeric or object dtype")

    return desc





@pf.register_series_accessor('wtd')
class WtdSeriesAccessor:
    def __init__(self, s: pd.Series):
        self._s = s
        # assert dtype is numeric
        if not pd.api.types.is_numeric_dtype(s):
            raise TypeError("Series must be of numeric dtype")


    def get_wts(self, wts=None) -> pd.Series:
        """
        Returns the weights for the series.

        Returns
        -------
        pd.Series
            The weights used for calculating the weighted statistics.

        Notes
        -----
        This function returns the weights for the series. The weights are used for calculating the weighted statistics. 
        If the weights are not specified, the function assumes equal weights for all elements.

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.get_wts()
        0    1
        1    1
        2    1
        3    1
        4    1
        5    1
        6    1
        7    1
        8    1
        9    1
        dtype: int64
        """
        if wts is not None:
            wts = pd.Series(wts)
        elif hasattr(self._s, 'wts'):
            wts = self._s.wts
        else:
            warnings.warn("Series has no `wts` attribute. Assuming equal weights for all element. Recommended to use the unweighted version.", UserWarning)
            wts = pd.Series(1, index=self._s.index)
        
        if len(self._s) != len(wts):
            raise ValueError("Length of series and weights must be the same")
        

        return wts
        



    # def set_wts(self, wts: pd.Series):
    #     """
    #     Sets the weights for the series.

    #     Parameters
    #     ----------
    #     wts : pd.Series
    #         The weights to be used for calculating the weighted statistics.

    #     Returns
    #     -------
    #     None

    #     Notes
    #     -----
    #     This function sets the weights for the series. The weights are used for calculating the weighted statistics. 
    #     The function does not return anything, but updates the weights attribute of the series.

    #     Examples
    #     --------
    #     >>> s = pd.Series(range(1, 11), dtype=float)
    #     >>> wts = pd.Series(range(1, 11), dtype=float)
    #     >>> s.wtd.set_wts(wts)
    #     >>> s.wtd.mean()
    #     7.0
    #     """
    #     if len(self._s) != len(wts):
    #         raise ValueError("Length of series and weights must be the same")
        
    #     # print("Setting weights")
    #     # print(wts)
        
    #     self._wts = wts


    def mean(self, wts: pd.Series | None=None) -> float:
        """
        Calculates the weighted mean of the input series: <s * wts> / <wts>

        Parameters
        ----------
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted mean. If not specified, the function assumes equal weights for all elements.

        Returns
        -------
        float
            The weighted mean of the input series.

        Notes
        -----
        This function calculates the weighted mean of the input series using the weights provided. If weights are not specified, the function assumes equal weights for all elements. 
        The function uses the `np.average` function from the numpy library to calculate the weighted mean.

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.mean()
        5.5
        >>> wts = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.mean(wts)
        7.0
        >>> s = pd.Series([1, 2, 3, 4, 5], dtype=float)
        >>> wts = pd.Series([0.1, 0.2, 0.3, 0.2, 0.2], dtype=float)
        >>> s.wtd.mean(wts)
        3.2
        """
        wts = self.get_wts(wts)

        if len(self._s) != len(wts):
            raise ValueError("Length of series and weights must be the same")
        
        # mask missing values
        mask = ~self._s.isnull() & ~wts.isnull()
        s, wts = self._s[mask], wts[mask]

        return np.average(s, weights=wts).item()
    
    def std(self, wts: pd.Series | None=None) -> float:
        """
        Calculates the weighted standard deviation of the input series.

        Parameters
        ----------
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted standard deviation. If not specified, the function assumes equal weights for all elements.

        Returns
        -------
        float
            The weighted standard deviation of the input series: sqrt(wtd_mean(sqr(s)) - sqr(wtd_mean(s)))

        Notes
        --------
        This is not the same as pd.Series.std() as that assumes ddof=1, while this function assumes ddof=0.

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> wts = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.std()
        2.8722813232690143
        >>> s.wts = wts
        >>> s.wtd.std()
        2.449489742783178
        >>> s = pd.Series([3] * 10, dtype=float)
        >>> s.wtd.std()
        0.0
        >>> s.wtd.std(wts)
        0.0
        """
        wts = self.get_wts(wts)
        var = (self._s**2).wtd.mean(wts) - (self._s.wtd.mean(wts))**2
        return np.sqrt(var).item()
    
    def quantile(self, wts: pd.Series | None=None, *, q: float | Sequence[float]=0.5, method: str='inverted_cdf') -> float | pd.Series:
        """
        Generalizes the concept of quantiles to weighted data.

        Parameters
        ----------
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted quantile. If not specified, the function assumes equal weights for all elements.
        q : float or sequence of floats
            The quantile or sequence of quantiles to be calculated. Values must be between 0 and 1.
        method : str, optional, default 'inverted_cdf'
            This parameter specifies the method to use for estimating the
            percentile.  There are many different methods, some unique to NumPy.
            See the notes for explanation.  The options sorted by their R type
            as summarized in the H&F paper [1]_ are:

            1. 'inverted_cdf' (default)
            2. 'averaged_inverted_cdf'
            3. 'closest_observation'
            4. 'interpolated_inverted_cdf'
            5. 'hazen'
            6. 'weibull'
            7. 'linear'
            8. 'median_unbiased'
            9. 'normal_unbiased'

            The first three methods are discontinuous.  NumPy further defines the
            following discontinuous variations of the default 'linear' (7.) option:

            * 'lower'
            * 'higher',
            * 'midpoint'
            * 'nearest'

        Returns
        -------
        float or pd.Series
            If ``q`` is an array, a Series will be returned where the
            index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.quantile()
        5.0
        >>> wts = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.quantile(wts, q=0.1)
        3.0
        >>> s.wtd.quantile(wts, q=[0.1, 0.5, 0.9])
        0.1     3.0
        0.5     7.0
        0.9    10.0
        dtype: float64
        """
        wts = self.get_wts(wts)

        if isinstance(q, float) or q == 0 or q == 1:
            return np.percentile(self._s, q*100, method=method, weights=wts).item()
        elif isinstance(q, Sequence):
            return pd.Series(
                {qi: np.percentile(self._s, qi*100, method=method, weights=wts).item() for qi in q}
                )
        else:
            raise TypeError("q must be a float or a sequence of floats")
        

    def size(self, wts: pd.Series | None=None) -> float:
        """
        Calculates the weighted size of the input series: sqrt(<s * wts * s> / <wts>)

        Parameters
        ----------
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted size. If not specified, the function assumes equal weights for all elements.

        Returns
        -------
        float
            The weighted size of the input series, which sqrt(wtd_mean(sqr(s)))

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.size()
        6.2048368229954285
        >>> s = pd.Series([3] * 10, dtype=float)
        >>> s.wtd.size()
        3.0
        """
        wts = self.get_wts(wts)
        return np.sqrt((self._s**2).wtd.mean(wts)).item()
    

    def corr(self, s2: pd.Series, wts: pd.Series | None=None) -> float:
        """
        Calculates the un-demeaned weighted correlation between two input series: <s1 * wts * s2> / sqrt(<s1 * wts * s1> * <s2 * wts * s2>)

        Parameters
        ----------
        s2 : pd.Series
            The second input series for which the weighted correlation is to be calculated.
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted correlation. If not specified, the function assumes equal weights for all elements.

        Returns
        -------
        float
            The weighted correlation between the two input series.

        Notes
        -----
        This function calculates the weighted correlation without demeaning the series. 
        Implicitly, it assumes that E[s1] = E[s2] = 0.


        Examples
        --------
        >>> s1 = pd.Series(range(1, 11), dtype=float)
        >>> s1.wtd.corr(s1) # self correlation
        1.0
        >>> wts = pd.Series(range(1, 11), dtype=float)
        >>> s1.wtd.corr(s1, wts)
        1.0
        >>> s1 = pd.Series([1, 0, -1, 1], dtype=float)
        >>> s2 = pd.Series([-1, 0, 1, 1], dtype=float)
        >>> s3 = pd.Series([1, 2, 1, 0], dtype=float)
        >>> wts = pd.Series([0.3, 0.4, 0.3, 0.0], dtype=float)
        >>> print(f"{s1.wtd.corr(s2, wts):.2f}")
        -1.00
        >>> s3.wtd.corr(s1, wts)
        0.0
        """
        wts = self.get_wts(wts)

        if len(self._s) != len(s2):
            raise ValueError("Length of series must be the same")

        
        mask = ~self._s.isnull() & ~s2.isnull() & ~wts.isnull()
        s1, s2, wts = self._s[mask], s2[mask], wts[mask]


        return (s1 * s2).wtd.mean(wts) / (s1.wtd.size(wts) * s2.wtd.size(wts))
    

    def describe(self, wts: pd.Series | None=None, percentiles: Sequence[float]=[0.25, 0.5, 0.75]) -> pd.Series:
        """
        Generates wtd descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution, excluding NaN values.

        Parameters
        ----------
        s : pd.Series
            The input series for which the descriptive statistics are to be calculated.
        wts : pd.Series, optional, default None
            The weights to be used for calculating the weighted descriptive statistics. If not specified, the function assumes equal weights for all elements.
        percentiles : sequence of percentiles
            The percentiles to include in the output. All should fall between 0 and 1.

        Returns
        -------
        pd.Series
            The descriptive statistics for the input series.

        Examples
        --------
        >>> s = pd.Series(range(1, 11), dtype=float)
        >>> wts = pd.Series(range(1, 11), dtype=float)
        >>> s.wtd.describe(wts, [0.1, 0.5, 0.9])
        count    10.000000
        mean      7.000000
        std       2.449490
        size      7.416198
        min       1.000000
        10%       3.000000
        50%       7.000000
        90%      10.000000
        max      10.000000
        dtype: float64
        """
        wts = self.get_wts(wts)

        # mask missing values
        mask = ~self._s.isnull() & ~wts.isnull()
        s, wts = self._s[mask], wts[mask]

        try:
            std = s.wtd.std(wts)
            size = s.wtd.size(wts)
        except TypeError:
            std = np.nan
            size = np.nan


        return pd.Series(
            {
                "count": len(s),
                "mean": s.wtd.mean(wts),
                "std": std,
                "size": size,
                "min": s.wtd.quantile(wts, q=0),
                **{f"{int(q*100)}%": s.wtd.quantile(wts, q=q) for q in percentiles},
                "max": s.wtd.quantile(wts, q=1),
            }
        )


        
# the below code is purely for type hinting purposes; DO NOT GENERALIZE IT ANY FURTHER
from typing import Callable

class Series(pd.Series):
    wtd: WtdSeriesAccessor
    qtl_clip: Callable[..., pd.Series]
# the above code is purely for type hinting purposes; DO NOT GENERALIZE IT ANY FURTHER

if __name__ == "__main__":
    # ignore warnings
    warnings.simplefilter("ignore")
    import doctest
    doctest.testmod()
