import numpy as np
import xarray as xr
from  bottleneck import rankdata
from xskillscore import pearson_r, pearson_r_p_value
#from xclim import run_length as rl
from typing import Sequence
from typing import Tuple
from typing import Union
from warnings import warn
import logging
from pandas import to_datetime
from scipy.stats import linregress
from scipy.stats import skew
from scipy import special
from scipy.stats import kurtosis as kt
from scipy.stats import kendalltau
from bottleneck import nanmean, nanmedian, nanmin, nanmax, nansum, nanstd, nanvar, anynan, ss

logging.captureWarnings(True)


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """
    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
        return func
    return decorate_func


 


@set_property("fctype", "simple")
def abs_energy(X,dim='time', **kwargs):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :param dim: core dimension in xarray to apply across
    :type x: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return pow(X,2).sum(dim)
 
     

@set_property("fctype", "simple")
def absolute_sum_of_changes(X, dim='time', **kwargs):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    :param x: the time series to calculate the feature of
    :param dim: core dimension in xarray to apply across
    :type x: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return np.abs(X.diff(dim)).sum(dim)
 


@set_property("fctype", "simple")
def mean_abs_change(X, dim='time', **kwargs):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n} \\sum_{i=1,\\ldots, n-1} | x_{i+1} - x_{i}|


    :param X: the time series to calculate the feature of
    :param dim: core dimension in xarray to apply across
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    
    return np.abs(X.diff(dim)).mean(dim)


@set_property("fctype", "ufunc")
def mean_change(X , dim='time', **kwargs):
    """
    Returns the mean over the differences between subsequent time series values which is
    
    .. math::
    
    \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})
    
    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    length= len(x)
    func = lambda x: (x[:,:,-1] - x[:,:,0]) / (length - 1) if length > 1 else np.NaN
    return xr.apply_ufunc(func, X,
                           input_core_dims=[[dim]],
                           dask='parallelized',
                           output_dtypes=[float])
    
    

@set_property("fctype", "simple")
def variance_larger_than_standard_deviation(X, dim='time', **kwargs):
    """
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: bool
    """
    y = X.var(dim)
    return (y > np.sqrt(y)).astype(np.int32)

 
    
@set_property("fctype", "simple")
def ratio_beyond_r_sigma(X, r=2, dim='time', **kwargs):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    
    return (np.abs(X - X.mean(dim)) > r * X.std(dim)).sum(dim)/len(X)


 
@set_property("fctype", "simple")
def large_standard_deviation(X, r=2, dim='time', **kwargs):
    """
    Boolean variable denoting if the standard dev of x is higher
    than 'r' times the range = difference between max and min of x.
    Hence it checks if

    .. math::

        std(x) > r * (max(X)-min(X))

    According to a rule of the thumb, the standard deviation should be a forth of the range of the values.

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    
    return (X.std(dim) > (r * (X.max(dim) - X.min(dim)))).astype(np.int32)





@set_property("fctype", "simple")
def symmetry_looking(X, r=0.1, dim='time', **kwargs):
    """
    Boolean variable denoting if the distribution of x *looks symmetric*. This is the case if

    .. math::

        | mean(X)-median(X)| < r * (max(X)-min(X))

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :param r: the percentage of the range to compare with
    :type r: float
    :return: the value of this feature
    :return type: bool
    """
    
    mean_median_difference = np.abs(X.mean(dim) - X.median(dim))
    max_min_difference = X.max(dim) - X.min(dim)
    return   (mean_median_difference < (r * max_min_difference)).astype(np.int32)
 
    

@set_property("fctype", "simple")
def sum_values(X, dim='time', **kwargs):
    """
    Calculates the sum over the time series values

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    
    if len(X[dim]) == 0:
        return 0
    return X.sum(dim)



# def autocorrelation(X, lag=1, dim="time", return_p=True, **kwargs):
#     from climpred.stats import autocorr
#     """
#     Calculated dim lagged correlation of a xr.Dataset.
    
    
#     :param X : xarray dataset/dataarray time series
#     :type x: xarray.DataArray
#     :param dim : name of time dimension/dimension to autocorrelate over
#     :type dim: str
#     :param return_p : boolean (default False)
#             if false, return just the correlation coefficient.
#             if true, return both the correlation coefficient and p-value.
#     :type return_p: boolean
#     :return: the value of this feature
#             r : Pearson correlation coefficient
#             p : (if return_p True) p-value
#     :return type: float
#     """
#     if return_p:
#         return autocorr(X, lag=lag, dim=dim, return_p=return_p)[1]
        
#     else:
#         return autocorr(X, lag=lag, dim=dim, return_p=return_p)

 



def _get_numpy_funcs(skipna, **kwargs):
    """
    Returns nansum and nanmean if skipna is True;
    Returns sum and mean if skipna is False.
    """
    if skipna:
        return nansum, nanmean
    else:
        return np.sum, np.mean
    

def _preprocess_dims(dim, **kwargs):
    """Preprocesses dimensions to prep for stacking.

    Parameters
    ----------
    dim : str, list
        The dimension(s) to apply the function along.
    """
    if isinstance(dim, str):
        dim = [dim]
    axis = tuple(range(-1, -len(dim) - 1, -1))
    return dim, axis

    
def _pearson_r(a, b, axis=-1, skipna=False, **kwargs):
    """
    ndarray implementation of scipy.stats.pearsonr.

    Parameters
    ----------
    a : xarray.DataArray
        Input array.
    b : xarray.DataArray
        Input array.
    axis : int
        The axis to apply the correlation along.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        Pearson's correlation coefficient.

    See Also
    --------
    scipy.stats.pearsonr

    """
    sumfunc, meanfunc = _get_numpy_funcs(skipna, **kwargs)
    
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)

    # Only do weighted sums if there are weights. Cannot have a
    # single generic function with weights of all ones, because
    # the denominator gets inflated when there are masked regions.
   
    ma = meanfunc(a, axis=0)
    mb = meanfunc(b, axis=0)

    am, bm = a - ma, b - mb

    r_num = sumfunc(am * bm, axis=0)
    r_den = np.sqrt(sumfunc(am * am, axis=0) * sumfunc(bm * bm, axis=0))

    r = r_num / r_den
    res = np.clip(r, -1.0, 1.0)
    return res


def _pearson_r_p_value(a, b, axis, skipna, **kwargs):
    """
    ndarray implementation of scipy.stats.pearsonr.

    Parameters
    ----------
    a : ndarray
        Input array.
    b : ndarray
        Input array.
    axis : int
        The axis to apply the correlation along.
    weights : ndarray
        Input array of weights for a and b.
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    res : ndarray
        2-tailed p-value.

    See Also
    --------
    scipy.stats.pearsonr

    """
    r = _pearson_r(a, b, axis, skipna)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    dof = np.apply_over_axes(np.sum, np.isnan(a * b), 0).squeeze() - 2
    dof = np.where(dof > 1.0, dof, a.shape[0] - 2)
    t_squared = r ** 2 * (dof / ((1.0 - r) * (1.0 + r)))
    _x = dof / (dof + t_squared)
    _x = np.asarray(_x)
    _x = np.where(_x < 1.0, _x, 1.0)
    _a = 0.5 * dof
    _b = 0.5
    res = special.betainc(_a, _b, _x)
    # reset masked values to nan
    all_nan = np.isnan(a.mean(axis=0) * b.mean(axis=0))
    res = np.where(all_nan, np.nan, res)
    return res


@set_property("fctype", "ufunc")
def pearson_r(a, b, dim='time',  skipna=False, **kwargs):
    """
    
    use corr from xarray  https://github.com/pydata/xarray/blob/master/xarray/core/computation.py
    
    Pearson's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Pearson's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.pearsonr
    xskillscore.core.np_deterministic._pearson_r

    Reference
    ---------
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})
        
    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        _pearson_r,
        a,
        b,
        input_core_dims=[[new_dim] , [new_dim]],  
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
    )
    
     

@set_property("fctype", "ufunc")
def pearson_r_p_value(a, b, dim = 'time',  skipna=False, **kwargs):
    """
    2-tailed p-value associated with pearson's correlation coefficient.

    Parameters
    ----------
    a : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    b : xarray.Dataset or xarray.DataArray
        Labeled array(s) over which to apply the function.
    dim : str, list
        The dimension(s) to apply the correlation along.
    weights : xarray.Dataset or xarray.DataArray
        Weights matching dimensions of ``dim`` to apply during the function.
        If None, an array of ones will be applied (i.e., no weighting).
    skipna : bool
        If True, skip NaNs when computing function.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        2-tailed p-value of Pearson's correlation coefficient.

    See Also
    --------
    xarray.apply_ufunc
    scipy.stats.pearsonr
    xskillscore.core.np_deterministic._pearson_r_p_value

    """
    dim, _ = _preprocess_dims(dim)
    if len(dim) > 1:
        new_dim = '_'.join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})

    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        _pearson_r_p_value,
        a,
        b,
        input_core_dims=[[new_dim] , [new_dim]], 
        kwargs={'axis': -1, 'skipna': skipna},
        dask='parallelized',
        output_dtypes=[float],
    )
    

@set_property("fctype", "simple")
def autocorr(X, lag=1, dim='time', return_p=True, **kwargs):
    """Calculate the lagged correlation of time series. Amended from xskillscore package
    Args:
        X (xarray object): Time series or grid of time series.
        lag (optional int): Number of time steps to lag correlate to.
        dim (optional str): Name of dimension to autocorrelate over.
        return_p (optional bool): If True, return correlation coefficients
                                  and p values.
    Returns:
        Pearson correlation coefficients.
        If return_p, also returns their associated p values.
    """
    X = X
    N = X[dim].size
    normal = X.isel({dim: slice(0, N - lag)})
    shifted = X.isel({dim: slice(0 + lag, N)})
    """
    xskillscore pearson_r looks for the dimensions to be matching, but we
    shifted them so they probably won't be. This solution doesn't work
    if the user provides a dataset without a coordinate for the main
    dimension, so we need to create a dummy dimension in that case.
    """
    if dim not in list(X.coords):
        normal[dim] = np.arange(1, N)
    shifted[dim] = normal[dim]
    r = pearson_r(normal, shifted, dim)
    if return_p:
        # NOTE: This assumes 2-tailed. Need to update eff_pearsonr
        # to utilize xskillscore's metrics but then compute own effective
        # p-value with option for one-tailed.
        p = pearson_r_p_value(normal, shifted, dim)
        return p
    else:
        return r

@set_property("fctype", "simple")
def ts_complexity_cid_ce(X, normalize=True, dim='time', **kwargs):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    
    if normalize:
        s = X.std(dim)
        #if s!=0:
        X = (X - X.mean(dim))/s
        #else:
        #    return 0.0

    X = X.diff(dim)
    return np.sqrt(X.dot(X,dims=dim) ).fillna(0)

# def _ts_complexity_cid_ce(X,normalize):
#     if normalize:
#         ##s = X.std(dim)
#         ##if s!=0:
#         #X = (X - X.mean(dim))/(X.std(dim))
#          X = (lambda X, dim: (X - X.mean(dim))/(X.std(dim)))(X,'time')
#         ##else:
#         ##    return 0.0

#     X = X.diff('time')
    
#     # return np.sqrt(X.dot(X,dims=dim) ).fillna(0)
#     return np.sqrt(X.dot(X,dims='time') ).fillna(0)
 
# @set_property("fctype", "simple")
# def ts_complexity_cid_ce(X, normalize=True, dim='time', **kwargs):
#     """
#     This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
#     valleys etc.). It calculates the value of

#     .. math::

#         \\sqrt{ \\sum_{i=0}^{n-2lag} ( x_{i} - x_{i+1})^2 }

#     .. rubric:: References

#     |  [1] Batista, Gustavo EAPA, et al (2014).
#     |  CID: an efficient complexity-invariant distance for time series.
#     |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

#     :param X: the time series to calculate the feature of
#     :type X: xarray.DataArray
#     :param normalize: should the time series be z-transformed?
#     :type normalize: bool

#     :return: the value of this feature
#     :return type: float
#     """

#     return xr.apply_ufunc(_ts_complexity_cid_ce, X,
#                             input_core_dims=[[dim]],
#                             kwargs={'normalize':normalize  },
#                             dask='parallelized',
#                              vectorize=True,
#                             output_dtypes=[float])
 
    # time_len = len( X )
    # func = lambda x: np.unique(x, 'time')[0].size / time_len
    
    # return xr.apply_ufunc(func, X,
    #                    input_core_dims=[[dim]],
    #                    vectorize=True,
    #                    kwargs={ },
    #                    dask='parallelized',
    #                    output_dtypes=[float])
    


@set_property("fctype", "ufunc")
def mean_second_derivative_central(X , dim='time', **kwargs):
    """
    Returns the mean over the differences between subsequent time series values which is
    
    .. math::
    
    \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})
    
    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    func = lambda x: (x[:,:,-1] - x[:,:,-2]  - x[:,:,1] + x[:,:,0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN

    return xr.apply_ufunc(func, X,
                           input_core_dims=[[dim]],
                           dask='parallelized',
                           output_dtypes=[float])
 

@set_property("fctype", "ufunc")
def median(X, dim='time', **kwargs):
    """
    Returns the median of x

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return xr.apply_ufunc(nanmedian, X,
                       input_core_dims=[[dim]],
                       kwargs={'axis': -1},
                       dask='parallelized',
                       output_dtypes=[float])

 
@set_property("fctype", "ufunc")
def mean(X, dim='time', **kwargs):
    """
    Returns the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return xr.apply_ufunc(nanmean, X,
                       input_core_dims=[[dim]],
                       kwargs={'axis': -1},
                       dask='parallelized',
                       output_dtypes=[float])
    
   
@set_property("fctype", "ufunc")
def length(X, dim='time', **kwargs):
    """
    Returns the mean of X

    :param X: the time series to calculate thfunce feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: floatde
    """
    return xr.apply_ufunc(np.size, X,
                       input_core_dims=[[dim]],
                       kwargs={ 'axis': -1},
                       vectorize=True,
                       dask='parallelized',
                       output_dtypes=[np.int32])



@set_property("fctype", "ufunc")
def standard_deviation(X, dim='time', **kwargs):
    """
    Returns the standard deviation of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    
    return xr.apply_ufunc(nanstd, X,
                       input_core_dims=[[dim]],
                       kwargs={'axis': -1},
                       dask='parallelized',
                       output_dtypes=[float])


@set_property("fctype", "ufunc")
def variance(X, dim='time', **kwargs):
    """
    Returns the variance of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return xr.apply_ufunc(nanvar, X,
                       input_core_dims=[[dim]],
                       kwargs={'axis': -1},
                       dask='parallelized',
                       output_dtypes=[float])
    

 

@set_property("fctype", "ufunc")
def skewness(X, dim='time', **kwargs):
    """
    Returns the sample skewness of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1). Normal value = 0, skewness > 0 means more weight in the left tail of 
    the distribution. 

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(skew, X,
                       input_core_dims=[[dim]],
                       kwargs={'axis': -1,'nan_policy':'omit'},
                       dask='parallelized',
                       output_dtypes=[float])

   
    
@set_property("fctype", "ufunc")
def kurtosis(X, dim='time', fisher=False, **kwargs):

    """
    Returns the kurtosis of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2). If fisher = True, returns fishers definition centered on 0. 

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :param X: If True, Fishers definition is used (normal =0), If False, Pearson's
            definition is used (normal = 3)
    :type X: boolean
    :return: the value of this feature
    :return type: float
    """
    
    return xr.apply_ufunc(kt, X,
                            input_core_dims=[[dim]],
                            kwargs={'axis': -1,'fisher':fisher,'nan_policy':'omit'},
                            dask='parallelized',
                            output_dtypes=[float])



@set_property("fctype", "ufunc")
def spearman_correlation(x, y, dim='time', **kwargs):
    """
    Returns the spearmans correlation of two xarray objects, which 
    must have the same dimensions.

    :param x: the time series to calculate the feature of
    :type x: xarray.DataArray
    :param y: the time series to calculate the feature of
    :type y: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    
    def _spearman_correlation_gufunc(x, y):
        x_ranks = rankdata(x, axis=-1)
        y_ranks = rankdata(y, axis=-1)
        return pearson_correlation_gufunc(x_ranks, y_ranks)
    
    return xr.apply_ufunc(
        _spearman_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])

@set_property("fctype", "ufunc")
def pearson_correlation(x, y, dim='time', **kwargs):
    """
    Returns the pearsons correlation of two xarray objects, which 
    must have the same dimensions.

    :param x: the time series to calculate the feature of
    :type x: xarray.DataArray
    :param y: the time series to calculate the feature of
    :type y: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """    
    def _covariance_gufunc(x, y):
        return ((x - x.mean(axis=-1, keepdims=True))
                * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

    def _pearson_correlation_gufunc(x, y):
        return _covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

    return xr.apply_ufunc(
        _pearson_correlation_gufunc, x, y,
        input_core_dims=[[dim], [dim]],
        dask='parallelized',
        output_dtypes=[float])
    
        

@set_property("fctype", "simple")
def longest_strike_below_mean(X, dim='time', **kwargs):
    """
    Returns the length of the longest consecutive subsequence in X that is smaller than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: longest period below the mean pixel value
    :return type: float
    """
    
    return longest_run(X < mean(X))

@set_property("fctype", "simple")
def longest_strike_above_mean(X, dim='time', **kwargs):
    """
    Returns the length of the longest consecutive subsequence in X that is smaller than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: longest period below the mean pixel value
    :return type: float
    """

     
    return longest_run(X > mean(X)) 


@set_property("fctype", "simple")
def count_above_mean(X, dim='time', **kwargs):
    """
    Returns the number of values in X that are higher than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return ((X > X.mean(dim)).sum(dim)).astype(np.int32)

@set_property("fctype", "simple")
def count_below_mean(X, dim='time', **kwargs):
    """
    Returns the number of values in X that are higher than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return ((X < X.mean(dim)).sum(dim)).astype(np.int32)


@set_property("fctype", "simple")
def last_doy_of_maximum(x,dim='time', band ='NDVI', **kwargs):
    """
    Returns the last day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """
    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime('%j')) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates,shp[-1]*shp[-2]).reshape(-1,1,shp[-2],shp[-1])

    #create xarrary to match x for doy 
    xr_dates = xr.DataArray(dates,  dims=['time','band', 'y', 'x'], coords={'time': x.time,'band':['doy'] ,'y':x.y,'x':x.x,} )
    x = xr.concat([x ,xr_dates], dim='band')

    # remove all but maximum values
    x_at_max = x.where(x.sel(band= band) == x.sel(band= band).max('time'))
    
    #change output band value to match others 
    out = x_at_max.sel(band='doy').max('time')  
    out.band.values = np.array(band, dtype='<U4')

    return out

@set_property("fctype", "simple")
def first_doy_of_maximum(x,dim='time', band ='ppt', **kwargs):
    """
    Returns the first day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """
    
    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime('%j')) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates,shp[-1]*shp[-2]).reshape(-1,1,shp[-2],shp[-1])

    #create xarrary to match x for doy 
    xr_dates = xr.DataArray(dates,  dims=['time','band', 'y', 'x'], coords={'time': x.time,'band':['doy'] ,'y':x.y,'x':x.x,} )
    x = xr.concat([x ,xr_dates], dim='band')

    # remove all but maximum values
    x_at_max = x.where(x.sel(band= band) == x.sel(band= band).max('time'))

    #change output band value to match others 
    out =   x_at_max.sel(band='doy').min('time') 
    out.band.values = np.array(band, dtype='<U4')
    
    return out

@set_property("fctype", "simple")
def last_doy_of_minimum(x,dim='time', band ='ppt', **kwargs):
    """
    Returns the last day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """

    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime('%j')) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates,shp[-1]*shp[-2]).reshape(-1,1,shp[-2],shp[-1])

    #create xarrary to match x for doy 
    xr_dates = xr.DataArray(dates,  dims=['time','band', 'y', 'x'], coords={'time': x.time,'band':['doy'] ,'y':x.y,'x':x.x,} )
    x = xr.concat([x ,xr_dates], dim='band')

    # remove all but maximum values
    x_at_min = x.where(x.sel(band= band) == x.sel(band= band).min('time'))

    #change output band value to match others 
    out =  x_at_min.sel(band='doy').max('time') 
    out.band.values = np.array(band, dtype='<U4')

    return out


@set_property("fctype", "simple")
def first_doy_of_minimum(x,dim='time', band ='ppt', **kwargs):
    """
    Returns the first day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """

    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime('%j')) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates,shp[-1]*shp[-2]).reshape(-1,1,shp[-2],shp[-1])

    #create xarrary to match x for doy 
    xr_dates = xr.DataArray(dates,  dims=['time','band', 'y', 'x'], coords={'time': x.time,'band':['doy'] ,'y':x.y,'x':x.x,} )
    x = xr.concat([x ,xr_dates], dim='band')

    # remove all but maximum values
    x_at_min = x.where(x.sel(band= band) == x.sel(band= band).min('time'))

    #change output band value to match others 
    out = x_at_min.sel(band='doy').min('time') 
    out.band.values = np.array(band, dtype='<U4')

    return out

@set_property("fctype", "ufunc")
def ratio_value_number_to_time_series_length(X,dim='time',  **kwargs):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # unique values / # values

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    time_len = len( X )
    func = lambda x: np.unique(x, 'time')[0].size / time_len
    
    return xr.apply_ufunc(func, X,
                       input_core_dims=[[dim]],
                       vectorize=True,
                       kwargs={ },
                       dask='parallelized',
                       output_dtypes=[float])
    
 
    
@set_property("fctype", "simple")
def _k_cor(x,y, pthres = 0.05, direction = True, **kwargs):
    """
    Uses the scipy stats module to calculate a Kendall correlation test
    :x vector: Input pixel vector to run tests on
    :y vector: The date input vector
    :pthres: Significance of the underlying test
    :direction: output only direction as output (-1 & 1)
    """
    # Check NA values
    co = np.count_nonzero(~np.isnan(x))
    if co < 4: # If fewer than 4 observations return -9999
        return -9999
    # Run the kendalltau test
    tau, p_value =  kendalltau(x, y)

    # Criterium to return results in case of Significance
    if p_value < pthres:
        # Check direction
        if direction:
            if tau < 0:
                return -1
            elif tau > 0:
                return 1
        else:
            return tau
    else:
      return 0  


@set_property("fctype", "ufunc")
def kendall_time_correlation(X, dim='time', direction = True, **kwargs):
    """
    Returns the significance of a kendall tau test across all time periods in x.
    
    If direction is True, return 1 for sigificant + time trend and -1 for 
    significant - time trend. 
    
    Note: this function is slow. Please use dask see:
        https://examples.dask.org/xarray.html
        https://xarray.pydata.org/en/stable/dask.html
        
    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """
    # The function we are going to use for applying our kendal test per pixel

    # x = Pixel value, y = a vector containing the date, dim == dimension
    
    y = xr.DataArray(np.arange(len(X[dim]))+1, dims=dim,
                 coords={dim: X[dim]})  
     
    return xr.apply_ufunc(
        _k_cor, X , y,
        input_core_dims=[[dim], [dim]],
        kwargs={'direction':direction},
        vectorize=True,  
        dask='parallelized',
        output_dtypes=[int])

def _timereg(x, t, param  ):
    
    linReg = linregress(x=t, y=x)
    
    return getattr(linReg, param) 
 
    
@set_property("fctype", "ufunc")
def linear_time_trend(x, param="slope", dim='time', **kwargs):
    

    """
    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param param: contains text of the attribute name of the regression model
    :type param: list
    :return: the value of this feature
    :return type: int
    """
    
    t = xr.DataArray(np.arange(len(x[dim]))+1, dims=dim,
             coords={dim: x[dim]})
    
    return xr.apply_ufunc( _timereg, x , t,
        input_core_dims=[[dim], [dim]],
        kwargs={ 'param':param},
        vectorize=True,  
        dask='parallelized',
        output_dtypes=[float])
 

@set_property("fctype", "simple")
def quantile(x, q, dim='time', **kwargs):

    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param q: the quantile to calculate [0,1]
    :type q: float
    :return: the value of this feature
    :return type: float
    """
    
    return x.quantile(q, dim).rename({'quantile':'band'})
    
@set_property("fctype", "simple")
def maximum(x,dim='time', **kwargs):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return x.max(dim)


@set_property("fctype", "simple")
def minimum(x,dim='time', **kwargs):
	
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return x.min(dim)


# from xclim https://github.com/Ouranosinc/xclim/blob/51123e0bbcaa5ad8882877f6905d9b285e63ddd9/xclim/run_length.py
@set_property("fctype", "simple")
def get_npts(da: xr.DataArray) -> int:

    """Return the number of gridpoints in a DataArray.
        Parameters
        ----------
        da : xarray.DataArray
          N-dimensional input array
        Returns
        -------
        int
          Product of input DataArray coordinate sizes excluding the dimension 'time'
        """

    coords = list(da.coords)
    coords.remove("time")
    npts = 1
    for c in coords:
        npts *= da[c].size
    return npts


@set_property("fctype", "simple")
def rle(da: xr.DataArray, dim: str = "time", max_chunk: int = 1_000_000):
    n = len(da[dim])
    i = xr.DataArray(np.arange(da[dim].size), dims=dim).chunk({"time": 1})
    ind = xr.broadcast(i, da)[0].chunk(da.chunks)
    b = ind.where(~da)  # find indexes where false
    end1 = (
        da.where(b[dim] == b[dim][-1], drop=True) * 0 + n
    )  # add additional end value index (deal with end cases)
    start1 = (
        da.where(b[dim] == b[dim][0], drop=True) * 0 - 1
    )  # add additional start index (deal with end cases)
    b = xr.concat([start1, b, end1], dim)

    # Ensure bfill operates on entire (unchunked) time dimension
    # Determine appropraite chunk size for other dims - do not exceed 'max_chunk' total size per chunk (default 1000000)
    ndims = len(b.shape)
    chunk_dim = b[dim].size
    # divide extra dims into equal size
    # Note : even if calculated chunksize > dim.size result will have chunk==dim.size
    chunksize_ex_dims = None
    if ndims > 1:
        chunksize_ex_dims = np.round(np.power(max_chunk / chunk_dim, 1 / (ndims - 1)))
    chunks = dict()
    chunks[dim] = -1
    for dd in b.dims:
        if dd != dim:
            chunks[dd] = chunksize_ex_dims
    b = b.chunk(chunks)

    # back fill nans with first position after
    z = b.bfill(dim=dim)
    # calculate lengths
    d = z.diff(dim=dim) - 1
    d = d.where(d >= 0)
    return d

npts_opt = 9000


@set_property("fctype", "simple")
def longest_run(
    da: xr.DataArray, dim: str = "time", ufunc_1dim: Union[str, bool] = "auto"
):
    """Return the length of the longest consecutive run of True values.
        Parameters
        ----------
        da : xr.DataArray
          N-dimensional array (boolean)
        dim : str
          Dimension along which to calculate consecutive run; Default: 'time'.
        ufunc_1dim : Union[str, bool]
          Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
          usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
          for dataarray with a small number of gridpoints.
        Returns
        -------
        N-dimensional array (int)
          Length of longest run of True values along dimension
        """
    if ufunc_1dim == "auto":
        npts = get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        rl_long = longest_run_ufunc(da)
    else:
        d = rle(da, dim=dim)
        rl_long = d.max(dim=dim)

    return rl_long



@set_property("fctype", "simple")
def rle_1d(arr: Union[int, float, bool, Sequence[Union[int, float, bool]]]
           ) -> Tuple[np.array, np.array, np.array]:
    """Return the length, starting position and value of consecutive identical values.
    Parameters
    ----------
    arr : Sequence[Union[int, float, bool]]
      Array of values to be parsed.
    Returns
    -------
    values : np.array
      The values taken by arr over each run
    run lengths : np.array
      The length of each run
    start position : np.array
      The starting index of each run
    Examples
    --------
    >>> a = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
    >>> rle_1d(a)
    (array([1, 2, 3]), array([2, 4, 6]), array([0, 2, 6]))
    """
    ia = np.asarray(arr)
    n = len(ia)

    if n == 0:
        e = "run length array empty"
        warn(e)
        # Returning None makes some other 1d func below fail.
        return np.array(np.nan), 0, np.array(np.nan)

    y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
    i = np.append(np.where(y), n - 1)  # must include last element position
    rl = np.diff(np.append(-1, i))  # run lengths
    pos = np.cumsum(np.append(0, rl))[:-1]  # positions
    return ia[i], rl, pos 


def _longest_run_1d(arr: Sequence[bool]) -> int:
    """Return the length of the longest consecutive run of identical values.
    Parameters
    ----------
    arr : Sequence[bool]
      Input array (bool)
    Returns
    -------
    int
      Length of longest run.
    """
    v, rl = rle_1d(arr)[:2]
    return np.where(v, rl, 0).max()


@set_property("fctype", "ufunc")
def longest_run_ufunc(x: Sequence[bool]) -> xr.apply_ufunc:
    """Dask-parallel version of longest_run_1d, ie the maximum number of consecutive true values in
    array.
    Parameters
    ----------
    x : Sequence[bool]
      Input array (bool)
    Returns
    -------
    out : func
      A function operating along the time dimension of a dask-array.
    """
    return xr.apply_ufunc(
        _longest_run_1d,
        x,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def dpp(ds, dim='time', m=10, chunk=True):
    """
    https://climpred.readthedocs.io/en/stable/_modules/climpred/stats.html
    Calculates the Diagnostic Potential Predictability (dpp)

    .. math::

        DPP_{\\mathrm{unbiased}}(m) = \\frac{\\sigma^{2}_{m} -
        \\frac{1}{m}\\cdot\\sigma^{2}}{\\sigma^{2}}

    Note:
        Resplandy et al. 2015 and Seferian et al. 2018 calculate unbiased DPP
        in a slightly different way: chunk=False.

    Args:
        ds (xr.DataArray): control simulation with time dimension as years.
        dim (str): dimension to apply DPP on. Default: time.
        m (optional int): separation time scale in years between predictable
                          low-freq component and high-freq noise.
        chunk (optional boolean): Whether chunking is applied. Default: True.
                    If False, then uses Resplandy 2015 / Seferian 2018 method.

    Returns:
        dpp (xr.DataArray): ds without time dimension.

    References:
        * Boer, G. J. “Long Time-Scale Potential Predictability in an Ensemble of
          Coupled Climate Models.” Climate Dynamics 23, no. 1 (August 1, 2004):
          29–44. https://doi.org/10/csjjbh.
        * Resplandy, L., R. Séférian, and L. Bopp. “Natural Variability of CO2 and
          O2 Fluxes: What Can We Learn from Centuries-Long Climate Models
          Simulations?” Journal of Geophysical Research: Oceans 120, no. 1
          (January 2015): 384–404. https://doi.org/10/f63c3h.
        * Séférian, Roland, Sarah Berthet, and Matthieu Chevallier. “Assessing the
          Decadal Predictability of Land and Ocean Carbon Uptake.” Geophysical
          Research Letters, March 15, 2018. https://doi.org/10/gdb424.

    """

    def _chunking(ds, dim='time', number_chunks=False, chunk_length=False):
        """
        Separate data into chunks and reshapes chunks in a c dimension.

        Specify either the number chunks or the length of chunks.
        Needed for dpp.

        Args:
            ds (xr.DataArray): control simulation with time dimension as years.
            dim (str): dimension to apply chunking to. Default: time
            chunk_length (int): see dpp(m)
            number_chunks (int): number of chunks in the return data.

        Returns:
            c (xr.DataArray): chunked ds, but with additional dimension c.

        """
        if number_chunks and not chunk_length:
            chunk_length = np.floor(ds[dim].size / number_chunks)
            cmin = int(ds[dim].min())
        elif not number_chunks and chunk_length:
            cmin = int(ds[dim].min())
            number_chunks = int(np.floor(ds[dim].size / chunk_length))
        else:
            raise KeyError('set number_chunks or chunk_length to True')
        c = ds.sel({dim: slice(cmin, cmin + chunk_length - 1)})
        c = c.expand_dims('c')
        c['c'] = [0]
        for i in range(1, number_chunks):
            c2 = ds.sel(
                {dim: slice(cmin + chunk_length * i, cmin + (i + 1) * chunk_length - 1)}
            )
            c2 = c2.expand_dims('c')
            c2['c'] = [i]
            c2[dim] = c[dim]
            c = xr.concat([c, c2], 'c')
        return c

    if not chunk:  # Resplandy 2015, Seferian 2018
        s2v = ds.rolling({dim: m}).mean().var(dim)
        s2 = ds.var(dim)

    if chunk:  # Boer 2004 ppvf
        # first chunk
        chunked_means = _chunking(ds, dim=dim, chunk_length=m).mean(dim)
        # sub means in chunks
        chunked_deviations = _chunking(ds, dim=dim, chunk_length=m) - chunked_means
        s2v = chunked_means.var('c')
        s2e = chunked_deviations.var([dim, 'c'])
        s2 = s2v + s2e
    dpp = (s2v - s2 / (m)) / s2
    return dpp



def decorrelation_time(da, r=20, dim='time'):

    """Calculate the decorrelaton time of a time series.
https://climpred.readthedocs.io/en/stable/_modules/climpred/stats.html
    .. math::
        \\tau_{d} = 1 + 2 * \\sum_{k=1}^{r}(\\alpha_{k})^{k}

    Args:
        da (xarray object): Time series.
        r (optional int): Number of iterations to run the above formula.
        dim (optional str): Time dimension for xarray object.

    Returns:
        Decorrelation time of time series.

    Reference:
        * Storch, H. v, and Francis W. Zwiers. Statistical Analysis in Climate
          Research. Cambridge ; New York: Cambridge University Press, 1999.,
          p.373

    """
    one = xr.ones_like(da.isel({dim: 0}))
    one = one.where(da.isel({dim: 0}).notnull())
    return one + 2 * xr.concat(
        [autocorr(da, dim=dim, lag=i) ** i for i in range(1, r)], 'it'
    ).sum('it')


def varweighted_mean_period(da, dim='time', **kwargs):
    """Calculate the variance weighted mean period of time series based on
    xrft.power_spectrum.
    https://climpred.readthedocs.io/en/stable/_modules/climpred/stats.html

    .. math::
        P_{x} = \\frac{\\sum_k V(f_k,x)}{\\sum_k f_k  \\cdot V(f_k,x)}

    Args:
        da (xarray object): input data including dim.
        dim (optional str): Name of time dimension.
        for **kwargs see xrft.power_spectrum

    Reference:
      * Branstator, Grant, and Haiyan Teng. “Two Limits of Initial-Value
        Decadal Predictability in a CGCM." Journal of Climate 23, no. 23
        (August 27, 2010): 6292-6311. https://doi.org/10/bwq92h.

    See also:
    https://xrft.readthedocs.io/en/latest/api.html#xrft.xrft.power_spectrum
    """
    # set nans to 0
    if isinstance(da, xr.Dataset):
        raise ValueError('require xr.Dataset')
    da = da.fillna(0.0)
    # dim should be list
    if isinstance(dim, str):
        dim = [dim]
    assert isinstance(dim, list)
    ps = power_spectrum(da, dim=dim, **kwargs)
    # take pos
    for d in dim:
        ps = ps.where(ps[f'freq_{d}'] > 0)
    # weighted average
    vwmp = ps
    for d in dim:
        vwmp = vwmp.sum(f'freq_{d}') / ((vwmp * vwmp[f'freq_{d}']).sum(f'freq_{d}'))
    for d in dim:
        del vwmp[f'freq_{d}_spacing']
    # try to copy coords
    try:
        vwmp = copy_coords_from_to(da.drop(dim), vwmp)
    except ValueError:
        warnings.warn("Couldn't keep coords.")
    return vwmp