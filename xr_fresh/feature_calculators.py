import numpy as np
import xarray as xr
from bottleneck import rankdata

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
from bottleneck import (
    nanmean,
    nanmedian,
    nansum,
    nanstd,
    nanvar,
    ss,
    allnan,
)  # nanmin, nanmax,anynan,

# from numba import jit, njit
from numba import float64, float32, int32, int16, guvectorize, int64, void


logging.captureWarnings(True)


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """

    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = (
                func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
            )
        return func

    return decorate_func


@set_property("fctype", "ufunc")
def abs_energy(X, dim="time", **kwargs):
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

    return xr.apply_ufunc(
        lambda x: pow(x, 2),
        X,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    ).sum(dim)


def _abs_energy(x):
    return __abs_energy(x)


@guvectorize([(float64[:], float64[:])], "(n) -> ()", nopython=True)
def __abs_energy(x, out):
    out[:] = np.sum(np.square(x))


@set_property("fctype", "simple")
def _abs_energy_slower(x, dim="time", **kwargs):
    """
    Not as fast as regular abs energy Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2

    :param x: the time series to calculate the feature of
    :param dim: core dimension in xarray to apply across
    :type x: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(
        _abs_energy,
        x,
        input_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


def test(test):
    """[summary]

    :param test: [description]
    :type test: [type]
    :raises KeyError: [description]
    :raises ValueError: [description]
    :return: [description]
    :rtype: [type]
    """


@set_property("fctype", "simple")
def absolute_sum_of_changes(X, dim="time", **kwargs):
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
def autocorr(X, lag=1, dim="time", return_p=True, **kwargs):
    """Calculate the lagged correlation of time series. Amended from xskillscore package

    :param X: Time series or grid of time series.
    :type X: xr.DataArray
    :param lag: Number of time steps to lag correlate to. optional, defaults to 1
    :type lag: int, optional
    :param dim: Name of dimension to autocorrelate over. optional, defaults to "time"
    :type dim: str, optional
    :param return_p: If True, return p values. optional, defaults to True
    :type return_p: bool, optional
    :return: Pearson correlation coefficients (r). If return_p, returns its associated p values.
    :rtype: float
    """

    X = X
    N = X[dim].size
    normal = X.isel({dim: slice(0, N - lag)})
    shifted = X.isel({dim: slice(0 + lag, N)})
    # """
    # xskillscore pearson_r looks for the dimensions to be matching, but we
    # shifted them so they probably won't be. This solution doesn't work
    # if the user provides a dataset without a coordinate for the main
    # dimension, so we need to create a dummy dimension in that case.
    # """
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
def count_above_mean(X, dim="time", **kwargs):
    """
    Returns the number of values in X that are higher than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return ((X > X.mean(dim)).sum(dim)).astype(np.float64)


@set_property("fctype", "simple")
def count_below_mean(X, dim="time", **kwargs):
    """
    Returns the number of values in X that are higher than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return ((X < X.mean(dim)).sum(dim)).astype(np.float64)


def decorrelation_time(da, r=20, dim="time"):
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
        [autocorr(da, dim=dim, lag=i) ** i for i in range(1, r)], "it"
    ).sum("it")


@set_property("fctype", "simple")
def doy_of_maximum_last(x, dim="time", band="NDVI", **kwargs):
    """
    Returns the last day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """
    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime("%j")) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates, shp[-1] * shp[-2]).reshape(-1, 1, shp[-2], shp[-1])

    # create xarrary to match x for doy
    xr_dates = xr.DataArray(
        dates,
        dims=["time", "band", "y", "x"],
        coords={
            "time": x.time,
            "band": ["doy"],
            "y": x.y,
            "x": x.x,
        },
    )
    x = xr.concat([x, xr_dates], dim="band")

    # remove all but maximum values
    x_at_max = x.where(x.sel(band=band) == x.sel(band=band).max("time"))

    # change output band value to match others
    out = x_at_max.sel(band="doy").max("time")
    out.band.values = np.array(band, dtype="<U4")

    return out


@set_property("fctype", "simple")
def doy_of_maximum_first(x, dim="time", band="ppt", **kwargs):
    """
    Returns the first day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """

    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime("%j")) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates, shp[-1] * shp[-2]).reshape(-1, 1, shp[-2], shp[-1])

    # create xarrary to match x for doy
    xr_dates = xr.DataArray(
        dates,
        dims=["time", "band", "y", "x"],
        coords={
            "time": x.time,
            "band": ["doy"],
            "y": x.y,
            "x": x.x,
        },
    )
    x = xr.concat([x, xr_dates], dim="band")

    # remove all but maximum values
    x_at_max = x.where(x.sel(band=band) == x.sel(band=band).max("time"))

    # change output band value to match others
    out = x_at_max.sel(band="doy").min("time")
    out.band.values = np.array(band, dtype="<U4")

    return out


@set_property("fctype", "simple")
def doy_of_minimum_last(x, dim="time", band="ppt", **kwargs):
    """
    Returns the last day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """

    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime("%j")) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates, shp[-1] * shp[-2]).reshape(-1, 1, shp[-2], shp[-1])

    # create xarrary to match x for doy
    xr_dates = xr.DataArray(
        dates,
        dims=["time", "band", "y", "x"],
        coords={
            "time": x.time,
            "band": ["doy"],
            "y": x.y,
            "x": x.x,
        },
    )
    x = xr.concat([x, xr_dates], dim="band")

    # remove all but maximum values
    x_at_min = x.where(x.sel(band=band) == x.sel(band=band).min("time"))

    # change output band value to match others
    out = x_at_min.sel(band="doy").max("time")
    out.band.values = np.array(band, dtype="<U4")

    return out


@set_property("fctype", "simple")
def doy_of_minimum_first(x, dim="time", band="ppt", **kwargs):
    """
    Returns the first day of the year (doy) location of the maximum value of x.
    The position is calculated relatively to the length of x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: int
    """

    # create doy array to match x
    dates = np.array([int(to_datetime(i).strftime("%j")) for i in x[dim].values])
    shp = x.shape
    dates = np.repeat(dates, shp[-1] * shp[-2]).reshape(-1, 1, shp[-2], shp[-1])

    # create xarrary to match x for doy
    xr_dates = xr.DataArray(
        dates,
        dims=["time", "band", "y", "x"],
        coords={
            "time": x.time,
            "band": ["doy"],
            "y": x.y,
            "x": x.x,
        },
    )
    x = xr.concat([x, xr_dates], dim="band")

    # remove all but maximum values
    x_at_min = x.where(x.sel(band=band) == x.sel(band=band).min("time"))

    # change output band value to match others
    out = x_at_min.sel(band="doy").min("time")
    out.band.values = np.array(band, dtype="<U4")

    return out


@set_property("fctype", "simple")
def _k_cor(x, y, pthres=0.05, direction=True, **kwargs):
    """
    Uses the scipy stats module to calculate a Kendall correlation test
    :x vector: Input pixel vector to run tests on
    :y vector: The date input vector
    :pthres: Significance of the underlying test
    :direction: output only direction as output (-1 & 1)
    """
    # Check NA values
    co = np.count_nonzero(~np.isnan(x))
    if co < 4:  # If fewer than 4 observations return -9999
        return -9999
    # Run the kendalltau test
    tau, p_value = kendalltau(x, y)

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
def kendall_time_correlation(X, dim="time", direction=True, **kwargs):
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

    y = xr.DataArray(np.arange(len(X[dim])) + 1, dims=dim, coords={dim: X[dim]})

    return xr.apply_ufunc(
        _k_cor,
        X,
        y,
        input_core_dims=[[dim], [dim]],
        kwargs={"direction": direction},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "ufunc")
def kurtosis(X, dim="time", fisher=False, **kwargs):
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

    return xr.apply_ufunc(
        kt,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1, "fisher": fisher, "nan_policy": "omit"},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def large_standard_deviation(X, r=2, dim="time", **kwargs):
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

    return (X.std(dim) > (r * (X.max(dim) - X.min(dim)))).astype(np.float64)


@set_property("fctype", "ufunc")
def length(X, dim="time", **kwargs):
    """
    Returns the length of X

    :param X: the time series to calculate thfunce feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: floatde
    """
    return xr.apply_ufunc(
        np.size,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64],
        keep_attrs=True,
    )


def _regression_gufunc(y_array):
    from scipy.stats import t

    y_array = np.moveaxis(y_array, -1, 0)
    x_array = np.empty(y_array.shape)
    for i in range(y_array.shape[0]):
        x_array[i, :, :] = (
            i + 1
        )  # This would be fine if time series is not too long. Or we can use i+yr (e.g. 2019).
    x_array[np.isnan(y_array)] = np.nan
    # Compute the number of non-nan over each (lon,lat) grid box.
    n = np.sum(~np.isnan(x_array), axis=0)
    # Compute mean and standard deviation of time series of x_array and y_array over each (lon,lat) grid box.
    x_mean = np.nanmean(x_array, axis=0)
    y_mean = np.nanmean(y_array, axis=0)
    x_std = np.nanstd(x_array, axis=0)
    y_std = np.nanstd(y_array, axis=0)
    # Compute co-variance between time series of x_array and y_array over each (lon,lat) grid box.
    cov = np.nansum((x_array - x_mean) * (y_array - y_mean), axis=0) / n
    # Compute correlation coefficients between time series of x_array and y_array over each (lon,lat) grid box.
    cor = cov / (x_std * y_std)
    # Compute slope between time series of x_array and y_array over each (lon,lat) grid box.
    slope = cov / (x_std**2)
    # Compute intercept between time series of x_array and y_array over each (lon,lat) grid box.
    intercept = y_mean - x_mean * slope
    # Compute tstats, stderr, and p_val between time series of x_array and y_array over each (lon,lat) grid box.
    tstats = cor * np.sqrt(n - 2) / np.sqrt(1 - cor**2)

    p_val = t.sf(tstats, n - 2) * 2
    # Compute r_square and rmse between time series of x_array and y_array over each (lon,lat) grid box.
    # r_square also equals to cor**2 in 1-variable lineare regression analysis, which can be used for checking.
    r_square = np.nansum(
        (slope * x_array + intercept - y_mean) ** 2, axis=0
    ) / np.nansum((y_array - y_mean) ** 2, axis=0)
    # rmse = np.sqrt(np.nansum((y_array - slope * x_array - intercept) ** 2, axis=0) / n)
    # Do further filteration if needed (e.g. We stipulate at least 3 data records are needed to do regression analysis) and return values
    n = n * 1.0  # convert n from integer to float to enable later use of np.nan
    n[n < 3] = np.nan
    slope[np.isnan(n)] = np.nan
    intercept[np.isnan(n)] = np.nan
    p_val[np.isnan(n)] = np.nan
    r_square[np.isnan(n)] = np.nan
    # rmse[np.isnan(n)] = np.nan

    return intercept, slope, p_val, r_square


def _timereg(x, param):
    # avoid all missing
    if allnan(x):
        if param != "all":
            return np.NaN
        else:
            return np.stack((np.NaN, np.NaN, np.NaN, np.NaN), axis=-1)

    else:
        intercept, slope, pvalue, rvalue = _regression_gufunc(x)
        reg_param = {
            "intercept": intercept,
            "slope": slope,
            "pvalue": pvalue,
            "rvalue": rvalue,
        }
        if param != "all":
            try:
                return reg_param[param]
            except KeyError:
                print(param, "Not available parameter")

        else:
            return np.stack(
                (
                    intercept,
                    slope,
                    pvalue,
                    rvalue,
                ),
                axis=-1,
            )


@set_property("fctype", "ufunc")
def linear_time_trend(x, param="all", dim="time", **kwargs):
    """
    # look at https://stackoverflow.com/questions/58719696/how-to-apply-a-xarray-u-function-over-netcdf-and-return-a-2d-array-multiple-new/62012973

    Calculate a linear least-squares regression for the values of the time series versus the sequence from 0 to
    length of the time series minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters control which of the characteristics are returned.

    Possible extracted attributes are "all", "pvalue", "rvalue", "intercept", "slope", "stderr", see the documentation of
    linregress for more information.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param param: contains text of the attribute name of the regression model
    :type param: list
    :return: the value of this feature
    :return type: int
    """
    if param == "all":
        out = (
            xr.apply_ufunc(
                _timereg,
                x,
                input_core_dims=[[dim]],
                kwargs={"param": param},
                dask="parallelized",
                output_dtypes=[float],
                output_core_dims=[["variable"]],
                output_sizes={"variable": 4},
                keep_attrs=True,
            )
            .to_dataset(dim="variable")
            .to_array()
        )
    else:
        out = xr.apply_ufunc(
            _timereg,
            x,
            input_core_dims=[[dim]],
            kwargs={"param": param},
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )

    return out


# another implementation https://stackoverflow.com/questions/52094320/with-xarray-how-to-parallelize-1d-operations-on-a-multidimensional-dataset
# def new_linregress(x, y):
#    # Wrapper around scipy linregress to use in apply_ufunc
#    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
#    return np.array([slope, intercept, r_value, p_value, std_err])
#
## return a new DataArray
# stats = xr.apply_ufunc(new_linregress, ds[x], ds[y],
#                       input_core_dims=[['year'], ['year']],
#                       output_core_dims=[["parameter"]],
#                       vectorize=True,
#                       dask="parallelized",
#                       output_dtypes=['float64'],
#                       output_sizes={"parameter": 5},
#                      )


# from xclim https://github.com/Ouranosinc/xclim/blob/51123e0bbcaa5ad8882877f6905d9b285e63ddd9/xclim/run_length.py
@set_property("fctype", "simple")
def _get_npts(da: xr.DataArray) -> int:
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
def longest_run(
    da: xr.DataArray,
    dim: str = "time",
    ufunc_1dim: Union[str, bool] = "auto",
    npts_opt=9000,
):
    """Return the length of the longest consecutive run of True values.


    :param da: N-dimensional array (boolean)
    :type da: xr.DataArray
    :param dim: Dimension along which to calculate consecutive run, defaults to "time"
    :type dim: str, optional
    :param ufunc_1dim: Use the 1d 'ufunc' version of this function : default (auto) will attempt to select optimal
          usage based on number of data points.  Using 1D_ufunc=True is typically more efficient
          for dataarray with a small number of gridpoints, defaults to "auto"
    :type ufunc_1dim: Union[str, bool], optional
    :param npts_opt: [description], defaults to 9000
    :type npts_opt: int, optional
    :return: Length of longest run of True values along dimension
    :rtype: ndarray (int)
    """

    if ufunc_1dim == "auto":
        npts = _get_npts(da)
        ufunc_1dim = npts <= npts_opt

    if ufunc_1dim:
        rl_long = longest_run_ufunc(da)
    else:
        d = rle(da, dim=dim)
        rl_long = d.max(dim=dim)

    return rl_long


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
def longest_strike_below_mean(X, dim="time", **kwargs):
    """
    Returns the length of the longest consecutive subsequence in X that is smaller than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: longest period below the mean pixel value
    :return type: float
    """

    return longest_run(X < mean(X))


@set_property("fctype", "simple")
def longest_strike_above_mean(X, dim="time", **kwargs):
    """
    Returns the length of the longest consecutive subsequence in X that is smaller than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: longest period below the mean pixel value
    :return type: float
    """

    return longest_run(X > mean(X))


@set_property("fctype", "simple")
def maximum(x, dim="time", **kwargs):
    """
    Calculates the highest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return x.max(dim)


@set_property("fctype", "ufunc")
def mean(X, dim="time", **kwargs):
    """
    Returns the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return xr.apply_ufunc(
        nanmean,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def mean_abs_change(X, dim="time", **kwargs):
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
def mean_change(X, dim="time", **kwargs):
    """
    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1,\\ldots, n-1}  x_{i+1} - x_{i} = \\frac{1}{n-1} (x_{n} - x_{1})

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    func = lambda x: (x[:, :, -1] - x[:, :, 0]) / (len(x) - 1)
    return xr.apply_ufunc(
        func,
        X,
        input_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


# @guvectorize(['void(float64[:], float64)',
#               'void(float32[:], float64)',
#               'void(int64[:], float64)',
#               'void(int32[:], float64)',
#               'void(int16[:], float64)',
#               ],
#               "(n) -> ()" )#, nopython=True )
# def _msdc(x, out):
#     x = x[~np.isnan(x)]
#     out[:] = (x[:,:,-1] - x[:,:,-2]  - x[:,:,1] + x[:,:,0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN


@set_property("fctype", "ufunc")
def mean_second_derivative_central(X, dim="time", **kwargs):
    """
    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1,\\ldots, n-1}  \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_i)

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    # func = lambda x: (x[:,:,-1] - x[:,:,-2]  - x[:,:,1] + x[:,:,0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN

    # return xr.apply_ufunc(_msdc, X,
    #                        input_core_dims=[[dim]],
    #                        dask='parallelized',
    #                        output_dtypes=[np.float64],
    #                        keep_attrs= True )

    return X.diff(dim).sum(dim) / (len(X) - 1)


@set_property("fctype", "ufunc")
def median(X, dim="time", **kwargs):
    """
    Returns the median of x

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(
        nanmedian,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def minimum(x, dim="time", **kwargs):
    """
    Calculates the lowest value of the time series x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return x.min(dim)


def _covariance_gufunc(x, y):
    return (
        (x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))
    ).mean(axis=-1)


def _pearson_correlation_gufunc(x, y):
    return _covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))


@set_property("fctype", "ufunc")
def pearson_correlation(x, y, dim="time", **kwargs):
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

    return xr.apply_ufunc(
        _pearson_correlation_gufunc,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


def _get_bottleneck_funcs(skipna, **kwargs):
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
    sumfunc, meanfunc = _get_bottleneck_funcs(skipna, **kwargs)

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

    :param a: Input array.
    :type a: ndarray
    :param b: Input array.
    :type b: ndarray
    :param axis: The axis to apply the correlation along.
    :type axis: int
    :param skipna: If True, skip NaNs when computing function.
    :type skipna: bool
    :return: 2-tailed p-value.
    :rtype: ndarray

    See Also
    --------
    scipy.stats.pearsonr
    """

    r = _pearson_r(a, b, axis, skipna)
    a = np.rollaxis(a, axis)
    b = np.rollaxis(b, axis)
    dof = np.apply_over_axes(np.sum, np.isnan(a * b), 0).squeeze() - 2
    dof = np.where(dof > 1.0, dof, a.shape[0] - 2)
    t_squared = r**2 * (dof / ((1.0 - r) * (1.0 + r)))
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
def pearson_r(a, b, dim="time", skipna=False, **kwargs):
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
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})

    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        _pearson_r,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "ufunc")
def pearson_r_p_value(a, b, dim="time", skipna=False, **kwargs):
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
        new_dim = "_".join(dim)
        a = a.stack(**{new_dim: dim})
        b = b.stack(**{new_dim: dim})

    else:
        new_dim = dim[0]

    return xr.apply_ufunc(
        _pearson_r_p_value,
        a,
        b,
        input_core_dims=[[new_dim], [new_dim]],
        kwargs={"axis": -1, "skipna": skipna},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def potential_predictability(ds, dim="time", m=10, chunk=True):
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

    def _chunking(ds, dim="time", number_chunks=False, chunk_length=False):
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
            raise KeyError("set number_chunks or chunk_length to True")
        c = ds.sel({dim: slice(cmin, cmin + chunk_length - 1)})
        c = c.expand_dims("c")
        c["c"] = [0]
        for i in range(1, number_chunks):
            c2 = ds.sel(
                {dim: slice(cmin + chunk_length * i, cmin + (i + 1) * chunk_length - 1)}
            )
            c2 = c2.expand_dims("c")
            c2["c"] = [i]
            c2[dim] = c[dim]
            c = xr.concat([c, c2], "c")
        return c

    if not chunk:  # Resplandy 2015, Seferian 2018
        s2v = ds.rolling({dim: m}).mean().var(dim)
        s2 = ds.var(dim)

    if chunk:  # Boer 2004 ppvf
        # first chunk
        chunked_means = _chunking(ds, dim=dim, chunk_length=m).mean(dim)
        # sub means in chunks
        chunked_deviations = _chunking(ds, dim=dim, chunk_length=m) - chunked_means
        s2v = chunked_means.var("c")
        s2e = chunked_deviations.var([dim, "c"])
        s2 = s2v + s2e
    dpp = (s2v - s2 / (m)) / s2
    return dpp


@set_property("fctype", "simple")
def ratio_beyond_r_sigma(X, r=2, dim="time", **kwargs):
    """
    Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return (np.abs(X - X.mean(dim)) > r * X.std(dim)).sum(dim) / len(X)


@set_property("fctype", "ufunc")
def skewness(X, dim="time", **kwargs):
    """
    Returns the sample skewness of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1). Normal value = 0, skewness > 0 means more weight in the left tail of
    the distribution.

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(
        skew,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1, "nan_policy": "omit"},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


def _spearman_correlation_gufunc(x, y):
    x_ranks = rankdata(x, axis=-1)
    y_ranks = rankdata(y, axis=-1)
    return _pearson_correlation_gufunc(x_ranks, y_ranks)


@set_property("fctype", "ufunc")
def spearman_correlation(x, y, dim="time", **kwargs):
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

    return xr.apply_ufunc(
        _spearman_correlation_gufunc,
        x,
        y,
        input_core_dims=[[dim], [dim]],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "ufunc")
def standard_deviation(X, dim="time", **kwargs):
    """
    Returns the standard deviation of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(
        nanstd,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def sum_values(X, dim="time", **kwargs):
    """
    Calculates the sum over the time series values

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    if allnan(X):
        return np.NaN
    return X.sum(dim)


@set_property("fctype", "simple")
def symmetry_looking(X, r=0.1, dim="time", **kwargs):
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
    return (mean_median_difference < (r * max_min_difference)).astype(np.float64)


@set_property("fctype", "simple")
def ts_complexity_cid_ce(X, normalize=True, dim="time", **kwargs):
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
        # if s!=0:
        X = (X - X.mean(dim)) / s
        # else:
        #    return 0.0

    return np.sqrt(X.dot(X.diff(dim), dims=dim))


@set_property("fctype", "ufunc")
def variance(X, dim="time", **kwargs):
    """
    Returns the variance of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """
    return xr.apply_ufunc(
        nanvar,
        X,
        input_core_dims=[[dim]],
        kwargs={"axis": -1},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


@set_property("fctype", "simple")
def variance_larger_than_standard_deviation(X, dim="time", **kwargs):
    """
    Boolean variable denoting if the variance of x is greater than its standard deviation. Is equal to variance of x
    being larger than 1

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: bool
    """
    y = X.var(dim)
    return (y > np.sqrt(y)).astype(np.float64)


@set_property("fctype", "simple")
def quantile_slow(x, q, skipna=True, dim="time", **kwargs):
    """
    Calculates the q quantile of x. This is the value of x greater than q% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param q: the quantile to calculate [0,1]
    :type q: float
    :return: the value of this feature
    :return type: float
    """

    return x.quantile(q, dim).rename({"quantile": "band"})


def _quantile(x, q):
    return __quantile(x, q)


@guvectorize(
    [
        "void(float64[:],float64, float64[:])",
        "void(float32[:],float64, float64[:])",
        "void(int64[:],float64, float64[:])",
        "void(int32[:],float64, float64[:])",
        "void(int16[:],float64, float64[:])",
    ],
    "(n), ()-> ()",
    nopython=True,
)
def __quantile(x, q: float, out):
    out[:] = np.nanquantile(x, q)


@set_property("fctype", "simple")
def quantile(x, q, dim="time", **kwargs):
    """
    Ultrafast implimentation of np.nanpercentile.
    Calculates the 95th percentile of x. This is the value of x greater than 95% of the ordered values from x.

    :param x: the time series to calculate the feature of
    :type x:  xarray.DataArray
    :param q: the quantile to calculate [0,1]
    :type q: float
    :return: the value of this feature
    :return type: float
    """

    return xr.apply_ufunc(
        _quantile,
        x,
        input_core_dims=[[dim]],
        dask="parallelized",
        kwargs={"q": q},
        output_dtypes=[np.float64],
        keep_attrs=True,
    )


@set_property("fctype", "ufunc")
def ratio_value_number_to_time_series_length(X, dim="time", **kwargs):
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
    time_len = len(X)
    func = lambda x: np.unique(x, "time")[0].size / time_len

    return xr.apply_ufunc(
        func,
        X,
        input_core_dims=[[dim]],
        vectorize=True,
        kwargs={},
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )


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


@set_property("fctype", "simple")
def rle_1d(
    arr: Union[int, float, bool, Sequence[Union[int, float, bool]]]
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


def varweighted_mean_period(da, dim="time", **kwargs):
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
        raise ValueError("require xr.Dataset")
    da = da.fillna(0.0)
    # dim should be list
    if isinstance(dim, str):
        dim = [dim]
    assert isinstance(dim, list)
    ps = power_spectrum(da, dim=dim, **kwargs)
    # take pos
    for d in dim:
        ps = ps.where(ps[f"freq_{d}"] > 0)
    # weighted average
    vwmp = ps
    for d in dim:
        vwmp = vwmp.sum(f"freq_{d}") / ((vwmp * vwmp[f"freq_{d}"]).sum(f"freq_{d}"))
    for d in dim:
        del vwmp[f"freq_{d}_spacing"]
    # try to copy coords
    try:
        vwmp = copy_coords_from_to(da.drop(dim), vwmp)
    except ValueError:
        warnings.warn("Couldn't keep coords.")
    return vwmp


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
