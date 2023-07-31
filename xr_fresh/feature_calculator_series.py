import jax.numpy as jnp
import numpy as np
import geowombat as gw
from datetime import datetime

# ratio_beyond_r_sigma


# Define a function to apply strftime('%j') to each element
def _get_day_of_year(dt):
    return int(dt.strftime("%j"))


def _check_valid_array(obj):
    # Check if the object is a NumPy or JAX array or list
    if not isinstance(obj, (np.ndarray, jnp.DeviceArray, list)):
        raise TypeError("Object must be a NumPy, JAX array or list.")

    # convert lists to numpy array
    if isinstance(obj, list):
        obj = np.array(obj)

    # Check if the array contains only integers or datetime objects
    if np.issubdtype(obj.dtype, np.integer):
        return jnp.array(obj)

    # datetime objects are converted to integers
    elif np.issubdtype(obj.dtype, datetime):
        return jnp.array(np.vectorize(_get_day_of_year)(obj))
    else:
        raise TypeError("Array must contain only integers, datetime objects.")


# %% WORKING


class abs_energy(gw.TimeModule):
    """
    Returns the absolute energy of the time series which is the sum over the squared values

    .. math::

        E = \\sum_{i=1,\\ldots, n} x_i^2
    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(abs_energy, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.square(array), axis=0).squeeze()


class absolute_sum_of_changes(gw.TimeModule):
    """
    Returns the sum over the absolute value of consecutive changes in the series x

    .. math::

        \\sum_{i=1, \\ldots, n-1} \\mid x_{i+1}- x_i \\mid

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(absolute_sum_of_changes, self).__init__()

    def calculate(self, array):
        return jnp.nansum(np.abs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()


class autocorrelation(gw.TimeModule):
    """Returns the autocorrelation of the time series data at a specified lag

    Args:
        gw (_type_): _description_
        lag (int): lag at which to calculate the autocorrelation (default: {1})
    """

    def __init__(self, lag=1):
        super(autocorrelation, self).__init__()
        self.lag = lag

    def calculate(self, array):
        # Extract the series and its lagged version
        series = array[: -self.lag]
        lagged_series = array[self.lag :]
        autocor = (
            jnp.sum(series * lagged_series, axis=0) / jnp.sum(series**2, axis=0)
        ).squeeze()

        return autocor


class count_above_mean(gw.TimeModule):
    """Returns the number of values in X that are higher than the mean of X

    Args:
        gw (_type_): _description_
        mean (int): An integer to use as the "mean" value of the raster
    """

    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            # Calculate the mean along the time dimension (axis=0) and broadcast it to match the shape of 'array'
            return jnp.nansum(array > jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class count_below_mean(gw.TimeModule):
    """Returns the number of values in X that are lower than the mean of X

    Args:
        gw (_type_): _description_
        mean (int): An integer to use as the "mean" value of the raster
    """

    def __init__(self, mean=None):
        super(count_below_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            # Calculate the mean along the time dimension (axis=0) and broadcast it to match the shape of 'array'
            return jnp.nansum(array < jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class doy_of_max(gw.TimeModule):
    """Returns the day of the year (doy) location of the maximum value of the series - treats all years as the same.

    pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
    files = sorted(glob(f"{pth}*.tif"))[0:10]
    strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
    dates = sorted(datetime.strptime(string, strp_glob) for string in files)
    dates

    with gw.series(
    files,
    nodata=9999,
        ) as src:
            print(src)
            src.apply(
                func=doy_of_max(dates),
                outfile=f"/home/mmann1123/Downloads/test.tif",
                num_workers=1,
                bands=1,
            )

    Args:
        gw (_type_): _description_
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.
    """

    def __init__(self, dates=None):
        super(doy_of_max, self).__init__()
        # check that dates is an array holding datetime objects or integers throw error if not
        dates = _check_valid_array(dates)
        self.dates = dates
        print("Day of the year found as:", self.dates)

    def calculate(self, array):
        # Find the indices of the maximum values along the time axis
        max_indices = np.argmax(array, axis=0)

        # Use the indices to extract the corresponding dates from the 'dates' array
        return self.dates[max_indices].squeeze()


class doy_of_min(gw.TimeModule):
    """Returns the day of the year (doy) location of the minimum value of the series - treats all years as the same.

    pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
    files = sorted(glob(f"{pth}*.tif"))[0:10]
    strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
    dates = sorted(datetime.strptime(string, strp_glob) for string in files)
    dates

    with gw.series(
    files,
    nodata=9999,
        ) as src:
            print(src)
            src.apply(
                func=doy_of_max(dates),
                outfile=f"/home/mmann1123/Downloads/test.tif",
                num_workers=1,
                bands=1,
            )

    Args:
        gw (_type_): _description_
        dates (np.array): An array holding the dates of the time series as integers or as datetime objects.
    """

    def __init__(self, dates=None):
        super(doy_of_min, self).__init__()
        # check that dates is an array holding datetime objects or integers throw error if not
        dates = _check_valid_array(dates)
        self.dates = dates
        print("Day of the year found as:", self.dates)

    def calculate(self, array):
        # Find the indices of the maximum values along the time axis
        min_indices = np.argmin(array, axis=0)

        # Use the indices to extract the corresponding dates from the 'dates' array
        return self.dates[min_indices].squeeze()


class kurtosis_excess(gw.TimeModule):
    """
    # https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
    Returns the excess kurtosis of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).
    Args:
        gw (_type_): _description_

    """

    def __init__(self):
        super(kurtosis_excess, self).__init__()

    def calculate(self, array):
        mean_ = jnp.mean(array, axis=0)

        mu4 = jnp.mean((array - mean_) ** 4, axis=0)
        mu2 = jnp.mean((array - mean_) ** 2, axis=0)
        beta2 = mu4 / (mu2**2)
        gamma2 = beta2 - 3
        return gamma2.squeeze()


class large_standard_deviation(gw.TimeModule):
    """Boolean variable denoting if the standard dev of x is higher than 'r' times the range.

    Args:
        gw (_type_): _description_
        r (float, optional): The percentage of the range to compare with. Default is 2.0.
    """

    def __init__(self, r=2):
        super(large_standard_deviation, self).__init__()
        self.r = r

    def calculate(self, array):
        std_dev = jnp.nanstd(array, axis=0)
        max_val = jnp.nanmax(array, axis=0)
        min_val = jnp.nanmin(array, axis=0)

        return (std_dev > self.r * (max_val - min_val)).astype(jnp.int8).squeeze()


class maximum(gw.TimeModule):
    """Calculate the highest value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the maximum.
                             Default is "time".
    """

    def __init__(self):
        super(maximum, self).__init__()

    def calculate(self, x):
        return jnp.nanmax(x, axis=0).squeeze()


class mean(gw.TimeModule):
    """Calculate the mean value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(mean, self).__init__()

    def calculate(self, x):
        return jnp.nanmean(x, axis=0).squeeze()


class mean_abs_change(gw.TimeModule):
    """Calculate the mean over the absolute differences between subsequent time series values.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(mean_abs_change, self).__init__()

    def calculate(self, x):
        abs_diff = jnp.abs(jnp.diff(x, axis=0))
        return jnp.nanmean(abs_diff, axis=0).squeeze()


class mean_change(gw.TimeModule):
    """Calculate the mean over the differences between subsequent time series values.

    Args:
        gw (_type_): _description_

    """

    def __init__(self):
        super(mean_change, self).__init__()

    def calculate(self, array):
        diff = array[1:] - array[:-1]
        return np.nanmean(diff, axis=0).squeeze()


class median(gw.TimeModule):
    """Calculate the mean value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(median, self).__init__()

    def calculate(self, x):
        return jnp.nanmedian(x, axis=0).squeeze()


class ratio_beyond_r_sigma(gw.TimeModule):
    """Ratio of values that are more than r*std(x) (so r sigma) away from the mean of x.

    Args:
        gw (_type_): _description_
        r (int, optional):   Defaults to 2.
    """

    def __init__(self, r=2):
        super(ratio_beyond_r_sigma, self).__init__()
        self.r = r

    def calculate(self, array):
        return (
            jnp.nansum(
                jnp.abs(array - jnp.nanmean(array, axis=0))
                > self.r * jnp.nanstd(array, axis=0),
                axis=0,
            )
            / len(array)
        ).squeeze()


class skewness(gw.TimeModule):
    """
    # https://medium.com/@pritul.dave/everything-about-moments-skewness-and-kurtosis-using-python-numpy-df305a193e46
    Returns the sample skewness of X (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G1). Normal value = 0, skewness > 0 means more weight in the left tail of
    the distribution.
    Args:
        gw (_type_): _description_
        axis (int, optional): Axis along which to compute the kurtosis. Default is 0.
        fisher (bool, optional): If True, Fisher's definition is used (normal=0).
                                 If False, Pearson's definition is used (normal=3).
                                 Default is False.
    """

    def __init__(self):
        super(skewness, self).__init__()

    def calculate(self, array):
        _mean = jnp.nanmean(array, axis=0)
        _diff = array - _mean
        _mu3 = jnp.nanmean(_diff**3, axis=0)
        _mu2 = jnp.nanmean(_diff**2, axis=0)
        beta = _mu3**2 / _mu2**3
        return jnp.sqrt(beta).squeeze()


class standard_deviation(gw.TimeModule):
    """Calculate the standard_deviation value of the time series.

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(standard_deviation, self).__init__()

    def calculate(self, x):
        return jnp.nanstd(x, axis=0).squeeze()


class variance_larger_than_standard_deviation(gw.TimeModule):
    """Calculate the variance of the time series is larger than the standard_deviation.

    Args:
        gw (_type_): _description_
    Returns:
        bool:
    """

    def __init__(self):
        super(variance_larger_than_standard_deviation, self).__init__()

    def calculate(self, x):
        return (jnp.var(x, axis=0) > jnp.nanstd(x, axis=0)).astype(np.int8).squeeze()


# skipped
# def pearson_r(a, b, dim="time", skipna=False, **kwargs):
# linear_time_trend
# longest_run
