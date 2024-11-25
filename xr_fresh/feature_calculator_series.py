import jax.numpy as jnp
import numpy as np
import geowombat as gw
from datetime import datetime
from typing import List, Union, Any

__all__ = [
    "abs_energy",
    "absolute_sum_of_changes",
    "autocorrelation",
    "count_above_mean",
    "count_below_mean",
    "doy_of_maximum",
    "doy_of_minimum",
    "kurtosis",
    "kurtosis_excess",
    "large_standard_deviation",
    "longest_strike_above_mean",
    "longest_strike_below_mean",
    "maximum",
    "minimum",
    "mean",
    "mean_abs_change",
    "mean_change",
    "mean_second_derivative_central",
    "median",
    "ols_slope_intercept",
    "quantile",
    "ratio_beyond_r_sigma",
    "skewness",
    "standard_deviation",
    "sum",
    "symmetry_looking",
    "ts_complexity_cid_ce",
    "unique_value_number_to_time_series_length",
    "variance",
    "variance_larger_than_standard_deviation",
]


def _get_day_of_year(dt):
    return int(dt.strftime("%j"))


def _check_valid_array(obj):
    # Check if the object is a NumPy or JAX array or list
    if not isinstance(obj, (np.ndarray, list)):  # jnp.DeviceArray,
        raise TypeError("Object must be a NumPy array or list.")

    # convert lists to numpy array
    if isinstance(obj, list):
        obj = np.array(obj)  # must be np array not jnp

    # Check if the array contains only integers or datetime objects
    if jnp.issubdtype(obj.dtype, np.integer):
        return jnp.array(obj)

    # datetime objects are converted to integers
    elif jnp.issubdtype(obj.dtype, datetime):
        return jnp.array(np.vectorize(_get_day_of_year)(obj))
    else:
        raise TypeError("Array must contain only integers, datetime objects.")


class abs_energy(gw.TimeModule):
    """
    Returns the absolute energy of the time series which is the sum over the squared values.

    .. math::

        E = \\sum_{i=1}^{n} x_i^2

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
    """

    def __init__(self):
        super(abs_energy, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.square(array), axis=0).squeeze()


class absolute_sum_of_changes(gw.TimeModule):
    """
    Returns the sum over the absolute value of consecutive changes in the series x.

    .. math::

        \\sum_{i=1}^{n-1} \\mid x_{i+1} - x_i \\mid

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
    """

    def __init__(self):
        super(absolute_sum_of_changes, self).__init__()

    def calculate(self, array):
        return jnp.nansum(jnp.abs(jnp.diff(array, n=1, axis=0)), axis=0).squeeze()


class autocorrelation(gw.TimeModule):
    """
    Calculates the autocorrelation of the specified lag, according to the formula [1].

    .. math::

        \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
    mean. `l` denotes the lag.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        lag (int): lag at which to calculate the autocorrelation (default: {1}).
    """

    def __init__(self, lag=1):
        super(autocorrelation, self).__init__()
        self.lag = lag

    def calculate(self, array):
        series = array[: -self.lag]
        lagged_series = array[self.lag :]
        autocor = (
            jnp.nansum(series * lagged_series, axis=0) / jnp.nansum(series**2, axis=0)
        ).squeeze()

        return autocor


class count_above_mean(gw.TimeModule):
    """
    Returns the number of values in x that are higher than the mean of x.

    .. math::

        N_{\\text{above}} = \\sum_{i=1}^n (x_i > \\bar{x})

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        mean (int): An integer to use as the "mean" value of the raster
    """

    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            return jnp.nansum(array > jnp.nanmean(array, axis=0), axis=0).squeeze()
        else:
            return jnp.nansum(array > self.mean, axis=0).squeeze()


class count_below_mean(gw.TimeModule):
    """
    Returns the number of values in x that are lower than the mean of x.

    .. math::

        N_{\\text{below}} = \\sum_{i=1}^n (x_i < \\bar{x})

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
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


class doy_of_maximum(gw.TimeModule):
    """
    Returns the day of the year (doy) location of the maximum value of the series - treats all years as the same.

    Args:
        dates (numpy.ndarray): An array holding the dates of the time series as integers or as datetime objects.
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        int: The day of the year of the maximum value.
    """

    def __init__(self, dates=None):
        super(doy_of_maximum, self).__init__()
        # check that dates is an array holding datetime objects or integers throw error if not
        dates = _check_valid_array(dates)
        self.dates = jnp.array(dates) if dates is not None else None

    def calculate(self, array):
        if self.dates is None:
            raise ValueError("Dates array is not provided.")
        # Find the indices of the maximum values along the time axis
        max_indices = jnp.argmax(array, axis=0)
        # Use the indices to extract the corresponding dates from the 'dates' array
        return self.dates[max_indices].squeeze()


class doy_of_minimum(gw.TimeModule):
    """
    Returns the day of the year (doy) location of the minimum value of the series - treats all years as the same.

    Args:
        dates (numpy.ndarray): An array holding the dates of the time series as integers or as datetime objects.
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        int: The day of the year of the minimum value.
    """

    def __init__(self, dates=None):
        super(doy_of_minimum, self).__init__()
        dates = _check_valid_array(dates)
        self.dates = jnp.array(dates) if dates is not None else None

    def calculate(self, array):
        if self.dates is None:
            raise ValueError("Dates array is not provided.")
        min_indices = jnp.argmin(array, axis=0)
        return self.dates[min_indices].squeeze()


class kurtosis(gw.TimeModule):
    """
    Compute the sample kurtosis of a given array along the time axis.

    .. math::

        G_2 = \\frac{\\mu_4}{\\sigma^4} - 3

    where :math:`\\mu_4` is the fourth central moment and :math:`\\sigma` is the standard deviation.

    Args:
        array (GeoWombat series object): An object that contains geospatial and temporal metadata.
        fisher (bool, optional): If True, Fisher’s definition is used (normal ==> 0.0).
                                 If False, Pearson’s definition is used (normal ==> 3.0).

    Returns:
        float: Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).
    """

    def __init__(self, fisher=True):
        super(kurtosis, self).__init__()
        self.fisher = fisher

    def calculate(self, array):
        mean_ = jnp.nanmean(array, axis=0)
        mu4 = jnp.nanmean((array - mean_) ** 4, axis=0)
        mu2 = jnp.nanmean((array - mean_) ** 2, axis=0)
        beta2 = mu4 / (mu2**2)
        gamma2 = beta2 - 3
        return gamma2.squeeze()


class kurtosis_excess(gw.TimeModule):
    """
    Compute the excess kurtosis of a given array along the time axis.

    .. math::

        G_2 = \\frac{\\mu_4}{\\sigma^4} - 3

    where :math:`\\mu_4` is the fourth central moment and :math:`\\sigma` is the standard deviation.

    Args:
        array (GeoWombat series object): An object that contains geospatial and temporal metadata.
        fisher (bool, optional): If True, Fisher’s definition is used (normal ==> 0.0).
                                 If False, Pearson’s definition is used (normal ==> 3.0).

    Returns:
        float: Returns the excess kurtosis of X (calculated with the adjusted Fisher-Pearson standardized moment coefficient G2).
    """

    def __init__(self, Fisher=True):
        super(kurtosis_excess, self).__init__()

    def calculate(self, array):
        mean_x = jnp.nanmean(array, axis=0)
        var_x = jnp.nanvar(array, axis=0)
        centered_x = array - mean_x
        fourth_moment = jnp.nanmean(centered_x**4, axis=0)

        kurt = fourth_moment / (var_x**2)
        return kurt.squeeze()


class large_standard_deviation(gw.TimeModule):
    """
    Boolean variable denoting if the standard dev of x is higher than 'r' times the range.

    Args:
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


def _count_longest_consecutive(values):
    max_count = 0
    current_count = 0

    for value in values:
        if value:
            current_count += 1
            max_count = jnp.nanmax(jnp.array([max_count, current_count]))
        else:
            current_count = 0

    return max_count


# try importing rle
try:
    from xr_fresh import rle

    longest_true_run = rle.longest_true_run
except ImportError:
    print("C++ rle module not found, using Python version")
    longest_true_run = _count_longest_consecutive


class longest_strike_above_mean(gw.TimeModule):
    """
    Returns the length of the longest consecutive subsequence in x that is bigger than the mean of x.

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    """

    def __init__(self, mean=None):
        super(longest_strike_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        # compare to mean
        if self.mean is None:
            below_mean = array > jnp.nanmean(array, axis=0)
        else:
            below_mean = array > self.mean
        # Count the longest consecutive True values along the time dimension
        consecutive_true = np.apply_along_axis(
            func1d=longest_true_run, axis=0, arr=below_mean
        ).squeeze()

        # Count the longest consecutive False values along the time dimension
        consecutive_false = np.apply_along_axis(
            func1d=longest_true_run, axis=0, arr=~below_mean
        ).squeeze()

        return jnp.maximum(consecutive_true, consecutive_false)


class longest_strike_below_mean(gw.TimeModule):
    """
    Returns the length of the longest consecutive subsequence in x that is smaller than the mean of x.

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    """

    def __init__(self, mean=None):
        super(longest_strike_below_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        # compare to mean
        if self.mean is None:
            below_mean = array < jnp.nanmean(array, axis=0)
        else:
            below_mean = array < self.mean

        # Count the longest consecutive True values along the time dimension
        consecutive_true = np.apply_along_axis(
            longest_true_run, axis=0, arr=below_mean
        ).squeeze()

        # Count the longest consecutive False values along the time dimension
        consecutive_false = np.apply_along_axis(
            longest_true_run, axis=0, arr=~below_mean
        ).squeeze()

        return jnp.maximum(consecutive_true, consecutive_false)


class maximum(gw.TimeModule):
    """
    Returns the maximum value of the time series x.

    .. math::

        x_{\\text{max}}

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The maximum value.
    """

    def __init__(self):
        super(maximum, self).__init__()

    def calculate(self, x):
        return jnp.nanmax(x, axis=0).squeeze()


class minimum(gw.TimeModule):
    """
    Returns the minimum value of the time series x.

    .. math::

        x_{\\text{min}}

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The minimum value.
    """

    def __init__(self):
        super(minimum, self).__init__()

    def calculate(self, x):
        return jnp.nanmin(x, axis=0).squeeze()


class mean(gw.TimeModule):
    """
    Returns the mean value of the time series x.

    .. math::

        \\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The mean value.
    """

    def __init__(self):
        super(mean, self).__init__()

    def calculate(self, x):
        return jnp.nanmean(x, axis=0).squeeze()


class mean_abs_change(gw.TimeModule):
    """
    Returns the mean over the absolute differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1}^{n-1} | x_{i+1} - x_{i} |

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The mean absolute change.
    """

    def __init__(self):
        super(mean_abs_change, self).__init__()

    def calculate(self, x):
        abs_diff = jnp.abs(jnp.diff(x, axis=0))
        return jnp.nanmean(abs_diff, axis=0).squeeze()


class mean_change(gw.TimeModule):
    """
    Returns the mean over the differences between subsequent time series values which is

    .. math::

        \\frac{1}{n-1} \\sum_{i=1}^{n-1} ( x_{i+1} - x_{i} )

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The mean change.
    """

    def __init__(self):
        super(mean_change, self).__init__()

    def calculate(self, array):
        diff = array[1:] - array[:-1]
        return jnp.nanmean(diff, axis=0).squeeze()


class mean_second_derivative_central(gw.TimeModule):
    """
    Returns the mean value of a central approximation of the second derivative of the time series.

    .. math::

        \\frac{1}{2(n-2)} \\sum_{i=1}^{n-2} \\frac{1}{2} (x_{i+2} - 2 \\cdot x_{i+1} + x_{i})

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The mean second derivative.
    """

    def __init__(self):
        super(mean_second_derivative_central, self).__init__()

    def calculate(self, array):
        series2 = array[:-2]
        lagged2 = array[2:]
        lagged1 = array[1:-1]
        msdc = jnp.nansum(0.5 * (lagged2 - 2 * lagged1 + series2), axis=0) / (
            (2 * (len(array) - 2))
        )

        return msdc.squeeze()


class median(gw.TimeModule):
    """
    Returns the median of the time series x.

    .. math::

        \\tilde{x}

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The median value.
    """

    def __init__(self):
        super(median, self).__init__()

    def calculate(self, x):
        return jnp.nanmedian(x, axis=0).squeeze()


def _lstsq(data):
    """
    Calculate the least-squares solution to a linear matrix equation.
    """

    M = data
    x = jnp.arange(0, M.shape[0])
    reg = jnp.linalg.lstsq(jnp.c_[x, jnp.ones_like(x)], M, rcond=None)
    slope_intercept = reg[0]
    residuals = reg[1]
    # Fit a least squares solution to each sample
    return slope_intercept, residuals


class ols_slope_intercept(gw.TimeModule):
    """
    Calculate the slope, intercept, and R2 of the time series using ordinary least squares.

    Args:
        gw (array): the time series data
        returns (str, optional): What to return, "slope", "intercept" or "rsquared". Defaults to "slope".

    Returns:
        array: Return desired time series property array.
    """

    def __init__(self, returns="slope"):
        super(ols_slope_intercept, self).__init__()

        allowed_values = ["slope", "intercept", "rsquared"]
        self.returns = returns

        if self.returns not in allowed_values:
            raise ValueError(f"Invalid argument. Allowed values are {allowed_values}")

    def calculate(self, array):
        if self.returns == "slope":
            array, residuals = jnp.apply_along_axis(_lstsq, axis=0, arr=array)
            slope = array[0, 0, :, :]
            return slope.squeeze()
        elif self.returns == "intercept":
            array, residuals = jnp.apply_along_axis(_lstsq, axis=0, arr=array)
            intercept = array[0, 1, :, :]
            return intercept.squeeze()
        elif self.returns == "rsquared":
            array, SSR = jnp.apply_along_axis(_lstsq, axis=0, arr=array)
            y = jnp.arange(0, array.shape[0])
            TSS = jnp.nansum((y - jnp.nanmean(y)) ** 2)
            return (1 - SSR / TSS).squeeze()


class quantile(gw.TimeModule):
    """
    Calculates the q-th quantile of x. This is the value of x greater than q% of the ordered values from x.

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        q (float): Probability or sequence of probabilities for the quantiles to compute. Values must be between 0 and 1 inclusive.

    Returns:
        float: The q-th quantile of x.
    """

    def __init__(self, q=None, method="linear"):
        super(quantile, self).__init__()
        self.q = q
        self.method = method

    def calculate(self, array):
        return jnp.nanquantile(array, q=self.q, method=self.method, axis=0).squeeze()


class ratio_beyond_r_sigma(gw.TimeModule):
    """
    Returns the ratio of values that are more than r times the standard deviation away from the mean of the time series.

    .. math::

        P_{r} = \\frac{1}{n} \\sum_{i=1}^{n} (| x_i - \\bar{x} | > r \\cdot \\sigma)

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        r (float): The number of standard deviations. Defaults to 2.

    Returns:
        float: The ratio of values beyond r sigma.
    """

    def __init__(self, r=2):
        super(ratio_beyond_r_sigma, self).__init__()
        self.r = r

    def calculate(self, array):
        out = (
            jnp.nansum(
                jnp.abs(array - jnp.nanmean(array, axis=0))
                > self.r * jnp.nanstd(array, axis=0),
                axis=0,
            )
            / len(array)
        ).squeeze()
        return jnp.where(jnp.isnan(out), 0, out)


class skewness(gw.TimeModule):
    """
    Returns the skewness of x.

    .. math::

        \\frac{n}{(n-1)(n-2)} \\sum \\left( \\frac{X_i - \\overline{X}}{s} \\right)^3

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        axis (int, optional): Axis along which to compute the kurtosis. Default is 0.
        fisher (bool, optional): If True, Fisher's definition is used (normal=0).
                                 If False, Pearson's definition is used (normal=3).
                                 Default is False.

    Returns:
        float: The skewness.
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
    """
    Returns the standard deviation of x.

    .. math::

        \\sqrt{ \\frac{1}{N} \\sum_{i=1}^{n} (x_i - \\bar{x})^2 }

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The standard deviation.
    """

    def __init__(self):
        super(standard_deviation, self).__init__()

    def calculate(self, x):
        return jnp.nanstd(x, axis=0).squeeze()


class sum(gw.TimeModule):
    """
    Returns the sum of all values in x.

    .. math::

        S = \\sum_{i=1}^{n} x_i

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The sum of values.
    """

    def __init__(self):
        super(sum, self).__init__()

    def calculate(self, x):
        return jnp.nansum(x, axis=0).squeeze()


class symmetry_looking(gw.TimeModule):
    """
    Measures the similarity of the time series when flipped horizontally. Boolean variable denoting if the distribution of x *looks symmetric*.

    .. math::

        | x_{\\text{mean}} - x_{\\text{median}} | < r \\cdot (x_{\\text{max}} - x_{\\text{min}} )

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        r (float): A threshold value, the percentage of the range to compare with (default: 0.1)

    Returns:
        float: The symmetry measure.
    """

    def __init__(self, r=0.1):
        super(symmetry_looking, self).__init__()
        self.r = r

    def calculate(self, array):
        out = (
            jnp.abs(jnp.nanmean(array, axis=0) - jnp.nanmedian(array, axis=0))
            < (self.r * (jnp.nanmax(array, axis=0) - jnp.nanmin(array, axis=0)))
        ).squeeze()
        return jnp.where(jnp.isnan(out), 0, out)


class ts_complexity_cid_ce(gw.TimeModule):
    """
    Returns the time series complexity measure CID CE.

    .. math::

        \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.
        normalize: should the time series be z-transformed? (default: True)

    Returns:
        float: The complexity measure.
    """

    def __init__(self, normalize=True):
        super(ts_complexity_cid_ce, self).__init__()
        self.normalize = normalize

    def calculate(self, array):
        if self.normalize:
            s = jnp.std(array, axis=0)
            array = jnp.where(s != 0, (array - jnp.nanmean(array, axis=0)) / s, array)
            array = jnp.where(s == 0, 0.0, array)
        x = jnp.diff(array, axis=0)
        try:
            dot_prod = jnp.einsum("tijk, tijk->jk", x, x)
        except:
            dot_prod = jnp.einsum("ijk, ijk->jk", x, x)
        return jnp.sqrt(dot_prod)


class unique_value_number_to_time_series_length(gw.TimeModule):
    """
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # of unique values / # of values
    """

    def __init__(self):
        super(unique_value_number_to_time_series_length, self).__init__()
        print("this is slow and needs more work")

    def calculate(self, array):
        # Count the number of unique values along the time axis (axis=0)
        unique_counts = jnp.sum(jnp.unique(array, axis=0), axis=0)

        return (unique_counts / len(array)).squeeze()


class variance(gw.TimeModule):
    """
    Returns the variance of x.

    .. math::

        \\sigma^2 = \\frac{1}{N} \\sum_{i=1}^{n} (x_i - \\bar{x})^2

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        float: The variance.
    """

    def __init__(self):
        super(variance, self).__init__()

    def calculate(self, x):
        return jnp.nanvar(x, axis=0).squeeze()


class variance_larger_than_standard_deviation(gw.TimeModule):
    """
    Returns 1 if variance of x is larger than its standard deviation and 0 otherwise.

    .. math::

        \\sigma^2 > 1

    Args:
        x (numpy.ndarray): Geowombat series object contain time series of images.

    Returns:
        int: 1 if variance is larger than standard deviation, 0 otherwise.
    """

    def __init__(self):
        super(variance_larger_than_standard_deviation, self).__init__()

    def calculate(self, x):
        out = (jnp.nanvar(x, axis=0) > jnp.nanstd(x, axis=0)).astype(np.int8).squeeze()

        return jnp.where(jnp.isnan(out), 0, out)


function_mapping = {
    "abs_energy": abs_energy,
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "autocorrelation": autocorrelation,
    "count_above_mean": count_above_mean,
    "count_below_mean": count_below_mean,
    "doy_of_maximum": doy_of_maximum,
    "doy_of_minimum": doy_of_minimum,
    "kurtosis": kurtosis,
    "kurtosis_excess": kurtosis_excess,
    "large_standard_deviation": large_standard_deviation,
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "maximum": maximum,
    "minimum": minimum,
    "mean": mean,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    "median": median,
    "ols_slope_intercept": ols_slope_intercept,
    "quantile": quantile,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "skewness": skewness,
    "standard_deviation": standard_deviation,
    "sum": sum,
    "symmetry_looking": symmetry_looking,
    "ts_complexity_cid_ce": ts_complexity_cid_ce,
    "unique_value_number_to_time_series_length": unique_value_number_to_time_series_length,
    "variance": variance,
    "variance_larger_than_standard_deviation": variance_larger_than_standard_deviation,
}
