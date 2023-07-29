import jax.numpy as jnp
import numpy as np
import geowombat as gw
import pandas as pd


class count_above_mean(gw.TimeModule):
    """Returns the number of values in X that are higher than the mean of X

    Args:
        gw (_type_): _description_
    """

    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            self.mean = jnp.nanmean(array, axis=0)

        return jnp.nansum(array > self.mean, axis=0).squeeze()


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


class CountBelowMean(gw.TimeModule):
    """Returns the number of values in X that are lower than the mean of X

    Args:
        gw (_type_): _description_
    """

    def __init__(self):
        super(CountBelowMean, self).__init__()
        self.mean = None

    def calculate(self, array):
        if self.mean is None:
            self.mean = jnp.nanmean(array, axis=0)

        return jnp.nansum(array < self.mean, axis=0).squeeze()


# not sure about this
class DoyOfMaximumFirst(gw.TimeModule):
    """Returns the last day of the year (doy) location of the maximum value of x.

    Args:
        gw (_type_): _description_
        band (str): Band name to consider.
    """

    def __init__(self, band="NDVI"):
        super(DoyOfMaximumLast, self).__init__()
        self.band = band

    def calculate(self, x):
        # Create doy array to match x
        dates = jnp.array(
            [int(pd.to_datetime(i).strftime("%j")) for i in x.time.values]
        )
        shp = x.shape
        dates = jnp.repeat(dates, shp[-1] * shp[-2]).reshape(-1, 1, shp[-2], shp[-1])

        # Create xarray to match x for doy
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

        # Remove all but maximum values
        x_at_max = x.where(x.sel(band=self.band) == x.sel(band=self.band).max("time"))

        # Change output band value to match others
        out = x_at_max.sel(band="doy").max("time")
        out.band.values = jnp.array(self.band, dtype="<U4")

        return out


import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import kurtosis as kt


class Kurtosis(gw.TimeModule):
    """Calculate the kurtosis of the input time series along a specific axis.

    Args:
        gw (_type_): _description_
        axis (int, optional): Axis along which to compute the kurtosis. Default is 0.
        fisher (bool, optional): If True, Fisher's definition is used (normal=0).
                                 If False, Pearson's definition is used (normal=3).
                                 Default is False.
    """

    def __init__(self, axis=0, fisher=False, nan_policy="omit"):
        super(Kurtosis, self).__init__()
        self.axis = axis
        self.fisher = fisher
        self.nan_policy = nan_policy

    def calculate(self, array):
        # Using jnp.apply_along_axis to compute kurtosis along the specified axis
        return jnp.apply_along_axis(
            kt, self.axis, array, fisher=self.fisher, nan_policy=self.nan_policy
        )


class LargeStandardDeviation(gw.TimeModule):
    """Boolean variable denoting if the standard dev of x is higher than 'r' times the range.

    Args:
        gw (_type_): _description_
        r (float, optional): The percentage of the range to compare with. Default is 2.0.
    """

    def __init__(self, r=2):
        super(LargeStandardDeviation, self).__init__()
        self.r = r

    def calculate(self, array):
        std_dev = jnp.nanstd(array, axis=0)
        max_val = jnp.nanmax(array, axis=0)
        min_val = jnp.nanmin(array, axis=0)

        return (std_dev > self.r * (max_val - min_val)).astype(jnp.int8).squeeze()


class Maximum(gw.TimeModule):
    """Calculate the highest value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the maximum.
                             Default is "time".
    """

    def __init__(self):
        super(Maximum, self).__init__()

    def calculate(self, x):
        return jnp.nanmax(x, axis=0).squeeze()


import jax.numpy as jnp


class Mean(gw.TimeModule):
    """Calculate the mean value of the time series.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(Mean, self).__init__()

    def calculate(self, x):
        return jnp.nanmean(x, axis=0).squeeze()


class MeanAbsChange(gw.TimeModule):
    """Calculate the mean over the absolute differences between subsequent time series values.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(MeanAbsChange, self).__init__()

    def calculate(self, x):
        abs_diff = jnp.abs(jnp.diff(x, axis=0))
        return jnp.nanmean(abs_diff, axis=0).squeeze()


class MeanChange(gw.TimeModule):
    """Calculate the mean over the differences between subsequent time series values.

    Args:
        gw (_type_): _description_
        dim (str, optional): The name of the dimension along which to calculate the mean.
                             Default is "time".
    """

    def __init__(self):
        super(MeanChange, self).__init__()

    def calculate(self, x):
        diff = x[..., 1:] - x[..., :-1]
        return np.mean(diff, axis=-1).squeeze()


# skipped
# def pearson_r(a, b, dim="time", skipna=False, **kwargs):
# def autocorr(X, lag=1, dim="time", return_p=True, **kwargs):
# def doy_of_minimum_last(x, dim="time", band="ppt", **kwargs):
# doy_of_minimum_first
# linear_time_trend
# longest_run
