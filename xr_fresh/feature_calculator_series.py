import jax.numpy as jnp
import numpy as np
import geowombat as gw


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


# skipped
# def pearson_r(a, b, dim="time", skipna=False, **kwargs):
