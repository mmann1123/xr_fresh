#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .backends import Cluster

from .feature_calculator_series import (
    interpolate_nan_dates,
    interpolate_nan,
    doy_of_maximum,
    abs_energy,
    abs_energy2,
    autocorrelation,
    doy_of_maximum,
    plot_interpolated_actual,
)

__all__ = ["Cluster"]

__version__ = "0.0.1"
__author__ = "Michael Mann"
__license__ = "MIT"
__maintainer__ = "Michael Mann"
__email__ = "mmann1123@gwu.edu"
