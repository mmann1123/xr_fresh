#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .backends import Cluster

from .feature_calculator_series import (
    abs_energy,
    absolute_sum_of_changes,
    autocorrelation,
    count_above_mean,
    count_below_mean,
    doy_of_maximum,
    doy_of_minimum,
    kurtosis,
    kurtosis_excess,
    large_standard_deviation,
    longest_strike_above_mean,
    longest_strike_below_mean,
    maximum,
    minimum,
    mean,
    mean_abs_change,
    mean_change,
    mean_second_derivative_central,
    median,
    ols_slope_intercept,
    quantile,
    ratio_beyond_r_sigma,
    skewness,
    standard_deviation,
    sum,
    symmetry_looking,
    ts_complexity_cid_ce,
    unique_value_number_to_time_series_length,
    variance,
    variance_larger_than_standard_deviation,
)

from .interpolate_series import interpolate_nan

__all__ = ["Cluster"]

__version__ = "0.0.1"
__author__ = "Michael Mann"
__license__ = "MIT"
__maintainer__ = "Michael Mann"
__email__ = "mmann1123@gwu.edu"
