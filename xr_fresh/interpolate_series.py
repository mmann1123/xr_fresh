
import numpy as np
from scipy.interpolate import interp1d, CubicSpline, UnivariateSpline
from datetime import datetime
import geowombat as gw

class interpolate_nan(gw.TimeModule):

    def __init__(self, missing_value=None, interp_type="linear", count=1, dates=None):
        super(interpolate_nan, self).__init__()
        if dates is None:
            print("NOTE: Dates are unknown, assuming regular time interval")
            self.dates = dates
        elif not isinstance(dates, list) or not all(isinstance(d, datetime) for d in dates):
            raise TypeError("dates must be a list of datetime objects")
        else:
            print("NOTE: Dates will be used to index the time series for interpolation")
            self.dates = dates
            self.date_indices = np.array([(date - self.dates[0]).days for date in self.dates])
        
        self.missing_value = missing_value
        self.interp_type = interp_type
        self.count = count

    @staticmethod
    def _interpolate_nans_interp1d(array, kind=None):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = interp1d(
                x=valid_indices,
                y=valid_values,
                kind=kind,
                bounds_error=False,
                fill_value="extrapolate",
            )
            return inter_fun(np.arange(len(array)))

    def _interpolate_nans_interp1d_with_dates(self, array):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]
            inter_fun = interp1d(
                x=valid_dates,
                y=valid_values,
                kind=self.interp_type,
                fill_value="extrapolate",
            )
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_linear(array):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            return np.interp(
                np.arange(len(array)),
                np.arange(len(array))[np.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    def _interpolate_nans_linear_with_dates(self, array):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            return np.interp(
                self.date_indices,
                self.date_indices[np.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    @staticmethod
    def _interpolate_nans_CubicSpline(array):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = CubicSpline(
                x=valid_indices, y=valid_values, bc_type="not-a-knot"
            )
            return inter_fun(np.arange(len(array)))

    def _interpolate_nans_CubicSpline_with_dates(self, array):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_dates = self.date_indices[valid_indices]
            valid_values = array[valid_indices]

            inter_fun = CubicSpline(x=valid_dates, y=valid_values, bc_type="not-a-knot")
            return inter_fun(self.date_indices)

    @staticmethod
    def _interpolate_nans_UnivariateSpline(array, s=1):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_values = array[valid_indices]
            inter_fun = UnivariateSpline(x=valid_indices, y=valid_values, s=s)
            return inter_fun(np.arange(len(array)))

    def _interpolate_nans_UnivariateSpline_with_dates(self, array, s=1):
        nan_check = np.isnan(array)
        if all(nan_check):
            return array
        elif all(~nan_check):
            return array
        else:
            valid_indices = np.where(np.isnan(array) == False)[0]
            valid_date_indices = self.date_indices[valid_indices]
            valid_values = array[valid_indices]
            inter_fun = UnivariateSpline(x=valid_date_indices, y=valid_values, s=s)
            return inter_fun(self.date_indices)

    def calculate(self, array):
        if self.missing_value is not None:
            if not np.isnan(self.missing_value):
                array = np.where(array == self.missing_value, np.NaN, array)
        
        if self.interp_type == "linear":
            if self.dates is None:
                array = np.apply_along_axis(self._interpolate_nans_linear, axis=0, arr=array)
            else:
                array = np.apply_along_axis(self._interpolate_nans_linear_with_dates, axis=0, arr=array, self=self)
        
        elif self.interp_type in ["nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next"]:
            raise TypeError("interp1d not supported - use splines or linear - ")
            if self.dates is None:
                array = np.apply_along_axis(self._interpolate_nans_interp1d, axis=0, arr=array, kind=self.interp_type)
            else:
                array = np.apply_along_axis(self._interpolate_nans_interp1d_with_dates, axis=0, arr=array, self=self, kind=self.interp_type)

        elif self.interp_type in ["cubicspline", "spline"]:
            if self.dates is None:
                array = np.apply_along_axis(self._interpolate_nans_CubicSpline, axis=0, arr=array)
            else:
                array = np.apply_along_axis(self._interpolate_nans_CubicSpline_with_dates, axis=0, arr=array, self=self)

        elif self.interp_type == "UnivariateSpline":
            if self.dates is None:
                array = np.apply_along_axis(self._interpolate_nans_UnivariateSpline, axis=0, arr=array, s=1)
            else:
                array = np.apply_along_axis(self._interpolate_nans_UnivariateSpline_with_dates, axis=0, arr=array, self=self, s=1)

        return array.squeeze()
