# This module contains some missing ops from jax
import jax.numpy as np
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from datetime import datetime
import geowombat as gw
import warnings


class interpolate_nan(gw.TimeModule):
    """
    Interpolate missing values in a geospatial time series. Without dates set this class assumes a
    regular time interval between observations. With dates set this class can handle irregular time,
    based on the DOY as an index.

    Args:
        missing_value (int or float, optional): The value to be replaced by NaNs. Default is None.
        interp_type (str, optional): The type of interpolation algorithm to use. Options include "linear",
                                      "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next",
                                      "cubicspline", "spline", and "UnivariateSpline". Default is "linear".
        dates (list[datetime]): List of datetime objects corresponding to each time slice.
        count (int, optional): Overrides the default output band count. Default is 1.

    Methods:
        calculate(array): Applies the interpolation on the input array.

    Example
    -------
    .. code-block:: python

        pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
        files = sorted(glob(f"{pth}*.tif"))[0:10]
        strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
        dates = sorted(datetime.strptime(string, strp_glob) for string in files)
        date_strings = [date.strftime("%Y-%m-%d") for date in dates]

        # window size controls RAM usage, transfer lab can be jax if using GPU
        with gw.series(files, window_size=[640, 640], transfer_lib="numpy") as src:
            src.apply(
                func=interpolate_nan(
                    missing_value=0,
                    count=len(src.filenames),
                    dates=dates,
                ),
                outfile="/home/mmann1123/Downloads/test.tif",
                num_workers=min(12, src.nchunks),
                bands=1,
            )
    """

    def __init__(self, missing_value=None, interp_type="linear", count=1, dates=None):
        super(interpolate_nan, self).__init__()
        # Validate dates is a list of datetime objects
        if dates is None:
            warnings.warn(
                "NOTE: Dates are unknown, assuming regular time interval", UserWarning
            )
            self.dates = dates
        elif not isinstance(dates, list) or not all(
            isinstance(d, datetime) for d in dates
        ):
            raise TypeError("dates must be a list of datetime objects")
        else:
            warnings.warn(
                "NOTE: Dates will be used to index the time series for interpolation",
                UserWarning,
            )

            self.dates = dates
            self.date_indices = np.array(
                [(date - self.dates[0]).days for date in self.dates]
            )

        self.missing_value = missing_value
        self.interp_type = interp_type
        self.count = count

    @staticmethod
    def _interpolate_nans_interp1d(array, kind=None):
        # TO DO: seems to overwrite the first band with bad values
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

    def _interpolate_nans_interp1d_with_dates(array, self):
        # TO DO: seems to overwrite the first band with bad values

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

    @staticmethod
    def _interpolate_nans_linear_with_dates(array, self):
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

    @staticmethod
    def _interpolate_nans_CubicSpline_with_dates(array, self):
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
    def _interpolate_nans_CubicSpline_with_dates(array, self):
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

    @staticmethod
    def _interpolate_nans_UnivariateSpline_with_dates(array, self, s=1):
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
        # check if missing_value is not None and not np.nan
        if self.missing_value is not None:
            if not np.isnan(self.missing_value):
                array = np.where(array == self.missing_value, np.NaN, array)
            if self.interp_type == "linear":
                # check for dates as index
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_linear, axis=0, arr=array
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_linear_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                    )

            elif self.interp_type in [
                "nearest",
                "nearest-up",
                "zero",
                "slinear",
                "quadratic",
                "cubic",
                "previous",
                "next",
            ]:
                # raise TypeError("interp1d not supported - use splines or linear - ")
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_interp1d,
                        axis=0,
                        arr=array,
                        kind=self.interp_type,
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_interp1d_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                        kind=self.interp_type,
                    )
            elif self.interp_type in [
                "cubicspline",
                "spline",
            ]:
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_CubicSpline,
                        axis=0,
                        arr=array,
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_CubicSpline_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                    )
            elif self.interp_type in [
                "UnivariateSpline",
            ]:
                if self.dates is None:
                    array = np.apply_along_axis(
                        self._interpolate_nans_UnivariateSpline, axis=0, arr=array, s=1
                    )
                else:
                    array = np.apply_along_axis(
                        self._interpolate_nans_UnivariateSpline_with_dates,
                        axis=0,
                        arr=array,
                        self=self,
                        s=1,
                    )
        # Return the interpolated array (3d -> time/bands x height x width)
        # If the array is (time x 1 x height x width) then squeeze to 3d
        return array.squeeze()
