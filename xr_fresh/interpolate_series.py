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

    Example Usage:

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


# from datetime import datetime, timedelta


# def apply_interpolation(input_file, dates=None, interp_type="linear"):
#     """
#     Apply interpolation to a geospatial time series.

#     Args:
#         input_file (str): The input file path.
#         dates (list[datetime], optional): List of datetime objects corresponding to each time slice.
#         interp_type (str, optional): The type of interpolation algorithm to use. Options include "linear",
#                                      "nearest", "zero", "slinear", "quadratic", "cubic", "previous", "next",
#                                      "cubicspline", "spline", and "UnivariateSpline". Default is "linear".

#     Example Usage:
#             apply_interpolation(['input-day1.tif','input-day2.tif'], interp_type='linear')
#     """

#     # Open the input file using geowombat
#     with gw.open(input_file) as src:
#         data = src.data
#         if dates is None:
#             # assume dates are consecutive days if none are provided
#             dates = [datetime.now() + timedelta(days=i) for i in range(src.shape[0])]

#     # Instantiate the interpolation class
#     interpolator = interpolate_nan(interp_type=interp_type, dates=dates)

#     # Apply the interpolation
#     interpolated_data = interpolator.calculate(data)

#     # Export individual files for each time period
#     for i, date in enumerate(dates):
#         output_filename = f"{input_file.stem}_{interp_type}_{date.strftime('%Y%m%d')}{input_file.suffix}"
#         with gw.open(output_filename, "w", **src.meta) as dst:
#             dst.write(interpolated_data[i, :, :], 1)


# __all__ = ["interp"]


# [docs] @ functools.partial(vmap, in_axes=(0, None, None))
# def interp(x, xp, fp):
#     """
#     Simple equivalent of np.interp that compute a linear interpolation.

#     We are not doing any checks, so make sure your query points are lying
#     inside the array.

#     TODO: Implement proper interpolation!

#     x, xp, fp need to be 1d arrays
#     """
#     # First we find the nearest neighbour
#     ind = np.argmin((x - xp) ** 2)

#     # Perform linear interpolation
#     ind = np.clip(ind, 1, len(xp) - 2)

#     xi = xp[ind]
#     # Figure out if we are on the right or the left of nearest
#     s = np.sign(np.clip(x, xp[1], xp[-2]) - xi).astype(np.int64)
#     a = (fp[ind + np.copysign(1, s).astype(np.int64)] - fp[ind]) / (
#         xp[ind + np.copysign(1, s).astype(np.int64)] - xp[ind]
#     )
#     b = fp[ind] - a * xp[ind]
#     return a * x + b


# @register_pytree_node_class
# class InterpolatedUnivariateSpline(object):
#     def __init__(self, x, y, k=3, endpoints="not-a-knot", coefficients=None):
#         """JAX implementation of kth-order spline interpolation.

#         This class aims to reproduce scipy's InterpolatedUnivariateSpline
#         functionality using JAX. Not all of the original class's features
#         have been implemented yet, notably
#         - `w`    : no weights are used in the spline fitting.
#         - `bbox` : we assume the boundary to always be [x[0], x[-1]].
#         - `ext`  : extrapolation is always active, i.e., `ext` = 0.
#         - `k`    : orders `k` > 3 are not available.
#         - `check_finite` : no such check is performed.

#         (The relevant lines from the original docstring have been included
#         in the following.)

#         Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
#         Spline function passes through all provided points. Equivalent to
#         `UnivariateSpline` with s = 0.

#         Parameters
#         ----------
#         x : (N,) array_like
#             Input dimension of data points -- must be strictly increasing
#         y : (N,) array_like
#             input dimension of data points
#         k : int, optional
#             Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
#         endpoints : str, optional, one of {'natural', 'not-a-knot'}
#             Endpoint condition for cubic splines, i.e., `k` = 3.
#             'natural' endpoints enforce a vanishing second derivative
#             of the spline at the two endpoints, while 'not-a-knot'
#             ensures that the third derivatives are equal for the two
#             left-most `x` of the domain, as well as for the two
#             right-most `x`. The original scipy implementation uses
#             'not-a-knot'.
#         coefficients: list, optional
#             Precomputed parameters for spline interpolation. Shouldn't be set
#             manually.

#         See Also
#         --------
#         UnivariateSpline : Superclass -- allows knots to be selected by a
#             smoothing condition
#         LSQUnivariateSpline : spline for which knots are user-selected
#         splrep : An older, non object-oriented wrapping of FITPACK
#         splev, sproot, splint, spalde
#         BivariateSpline : A similar class for two-dimensional spline interpolation

#         Notes
#         -----
#         The number of data points must be larger than the spline degree `k`.

#         The general form of the spline can be written as
#           f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
#           i = 0, ..., n-1,
#         where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

#         The unknown coefficients (a, b, c, d) define a symmetric, diagonal
#         linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
#         and z = c for `k` = 3. In each case, the coefficients defining each
#         spline piece can be expressed in terms of only z[i], z[i+1],
#         y[i], and y[i+1]. The coefficients are solved for using
#         `np.linalg.solve` when `k` = 2 and `k` = 3.

#         """
#         # Verify inputs
#         k = int(k)
#         assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
#         x = np.atleast_1d(x)
#         y = np.atleast_1d(y)
#         assert len(x) == len(y), "Input arrays must be the same length."
#         assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
#         n_data = len(x)

#         # Difference vectors
#         h = np.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
#         p = np.diff(y)  # y[i+1] - y[i]

#         if coefficients is None:
#             # Build the linear system of equations depending on k
#             # (No matrix necessary for k=1)
#             if k == 1:
#                 assert n_data > 1, "Not enough input points for linear spline."
#                 coefficients = p / h

#             if k == 2:
#                 assert n_data > 2, "Not enough input points for quadratic spline."
#                 assert endpoints == "not-a-knot"  # I have only validated this
#                 # And actually I think it's probably the best choice of border condition

#                 # The knots are actually in between data points
#                 knots = (x[1:] + x[:-1]) / 2.0
#                 # We add 2 artificial knots before and after
#                 knots = np.concatenate(
#                     [
#                         np.array([x[0] - (x[1] - x[0]) / 2.0]),
#                         knots,
#                         np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
#                     ]
#                 )
#                 n = len(knots)
#                 # Compute interval lenghts for these new knots
#                 h = np.diff(knots)
#                 # postition of data point inside the interval
#                 dt = x - knots[:-1]

#                 # Now we build the system natrix
#                 A = np.diag(
#                     np.concatenate(
#                         [
#                             np.ones(1),
#                             (
#                                 2 * dt[1:]
#                                 - dt[1:] ** 2 / h[1:]
#                                 - dt[:-1] ** 2 / h[:-1]
#                                 + h[:-1]
#                             ),
#                             np.ones(1),
#                         ]
#                     )
#                 )

#                 A += np.diag(
#                     np.concatenate([-np.array([1 + h[0] / h[1]]), dt[1:] ** 2 / h[1:]]),
#                     k=1,
#                 )
#                 A += np.diag(
#                     np.concatenate([np.atleast_1d(h[0] / h[1]), np.zeros(n - 3)]), k=2
#                 )

#                 A += np.diag(
#                     np.concatenate(
#                         [
#                             h[:-1] - 2 * dt[:-1] + dt[:-1] ** 2 / h[:-1],
#                             -np.array([1 + h[-1] / h[-2]]),
#                         ]
#                     ),
#                     k=-1,
#                 )
#                 A += np.diag(
#                     np.concatenate([np.zeros(n - 3), np.atleast_1d(h[-1] / h[-2])]),
#                     k=-2,
#                 )

#                 # And now we build the RHS vector
#                 s = np.concatenate([np.zeros(1), 2 * p, np.zeros(1)])

#                 # Compute spline coefficients by solving the system
#                 coefficients = np.linalg.solve(A, s)

#             if k == 3:
#                 assert n_data > 3, "Not enough input points for cubic spline."
#                 if endpoints not in ("natural", "not-a-knot"):
#                     print("Warning : endpoints not recognized. Using natural.")
#                     endpoints = "natural"

#                 # Special values for the first and last equations
#                 zero = array([0.0])
#                 one = array([1.0])
#                 A00 = one if endpoints == "natural" else array([h[1]])
#                 A01 = zero if endpoints == "natural" else array([-(h[0] + h[1])])
#                 A02 = zero if endpoints == "natural" else array([h[0]])
#                 ANN = one if endpoints == "natural" else array([h[-2]])
#                 AN1 = (
#                     -one if endpoints == "natural" else array([-(h[-2] + h[-1])])
#                 )  # A[N, N-1]
#                 AN2 = zero if endpoints == "natural" else array([h[-1]])  # A[N, N-2]

#                 # Construct the tri-diagonal matrix A
#                 A = np.diag(concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
#                 upper_diag1 = np.diag(concatenate((A01, h[1:])), k=1)
#                 upper_diag2 = np.diag(concatenate((A02, zeros(n_data - 3))), k=2)
#                 lower_diag1 = np.diag(concatenate((h[:-1], AN1)), k=-1)
#                 lower_diag2 = np.diag(concatenate((zeros(n_data - 3), AN2)), k=-2)
#                 A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

#                 # Construct RHS vector s
#                 center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
#                 s = concatenate((zero, center, zero))
#                 # Compute spline coefficients by solving the system
#                 coefficients = np.linalg.solve(A, s)

#         # Saving spline parameters for evaluation later
#         self.k = k
#         self._x = x
#         self._y = y
#         self._coefficients = coefficients

#     # Operations for flattening/unflattening representation
#     def tree_flatten(self):
#         children = (self._x, self._y, self._coefficients)
#         aux_data = {"endpoints": self._endpoints, "k": self.k}
#         return (children, aux_data)

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         x, y, coefficients = children
#         return cls(x, y, coefficients=coefficients, **aux_data)

#     def __call__(self, x):
#         """Evaluation of the spline.

#         Notes
#         -----
#         Values are extrapolated if x is outside of the original domain
#         of knots. If x is less than the left-most knot, the spline piece
#         f[0] is used for the evaluation; similarly for x beyond the
#         right-most point.

#         """
#         if self.k == 1:
#             t, a, b = self._compute_coeffs(x)
#             result = a + b * t

#         if self.k == 2:
#             t, a, b, c = self._compute_coeffs(x)
#             result = a + b * t + c * t**2

#         if self.k == 3:
#             t, a, b, c, d = self._compute_coeffs(x)
#             result = a + b * t + c * t**2 + d * t**3

#         return result

#     def _compute_coeffs(self, xs):
#         """Compute the spline coefficients for a given x."""
#         # Retrieve parameters
#         x, y, coefficients = self._x, self._y, self._coefficients

#         # In case of quadratic, we redefine the knots
#         if self.k == 2:
#             knots = (x[1:] + x[:-1]) / 2.0
#             # We add 2 artificial knots before and after
#             knots = np.concatenate(
#                 [
#                     np.array([x[0] - (x[1] - x[0]) / 2.0]),
#                     knots,
#                     np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
#                 ]
#             )
#         else:
#             knots = x

#         # Determine the interval that x lies in
#         ind = np.digitize(xs, knots) - 1
#         # Include the right endpoint in spline piece C[m-1]
#         ind = np.clip(ind, 0, len(knots) - 2)
#         t = xs - knots[ind]
#         h = np.diff(knots)[ind]

#         if self.k == 1:
#             a = y[ind]
#             result = (t, a, coefficients[ind])

#         if self.k == 2:
#             dt = (x - knots[:-1])[ind]
#             b = coefficients[ind]
#             b1 = coefficients[ind + 1]
#             a = y[ind] - b * dt - (b1 - b) * dt**2 / (2 * h)
#             c = (b1 - b) / (2 * h)
#             result = (t, a, b, c)

#         if self.k == 3:
#             c = coefficients[ind]
#             c1 = coefficients[ind + 1]
#             a = y[ind]
#             a1 = y[ind + 1]
#             b = (a1 - a) / h - (2 * c + c1) * h / 3.0
#             d = (c1 - c) / (3 * h)
#             result = (t, a, b, c, d)

#         return result

#     def derivative(self, x, n=1):
#         """Analytic nth derivative of the spline.

#         The spline has derivatives up to its order k.

#         """
#         assert n in range(self.k + 1), "Invalid n."

#         if n == 0:
#             result = self.__call__(x)
#         else:
#             # Linear
#             if self.k == 1:
#                 t, a, b = self._compute_coeffs(x)
#                 result = b

#             # Quadratic
#             if self.k == 2:
#                 t, a, b, c = self._compute_coeffs(x)
#                 if n == 1:
#                     result = b + 2 * c * t
#                 if n == 2:
#                     result = 2 * c

#             # Cubic
#             if self.k == 3:
#                 t, a, b, c, d = self._compute_coeffs(x)
#                 if n == 1:
#                     result = b + 2 * c * t + 3 * d * t**2
#                 if n == 2:
#                     result = 2 * c + 6 * d * t
#                 if n == 3:
#                     result = 6 * d

#         return result

#     def antiderivative(self, xs):
#         """
#         Computes the antiderivative of first order of this spline
#         """
#         # Retrieve parameters
#         x, y, coefficients = self._x, self._y, self._coefficients

#         # In case of quadratic, we redefine the knots
#         if self.k == 2:
#             knots = (x[1:] + x[:-1]) / 2.0
#             # We add 2 artificial knots before and after
#             knots = np.concatenate(
#                 [
#                     np.array([x[0] - (x[1] - x[0]) / 2.0]),
#                     knots,
#                     np.array([x[-1] + (x[-1] - x[-2]) / 2.0]),
#                 ]
#             )
#         else:
#             knots = x

#         # Determine the interval that x lies in
#         ind = np.digitize(xs, knots) - 1
#         # Include the right endpoint in spline piece C[m-1]
#         ind = np.clip(ind, 0, len(knots) - 2)
#         t = xs - knots[ind]

#         if self.k == 1:
#             a = y[:-1]
#             b = coefficients
#             h = np.diff(knots)
#             cst = np.concatenate([np.zeros(1), np.cumsum(a * h + b * h**2 / 2)])
#             return cst[ind] + a[ind] * t + b[ind] * t**2 / 2

#         if self.k == 2:
#             h = np.diff(knots)
#             dt = x - knots[:-1]
#             b = coefficients[:-1]
#             b1 = coefficients[1:]
#             a = y - b * dt - (b1 - b) * dt**2 / (2 * h)
#             c = (b1 - b) / (2 * h)
#             cst = np.concatenate(
#                 [np.zeros(1), np.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3)]
#             )
#             return cst[ind] + a[ind] * t + b[ind] * t**2 / 2 + c[ind] * t**3 / 3

#         if self.k == 3:
#             h = np.diff(knots)
#             c = coefficients[:-1]
#             c1 = coefficients[1:]
#             a = y[:-1]
#             a1 = y[1:]
#             b = (a1 - a) / h - (2 * c + c1) * h / 3.0
#             d = (c1 - c) / (3 * h)
#             cst = np.concatenate(
#                 [
#                     np.zeros(1),
#                     np.cumsum(a * h + b * h**2 / 2 + c * h**3 / 3 + d * h**4 / 4),
#                 ]
#             )
#             return (
#                 cst[ind]
#                 + a[ind] * t
#                 + b[ind] * t**2 / 2
#                 + c[ind] * t**3 / 3
#                 + d[ind] * t**4 / 4
#             )

#     def integral(self, a, b):
#         """
#         Compute a definite integral over a piecewise polynomial.
#         Parameters
#         ----------
#         a : float
#             Lower integration bound
#         b : float
#             Upper integration bound
#         Returns
#         -------
#         ig : array_like
#             Definite integral of the piecewise polynomial over [a, b]
#         """
#         # Swap integration bounds if needed
#         sign = 1
#         if b < a:
#             a, b = b, a
#             sign = -1
#         xs = np.array([a, b])
#         return sign * np.diff(self.antiderivative(xs))
