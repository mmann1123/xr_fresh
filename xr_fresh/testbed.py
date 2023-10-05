# %% env:testbed laptop

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
import jax.numpy as jnp
from datetime import datetime
from geowombat.data import l8_224078_20200518
from scipy.interpolate import interp1d, CubicSpline


sys.path.append("/home/mmann1123/Documents/github/xr_fresh/xr_fresh")
sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
import xr_fresh as xf

from xr_fresh.feature_calculator_series import (
    interpolate_nan,
    doy_of_maximum,
    abs_energy,
    abs_energy2,
)

pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"
pth = "/home/mmann1123/extra_space/Dropbox/Africa_data/Temperature/"

files = sorted(glob(f"{pth}*.tif"))[0:3]
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]
date_strings


# %% Simiple without arguments

with gw.series(
    files,
    nodata=9999,
) as src:
    src.apply(
        func=interpolate_nan(missing_value=0),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )


# %% with date argument

with gw.series(
    files,
    nodata=9999,
) as src:
    for i, name in enumerate(date_strings):
        src.apply(
            func=interpolate_nan(
                missing_value=0, interp_type="linear", index_to_write=i
            ),
            outfile=f"/home/mmann1123/Downloads/Temperature_{name}.tif",
            num_workers=5,
            bands=1,
        )


# %%


# %%
class interpolate_nan(gw.TimeModule):
    def __init__(self, missing_value=None, interp_type="linear", count=1):
        super(interpolate_nan, self).__init__()
        self.missing_value = missing_value
        self.interp_type = interp_type
        # Overrides the default output band count
        self.count = count

    @staticmethod
    def _interpolate_nans_linear(array):
        if all(np.isnan(array)):
            return array
        else:
            return np.interp(
                np.arange(len(array)),
                np.arange(len(array))[jnp.isnan(array) == False],
                array[np.isnan(array) == False],
            )

    @staticmethod
    def _interpolate_nans_interp1d(array, kind=None):
        if all(np.isnan(array)):
            return array
        else:
            inter_fun = interp1d(x=np.arange(len(array)), y=array, kind=kind)
            return inter_fun(np.arange(len(array)))

    @staticmethod
    def _interpolate_nans_CubicSpline(array, kind=None):
        if all(np.isnan(array)):
            return array
        else:
            inter_fun = CubicSpline(
                x=np.arange(len(array)), y=array, bc_type="not-a-knot"
            )
            return inter_fun(np.arange(len(array)))

    def calculate(self, array):
        # check if missing_value is not None and not np.nan
        if self.missing_value is not None:
            if not np.isnan(self.missing_value):
                array = jnp.where(array == self.missing_value, np.NaN, array)
            if self.interp_type == "linear":
                array = np.apply_along_axis(
                    self._interpolate_nans_linear, axis=0, arr=array
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
                array = np.apply_along_axis(
                    self._interpolate_nans_interp1d,
                    axis=0,
                    arr=array,
                    kind=self.interp_type,
                )
            elif self.interp_type in [
                "cubicspline",
                "spline",
            ]:
                array = np.apply_along_axis(
                    self._interpolate_nans_CubicSpline,
                    axis=0,
                    arr=array,
                    kind=self.interp_type,
                )
        # Return the interpolated array (3d -> time/bands x height x width)
        # If the array is (time x 1 x height x width) then squeeze to 3d
        return array.squeeze()


with gw.series(files) as src:
    src.apply(
        func=interpolate_nan(
            missing_value=9999,
            interp_type="spline",
            # output band count
            count=len(src.filenames),
        ),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=10,
        # number of bands to read
        bands=1,
    )


# %%
# not sure this works
# print("filled", filled)
# # check if out directory exists and create if not
# if outdir is None:
#     outdir = os.path.dirname("smoothed_values")
# if not os.path.exists(outdir):
#     os.makedirs(outdir)

# # write files
# filled.attrs = src.attrs.copy()
# file_names = src.attrs["filename"]
# file_names = concat_file_name(file_names, text=name_append)

# for i, file in enumerate(file_names):
#     gw.to_raster(
#         filled.sel(time=i),
#         f"{outdir}/{file}",
#         nodata=missing_value,
#     )


# interpolate_missing(
#     files,
#     missing_value=9999,
#     method="linear",
#     limit=3,
#     outdir="/home/mmann1123/Downloads/smoothed_values",
#     name_append="smoothed",
# )
# %%


# %%
