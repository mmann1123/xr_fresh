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
        func=abs_energy(),
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
            func=doy_of_maximum(dates),
            outfile=f"/home/mmann1123/Downloads/test.tif",
            num_workers=1,
            bands=1,
        )


# %% for missing values
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
