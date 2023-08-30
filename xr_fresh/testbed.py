# %% env:testbed laptop

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
import jax.numpy as jnp
from datetime import datetime

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

files = sorted(glob(f"{pth}*.tif"))[0:10]
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]
date_strings

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


# %% Simiple without arguments

with gw.series(
    files,
    nodata=9999,
) as src:
    print(src)
    src.apply(
        func=interpolate_nan(missing_value=0),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )

# %%
import numpy as np

xp = [1, 2, 3]
y = [3, np.nan, 0, -2, -3]
np.interp(
    np.arange(len(y)), np.arange(len(y))[np.isnan(y) == False], y[np.isnan(y) == False]
)

# %%
x = np.array([1, np.nan, np.nan, 2, 2, np.nan, 3])
np.interp(
    np.arange(len(x)), np.arange(len(x))[np.isnan(x) == False], x[np.isnan(x) == False]
)
# %%
