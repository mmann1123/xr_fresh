# %%
# read in RadT and create a missing date
import geowombat as gw
import numpy as np

with gw.open(
    "/home/mmann1123/Documents/github/xr_fresh/tests/data/RadT_tavg_202212.tif",
    nodata=np.nan,
) as src:
    # print(np.nanmean(src.values))
    src = src.where(src > 297)
    src.gw.save(
        "/home/mmann1123/Documents/github/xr_fresh/tests/data/RadT_tavg_202203.tif",
        overwrite=True,
    )

# %%
np.nanmean
