# %%
import geowombat as gw
import matplotlib.pyplot as plt
from pathlib import Path
import os
from glob import glob

# Define the directory containing raster files
data_dir = os.chdir("../tests/data")

files = sorted(glob("values_equal_small*.tif"))
# Print file paths for debugging
print("Raster files:", files)

with gw.open(files) as ds:
    ds.plot(col="time", col_wrap=4, cmap="viridis", robust=True)

# %%
from xr_fresh.interpolate_series import interpolate_nan
import numpy as np
import tempfile

temp_dir = tempfile.mkdtemp()

# Output path for the interpolated raster
output_file = os.path.join(temp_dir, "interpolated_time_series.tif")

# Apply interpolation
with gw.series(files, transfer_lib="jax", window_size=[256 * 2, 256 * 2]) as src:
    src.apply(
        func=interpolate_nan(
            interp_type="slinear",  # Interpolation type
            missing_value=np.nan,  # Value representing missing data
            count=len(src.filenames),  # Number of time steps
        ),
        outfile=output_file,
        bands=1,  # Apply interpolation to the first band
    )

print("Interpolation completed. Output saved to:", output_file)

# Define the directory containing raster files
interpolated_file = Path(temp_dir, "interpolated_time_series.tif")

with gw.open(interpolated_file) as ds:
    # display(ds)
    ds.plot(col="band", col_wrap=4, cmap="viridis", robust=True)
# %%
