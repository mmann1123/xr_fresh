# %%
# create temp folder
import tempfile
import os
from glob import glob
from datetime import datetime
import unittest
import geowombat as gw
from pathlib import Path
from xr_fresh.feature_calculator_series import (
    interpolate_nan_dates,
    interpolate_nan,
    doy_of_maximum,
    abs_energy,
    abs_energy2,
    autocorrelation,
    doy_of_maximum,
)

# 202303 is all nan's

# Create a temporary directory

# with tempfile.TemporaryDirectory() as temp_dir:
#     print(f"Created temporary directory: {temp_dir}")
# pth = r"../tests/data/"
files = glob(f"{pth}*.tif")
print(files)
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]
date_strings
# %%


# %%


class TestSeries(unittest.TestCase):
    def test_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "test.tif"
        with gw.series(files) as src:
            src.apply(abs_energy(), bands=1, num_workers=2, outfile=out_path)
        with gw.open(out_path) as dst:
            self.assertEqual(dst.gw.nbands, 1)
            self.assertEqual(dst.shape, (1, 1613, 2313))


if __name__ == "__main__":
    unittest.main()
