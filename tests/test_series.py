# %%
# create temp folder
import tempfile
import os
from glob import glob
from datetime import datetime
import unittest
import geowombat as gw
from pathlib import Path
import unittest
from xr_fresh.feature_calculator_series import abs_energy, maximum
import jax

# %% check for cpu or gpu
# Set JAX to use the determined backend
# jax_backend = _get_jax_backend()
# jax.config.update("jax_platform_name", jax_backend)


# set change directory to location of this file
pth = os.path.dirname(os.path.abspath(__file__))
os.chdir(pth)
pth = f"{pth}/data/"
files = glob(f"{pth}*.tif")
print(files)
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
date_strings = [date.strftime("%Y-%m-%d") for date in dates]


class TestSeries(unittest.TestCase):
    def test_series_abs_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(files) as src:
                src.apply(abs_energy(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_series_maximum(self):
        with tempfile.TemporaryDirectory() as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(files) as src:
                src.apply(maximum(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))


if __name__ == "__main__":
    unittest.main()
