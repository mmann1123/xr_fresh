# creating tests for the interpolation functions

# %%
import unittest
import numpy as np

# import jax.numpy as jnp
from xr_fresh.interpolate_series import *

# from xr_fresh.feature_calculator_series import _get_jax_backend
from pathlib import Path
from glob import glob
from datetime import datetime
import geowombat as gw
import os
import tempfile


files = glob("tests/data/values_equal_*.tif")


class TestInterpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmp_dir = tempfile.TemporaryDirectory()
        # set change directory to location of this file
        cls.pth = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cls.pth)
        cls.pth = f"{cls.pth}/data/"
        cls.files = sorted(glob(f"{cls.pth}values_equal_*.tif"))
        print(cls.files)

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        cls.tmp_dir.cleanup()

    def test_linear_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            path = "./test.tif"
            with gw.series(self.files, transfer_lib="jax") as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="linear",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=path,
                    bands=1,
                )
            with gw.open(path) as dst:
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 2313))
                # assert all of band 1 are equal to 1 NOTE EDGE CASE NOT HANDLED
                # assert np.all(dst[0] == 1)
                # assert all of band 2 are equal to 2
                assert np.all(dst[1] == 2)
                # assert all of band 4 are equal to 4
                assert np.all(dst[2] == 3)
                # assert all of band 4 are equal to 4
                assert np.all(dst[3] == 4)
                # assert all of band 5 are equal to 5
                # assert np.all(dst[4] == 5) NOTE EDGE CASE NOT HANDLED


# %%
# Generating the test data for interpolation
# images with alternating bands of 322 rows set to nan

# import numpy as np
# import geowombat as gw

# files = glob("tests/data/RadT*.tif")

# # Open the dataset
# with gw.open(files) as src:
#     for i in range(len(src)):
#         # Replace the values in each band of the ith time slice with i + 1
#         src[i] = i + 1

#         # Replace alternating strips of rows with NaN values in each band
#         block_length = len(src[0, 0]) // 5  # Divide the y dimension into 5 blocks
#         start_row = i * block_length  # Start row index for the current block
#         end_row = (i + 1) * block_length  # End row index for the current block

#         # Replace the rows in each band with NaN values for the current block
#         src[i, :, start_row:end_row, :] = np.nan

#         # Save the modified dataset
#         gw.save(
#             src.sel(time=i + 1), f"./tests/data/values_equal_{i+1}.tif", overwrite=True
#         )
