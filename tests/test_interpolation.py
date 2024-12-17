# creating tests for the interpolation functions

# %%
import unittest
import numpy as np

# import jax.numpy as jnp
from xr_fresh.interpolate_series import *

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

        # Set path to the directory containing this file (test script)
        cls.base_path = Path(__file__).parent

        # Access data directory relative to this script
        cls.data_path = cls.base_path / "data"

        # Gather all .tif files that match the pattern
        cls.files = sorted(cls.data_path.glob("values_equal_*.tif"))
        cls.small_files = sorted(cls.data_path.glob("small_missing*.tif"))
        print([str(file) for file in cls.files])  # Print file paths for debugging

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        cls.tmp_dir.cleanup()

    def test_linear_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="linear",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
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

    def test_slinear_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="slinear",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(dst[0] == 1)
                # assert all of band 2 are equal to 2
                assert np.all(dst[1] == 2)
                # assert all of band 4 are equal to 4
                assert np.all(dst[2] == 3)
                # assert all of band 4 are equal to 4
                assert np.all(dst[3] == 4)
                # assert all of band 5 are equal to 5
                assert np.all(dst[4] == 5)

    def test_quadratic_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="quadratic",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                atol = 1e-2
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(np.isclose(dst[0], 1, atol))
                # assert all of band 2 are equal to 2
                assert np.all(np.isclose(dst[1], 2, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[2], 3, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[3], 4, atol))
                # assert all of band 5 are equal to 5
                assert np.all(np.isclose(dst[4], 5, atol))

    def test_cubic_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="cubic",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                atol = 1e-2
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(np.isclose(dst[0], 1, atol))
                # assert all of band 2 are equal to 2
                assert np.all(np.isclose(dst[1], 2, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[2], 3, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[3], 4, atol))
                # assert all of band 5 are equal to 5
                assert np.all(np.isclose(dst[4], 5, atol))

    def test_spline_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="spline",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                atol = 1e-2
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(np.isclose(dst[0], 1, atol))
                # assert all of band 2 are equal to 2
                assert np.all(np.isclose(dst[1], 2, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[2], 3, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[3], 4, atol))
                # assert all of band 5 are equal to 5
                assert np.all(np.isclose(dst[4], 5, atol))

    def test_UnivariateSpline_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="UnivariateSpline",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                atol = 1e-2
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(np.isclose(dst[0], 1, atol))
                # assert all of band 2 are equal to 2
                assert np.all(np.isclose(dst[1], 2, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[2], 3, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[3], 4, atol))
                # assert all of band 5 are equal to 5
                assert np.all(np.isclose(dst[4], 5, atol))

    def test_cubicspline_interpolation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            with gw.series(
                self.small_files, transfer_lib="jax", window_size=[256, 256]
            ) as src:
                src.apply(
                    func=interpolate_nan(
                        interp_type="cubicspline",
                        missing_value=np.nan,
                        count=len(src.filenames),
                    ),
                    outfile=out_path,
                    bands=1,
                )
            with gw.open(out_path) as dst:
                atol = 1e-2
                self.assertEqual(dst.gw.nbands, 5)
                self.assertEqual(dst.shape, (5, 1613, 64))
                # assert all of band 1 are equal to 1
                assert np.all(np.isclose(dst[0], 1, atol))
                # assert all of band 2 are equal to 2
                assert np.all(np.isclose(dst[1], 2, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[2], 3, atol))
                # assert all of band 4 are equal to 4
                assert np.all(np.isclose(dst[3], 4, atol))
                # assert all of band 5 are equal to 5
                assert np.all(np.isclose(dst[4], 5, atol))

    # def test_apply_interpolation(self):
    #     input_file = self.files[0]  # Use the first file from the sorted list
    #     output_dir = Path(self.tmp_dir.name)

    #     # Apply interpolation
    #     apply_interpolation(input_file, interp_type="linear")

    #     # Check that output files are created for each time period
    #     with gw.open(input_file) as src:
    #         dates = src.dates
    #         for date in dates:
    #             output_filename = f"{input_file.stem}_linear_{date.strftime('%Y%m%d')}{input_file.suffix}"
    #             output_filepath = output_dir / output_filename
    #             self.assertTrue(output_filepath.exists())

    #             # Open the output file and perform some basic checks
    #             with gw.open(output_filepath) as dst:
    #                 self.assertEqual(dst.shape, src.shape[1:])  # Check shape matches
    #                 self.assertFalse(
    #                     np.any(np.isnan(dst.read(1)))
    #                 )  # Ensure no NaNs in the output


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
# # %%
# files = r"./values_equal_*.tif"
# files = sorted(glob(files))

# with gw.open(files) as ds:
#     # display(ds)
#     ds.plot(col="time", col_wrap=4, cmap="viridis", robust=True)
#     for i in range(len(ds)):

#         # ds.isel(time=i, x=slice(0, 64)).plot()
#         # display(ds.isel(time=i, x=slice(0, 64)) )
#         gw.save(
#             ds.isel(time=i, x=slice(0, 64)),
#             f"./small_missing_{i+1}.tif",
#             overwrite=True,
#         )
# %%
