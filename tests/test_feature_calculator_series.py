import unittest
import numpy as np

# import jax.numpy as jnp
from xr_fresh.feature_calculator_series import *

from pathlib import Path
from glob import glob
from datetime import datetime
import geowombat as gw
import warnings
import os
import jax
import tempfile


class TestFeatureCalculators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmp_dir = tempfile.TemporaryDirectory()
        # set change directory to location of this file
        cls.pth = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cls.pth)
        cls.pth = f"{cls.pth}/data/"
        cls.files = sorted(glob(f"{cls.pth}RadT_tavg_*.tif"))
        print(cls.files)
        cls.strp_glob = f"{cls.pth}RadT_tavg_%Y%m.tif"
        cls.dates = sorted(
            datetime.strptime(
                os.path.basename(string).split("_")[2].split(".")[0], "%Y%m"
            )
            for string in cls.files
        )
        cls.date_strings = [date.strftime("%Y-%m-%d") for date in cls.dates]

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary directory
        cls.tmp_dir.cleanup()

    def test_abs_energy(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(abs_energy(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_absolute_sum_of_changes(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    absolute_sum_of_changes(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_autocorrelation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(autocorrelation(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_count_above_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(count_above_mean(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_count_below_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(count_below_mean(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_doy_of_maximum(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    doy_of_maximum(dates=self.dates),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_doy_of_minimum(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    doy_of_minimum(dates=self.dates),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_kurtosis(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(kurtosis(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_kurtosis_excess(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(kurtosis_excess(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                print("dst", dst.shape)
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_large_standard_deviation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    large_standard_deviation(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_longest_strike_above_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    longest_strike_above_mean(),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_longest_strike_above_set_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    longest_strike_above_mean(mean=8),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_longest_strike_below_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    longest_strike_below_mean(),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_maximum(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(maximum(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_minimum(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(minimum(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(mean(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_mean_abs_change(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(mean_abs_change(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_mean_change(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(mean_change(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_mean_second_derivative_central(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    mean_second_derivative_central(),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_median(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(median(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_ols_slope_intercept(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    ols_slope_intercept(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_quantile(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(quantile(q=0.05), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_ratio_beyond_r_sigma(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    ratio_beyond_r_sigma(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_skewness(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(skewness(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_standard_deviation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    standard_deviation(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_sum(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(sum(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_symmetry_looking(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(symmetry_looking(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_ts_complexity_cid_ce(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    ts_complexity_cid_ce(), bands=1, num_workers=2, outfile=out_path
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    # Error=4 "Infinite loop run"
    # def test_unique_value_number_to_time_series_length(self):
    #     with self.tmp_dir as tmp:
    #         if not os.path.exists(tmp):
    #             os.mkdir(tmp)
    #         out_path = Path(tmp) / "test.tif"
    #         # use rasterio to create a new file tif file

    #         with gw.series(self.files) as src:
    #             src.apply(unique_value_number_to_time_series_length(), bands=1, num_workers=2, outfile=out_path)
    #         with gw.open(out_path) as dst:
    #             self.assertEqual(dst.gw.nbands, 1)
    #             self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_variance(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(variance(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

    def test_variance_larger_than_standard_deviation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(
                    variance_larger_than_standard_deviation(),
                    bands=1,
                    num_workers=2,
                    outfile=out_path,
                )
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))


if __name__ == "__main__":
    unittest.main()
