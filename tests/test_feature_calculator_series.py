import unittest
import numpy as np
#import jax.numpy as jnp
from xr_fresh.feature_calculator_series import *
#from xr_fresh.feature_calculator_series import _get_jax_backend
from pathlib import Path
from glob import glob
from datetime import datetime
import geowombat as gw
import warnings
import os
import jax
import tempfile


# jax_backend = _get_jax_backend()
# jax.config.update("jax_platform_name", jax_backend)


class TestFeatureCalculators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary directory
        cls.tmp_dir = tempfile.TemporaryDirectory()
        # set change directory to location of this file
        cls.pth = os.path.dirname(os.path.abspath(__file__))
        os.chdir(cls.pth)
        cls.pth = f"{cls.pth}/data/"
        cls.files = glob(f"{cls.pth}*.tif")
        print(cls.files)
        cls.strp_glob = f"{cls.pth}RadT_tavg_%Y%m.tif"
        cls.dates = sorted(datetime.strptime(os.path.basename(string).split("_")[2].split(".")[0], "%Y%m") for string in cls.files)
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
                src.apply(absolute_sum_of_changes(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(doy_of_maximum(dates=self.dates), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(doy_of_minimum(dates=self.dates), bands=1, num_workers=2, outfile=out_path)
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
#Error= 1
    def test_kurtosis_excess(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(kurtosis_excess(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                #print(dst.shape)
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))

# ERROR: test_kurtosis_excess (test_feature_calculator_series.TestFeatureCalculators)
# ----------------------------------------------------------------------
# Traceback (most recent call last):
#   File "C:\Users\jithe\Documents\github\xr_fresh\tests\test_feature_calculator_series.py", line 155, in test_kurtosis_excess
#     src.apply(kurtosis_excess(), bands=1, num_workers=2, outfile=out_path)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\api.py", line 1179, in apply
#     self._write_window(dst, res, apply_func_.count, w)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\api.py", line 1230, in _write_window
#     dst_.write(
#   File "rasterio\_io.pyx", line 1700, in rasterio._io.DatasetWriterBase.write
# ValueError: Source shape (1,) is inconsistent with given indexes 1

    def test_large_standard_deviation(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(large_standard_deviation(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))
#Error= 2
    def test_longest_strike_above_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(longest_strike_above_mean(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))
# ======================================================================
# ERROR: test_longest_strike_below_mean (test_feature_calculator_series.TestFeatureCalculators)
# ----------------------------------------------------------------------
# Traceback (most recent call last):
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\threading.py", line 973, in _bootstrap
#     self._bootstrap_inner()
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\threading.py", line 1016, in _bootstrap_inner
#     self.run()
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\threading.py", line 953, in run
#     self._target(*self._args, **self._kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\thread.py", line 83, in _worker
#     work_item.run()
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\thread.py", line 58, in run
#     result = self.fn(*self.args, **self.kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\api.py", line 1174, in <lambda>
#     executor.map(lambda f: apply_func_(*f), data_gen),
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\series.py", line 294, in __call__
#     return w, self.calculate(array)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\xr_fresh\feature_calculator_series.py", line 321, in calculate
#     consecutive_true = jnp.apply_along_axis(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\numpy\lax_numpy.py", line 3032, in apply_along_axis
#     return func(arr)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\traceback_util.py", line 166, in reraise_with_filtered_traceback
#     return fun(*args, **kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\api.py", line 1240, in vmap_f
#     out_flat = batching.batch(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\linear_util.py", line 188, in call_wrapped
#     ans = self.f(*args, **dict(self.params, **kwargs))
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\traceback_util.py", line 166, in reraise_with_filtered_traceback
#     return fun(*args, **kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\api.py", line 1240, in vmap_f
#     out_flat = batching.batch(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\linear_util.py", line 188, in call_wrapped
#     ans = self.f(*args, **dict(self.params, **kwargs))
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\traceback_util.py", line 166, in reraise_with_filtered_traceback
#     return fun(*args, **kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\api.py", line 1240, in vmap_f
#     out_flat = batching.batch(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\linear_util.py", line 188, in call_wrapped
#     ans = self.f(*args, **dict(self.params, **kwargs))
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\numpy\lax_numpy.py", line 3027, in <lambda>
#     func = lambda arr: func1d(arr, *args, **kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\xr_fresh\feature_calculator_series.py", line 266, in _count_longest_consecutive   
#     if value:
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\core.py", line 667, in __bool__
#     def __bool__(self): return self.aval._bool(self)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\core.py", line 1370, in error
#     raise ConcretizationTypeError(arg, fname_context)
# jax._src.traceback_util.UnfilteredStackTrace: jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[].
# The problem arose with the `bool` function.
# This BatchTracer with object id 1915579993520 was created on line:


# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError

# The stack trace below excludes JAX-internal frames.
# The preceding is the original exception that occurred, unmodified.

# --------------------

# The above exception was the direct cause of the following exception:

# Traceback (most recent call last):
#   File "C:\Users\jithe\Documents\github\xr_fresh\tests\test_feature_calculator_series.py", line 217, in test_longest_strike_below_mean
#     src.apply(longest_strike_below_mean(), bands=1, num_workers=2, outfile=out_path)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\api.py", line 1173, in apply
#     for w, res in tqdm_obj(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\tqdm\std.py", line 1181, in __iter__
#     for obj in iterable:
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\_base.py", line 621, in result_iterator
#     yield _result_or_cancel(fs.pop())
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\_base.py", line 319, in _result_or_cancel
#     return fut.result(timeout)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\_base.py", line 451, in result
#     return self.__get_result()
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\_base.py", line 403, in __get_result
#     raise self._exception
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\concurrent\futures\thread.py", line 58, in run
#     result = self.fn(*self.args, **self.kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\api.py", line 1174, in <lambda>
#     executor.map(lambda f: apply_func_(*f), data_gen),
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\geowombat\core\series.py", line 294, in __call__
#     return w, self.calculate(array)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\xr_fresh\feature_calculator_series.py", line 321, in calculate
#     consecutive_true = jnp.apply_along_axis(
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\numpy\lax_numpy.py", line 3032, in apply_along_axis
#     return func(arr)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\jax\_src\numpy\lax_numpy.py", line 3027, in <lambda>
#     func = lambda arr: func1d(arr, *args, **kwargs)
#   File "C:\Users\jithe\anaconda3\envs\xr_fresh\lib\site-packages\xr_fresh\feature_calculator_series.py", line 266, in _count_longest_consecutive   
#     if value:
# jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[].
# The problem arose with the `bool` function.
# This BatchTracer with object id 1915579993520 was created on line:


# See https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError

# ----------------------------------------------------------------------

#Error= 3
    def test_longest_strike_below_mean(self):
        with self.tmp_dir as tmp:
            if not os.path.exists(tmp):
                os.mkdir(tmp)
            out_path = Path(tmp) / "test.tif"
            # use rasterio to create a new file tif file

            with gw.series(self.files) as src:
                src.apply(longest_strike_below_mean(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(mean_second_derivative_central(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(ols_slope_intercept(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(ratio_beyond_r_sigma(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(standard_deviation(), bands=1, num_workers=2, outfile=out_path)
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
                src.apply(ts_complexity_cid_ce(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))
#Error=4 "Infinite loop run"
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
                src.apply(variance_larger_than_standard_deviation(), bands=1, num_workers=2, outfile=out_path)
            with gw.open(out_path) as dst:
                self.assertEqual(dst.gw.nbands, 1)
                self.assertEqual(dst.shape, (1, 1613, 2313))


if __name__ == '__main__':
    unittest.main()

