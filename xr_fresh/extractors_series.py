import geowombat as gw
from .feature_calculator_series import *

function_mapping = {
    "abs_energy": abs_energy,
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "autocorrelation": autocorrelation,
    "count_above_mean": count_above_mean,
    "count_below_mean": count_below_mean,
    "doy_of_maximum": doy_of_maximum,
    "doy_of_minimum": doy_of_minimum,
    "kurtosis": kurtosis,
    "kurtosis_excess": kurtosis_excess,
    "large_standard_deviation": large_standard_deviation,
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "maximum": maximum,
    "minimum": minimum,
    "mean": mean,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    "median": median,
    "ols_slope_intercept": ols_slope_intercept,
    "quantile": quantile,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "skewness": skewness,
    "standard_deviation": standard_deviation,
    "sum": sum,
    "symmetry_looking": symmetry_looking,
    "ts_complexity_cid_ce": ts_complexity_cid_ce,
    "unique_value_number_to_time_series_length": unique_value_number_to_time_series_length,
    "variance": variance,
    "variance_larger_than_standard_deviation": variance_larger_than_standard_deviation,
}


def extract_features_series(
    gw_series,
    feature_dict,
    band,
    filepath=None,
    postfix="",
    persist=False,
    *args,
):
    # Loop through the dictionary and apply functions
    with gw.series(gw_series, window_size=[512, 512]) as src:
        for func_name, param_list in feature_dict.items():
            for params in param_list:
                func_class = function_mapping.get(func_name)
                if func_class:
                    func_instance = func_class(**params)  # Instantiate with parameters
                    outfile = f"/home/mmann1123/Downloads/{func_name}.tif"
                    src.apply(
                        func=func_instance,
                        outfile=outfile,
                        num_workers=1,  # Adjust as needed
                        bands=1,
                        kwargs={"BIGTIFF": "YES"},
                    )




# from pathlib import Path
# from xr_fresh.feature_calculator_series import *
# from xr_fresh.feature_calculator_series import function_mapping
# import geowombat as gw
# from glob import glob
# from datetime import datetime
# import re
# import os
# import numpy as np
# import logging

# os.chdir(
#     "/home/mmann1123/extra_space/Dropbox/Tanzania_data/Projects/YM_Tanzania_Field_Boundaries/Land_Cover/northern_tz_data/interpolated_monthly"
# )

# # Set up logging
# logging.basicConfig(
#     filename="../features/error_log.log",
#     level=logging.ERROR,
#     format="%(asctime)s:%(levelname)s:%(message)s",
# )


# complete_times_series_list = {
#     "abs_energy": [{}],
#     "absolute_sum_of_changes": [{}],
#     "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
#     "count_above_mean": [{}],
#     "count_below_mean": [{}],
#     "doy_of_maximum": [{}],
#     "doy_of_minimum": [{}],
#     "kurtosis": [{}],
#     "large_standard_deviation": [{}],
#     # # # "longest_strike_above_mean": [{}],  # not working with jax GPU ram issue
#     # # # "longest_strike_below_mean": [{}],  # not working with jax GPU ram issue
#     "maximum": [{}],
#     "mean": [{}],
#     "mean_abs_change": [{}],
#     "mean_change": [{}],
#     "mean_second_derivative_central": [{}],
#     "median": [{}],
#     "minimum": [{}],
#     # "ols_slope_intercept": [
#     #     {"returns": "intercept"},
#     #     {"returns": "slope"},
#     #     {"returns": "rsquared"},
#     # ],  # not working
#     "quantile": [{"q": 0.05}, {"q": 0.95}],
#     "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
#     "skewness": [{}],
#     "standard_deviation": [{}],
#     "sum": [{}],
#     "symmetry_looking": [{}],
#     "ts_complexity_cid_ce": [{}],
#     "variance": [{}],
#     "variance_larger_than_standard_deviation": [{}],
# }


# for band_name in ["B12", "B11", "hue", "B6", "EVI", "B2"][-1:]:
#     file_glob = f"**/*{band_name}*.tif"

#     f_list = sorted(glob(file_glob))

#     # Get unique grid codes
#     pattern = r"S2_SR_[A-Za-z0-9]+_M_[0-9]{4}-[0-9]{2}-[A-Za-z0-9]+_([0-9]+-[0-9]+(?:-part[12])?)\.tif"

#     unique_grids = list(
#         set(
#             [
#                 re.search(pattern, file_path).group(1)
#                 for file_path in f_list
#                 if re.search(pattern, file_path)
#             ]
#         )
#     )

#     # # add data notes
#     try:
#         Path(f".//features").mkdir(parents=True)
#     except FileExistsError:
#         print(f"The interpolation directory already exists. Skipping.")

#     with open(f".//features/0_notes.txt", "a") as the_file:
#         the_file.write(
#             "Gererated by  github/YM_TZ_crop_classifier/2_xr_fresh_extraction.py \t"
#         )
#         the_file.write(str(datetime.now()))

#     # iterate across grids
#     for grid in unique_grids:
#         print("working on band", band_name, " grid ", grid)
#         a_grid = sorted([f for f in f_list if grid in f])
#         print(a_grid)

#         try:
#             # get dates
#             date_pattern = r"S2_SR_[A-Za-z0-9]+_M_(\d{4}-\d{2})-[A-Za-z0-9]+_.*\.tif"
#             dates = [
#                 datetime.strptime(re.search(date_pattern, filename).group(1), "%Y-%m")
#                 for filename in a_grid
#                 if re.search(date_pattern, filename)
#             ]
#         except Exception as e:
#             logging.error(f"Error parsing name from grid {grid}: {e}")
#             print(f"Error parsing name from grid {grid}: {e}")
#             continue

#         # update doy with dates
#         complete_times_series_list["doy_of_maximum"] = [{"dates": dates}]
#         complete_times_series_list["doy_of_minimum"] = [{"dates": dates}]

#         print(f"working on {band_name} {grid}")
#         with gw.series(
#             a_grid,
#             window_size=[512, 512],  # transfer_lib="numpy"
#             nodata=np.nan,
#         ) as src:
#             # iterate across functions
#             for func_name, param_list in complete_times_series_list.items():
#                 for params in param_list:
#                     # instantiate function
#                     func_class = function_mapping.get(func_name)
#                     if func_class:
#                         func_instance = func_class(
#                             **params
#                         )  # Instantiate with parameters
#                         if len(params) > 0:
#                             print(f"Instantiated {func_name} with  {params}")
#                         else:
#                             print(f"Instantiated {func_name} ")

#                     # create output file name
#                     if len(list(params.keys())) > 0:
#                         key_names = list(params.keys())[0]
#                         value_names = list(params.values())[0]
#                         outfile = f"../features/{band_name}/{band_name}_{func_name}_{key_names}_{value_names}_{grid}.tif"
#                         # avoid issue with all dates
#                         if func_name in ["doy_of_maximum", "doy_of_minimum"]:
#                             outfile = f"../features/{band_name}/{band_name}_{func_name}_{key_names}_{grid}.tif"
#                     else:
#                         outfile = f"../features/{band_name}/{band_name}_{func_name}_{grid}.tif"
#                     # extract features
#                     try:
#                         src.apply(
#                             func=func_instance,
#                             outfile=outfile,
#                             num_workers=3,
#                             processes=False,
#                             bands=1,
#                             kwargs={"BIGTIFF": "YES", "compress": "LZW"},
#                         )
#                     except Exception as e:
#                         logging.error(
#                             f"Error extracting features from {band_name} {func_name} {grid}: {e}"
#                         )
#                         continue
