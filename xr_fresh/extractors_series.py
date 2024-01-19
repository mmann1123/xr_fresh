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
