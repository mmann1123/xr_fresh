import geowombat as gw
from xr_fresh.feature_calculator_series import *
import numpy as np
import logging
import re
import os
from glob import glob
from datetime import datetime
from pathlib import Path

# Mapping of feature names to corresponding functions
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

# class extractors_series(gw.TimeModule):


def extract_features_series(gw_series, feature_dict, band_name, output_dir):
    """
    Extracts features from a geospatial time series and saves them as TIFF files.

    Parameters:
        gw_series (geowombat.Dataset): Geospatial time series dataset.
        feature_dict (dict): Dictionary containing feature names and parameters.
        band_name (str): Name of the band.
        output_dir (str): Directory to save the output TIFF files.

    Returns:
        None
    """
    # Create output directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        filename=Path(output_dir) / "error_log.log",
        level=logging.ERROR,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )

    # Apply the function to the geospatial time series
    with gw.series(gw_series, window_size=[512, 512]) as src:
        # Iterate over each feature in the dictionary
        for feature_name, params_list in feature_dict.items():
            for params in params_list:
                # Get the corresponding function from the mapping
                func = function_mapping.get(feature_name)
                if func:
                    # Instantiate the function with parameters
                    feature_func = func(**params)
                    # Output file name
                    key_names, value_names = extract_key_value_names(band_name)
                    grid = extract_grid(band_name)
                    output_file = (
                        Path(output_dir)
                        / f"{band_name}_{feature_name}_{key_names}_{value_names}_{grid}.tif"
                    )

                    try:
                        src.apply(
                            func=feature_func,
                            outfile=output_file,
                            num_workers=1,
                            bands=1,
                            kwargs={"BIGTIFF": "YES"},
                        )
                    except Exception as e:
                        logging.error(
                            f"Error extracting feature {feature_name} for band {band_name}: {e}"
                        )


def extract_key_value_names(band_name):
    """
    Extracts key_names and value_names from the band_name using regular expressions.

    Parameters:
        band_name (str): Name of the band.

    Returns:
        key_names (str): Extracted key names.
        value_names (str): Extracted value names.
    """
    # Define the regular expressions to capture key_names and value_names
    key_names_pattern = r"key_(\w+)"
    value_names_pattern = r"value_(\w+)"
    # Use regular expressions to extract key_names and value_names from the band_name
    key_names_match = re.search(key_names_pattern, band_name)
    value_names_match = re.search(value_names_pattern, band_name)
    # Check if matches are found
    if key_names_match and value_names_match:
        key_names = key_names_match.group(1)
        value_names = value_names_match.group(1)
        return key_names, value_names
    else:
        # Return default values if matches are not found
        return "default_key", "default_value"


def extract_grid(band_name):
    """
    Extracts grid value from the band_name using regular expressions.

    Parameters:
        band_name (str): Name of the band.

    Returns:
        grid (str): Extracted grid value.
    """
    # Define the regular expression pattern to capture the grid value
    grid_pattern = r"grid_(\d+)"
    # Use regular expression to extract the grid value from the band_name
    grid_match = re.search(grid_pattern, band_name)
    # Check if match is found
    if grid_match:
        grid = grid_match.group(1)
        return grid
    else:
        # Return default value if match is not found
        return "default_grid"


#################################################################################################################################
#################################################################################################################################

# Example usage
if __name__ == "__main__":
    # Define the geospatial time series dataset
    gw_series = ...

    # Define the feature dictionary
    feature_dict = {
        "abs_energy": [{}],
        "absolute_sum_of_changes": [{}],
        "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
        "count_above_mean": [{}],
        "count_below_mean": [{}],
        "doy_of_maximum": [{}],
        "doy_of_minimum": [{}],
        "kurtosis": [{}],
        "large_standard_deviation": [{}],
        # # # "longest_strike_above_mean": [{}],  # not working with jax GPU ram issue
        # # # "longest_strike_below_mean": [{}],  # not working with jax GPU ram issue
        "maximum": [{}],
        "mean": [{}],
        "mean_abs_change": [{}],
        "mean_change": [{}],
        "mean_second_derivative_central": [{}],
        "median": [{}],
        "minimum": [{}],
        # "ols_slope_intercept": [
        #     {"returns": "intercept"},
        #     {"returns": "slope"},
        #     {"returns": "rsquared"},
        # ],  # not working
        "quantile": [{"q": 0.05}, {"q": 0.95}],
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
        "skewness": [{}],
        "standard_deviation": [{}],
        "sum": [{}],
        "symmetry_looking": [{}],
        "ts_complexity_cid_ce": [{}],
        "variance": [{}],
        "variance_larger_than_standard_deviation": [{}],
    }

    # Define the band name and output directory
    band_name = "B2"

    # Create the output directory if it doesn't exist
    output_directory = "../features"

    # Extract features from the geospatial time series
    extract_features_series(gw_series, feature_dict, band_name, output_directory)
