import geowombat as gw
from xr_fresh.feature_calculator_series import *
import numpy as np
import logging
import re
import os
from glob import glob
from datetime import datetime
from pathlib import Path
import inspect
from xr_fresh.feature_calculator_series import function_mapping


def extract_features_series(
    gw_series, feature_dict, band_name, output_dir, num_workers=-1, nodata=np.nan
):
    """
    Extracts features from a geospatial time series and saves them as TIFF files.

    Args:
        gw_series (geowombat.Dataset): Geospatial time series dataset.
        feature_dict (dict): Dictionary containing feature names and parameters.
        band_name (str): Name of the band.
        output_dir (str): Directory to save the output TIFF files.

    Returns:
        None

    Example
    -------
    .. code-block:: python

        # Define the feature dictionary
        feature_dict = {
        "abs_energy": [{}],
        "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
        "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
        "skewness": [{}],
        }

        # Define the band name and output directory
        band_name = "B2"

        # Create the output directory if it doesn't exist
        output_directory = "../features"

        # Extract features from the geospatial time series
        extract_features_series(gw_series, feature_dict, band_name, output_directory)
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
    with gw.series(gw_series, nodata=nodata, window_size=[256, 256]) as src:
        # Iterate over each feature in the dictionary
        for feature_name, params_list in feature_dict.items():
            for params in params_list:
                # Get the corresponding function from the mapping
                func = function_mapping.get(feature_name)
                if inspect.isclass(func):
                    # Instantiate the function with parameters
                    feature_func = func(**params)

                    # create output file name if parameters exist
                    # avoid issue with all dates
                    if feature_name in ["doy_of_maximum", "doy_of_minimum"]:
                        key_names = list(params.keys())[0]
                        value_names = list(params.values())[0]
                        output_file = os.path.join(
                            output_dir,
                            f"{band_name}_{feature_name}_{key_names}.tif",
                        )
                    elif len(list(params.keys())) > 0:
                        key_names = list(params.keys())[0]
                        value_names = list(params.values())[0]
                        output_file = os.path.join(
                            output_dir,
                            f"{band_name}_{feature_name}_{key_names}_{value_names}.tif",
                        )
                    else:
                        output_file = os.path.join(
                            output_dir, f"{band_name}_{feature_name}.tif"
                        )

                    try:
                        src.apply(
                            func=feature_func,
                            outfile=output_file,
                            num_workers=num_workers,
                            bands=1,
                            kwargs={
                                "BIGTIFF": "IFNEEDED",
                                "compress": "LZW",
                            },
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
