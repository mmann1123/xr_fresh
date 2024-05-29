import logging
import warnings
import xarray as xr
from xr_fresh import feature_calculators
from xr_fresh.utils import xarray_to_rasterio
from itertools import chain

_logger = logging.getLogger(__name__)


def _exists(var):
    return var in globals()


def _append_dict(join_dict, string="_"):
    """
    Creates strings from dictionary key and value pairs if dictionary exists.
    """
    assert isinstance(join_dict, str)
    return "_" * _exists(join_dict) + string.join(
        [
            "_".join(map(str, chain.from_iterable(globals()[join_dict].items())))
            if _exists(join_dict)
            else ""
        ]
    )


def _get_xr_attr(function_name):
    return getattr(feature_calculators, function_name)


def _month_subset(xr_data, args):
    xr_data = xr_data.where(
        (xr_data["time.month"] >= args["start_month"])
        & (xr_data["time.month"] <= args["end_month"]),
        drop=True,
    )

    months = {k: args[k] for k in args.keys() & {"start_month", "end_month"}}

    for x in ["end_month", "start_month"]:
        args.pop(x, None)

    return xr_data, args, months


def _apply_fun_name(function_name, xr_data, band, args):
    print("Extracting:  " + function_name)

    global months

    if "start_month" and "end_month" in args:
        print("Subsetting to month bounds")
        xr_data, args, months = _month_subset(xr_data, args)

    out = _get_xr_attr(function_name)(xr_data.sel(band=band), **args).compute()

    if (
        function_name == "linear_time_trend"
        or function_name == "linear_time_trend2"
        and args == {"param": "all"}
    ):
        out.coords["variable"] = [
            band + "__" + function_name + "__" + x + _append_dict(join_dict="months")
            for x in ["intercept", "slope", "pvalue", "rvalue"]
        ]

    else:
        out.coords["variable"] = (
            band
            + "__"
            + function_name
            + "_"
            + "_".join(map(str, chain.from_iterable(args.items())))
            + _append_dict(join_dict="months")
        )

    return out


def check_dictionary(arguments):
    for func, args in arguments.items():
        if type(args) == list and len(args) == 0:
            warnings.warn(
                "Problem with feature_dict, should take the following form: feature_dict = { 'maximum':[{}] ,'quantile': [{'q':'0.5'},{'q':'0.95'}]} Not all functions will be calculated"
            )
            print(
                """Problem with feature_dict, should take the following form: 
                    feature_dict = { 'maximum':[{}] ,'quantile': [{'q':'0.5'},{'q':'0.95'}]} 
                    ***Not all functions will be calculated***"""
            )


def extract_features(
    xr_data,
    feature_dict,
    band,
    na_rm=False,
    filepath=None,
    postfix="",
    dim="variable",
    persist=False,
    *args
):
    """
    Extract features from a xarray.DataArray containing a time series of rasters.

    Args:
        xr_data (xarray.DataArray): The xarray.DataArray with a time series of rasters to compute the features for.
        feature_dict (dict): Mapping from feature calculator names to parameters.
        band (str): The name of the variable to create features for.
        na_rm (bool): If True, all missing values are masked using .attrs['nodatavals'].
        filepath (str): If not None, writes each feature to filepath.
        postfix (str): If filepath is not None, appends postfix to the end of the feature name.
        dim (str): The name of the dimension used to collect outputed features.
        persist (bool): If xr_data can easily fit in memory, set to True; if not, keep False.

    Returns:
        xarray.DataArray: The DataArray containing extracted features in `dim`.
    """
    check_dictionary(feature_dict)

    if na_rm is True:
        nodataval = xr_data.attrs["nodatavals"]
        xr_data = xr_data.where(xr_data.sel(band=band) != nodataval)

    if persist:
        print("IMPORTANT: Persisting, pulling all data into memory")
        xr_data = xr_data.persist()

    if filepath != None:
        for func, args in feature_dict.items():
            feature = [
                _apply_fun_name(
                    function_name=func, xr_data=xr_data, band=band, args=arg
                )
                for arg in args
            ]

            feature = xr.concat(feature, dim)

            if hasattr(xr_data, "gw"):
                feature = feature.gw.match_data(
                    xr_data, band_names=feature["variable"].values.tolist()
                )

            xarray_to_rasterio(feature, path=filepath, postfix=postfix)

        return None

    else:
        features = [
            _apply_fun_name(function_name=func, xr_data=xr_data, band=band, args=arg)
            for func, args in feature_dict.items()
            for arg in args
        ]

        features = xr.concat(features, dim)

        if hasattr(xr_data, "gw"):
            features = features.gw.match_data(
                xr_data, band_names=features["variable"].values.tolist()
            )

        return features
