import logging
import warnings
import xarray as xr

# from numpy import where
from xr_fresh import feature_calculators
from xr_fresh.utils import xarray_to_rasterio
from itertools import chain

# from os.path import expanduser
# from os.path import join as path_join
# from dask.distributed import progress
# from dask.diagnostics import ProgressBar

_logger = logging.getLogger(__name__)


def _exists(var):
    return var in globals()


# use of globals is problematic, applies to all


def _append_dict(join_dict, string="_"):
    """
    creates strings from dictionary key and value pairs if dictionary exists 
    """

    assert isinstance(join_dict, str)

    return "_" * _exists(join_dict) + string.join(
        [
            "_".join(map(str, chain.from_iterable(globals()[join_dict].items())))
            if _exists(join_dict)
            else ""
        ]
    )


def _stringr(notstring):
    "_".join(str(notstring))


def _get_xr_attr(function_name):
    return getattr(feature_calculators, function_name)


def _month_subset(xr_data, args):
    # subset by month
    xr_data = xr_data.where(
        (xr_data["time.month"] >= args["start_month"])
        & (xr_data["time.month"] <= args["end_month"]),
        drop=True,
    )

    months = {k: args[k] for k in args.keys() & {"start_month", "end_month"}}

    for x in ["end_month", "start_month"]:
        args.pop(x, None)  # remove unneeded args

    return xr_data, args, months


def _apply_fun_name(function_name, xr_data, band, args):
    # apply function for large objects lazy
    print("Extracting:  " + function_name)

    global months  # required to check if exists w _exists

    if "start_month" and "end_month" in args:
        print("subsetting to month bounds")
        xr_data, args, months = _month_subset(xr_data, args)

    out = _get_xr_attr(function_name)(xr_data.sel(band=band), **args).compute()

    if (
        function_name == "linear_time_trend"
        or function_name == "linear_time_trend2"
        and args == {"param": "all"}
    ):
        # handle exception for regression
        out.coords["variable"] = [
            band + "__" + function_name + "__" + x + _append_dict(join_dict="months")
            for x in ["intercept", "slope", "pvalue", "rvalue"]
        ]

    else:

        # out.coords['variable'] = band + "__" + function_name+'_'   + _append_dict(join_dict='args') + _append_dict(join_dict='months')
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
                " Problem with feature_dict, should take the following form: feature_dict = { 'maximum':[{}] ,'quantile': [{'q':'0.5'},{'q':'0.95'}]} Not all functions will be calculated"
            )
            print(
                """ Problem with feature_dict, should take the following form: 
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
    Extract features from

    * a :class:`xarray.DataArray` containing a time series of rasters

    A :class:`xarray.DataArray` with the calculated features will be returned a 'variable'.

    Examples
    ========

    >>>  f_dict = { 'maximum':[{}] ,  
                   'quantile': [{'q':"0.5"},{'q':'0.95'}]}
    >>>  f_dict2 = { 'maximum':[{}] ,  
                   'quantile': [{'q':"0.5", 'start_month':1, 'end_month':6}]}
    >>>  features = extract_features(xr_data=ds,
    >>>                     feature_dict=f_dict,
    >>>                     band='aet', 
    >>>                     na_rm = True)

    :param xr_data: The xarray.DataArray with a time series of rasters to compute the features for.
    :type xr_data: xarray.DataArray

    :param feature_dict: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See examples above. 
    :type feature_dict: dict

    :param band: The name of the variable to create feature for.
    :type band: str

    :param na_rm: If True (default), all missing values are masked using .attrs['nodatavals']
    :type na_rm: bool

    :param filepath: If not none, assuming xarrays being used, writes each feature to filepath 
    :type filepath: str

    :param postfix: If filepath not none, appends postfix to the end of the feature name 
    :type postfix: str

    :param dim: The name of the dimension used to collect outputed features
    :type dim: str
    
    :param persist: (optional) If xr_data can easily fit in memory, set to True, if not keep False
    :type persist: bool
    
    :return: The DataArray containing extracted features in `dim`.
    :rtype: xarray.DataArray
    
    """
    check_dictionary(feature_dict)

    # improvement: check cluster status, have attribute "persist" for setting
    # persistence of small in memory objects.
    # if Cluster.type = 'large_object', no persist

    if na_rm is True:
        nodataval = xr_data.attrs["nodatavals"]
        xr_data = xr_data.where(xr_data.sel(band=band) != nodataval)

    if persist:
        print("IMPORTANT: Persisting pulling all data into memory")
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

            # out = feature[0]
            # out.gw.imshow()

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
            )  # place postfix here? or time assigned by dim?

        return features

