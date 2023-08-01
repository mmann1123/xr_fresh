# %% env:testbed laptop

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
import jax.numpy as jnp
from datetime import datetime

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/xr_fresh")
sys.path.append("/home/mmann1123/Documents/github/xr_fresh/")
import xr_fresh as xf

pth = "/home/mmann1123/Dropbox/Africa_data/Temperature/"


# files = glob("/home/mmann1123/extra_space/Dropbox/Africa_data/Temperature/*.tif")[0:10]
files = sorted(glob(f"{pth}*.tif"))[0:10]
strp_glob = f"{pth}RadT_tavg_%Y%m.tif"
dates = sorted(datetime.strptime(string, strp_glob) for string in files)
dates

#  %%


class unique_value_number_to_time_series_length(gw.TimeModule):
    """SLOW
    Returns a factor which is 1 if all values in the time series occur only once,
    and below one if this is not the case.
    In principle, it just returns

        # of unique values / # of values
    """

    def __init__(self):
        super(unique_value_number_to_time_series_length, self).__init__()
        print("this is slow and needs more work")

    def calculate(self, array):
        # Count the number of unique values along the time axis (axis=0)
        unique_counts = jnp.sum(jnp.unique(array, axis=0), axis=0)

        return (unique_counts / len(array)).squeeze()

        # def count_unique_values(arr):
        #     unique_counts = jnp.sum(np.unique(arr, axis=0), axis=0)
        #     return unique_counts

        # # Apply the function along the time axis (axis=0)
        # result = jnp.apply_along_axis(count_unique_values, axis=0, arr=array)

        # # Divide the count of unique values by the length of time
        # result /= array.shape[0]

        # return result.squeeze()


with gw.series(
    files,
    nodata=9999,
) as src:
    print(src)
    src.apply(
        func=unique_value_number_to_time_series_length(),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=5,
        bands=1,
    )


# %%
# %%


def count_above_mean(X, dim="time", **kwargs):
    """
    Returns the number of values in X that are higher than the mean of X

    :param X: the time series to calculate the feature of
    :type X: xarray.DataArray
    :return: the value of this feature
    :return type: float
    """

    return ((X > X.mean(dim)).sum(dim)).astype(np.float64)


# predict to stack
def user_func(w, block):
    pred_shape = list(block.shape)
    X = block.reshape(pred_shape[0], -1).T
    pred_shape[0] = 1
    y_hat = count_above_mean(X)
    X_reshaped = y_hat.T.reshape(pred_shape)
    return w, X_reshaped


# %%
# with gw.open(select_images, nodata=9999, stack_dim="band") as src:
#     src = pipeline_scale_clean.fit_transform(src)

#     src.gw.save(
#         "outputs/pred_stack.tif",
#         compress="lzw",
#         overwrite=True,  # bigtiff=True
#     )


# # predict to stack
# def user_func(w, block, model):
#     pred_shape = list(block.shape)
#     X = block.reshape(pred_shape[0], -1).T
#     pred_shape[0] = 1
#     y_hat = model.predict(X)
#     X_reshaped = y_hat.T.reshape(pred_shape)
#     return w, X_reshaped


# gw.apply(
#     "outputs/pred_stack.tif",
#     f"outputs/final_model_rf{len(select_images)}.tif",
#     user_func,
#     args=(pipeline_performance,),
#     n_jobs=16,
#     count=1,
# )
