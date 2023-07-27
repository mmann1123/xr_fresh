# %% env:testbed

import os
import sys
import numpy as np
from glob import glob
import geowombat as gw
import jax.numpy as jnp

sys.path.append("/home/mmann1123/Documents/github/xr_fresh/xr_fresh")
import xr_fresh as xf

files = glob("/home/mmann1123/extra_space/Dropbox/Africa_data/Temperature/*.tif")[0:10]


class count_above_mean(gw.TimeModule):
    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            self.mean = jnp.nanmean(array, axis=0)

        return jnp.nansum(array > self.mean, axis=0).squeeze()


class count_above_mean(gw.TimeModule):
    def __init__(self, mean=None):
        super(count_above_mean, self).__init__()
        self.mean = mean

    def calculate(self, array):
        if self.mean is None:
            self.mean = jnp.nanmean(array, axis=0)

        return jnp.nansum(array > self.mean, axis=0).squeeze()


with gw.series(
    files,
    nodata=9999,
) as src:
    src.apply(
        func=count_above_mean(mean=295),
        outfile=f"/home/mmann1123/Downloads/test.tif",
        num_workers=1,
        bands=1,
    )


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
