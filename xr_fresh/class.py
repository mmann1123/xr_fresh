# %%
from geowombat.core.geoxarray import GeoWombatAccessor
import xarray as xr
from geowombat.core.conversion import Converters
import typing as T
import itertools
import logging
import multiprocessing as multi
import os
import tempfile
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from ray.util import ActorPool
import ray
import rasterio as rio

import dask
import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from dask.distributed import Client, LocalCluster
from pyproj.enums import WktVersion
from pyproj.exceptions import CRSError
from rasterio import features
from rasterio.coords import BoundingBox
from scipy.spatial import cKDTree
from scipy.stats import mode as sci_mode
from shapely.geometry import Polygon
from geowombat.core.parallel import ParallelTask


@ray.remote
class Actor(object):
    def __init__(self, aoi_id=None, id_column=None, band_names=None):
        self.aoi_id = aoi_id
        self.id_column = id_column
        self.band_names = band_names

    # While the names can differ, these three arguments are required.
    # For ``ParallelTask``, the callable function within an ``Actor`` must be named exec_task.
    def exec_task(self, data_block_id, data_slice, window_id):
        data_block = data_block_id[data_slice]
        left, bottom, right, top = data_block.gw.bounds
        aoi_sub = self.aoi_id.cx[left:right, bottom:top]

        if aoi_sub.empty:
            return aoi_sub

        # Return a GeoDataFrame for each actor
        return gw.extract(
            data_block, aoi_sub, id_column=self.id_column, band_names=self.band_names
        )


def _calculate_n_chunks(array_shape, min_chunks=1, max_chunks=1000):
    bands, rows, cols = array_shape
    array_size = rows * cols * bands

    # Example strategy: linear scaling
    # Adjust these parameters based on your specific needs and performance tuning
    scale_factor = (
        100000  # Adjust this based on your array sizes and system capabilities
    )

    n_chunks = int(array_size / scale_factor)
    n_chunks = max(min_chunks, min(n_chunks, max_chunks))

    return n_chunks


@xr.register_dataarray_accessor("fresh")
class ExtendedGeoWombat(GeoWombatAccessor):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        # Initialize any additional attributes or call methods necessary for your extension

    # Define additional methods as needed
    def fast_extract(
        self,
        data: xr.DataArray,
        aoi: T.Union[str, Path, gpd.GeoDataFrame],
        bands: T.Union[int, T.Sequence[int]] = None,
        time_names: T.Sequence[T.Any] = None,
        band_names: T.Sequence[T.Any] = None,
        frac: float = 1.0,
        min_frac_area: T.Optional[T.Union[float, int]] = None,
        all_touched: T.Optional[bool] = False,
        id_column: T.Optional[str] = "id",
        time_format: T.Optional[str] = "%Y%m%d",
        mask: T.Optional[T.Union[Polygon, gpd.GeoDataFrame]] = None,
        n_jobs: T.Optional[int] = 8,
        verbose: T.Optional[int] = 0,
        n_workers: T.Optional[int] = 1,
        n_threads: T.Optional[int] = -1,
        use_client: T.Optional[bool] = False,
        address: T.Optional[str] = None,
        total_memory: T.Optional[int] = 24,
        processes: T.Optional[bool] = False,
        pool_kwargs: T.Optional[dict] = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        with rio.Env(GDAL_CACHEMAX=256 * 1e6) as env:
            df_id = ray.put(gpd.read_file(aoi).to_crs(f"EPSG:{data.crs}"))

            if not band_names:
                band_names = data.band.values.tolist()

            # Setup the pool of actors, one for each resource available to ``ray``.
            actor_pool = ActorPool(
                [
                    Actor.remote(aoi_id=df_id, id_column="id", band_names=band_names)
                    for n in range(0, int(ray.cluster_resources()["CPU"]))
                ]
            )
            nchunks = _calculate_n_chunks(data.shape)

            # Setup the task object
            pt = ParallelTask(
                data,
                # row_chunks=data.shape[0] // 16,
                # col_chunks=data.shape[1] // 16,
                # row_chunks=4096,
                # col_chunks=4096,
                scheduler="ray",
                n_chunks=nchunks,
            )
            results = pt.map(actor_pool)

            del df_id, actor_pool
            ray.shutdown()
            # results2 = [df.reset_index(drop=True) for df in results if len(df) > 0]
            # pd.concat(results2, ignore_index=True, axis=0)

            return results


# %%
import unittest

import geowombat as gw
from geowombat.data import (
    l8_224078_20200518_points,
    l8_224078_20200518_polygons,
    l8_224078_20200518,
    l8_224078_20200518_B2,
)

import numpy as np
import geopandas as gpd
from sklearn.preprocessing import LabelEncoder


aoi = gpd.read_file(l8_224078_20200518_points)
aoi["id"] = LabelEncoder().fit_transform(aoi.name)
aoi = aoi.drop(columns=["name"])
aoi = "/home/mmann1123/miniconda3/envs/xrfresh_statsmooth/lib/python3.9/site-packages/geowombat/data/LC08_L1TP_224078_20200518_20200518_01_RT_polygons.gpkg"

l8_224078_20200518_B2_values = np.array(
    [7966, 8030, 7561, 8302, 8277, 7398], dtype="float64"
)

l8_224078_20200518_values = np.array(
    [
        [7966, 8030, 7561, 8302, 8277, 7398],
        [7326, 7490, 6874, 8202, 7982, 6711],
        [6254, 8080, 6106, 8111, 7341, 6007],
    ],
    dtype="float64",
)


def test_single_image_single_band(aoi):
    with gw.open(l8_224078_20200518_B2) as src:
        df = src.fresh.fast_extract(
            data=src,
            aoi=aoi,
        )  # band_names=["blue"])

    assert (np.allclose(df.blue.values, l8_224078_20200518_B2_values)) == True


test_single_image_single_band(aoi)


# with gw.open(l8_224078_20200518_B2) as src:
#     print(type(src.crs))
# %%
# test fast_extract
import xarray as xr
import numpy as np
import geowombat as gw

# Create a sample xarray
arr = np.random.rand(100, 100)
xarr = xr.DataArray(arr)

# Register the accessor

xarr.fresh.fast_extract()


# %%
