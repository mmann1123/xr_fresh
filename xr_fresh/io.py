import dask.array as da
from pathlib import Path
from .transformers import Stackerizer
import xarray as xr
import geowombat as gw
import numpy as np


def stack_to_pandas(data, src, t, b, y, x):

    # The data from `da.store` are numpy arrays

    da = xr.DataArray(
        data=data,
        coords={
            "time": src.time.values[t],
            "band": src.band.values[b],
            "y": src.y.values[y],
            "x": src.x.values[x],
        },
        dims=("time", "band", "y", "x"),
    )

    X = Stackerizer(stack_dims=("y", "x", "time"), direction="stack").fit_transform(da)

    X = X.to_pandas()
    X.columns = X.columns.astype(str)

    return X


class WriteDaskArray(object):
    def __init__(self, filename, src, overwrite=True):

        self.filename = filename
        self.overwrite = overwrite

        fpath = Path(self.filename)
        self.parent = fpath.parent
        self.stem = fpath.stem
        self.suffix = fpath.suffix

        self.src = src
        self.gw = src.gw

    def __setitem__(self, key, item):

        if len(key) == 4:
            t, b, y, x = key
        elif len(key) == 3:
            b, y, x = key
        else:
            y, x = key

        row_chunks = list(self.src.chunks[-2])
        row_chunks[0] = 0
        row_id = np.add.accumulate(row_chunks)

        self.src.attrs["chunk_ids"] = dict(
            zip([f"{i:09d}" for i in row_id], range(0, len(row_id)),)
        )
        # note not working yet
        # chunk_id = self.src.attrs["chunk_ids"][f"{y.start:09d}"]

        out_filename = (
            self.parent
            / f"{self.stem}_row{y.start:09d}{self.suffix}"
            # / f"{self.stem}_chunk{chunk_id:09d}{self.suffix}"
        )

        if self.overwrite:

            if out_filename.is_file():
                out_filename.unlink()

        item = stack_to_pandas(item, self.src, t, b, y, x)

        item.to_parquet(out_filename)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def WriteStackedArray(src: xr.DataArray, file_path="/tmp/test.parquet"):
    """Writes stacked ie. flattened by (y,x,time) to parquet in chunks.

    :param src: [description]
    :type src: xr.DataArray
    :param file_path: [description], defaults to "/tmp/test.parquet":path
    :type file_path: [type], optional
    """
    # This could be used within a client context
    with WriteDaskArray(file_path, src) as dst:

        src = src.chunk(
            {"time": -1, "band": -1, "y": "auto", "x": -1}
        )  # rechunk to time and iterate across rows

        res = da.store(src.data, dst, lock=False, compute=False)

        res.compute()


def parquet_append(
    file_list: list, out_path: str, filters: list,
):
    """
    Read, filter and append large set of parquet files to a single file. Note: resulting file must be read with pd.read_parquet(engine='pyarrow')

    `See read_table docs <https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html#pyarrow.parquet.read_table>`_
    
    :param file_list: list of file paths to .parquet files
    :type file_list: list
    :param out_path: path and name of output parquet file
    :type out_path: str
    :param filters: list of 
    :type filters: list

    .. highlight:: python
    .. code-block:: python

        ('x', '=', 0)
        ('y', 'in', ['a', 'b', 'c'])
        ('z', 'not in', {'a','b'})
    ...

    """
    pqwriter = None
    for i, df in enumerate(file_list):
        table = pq.read_table(df, filters=filters, use_pandas_metadata=True,)
        # for the first chunk of records
        if i == 0:
            # create a parquet write object giving it an output file
            pqwriter = pq.ParquetWriter(out_path, table.schema,)
        pqwriter.write_table(table)

    # close the parquet writer
    if pqwriter:
        pqwriter.close()


# old example using parallel task, much slower, but shouldn't be.
# # add variables and stackerize
# def user_func(*args, name_append=os.path.basename(image_stack_folder)):
#     # Gather function arguments
#     data, window_id, num_workers = list(itertools.chain(*args))  # num_workers=1

#     # Send the computation to Dask
#     X = Stackerizer(stack_dims=("y", "x", "time"), direction="stack").fit_transform(
#         data
#     )  # NOTE stack y before x!!!
#     X.compute(scheduler="threads", num_workers=num_workers)

#     with open(unstacked_folder + "/log.txt", "a") as the_file:
#         the_file.write("working on %08d \t" % window_id)
#         the_file.write(str(datetime.now()) + "\n")

#     X = X.to_pandas()

#     # # downcast to save memory
#     X = downcast_pandas(X)
#     X.columns = X.columns.astype(str)

#     # save this approach wont write files in order
#     X.to_parquet(
#         unstacked_folder
#         + "/NDVI_2010-2016_Xy_"
#         + name_append
#         + "_"
#         + os.path.splitext(block)[0]
#         + "_%05d.parquet" % window_id,
#         compression="snappy",
#     )

#     return None


# # import matplotlib.pyplot as plt
# # fig, ax = plt.subplots(dpi=350)
# # labels.sel(band='r_code').plot.imshow(robust=False, ax=ax)
# # plt.show(plt.tight_layout(pad=1))


# #%%
# name_append = os.path.basename(image_stack_folder)

# lc = os.path.join(data_path, "Ethiopia_Land_Cover_2017/LandCoverMap/LC_ethiopia_RF.tif")


# for block in unique_blocks[1:2]:  # 0 completed
#     print("working on block:" + block)

#     vrts = sorted(glob(image_stack_folder + "/**/" + block))
#     vrts = vrts[:-3]  # limit to 2010-2016

#     time_names = [x for x in range(2010, 2017)]  # NEED TO LIMIT 2012 to 2016

#     with gw.open(
#         vrts, time_names=time_names, chunks=400
#     ) as ds:  # chunks is the pixel size of a chunk

#         ds = add_categorical(
#             data=ds,
#             labels=regions,
#             col="NAME_1",
#             variable_name="region_code",
#             missing_value=-9999,
#         )  # use -9999 going forward
#         ds = add_categorical(
#             data=ds,
#             labels=zones,
#             col="NAME_2",
#             variable_name="zone_code",
#             missing_value=-9999,
#         )
#         ds = add_categorical(
#             data=ds,
#             labels=loss_poly,
#             col="rk_code",
#             variable_name="rk_code",
#             missing_value=-9999,
#         )
#         # add target data by year
#         ds = add_time_targets(
#             data=ds,
#             target=loss_poly,
#             target_col_list=[
#                 "w_dam_2010",
#                 "w_dam_2011",
#                 "w_dam_2012",
#                 "w_dam_2013",
#                 "w_dam_2014",
#                 "w_dam_2015",
#                 "w_dam_2016",
#             ],
#             target_name="weather_damage",
#             missing_value=np.NaN,
#             append_to_X=True,
#         )

#         ds = add_categorical(
#             data=ds, labels=lc, variable_name="land_cover", missing_value=-9999,
#         )

#         ds = ds.chunk(
#             {"time": -1, "band": -1, "y": "auto", "x": "auto"}
#         )  # rechunk to time
#         print(ds)

#         pt = ParallelTask(ds, scheduler="threads", n_workers=2)
#         print("n chunks %s" % pt.n_chunks)
#         print("n windows %s" % pt.n_windows)
#         res = pt.map(user_func, 6)


#####################
# dask delayed write
#####################

# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar, progress

# ProgressBar().register()
# from glob import glob
# import dask

# files = sorted(glob(os.path.join(unstacked_folder, "Xy_2010_2017_ea_lu_*.parquet")))
# out_files = os.path.join(unstacked_folder, "training/")


# @dask.delayed
# def dask_dropna(file_in, column):
#     ddf = pd.read_parquet(file_in)
#     ddf.dropna(
#         axis="rows", inplace=True, subset=[column]
#     )  # loc[ddf[column].isnull(),:]
#     # ddf.to_parquet(os.path.join(out_files,os.path.basename(file_in) ))
#     return ddf


# results = []
# for x in files:
#     # Note the difference in way the inc function is called!
#     y = dask_dropna(file_in=x, column="weather_damage")
#     results.append(y)


# results = dask.compute(*results)
# results = pd.concat(results, ignore_index=False)
# print(results.head())

# # drop areas outside of sub kebeles
# results = results[results.rk_code != 0]
# results.to_parquet(
#     os.path.join(out_files, "Xy_2010_2017_ea_lu__training.parquet"),
#     engine="pyarrow",
#     index=True,
# )
