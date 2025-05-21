import dask.array as da
from pathlib import Path
from .transformers import Stackerizer
import xarray as xr
import numpy as np
import pyarrow.parquet as pq


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
            zip(
                [f"{i:09d}" for i in row_id],
                range(0, len(row_id)),
            )
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
    file_list: list,
    out_path: str,
    filters: list,
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

    """
    pqwriter = None
    for i, df in enumerate(file_list):
        table = pq.read_table(
            df,
            filters=filters,
            use_pandas_metadata=True,
        )
        # for the first chunk of records
        if i == 0:
            # create a parquet write object giving it an output file
            pqwriter = pq.ParquetWriter(
                out_path,
                table.schema,
            )
        pqwriter.write_table(table)

    # close the parquet writer
    if pqwriter:
        pqwriter.close()
