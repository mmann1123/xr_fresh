import dask.array as da
from pathlib import Path
from .transformers import Stackerizer
import xarray as xr


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

    def __setitem__(self, key, item):

        if len(key) == 4:
            t, b, y, x = key
        elif len(key) == 3:
            b, y, x = key
        else:
            y, x = key

        out_filename = (
            self.parent
            # / f"{self.stem}_y{y.start:09d}_x{x.start:09d}_h{y.stop - y.start:09d}_w{x.stop - x.start:09d}{self.suffix}"
            / f"{self.stem}_row{y.start:09d}{self.suffix}"
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
