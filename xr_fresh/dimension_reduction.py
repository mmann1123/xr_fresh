# %%
# note alternative outlined here: https://github.com/jgrss/geowombat/discussions/318
import numba
import ray
import geowombat as gw
from glob import glob
from geowombat.core.parallel import ParallelTask
from sklearn.decomposition import KernelPCA
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import cProfile

with gw.open(
    sorted(
        [
            "./tests/data/RadT_tavg_202301.tif",
            "./tests/data/RadT_tavg_202302.tif",
            "./tests/data/RadT_tavg_202304.tif",
            "./tests/data/RadT_tavg_202305.tif",
        ]
    ),
    stack_dim="band",
    band_names=[0, 1, 2, 3],
) as src:
    print(src)
    print(src.values.shape)
    src.sel(band=0).gw.imshow()

    num_features, height, width = src.shape
    data = src.values

transposed_data = np.transpose(data, (1, 2, 0))
features = transposed_data.reshape(-1, num_features)

# # drop rows with nans
features = features[~np.isnan(features).any(axis=1)]
features

# Number of random coordinates to select
num_samples = 10000

# Generate random coordinates
np.random.seed(42)  # For reproducibility

# select num samples rows
sampled_features = features[
    np.random.choice(features.shape[0], num_samples, replace=False)
]

# Fit Kernel PCA on the sampled features
print("fitting kpca")
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=4, n_jobs=-1)
kpca.fit(sampled_features)

# Extract necessary attributes from kpca for transformation
X_fit_ = kpca.X_fit_
eigenvector = kpca.eigenvectors_[:, 3]  # Take the first component
eigenvalue = kpca.eigenvalues_[3]
gamma = kpca.gamma
# Extract necessary attributes from kpca for transformation
X_fit_ = kpca.X_fit_
eigenvector = kpca.eigenvectors_[:, 0]  # Take the first component
eigenvalue = kpca.eigenvalues_[0]
gamma = kpca.gamma

# %%


@numba.jit(nopython=True, parallel=True)
def transform_entire_dataset_numba(data, X_fit_, eigenvector, eigenvalue, gamma):
    height, width = data.shape[1], data.shape[2]
    transformed_data = np.zeros((height, width))

    for i in numba.prange(height):
        for j in range(width):
            feature_vector = data[:, i, j]
            k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
            transformed_feature = np.dot(k, eigenvector / np.sqrt(eigenvalue))
            transformed_data[i, j] = transformed_feature

    return transformed_data


@ray.remote
def process_window(
    data_block_id,
    data_slice,
    window_id,
    X_fit_,
    eigenvector,
    eigenvalue,
    gamma,
    num_workers,
):
    data_chunk = data_block_id[
        data_slice
    ].data.compute()  # Convert Dask array to NumPy array
    return transform_entire_dataset_numba(
        data_chunk, X_fit_, eigenvector, eigenvalue, gamma
    )


# Open the raster file with GeoWombat
with gw.open(
    sorted(
        [
            "./tests/data/RadT_tavg_202301.tif",
            "./tests/data/RadT_tavg_202302.tif",
            "./tests/data/RadT_tavg_202304.tif",
            "./tests/data/RadT_tavg_202305.tif",
        ]
    ),
    stack_dim="band",
    band_names=[0, 1, 2, 3],
) as src:
    pt = ParallelTask(src, row_chunks=256, col_chunks=256, scheduler="ray", n_workers=8)

    # Map the process_window function to each chunk of the dataset
    futures = pt.map(process_window, X_fit_, eigenvector, eigenvalue, gamma, 8)


# Combine the results
transformed_data = np.zeros((height, width))
for window_id, future in enumerate(ray.get(futures)):
    window = pt.windows[window_id]
    row_start, col_start = window.row_off, window.col_off
    row_end, col_end = row_start + window.height, col_start + window.width
    transformed_data[row_start:row_end, col_start:col_end] = future
