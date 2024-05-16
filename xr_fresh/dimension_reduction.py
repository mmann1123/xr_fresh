# %%

import geowombat as gw
from glob import glob
import os
import itertools
import numpy as np
import numba
from sklearn.decomposition import KernelPCA, PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile

os.getcwd()
with gw.open(
    sorted(glob("./tests/data/R*.tif")), stack_dim="band", band_names=[0, 1, 2, 3]
) as src:
    print(src)
    print(src.values.shape)
    src.sel(band=0).gw.imshow()
# %%

num_features, height, width = src.shape
data = src.values

transposed_data = np.transpose(data, (1, 2, 0))
features = transposed_data.reshape(-1, num_features)

# # drop rows with nans
features = features[~np.isnan(features).any(axis=1)]
features
# %%

# Number of random coordinates to select
num_samples = 10000

# Generate random coordinates
np.random.seed(42)  # For reproducibility

# select num samples rows
sampled_features = features[
    np.random.choice(features.shape[0], num_samples, replace=False)
]

# Fit Kernel PCA on the sampled features
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=4, n_jobs=-1)
kpca.fit(sampled_features)

# Extract necessary attributes from kpca for transformation
X_fit_ = kpca.X_fit_
eigenvector = kpca.eigenvectors_[:, 3]  # Take the first component
eigenvalue = kpca.eigenvalues_[3]
gamma = kpca.gamma
# %%


@numba.jit(nopython=True, parallel=True)
def transform_entire_dataset_numba(data, X_fit_, eigenvector, eigenvalue, gamma):
    num_features, height, width = data.shape
    transformed_data = np.zeros((height, width))

    for i in numba.prange(height):
        for j in range(width):
            feature_vector = data[:, i, j]
            k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
            transformed_feature = np.dot(k, eigenvector / np.sqrt(eigenvalue))
            transformed_data[i, j] = transformed_feature

    return transformed_data


# Apply transformation to the entire datasetaset
cProfile.run(
    "transform_entire_dataset_numba(data, X_fit_, eigenvector, eigenvalue, gamma)"
)

# %%
# Apply transformation to the entire dataset
transformed_data = transform_entire_dataset_numba(
    data, X_fit_, eigenvector, eigenvalue, gamma
)
# Reshape to (4000, 3000, 1)
transformed_data = transformed_data.reshape(height, width, 1)

# Plot the transformed data for visualization
plt.imshow(transformed_data[:, :, 0], cmap="viridis")
plt.colorbar()
plt.title("Transformed Data")
plt.show()

# %% apply is slower


# class transform_PCA_kernel(gw.TimeModule):
#     """
#     Returns the absolute energy of the time series which is the sum over the squared values

#     Args:
#         gw (_type_): _description_
#     """

#     def __init__(self, X_fit_=None, eigenvector=None, eigenvalue=None, gamma=None):
#         super(transform_PCA_kernel, self).__init__()
#         self.X_fit_ = X_fit_
#         self.eigenvector = eigenvector
#         self.eigenvalue = eigenvalue
#         self.gamma = gamma

#     def calculate(self, data):
#         data = data[:, 0, :, :]
#         num_features, height, width = data.shape
#         transformed_data = np.zeros((height, width))

#         for i in numba.prange(height):
#             for j in range(width):
#                 feature_vector = data[:, i, j]
#                 k = np.exp(
#                     -self.gamma * np.sum((feature_vector - self.X_fit_) ** 2, axis=1)
#                 )
#                 transformed_feature = np.dot(
#                     k, self.eigenvector / np.sqrt(self.eigenvalue)
#                 )
#                 transformed_data[i, j] = transformed_feature

#         return transformed_data


# with gw.series(sorted(glob("./tests/data/R*.tif"))) as src:
#     src.apply(
#         func=transform_PCA_kernel(
#             X_fit_=X_fit_,
#             eigenvector=eigenvector,
#             eigenvalue=eigenvalue,
#             gamma=gamma,
#         ),
#         outfile="./transformed.tif",
#         bands=1,
#     )

# # %%
# ray.init()

# # %%
# import ray


# # @numba.jit(nopython=True, parallel=True)
# @ray.remote
# def transform_entire_dataset_numba(
#     data, X_fit_, eigenvector, eigenvalue, gamma, height, width
# ):
#     data = data.values
#     # num_features, height, width = data.shape
#     transformed_data = np.zeros((height, width))

#     for i in numba.prange(height):
#         for j in range(width):
#             feature_vector = data[:, i, j]
#             k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
#             transformed_feature = np.dot(k, eigenvector / np.sqrt(eigenvalue))
#             transformed_data[i, j] = transformed_feature

#     return transformed_data


# # Open the raster file with GeoWombat
# with gw.open(
#     glob("./tests/data/R*.tif"), stack_dim="band", band_names=[0, 1, 2, 3]
# ) as src:

#     w = src.gw.windows(256, 256)
#     print(w)
#     for i in w:
#         print(i)
#         print(type(i))
#         src.read(1, window=i)

#     # Define the window size
#     window_size = (256, 256)  # Example window size
#     height, width = (256, 256)
#     # Get the dimensions of the raster
#     height, width = src.gw.nrows, src.gw.ncols

#     # Create a list to store Ray tasks
#     ray_tasks = []

#     for w in src.gw.windows(256, 256):

#         # Submit the Ray task
#         ray_tasks.append(
#             transform_entire_dataset_numba.remote(
#                 w, X_fit_, eigenvector, eigenvalue, gamma, height, width
#             )
#         )
#     # Get the results from Ray tasks
#     results = ray.get(ray_tasks)

#     # Process the results (e.g., print the mean values)
#     for result in results:
#         print(result)

# # %%
# ray.shutdown()

# # %%
