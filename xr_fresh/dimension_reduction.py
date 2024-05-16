# %%
import numba
import ray
import geowombat as gw
from geowombat.core.parallel import ParallelTask
from sklearn.decomposition import KernelPCA
import numpy as np
import xarray as xr
import logging

# Import the original GeoWombatAccessor class
from geowombat.core.geoxarray import GeoWombatAccessor


# Extend the GeoWombatAccessor class
class ExtendedGeoWombatAccessor(GeoWombatAccessor):

    def k_pca(
        self,
        gamma: float,
        n_component: int,
        n_workers: int,
        chunk_size: int,
    ) -> xr.DataArray:
        """
        Applies Kernel PCA to the dataset and returns a DataArray with the components as bands.

        Args:
            gamma (float): The gamma parameter for the RBF kernel.
            n_component (int): The number of component that will be kept
            n_workers (int): The number of parallel jobs for KernelPCA and ParallelTask.
            chunk_size (int): The size of the chunks for processing.

        Returns:
            xr.DataArray: A DataArray with the Kernel PCA components as bands.

        Examples:
        # Initialize Ray
        with ray.init(num_cpus=8) as rays:


            # Example usage
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
                # get third k principal components - base zero counting
                transformed_dataarray = src.gw_ext.k_pca(
                    gamma=15, n_component=3, n_workers=8, chunk_size=256
                )
                transformed_dataarray.sel(component=3).plot()
        """

        # Transpose data to have shape (height, width, num_features)
        data = self._obj.transpose("y", "x", "band").values
        height, width, num_features = data.shape

        # Reshape data to 2D array
        transposed_data = data.reshape(-1, num_features)

        # Drop rows with NaNs
        transposed_data = transposed_data[~np.isnan(transposed_data).any(axis=1)]

        # Sample data for fitting Kernel PCA
        num_samples = 10000
        np.random.seed(42)  # For reproducibility
        sampled_features = transposed_data[
            np.random.choice(transposed_data.shape[0], num_samples, replace=False)
        ]

        # Fit Kernel PCA on the sampled features
        kpca = KernelPCA(
            kernel="rbf", gamma=gamma, n_components=n_component + 1, n_jobs=n_workers
        )
        kpca.fit(sampled_features)

        # Extract necessary attributes from kpca for transformation
        X_fit_ = kpca.X_fit_
        eigenvectors = kpca.eigenvectors_[:, n_component - 1]
        eigenvalues = kpca.eigenvalues_[n_component - 1]

        @numba.jit(nopython=True, parallel=True)
        def transform_entire_dataset_numba(
            data, X_fit_, eigenvector, eigenvalue, gamma
        ):
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
            num_workers=n_workers,
        ):
            data_chunk = data_block_id[
                data_slice
            ].data.compute()  # Convert Dask array to NumPy array
            return transform_entire_dataset_numba(
                data_chunk, X_fit_, eigenvector, eigenvalue, gamma
            )

        # Perform transformation in parallel
        pt = ParallelTask(
            self._obj,
            row_chunks=chunk_size,
            col_chunks=chunk_size,
            scheduler="ray",
            n_workers=n_workers,
        )

        # Map the process_window function to each chunk of the dataset
        futures = pt.map(process_window, X_fit_, eigenvectors, eigenvalues, gamma)

        # Combine the results
        transformed_data = np.zeros((height, width, 1), dtype=np.float64)

        # Combine the results
        transformed_data = np.zeros((height, width))
        for window_id, future in enumerate(ray.get(futures)):
            window = pt.windows[window_id]
            row_start, col_start = window.row_off, window.col_off
            row_end, col_end = row_start + window.height, col_start + window.width
            transformed_data[row_start:row_end, col_start:col_end] = future

        # extend dimension of transformed_data
        if len(transformed_data.shape) == 2:
            transformed_data = np.expand_dims(transformed_data, axis=2)

        # Create a new DataArray with the transformed data
        transformed_dataarray = xr.DataArray(
            transformed_data,
            dims=("y", "x", "component"),
            coords={
                "y": self._obj.y,
                "x": self._obj.x,
                "component": [
                    n_component
                ],  # [f"component_{i+1}" for i in range(n_components)],
            },
            attrs=self._obj.attrs,
        )

        return transformed_dataarray


# Register the new accessor
xr.register_dataarray_accessor("gw_ext")(ExtendedGeoWombatAccessor)

# %%
