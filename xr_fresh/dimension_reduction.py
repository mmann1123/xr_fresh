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
        n_components: int,
        n_workers: int,
        chunk_size: int,
    ) -> xr.DataArray:
        """
        Applies Kernel PCA to the dataset and returns a DataArray with the components as bands.

        Args:
            gamma (float): The gamma parameter for the RBF kernel.
            n_components (int): The number of components to keep.
            n_workers (int): The number of parallel jobs for KernelPCA and ParallelTask.
            chunk_size (int): The size of the chunks for processing.

        Returns:
            xr.DataArray: A DataArray with the Kernel PCA components as bands.


        Examples:
        .. code-block:: python

            import xr_fresh.dimension_reduction  # This registers the accessor

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
                    # get 3 k principal components - base zero counting
                    transformed_dataarray = src.gw_ext.k_pca(
                        gamma=15, n_components=3, n_workers=8, chunk_size=256
                    )
                    transformed_dataarray.plot.imshow(col='component', col_wrap=1, figsize=(8, 12))
                    plt.show()

        """

        # Transpose data to have shape (num_features, height, width)
        data = self._obj.transpose("band", "y", "x").values
        num_features, height, width = data.shape

        # Reshape data to 2D array (pixels, features)
        transposed_data = data.reshape(num_features, -1).T

        # Drop rows with NaNs
        valid_indices = ~np.isnan(transposed_data).any(axis=1)
        transposed_data_valid = transposed_data[valid_indices]

        # Sample data for fitting Kernel PCA
        num_samples = min(10000, transposed_data_valid.shape[0])
        np.random.seed(42)  # For reproducibility
        sampled_indices = np.random.choice(
            transposed_data_valid.shape[0], num_samples, replace=False
        )
        sampled_features = transposed_data_valid[sampled_indices]

        # Fit Kernel PCA on the sampled features
        kpca = KernelPCA(
            kernel="rbf", gamma=gamma, n_components=n_components, n_jobs=n_workers
        )
        kpca.fit(sampled_features)

        # Extract necessary attributes from kpca for transformation
        X_fit_ = kpca.X_fit_
        eigenvectors = kpca.eigenvectors_
        eigenvalues = kpca.eigenvalues_

        @numba.jit(nopython=True, parallel=True)
        def transform_entire_dataset_numba(
            data, X_fit_, eigenvectors, eigenvalues, gamma
        ):
            num_features, height, width = data.shape
            n_components = eigenvectors.shape[1]
            transformed_data = np.zeros((height, width, n_components))

            for i in numba.prange(height):
                for j in range(width):
                    feature_vector = data[:, i, j]
                    if np.isnan(feature_vector).any():
                        transformed_data[i, j, :] = np.nan
                        continue
                    k = np.exp(-gamma * np.sum((feature_vector - X_fit_) ** 2, axis=1))
                    for c in range(n_components):
                        transformed_feature = np.dot(
                            k, eigenvectors[:, c] / np.sqrt(eigenvalues[c])
                        )
                        transformed_data[i, j, c] = transformed_feature

            return transformed_data

        @ray.remote
        def process_window(
            data_block_id,
            data_slice,
            window_id,
            X_fit_,
            eigenvectors,
            eigenvalues,
            gamma,
        ):
            data_chunk = data_block_id[data_slice].data.compute()
            return transform_entire_dataset_numba(
                data_chunk, X_fit_, eigenvectors, eigenvalues, gamma
            )

        # Perform transformation in parallel
        pt = ParallelTask(
            self._obj.transpose("band", "y", "x"),
            row_chunks=chunk_size,
            col_chunks=chunk_size,
            scheduler="ray",
            n_workers=n_workers,
        )

        # Map the process_window function to each chunk of the dataset
        futures = pt.map(process_window, X_fit_, eigenvectors, eigenvalues, gamma)

        # Combine the results
        transformed_data = np.zeros((height, width, n_components), dtype=np.float64)
        results = ray.get(futures)
        for window_id, future in enumerate(results):
            window = pt.windows[window_id]
            row_start, col_start = window.row_off, window.col_off
            row_end, col_end = row_start + window.height, col_start + window.width
            transformed_data[row_start:row_end, col_start:col_end, :] = future

        # Create a new DataArray with the transformed data
        transformed_dataarray = xr.DataArray(
            transformed_data,
            dims=("y", "x", "component"),
            coords={
                "y": self._obj.y,
                "x": self._obj.x,
                "component": [f"component_{i+1}" for i in range(n_components)],
            },
            attrs=self._obj.attrs,
        )

        # add chunksize
        chunk_size = transformed_dataarray.gw.check_chunksize(
            512, transformed_dataarray.gw.ncols
        )

        return transformed_dataarray.chunk(chunk_size)


# Register the new accessor
xr.register_dataarray_accessor("gw_ext")(ExtendedGeoWombatAccessor)
