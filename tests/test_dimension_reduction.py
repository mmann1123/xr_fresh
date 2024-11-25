import unittest
import geowombat as gw
from xr_fresh.dimension_reduction import ExtendedGeoWombatAccessor
import ray
import numpy as np


class TestDimensionReduction(unittest.TestCase):

    def setUp(self):
        # Initialize Ray
        ray.init(num_cpus=8)

    def tearDown(self):
        # Shutdown Ray
        ray.shutdown()

    def test_k_pca(self):
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
                gamma=15, n_components=3, n_workers=8, chunk_size=256
            )

            # Check the shape of the transformed data
            self.assertEqual(transformed_dataarray.shape, (src.y.size, src.x.size, 3))

            # Check the attributes of the transformed data
            self.assertEqual(transformed_dataarray.attrs["crs"], src.attrs["crs"])
            self.assertEqual(
                transformed_dataarray.attrs["transform"], src.attrs["transform"]
            )

    def test_k_pca_invalid_gamma(self):
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
            gamma = -1
            n_components = 3
            n_workers = 8
            chunk_size = 256
            with self.assertRaises(ValueError):
                src.gw_ext.k_pca(
                    gamma=gamma,
                    n_components=n_components,
                    n_workers=n_workers,
                    chunk_size=chunk_size,
                )

    def test_k_pca_no_equal_components(self):
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
            gamma = 15
            n_components = 3
            n_workers = 8
            chunk_size = 256
            transformed_dataarray = src.gw_ext.k_pca(
                gamma=gamma,
                n_components=n_components,
                n_workers=n_workers,
                chunk_size=chunk_size,
            )

            for comp in transformed_dataarray.component.values:
                component_data = transformed_dataarray.sel(component=comp).values
                unique_values = np.unique(component_data)
                self.assertGreater(
                    len(unique_values), 1, f"Component {comp} has all equal values"
                )


if __name__ == "__main__":
    unittest.main()
