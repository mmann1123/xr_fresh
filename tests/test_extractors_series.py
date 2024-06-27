# import unittest
# from xr_fresh.extractors_series import extract_features_series, extract_key_value_names, extract_grid
# from pathlib import Path
# import shutil
# import os

# # Define a test class for extractors_series.py
# class TestExtractorsSeries(unittest.TestCase):
#     # Setup method to run before each test
#     def setUp(self):
#         # Define a geospatial time series dataset for testing
#         # You'll need to replace this with your actual geowombat.Dataset object
#         self.gw_series = ...

#         # Define a feature dictionary for testing
#         self.feature_dict = {
#             "abs_energy": [{}],
#             "absolute_sum_of_changes": [{}],
#             "autocorr": [{"lag": 1}, {"lag": 2}, {"lag": 3}],
#             # Add more features for testing
#         }

#         # Define a band name for testing
#         self.band_name = "B2"

#         # Create an output directory for testing
#         self.output_directory = "../test_output"
#         Path(self.output_directory).mkdir(parents=True, exist_ok=True)

#     # Teardown method to run after each test
#     def tearDown(self):
#         # Clean up the output directory after each test
#         if os.path.exists(self.output_directory):
#             shutil.rmtree(self.output_directory)

#     # Test the extract_key_value_names function
#     def test_extract_key_value_names(self):
#         band_name = "B2_key_name_value_name_grid_512"
#         key_names, value_names = extract_key_value_names(band_name)
#         self.assertEqual(key_names, "name")
#         self.assertEqual(value_names, "name")

#     # Test the extract_grid function
#     def test_extract_grid(self):
#         band_name = "B2_key_name_value_name_grid_512"
#         grid = extract_grid(band_name)
#         self.assertEqual(grid, "512")

#     # Test the extract_features_series function
#     def test_extract_features_series(self):
#         # Call the extract_features_series function
#         extract_features_series(self.gw_series, self.feature_dict, self.band_name, self.output_directory)

#         # Assert that the output files are created
#         output_path = Path(self.output_directory)
#         for feature_name in self.feature_dict.keys():
#             for params in self.feature_dict[feature_name]:
#                 key_names, value_names = extract_key_value_names(self.band_name)
#                 grid = extract_grid(self.band_name)
#                 output_file = output_path / f"{self.band_name}_{feature_name}_{key_names}_{value_names}_{grid}.tif"
#                 self.assertTrue(output_file.exists())

# if __name__ == "__main__":
#     unittest.main()
