import unittest
from unittest.mock import patch, MagicMock
from xr_fresh.extractors_series import extract_features_series



#@patch: This decorator is used to replace an object with a mock object for the duration of a test function. 
#        It allows you to temporarily replace the specified object (usually a function or class) with a mock version. 
#        When the test function finishes, the original object is restored. This is useful for isolating the behavior of a specific function 
#        or method without affecting other parts of the code. In the provided example, @patch is used to mock the xr module 
#        and the _apply_fun_name function during the execution of the test cases.

#MagicMock: This is a subclass of Mock that provides default implementations of all magic methods. It allows you to 
#           create mock objects with minimal setup, and it automatically generates stubs for all attribute accesses and method calls. 
#           MagicMock objects are highly flexible and can be used to simulate the behavior of any object or function. 
#           In the provided example, MagicMock is used to create mock objects for the xarray.DataArray class and its methods, 
#           which are then used as the return values of the mocked xr module. This allows the test cases to simulate the behavior of 
#           xarray.DataArray objects without actually creating them.


# class TestExtractFeaturesSeries(unittest.TestCase):
#     @patch('extractors_series.Path')
#     @patch('extractors_series.logging')
#     @patch('extractors_series.gw')
#     def test_extract_features_series(self, mock_gw, mock_logging, mock_path):
#         # Mock dataset and series
#         mock_series = MagicMock()
#         mock_gw.series.return_value = mock_series

#         # Define input parameters
#         gw_series = mock_series
#         feature_dict = {
#             "abs_energy": [{}],
#             "autocorr": [{"lag": 1}],
#             "mean": [{}]
#         }
#         band_name = "B2"
#         output_dir = "output"

#         # Call the function
#         extract_features_series(gw_series, feature_dict, band_name, output_dir)

#         # Assertions
#         mock_path.assert_called_once_with(output_dir)
#         mock_path().mkdir.assert_called_once_with(parents=True, exist_ok=True)
#         self.assertEqual(mock_gw.series.call_count, 1)
#         self.assertEqual(mock_series.apply.call_count, 3)
#         self.assertEqual(mock_logging.error.call_count, 0)

#     @patch('extractors_series.Path')
#     @patch('extractors_series.logging')
#     @patch('extractors_series.gw')
#     def test_extract_features_series_with_exception(self, mock_gw, mock_logging, mock_path):
#         # Mock dataset and series
#         mock_series = MagicMock()
#         mock_gw.series.return_value = mock_series
#         mock_series.apply.side_effect = Exception("Test exception")

#         # Define input parameters
#         gw_series = mock_series
#         feature_dict = {
#             "abs_energy": [{}],
#         }
#         band_name = "B2"
#         output_dir = "output"

#         # Call the function
#         extract_features_series(gw_series, feature_dict, band_name, output_dir)

#         # Assertions
#         mock_logging.error.assert_called_once()
#         self.assertEqual(mock_gw.series.call_count, 1)
#         self.assertEqual(mock_series.apply.call_count, 1)

if __name__ == '__main__':
    unittest.main()
