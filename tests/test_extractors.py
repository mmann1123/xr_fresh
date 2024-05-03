import unittest
from unittest.mock import patch, MagicMock
from xr_fresh.extractors import extract_features, check_dictionary
import xarray as xr

# class TestExtractFeatures(unittest.TestCase):
#     @patch('extractors._apply_fun_name')
#     @patch('extractors.xr')
#     def test_extract_features(self, mock_xr, mock_apply_fun_name):
#         # Mock DataArray
#         mock_dataarray = MagicMock()
#         mock_xr.DataArray.return_value = mock_dataarray

#         # Define input parameters
#         xr_data = mock_dataarray
#         feature_dict = {
#             "maximum": [{}],
#             "quantile": [{"q": "0.5"}, {"q": "0.95"}]
#         }
#         band = "B2"
#         na_rm = False
#         filepath = None
#         postfix = ""
#         dim = "variable"
#         persist = False

#         # Call the function
#         result = extract_features(xr_data, feature_dict, band, na_rm, filepath, postfix, dim, persist)

#         # Assertions
#         self.assertEqual(mock_xr.DataArray.call_count, 0)
#         self.assertEqual(mock_apply_fun_name.call_count, 3)
#         self.assertIsNone(result)

#     def test_check_dictionary(self):
#         # Define input parameters
#         arguments = {
#             "maximum": [{}],
#             "quantile": []
#         }

#         # Call the function
#         with self.assertWarns(UserWarning):
#             check_dictionary(arguments)

if __name__ == '__main__':
    unittest.main()
