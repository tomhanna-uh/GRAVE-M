import unittest
import json
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

class TestCalculateMissingAtes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Extract the function from the notebook
        with open('tft_alba_02272026.ipynb', 'r') as f:
            notebook = json.load(f)

        target_code = None
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if 'def calculate_missing_ates' in source and 'ROBUST ATE ESTIMATION FOR MISSING VARIABLES' in source:
                    target_code = source
                    break

        if target_code is None:
            raise ValueError("Could not find calculate_missing_ates in the notebook")

        # Execute the code in a new namespace dictionary
        cls.namespace = {}
        # We need to provide the necessary imports for the exec to work
        exec_code = f"""
import pandas as pd
import numpy as np
import torch
import os
from datetime import datetime
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting.metrics import RMSE

{target_code}
"""
        exec(exec_code, cls.namespace)

        # Attach the extracted function to the class
        cls.calculate_missing_ates = staticmethod(cls.namespace['calculate_missing_ates'])

    def test_extracted_function_exists(self):
        self.assertTrue(callable(self.calculate_missing_ates))

    def test_robust_state_dict_filtering(self):
        # We mock TimeSeriesDataSet and TemporalFusionTransformer inside the namespace
        mock_tft = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.load_state_dict.return_value = ([], []) # (missing_keys, unexpected_keys)
        mock_tft.from_dataset.return_value = mock_model_instance

        mock_ts_dataset = MagicMock()

        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # We will inject these mocks into the namespace where the function runs
        original_tft = self.namespace['TemporalFusionTransformer']
        original_ts_dataset = self.namespace['TimeSeriesDataSet']
        original_pl = self.namespace['pl']

        self.namespace['TemporalFusionTransformer'] = mock_tft
        self.namespace['TimeSeriesDataSet'] = mock_ts_dataset
        self.namespace['pl'].Trainer = mock_trainer_cls

        try:
            # 1. Setup minimal dummy data
            safe_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'year': list(range(2000, 2015)),
                'v2x_libdem': np.random.rand(15), # Need a target to process
                'alba_member': ['0'] * 15
            })

            causal_sub_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'time_idx': list(range(2000, 2015)),
                'propensity_score': np.random.rand(15),
                'alba_member': ['0'] * 15
            })

            # 2. Setup mock best_tft_model with a specific state_dict
            mock_best_tft_model = MagicMock()
            mock_best_tft_model.state_dict.return_value = {
                'keep_layer.weight': 'keep_value_1',
                'another_keep_layer.bias': 'keep_value_2',
                'output_layer.weight': 'skip_value_1',
                'variable_selection.module.weight': 'skip_value_2',
                'input_embeddings.time_idx.weight': 'skip_value_3'
            }

            # Since we just want to test state_dict filtering, we can mock predict to return something small
            mock_pred = MagicMock()
            mock_pred.cpu.return_value.numpy.return_value.flatten.return_value = np.zeros(15)
            mock_model_instance.predict.return_value = mock_pred

            # Prevent os.path.exists and to_csv from actually hitting the disk in testing
            original_os_path_exists = self.namespace['os'].path.exists
            original_to_csv = self.namespace['pd'].DataFrame.to_csv

            self.namespace['os'].path.exists = MagicMock(return_value=False)
            self.namespace['pd'].DataFrame.to_csv = MagicMock()

            try:
                import sys
                from io import StringIO
                captured_output = StringIO()
                original_stdout = sys.stdout
                try:
                    sys.stdout = captured_output
                    # Execute the function
                    self.calculate_missing_ates(safe_data, mock_best_tft_model, causal_sub_data, 2)
                finally:
                    sys.stdout = original_stdout
            finally:
                self.namespace['os'].path.exists = original_os_path_exists
                self.namespace['pd'].DataFrame.to_csv = original_to_csv

            # Assert load_state_dict was called
            self.assertTrue(mock_model_instance.load_state_dict.called)

            # Get the state_dict that was passed to load_state_dict
            args, kwargs = mock_model_instance.load_state_dict.call_args
            new_state = args[0]

            # Assert the filtering worked correctly
            self.assertIn('keep_layer.weight', new_state)
            self.assertIn('another_keep_layer.bias', new_state)

            self.assertNotIn('output_layer.weight', new_state)
            self.assertNotIn('variable_selection.module.weight', new_state)
            self.assertNotIn('input_embeddings.time_idx.weight', new_state)

            # Verify strict=False was passed
            self.assertEqual(kwargs.get('strict'), False)

        finally:
            # Restore the namespace to avoid side effects for other tests
            self.namespace['TemporalFusionTransformer'] = original_tft
            self.namespace['TimeSeriesDataSet'] = original_ts_dataset
            self.namespace['pl'] = original_pl

    def test_processing_loop_exception_handling(self):
        # We mock TimeSeriesDataSet to raise an Exception
        mock_ts_dataset = MagicMock(side_effect=Exception("Simulated TimeSeriesDataSet Error"))

        # We also need a mock TFT model just to pass to the function
        mock_best_tft_model = MagicMock()

        original_ts_dataset = self.namespace['TimeSeriesDataSet']
        self.namespace['TimeSeriesDataSet'] = mock_ts_dataset

        try:
            # 1. Setup dummy data with multiple targets
            safe_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'year': list(range(2000, 2015)),
                'v2x_libdem': np.random.rand(15),
                'fraser_bmp_score': np.random.rand(15), # Second target
                'alba_member': ['0'] * 15
            })

            causal_sub_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'time_idx': list(range(2000, 2015)),
                'propensity_score': np.random.rand(15),
                'alba_member': ['0'] * 15
            })

            # Use patch to capture stdout so we can verify the error was caught and printed
            import sys
            from io import StringIO
            captured_output = StringIO()
            original_stdout = sys.stdout

            try:
                sys.stdout = captured_output

                # Execute the function. If exception handling works, this will not raise an exception
                self.calculate_missing_ates(safe_data, mock_best_tft_model, causal_sub_data, 2)

            finally:
                sys.stdout = original_stdout

            output = captured_output.getvalue()

            # Verify the output indicates both targets were attempted and failed gracefully
            self.assertIn("--- Processing: v2x_libdem ---", output)
            self.assertIn("Failed processing v2x_libdem: Simulated TimeSeriesDataSet Error", output)

            self.assertIn("--- Processing: fraser_bmp_score ---", output)
            self.assertIn("Failed processing fraser_bmp_score: Simulated TimeSeriesDataSet Error", output)

        finally:
            self.namespace['TimeSeriesDataSet'] = original_ts_dataset

    def test_successful_execution_flow(self):
        # Mocks
        mock_tft = MagicMock()
        mock_model_instance = MagicMock()
        # Mock load_state_dict to return empty missing/unexpected
        mock_model_instance.load_state_dict.return_value = ([], [])

        # We need mock_model_instance.predict to return something we can iterate or use `.cpu().numpy().flatten()` on
        mock_pred = MagicMock()
        # Return a list or array that has len=15 (to match dummy data)
        # Note: the dataset size after `filter for survivors (min 15 years)` is going to be 15
        mock_pred_tensor = MagicMock()
        mock_pred_tensor.numpy.return_value.flatten.return_value = np.array([0.5] * 15)
        mock_pred.cpu.return_value = mock_pred_tensor
        mock_model_instance.predict.return_value = mock_pred

        mock_tft.from_dataset.return_value = mock_model_instance

        mock_ts_dataset_cls = MagicMock()
        mock_ts_dataset_instance = MagicMock()

        # We also need TimeSeriesDataSet.from_dataset
        mock_ts_dataset_cls.from_dataset.return_value = mock_ts_dataset_instance
        mock_ts_dataset_cls.return_value = mock_ts_dataset_instance

        # We need to mock the index attribute of the dataset to be a DataFrame for merging
        mock_index_df = pd.DataFrame({
            'COWcode': ['100'] * 15,
            'time_idx': list(range(2000, 2015))
        })
        mock_ts_dataset_instance.index = mock_index_df

        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        original_tft = self.namespace['TemporalFusionTransformer']
        original_ts_dataset = self.namespace['TimeSeriesDataSet']
        original_pl = self.namespace['pl']

        self.namespace['TemporalFusionTransformer'] = mock_tft
        self.namespace['TimeSeriesDataSet'] = mock_ts_dataset_cls
        self.namespace['pl'].Trainer = mock_trainer_cls

        original_os_path_exists = self.namespace['os'].path.exists
        original_to_csv = self.namespace['pd'].DataFrame.to_csv
        original_read_csv = self.namespace['pd'].read_csv

        self.namespace['os'].path.exists = MagicMock(return_value=False)
        self.namespace['pd'].DataFrame.to_csv = MagicMock()
        self.namespace['pd'].read_csv = MagicMock()

        try:
            # Setup dummy data for one target
            safe_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'year': list(range(2000, 2015)),
                'v2x_libdem': np.random.rand(15),
                'alba_member': ['0'] * 15
            })

            causal_sub_data = pd.DataFrame({
                'COWcode': ['100'] * 15,
                'time_idx': list(range(2000, 2015)),
                'propensity_score': np.random.rand(15),
                'alba_member': ['0'] * 15
            })

            mock_best_tft_model = MagicMock()
            mock_best_tft_model.state_dict.return_value = {}

            import sys
            from io import StringIO
            captured_output = StringIO()
            original_stdout = sys.stdout

            try:
                sys.stdout = captured_output
                # Execute the function
                self.calculate_missing_ates(safe_data, mock_best_tft_model, causal_sub_data, 2)
            finally:
                sys.stdout = original_stdout

            # Assert to_csv was called once (since there's one target)
            self.assertTrue(self.namespace['pd'].DataFrame.to_csv.called)

            # Verify it's saving to "mediation_master_summary.csv"
            args, kwargs = self.namespace['pd'].DataFrame.to_csv.call_args
            self.assertEqual(args[0], "mediation_master_summary.csv")
            self.assertFalse(kwargs.get('index'))

        finally:
            self.namespace['TemporalFusionTransformer'] = original_tft
            self.namespace['TimeSeriesDataSet'] = original_ts_dataset
            self.namespace['pl'] = original_pl

            self.namespace['os'].path.exists = original_os_path_exists
            self.namespace['pd'].DataFrame.to_csv = original_to_csv
            self.namespace['pd'].read_csv = original_read_csv

if __name__ == '__main__':
    unittest.main()
