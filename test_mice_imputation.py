import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
import json
import ast

def extract_notebook_function(notebook_path, function_signature):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if function_signature in source:
                # filter out magic commands
                filtered_source = "\n".join([line for line in source.split('\n') if not line.strip().startswith('!') and not line.strip().startswith('%')])
                # Remove if __name__ == "__main__": block
                lines = filtered_source.split('\n')
                main_index = -1
                for i, line in enumerate(lines):
                    if line.startswith('if __name__ == "__main__":'):
                        main_index = i
                        break
                if main_index != -1:
                    lines = lines[:main_index]
                filtered_source = "\n".join(lines)

                try:
                    ast.parse(filtered_source)
                    return filtered_source
                except SyntaxError:
                    continue
    return None

# Extract the function
source_code = extract_notebook_function('GRAVE_M_January_12_2026.ipynb', 'def run_mice_step_1')
if source_code:
    # We pass the globals() namespace to exec so that the extracted code
    # can access pd, np, os, sys, etc.
    notebook_namespace = globals().copy()
    exec(source_code, notebook_namespace)
else:
    raise RuntimeError("Could not find function run_mice_step_1 in notebook.")


class TestMiceStep1(unittest.TestCase):
    def setUp(self):
        self.test_input = "test_factors.csv"
        self.test_output = "test_imputed.csv"
        # We need to provide the target_cols and predictor_cols
        data = {
            "fraser_bmp_score": [5.0, np.nan, 6.0],
            "gini_disp": [40.0, np.nan, 42.0],
            "resource_rents": [np.nan, 10.0, 12.0],
            "log_oil_gas_wealth": [2.0, np.nan, 2.1],
            "unified_gdp_pc": [5000, 5200, np.nan],
            "unified_pop": [1e6, np.nan, 1.2e6],
            "year": [2000, 2001, 2002],
            "v2x_libdem": [0.5, 0.6, 0.55],
            "is_petro_state": [1, 1, 1],
            "unified_corruption": [0.5, 0.4, 0.6],
            "COWcode": [100, 100, 100]
        }
        pd.DataFrame(data).to_csv(self.test_input, index=False)

    def tearDown(self):
        if os.path.exists(self.test_input):
            os.remove(self.test_input)
        if os.path.exists(self.test_output):
            os.remove(self.test_output)

    def test_local_file_imputation(self):
        # Capture stdout to avoid noise
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            df_clean = notebook_namespace['run_mice_step_1'](input_filename=self.test_input, output_filename=self.test_output)

        target_cols = [
            "fraser_bmp_score", "gini_disp", "resource_rents",
            "log_oil_gas_wealth", "unified_gdp_pc", "unified_pop"
        ]

        for col in target_cols:
            self.assertEqual(df_clean[col].isna().sum(), 0, f"Column {col} has missing values")

        self.assertTrue(os.path.exists(self.test_output))

        # Verify derived columns are created
        self.assertIn("log_gdp_pc", df_clean.columns)
        self.assertIn("log_pop", df_clean.columns)

    def test_colab_upload_fallback(self):
        # Create a mock for google.colab
        colab_mock = MagicMock()
        colab_mock.files.upload.return_value = {self.test_input: b''}

        # We need to mock sys.modules to simulate Colab
        with patch.dict('sys.modules', {'google.colab': colab_mock}):
            # The function checks 'google.colab' in sys.modules inside run_mice_step_1
            # But it uses `files.upload()` from the global scope where it was defined.
            # So we must inject `files` into the notebook's namespace for this test
            notebook_namespace['files'] = colab_mock.files

            import io
            from contextlib import redirect_stdout
            with redirect_stdout(io.StringIO()):
                df_clean = notebook_namespace['run_mice_step_1'](input_filename="non_existent.csv", output_filename=self.test_output)

            colab_mock.files.upload.assert_called_once()
            self.assertEqual(df_clean["fraser_bmp_score"].isna().sum(), 0)

            # Clean up the injected mock
            del notebook_namespace['files']

    def test_file_not_found_local(self):
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(FileNotFoundError):
                notebook_namespace['run_mice_step_1'](input_filename="non_existent_local.csv", output_filename=self.test_output)

if __name__ == '__main__':
    unittest.main()
