import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
import json
import os
import ast
import torch
from datetime import datetime
from pytorch_forecasting.data import NaNLabelEncoder, TorchNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
import lightning.pytorch as pl

def extract_function_from_notebook(notebook_path, function_name):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_lines = cell['source']
            # Filter out IPython magic commands
            filtered_source_lines = [line for line in source_lines if not line.strip().startswith('!') and not line.strip().startswith('%')]
            source = "".join(filtered_source_lines)

            if f"def {function_name}" in source:
                # Basic syntax check to ensure it's valid Python
                try:
                    ast.parse(source)
                    return source
                except SyntaxError:
                    continue
    return None

source_code = extract_function_from_notebook('tft_alba_02272026_2.ipynb', 'run_causal_analysis')
if source_code:
    exec(source_code, globals())
else:
    raise RuntimeError("Could not find run_causal_analysis in notebook.")

@pytest.fixture
def dummy_safe_data():
    df = pd.DataFrame({
        'COWcode': ['100']*20 + ['200']*20,
        'year': list(range(1990, 2010)) * 2,
        'time_idx': list(range(1990, 2010)) * 2,
        'alba_member': [0]*15 + [1]*5 + [0]*20,
        'unified_gdp_pc': np.random.randn(40),
        'log_gdp_pc': np.random.randn(40),
        'unified_pop': np.random.randn(40),
        'resource_rents': np.random.randn(40),
        'gini_disp': np.random.randn(40),
        'v2x_libdem': np.random.randn(40),
        'fraser_bmp_score': np.random.randn(40),
        'is_petro_state': [0]*40,
        'is_aut_episode': [0]*40,
        'is_dem_episode': [1]*40,
        'mid_count_total': [0]*40,
        'mid_high_fatality_event': [0]*40,
        'is_leftist_leader': [0]*40,
        'is_rightist_leader': [0]*40,
        'gli_leader_ideology_num': [0]*40,
        'mid_max_fatality_cat': [0]*40,
        'mid_max_hostility': [0]*40,
    })
    return df

@pytest.fixture
def dummy_causal_sub_data():
    df = pd.DataFrame({
        'COWcode': ['100']*20 + ['200']*20,
        'time_idx': list(range(1990, 2010)) * 2,
        'propensity_score': np.random.uniform(0.1, 0.9, 40)
    })
    return df

@pytest.fixture
def dummy_tft_model():
    mock_model = MagicMock()
    mock_model.dataset_parameters = {
        "categorical_encoders": {}
    }
    mock_model.hparams = MagicMock()
    mock_model.hparams.hidden_size = 16
    mock_model.hparams.attention_head_size = 4
    mock_model.hparams.dropout = 0.1
    mock_model.hparams.hidden_continuous_size = 8
    mock_model.state_dict.return_value = {}
    return mock_model


@patch('os.path.exists')
@patch('pandas.read_csv')
@patch('pandas.DataFrame.to_csv')
@patch('lightning.pytorch.Trainer')
@patch('pytorch_forecasting.TemporalFusionTransformer.from_dataset')
@patch('builtins.print')
def test_data_preparation_and_target_forcing(mock_print, mock_tft, mock_trainer, mock_to_csv, mock_read_csv, mock_exists, dummy_safe_data, dummy_causal_sub_data, dummy_tft_model):
    # Simulate missing CSV so safe_data is used
    mock_exists.return_value = False

    # Mock model predict to avoid running actual prediction logic
    mock_model_instance = MagicMock()
    # Need to return valid tuple for get_preds helper
    # get_preds handles: ret.index, ret.output['prediction'] OR ret[-1], ret[0] OR ds.index, ret
    mock_idx = pd.DataFrame({'COWcode': ['100']*40, 'time_idx': list(range(1990, 2010))*2})
    mock_pred = torch.zeros((40, 1))
    mock_model_instance.predict.return_value = (mock_pred, mock_idx)
    mock_tft.return_value = mock_model_instance

    # Target is continuous/real for outcome analysis. Forcing dummy target to be numeric
    target = 'fraser_bmp_score'

    run_causal_analysis(target, dummy_safe_data, dummy_tft_model, dummy_causal_sub_data, n_bootstrap=2)

    # target forcing is handled via cols_to_fill in this notebook version
    # It converts the cols_to_fill (which includes outcome_target) to numeric.

    # Check that model was created
    assert mock_tft.called

@patch('os.path.exists')
@patch('pandas.DataFrame.to_csv')
@patch('lightning.pytorch.Trainer')
@patch('pytorch_forecasting.TemporalFusionTransformer.from_dataset')
def test_missing_target(mock_tft, mock_trainer, mock_to_csv, mock_exists, dummy_safe_data, dummy_causal_sub_data, dummy_tft_model, capsys):
    mock_exists.return_value = False

    # Target not in dummy_safe_data
    target = 'missing_target_col'

    run_causal_analysis(target, dummy_safe_data, dummy_tft_model, dummy_causal_sub_data, n_bootstrap=2)

    captured = capsys.readouterr()
    assert f"SKIP: Target {target} not found in dataset." in captured.out

    # Model should not be trained
    assert not mock_tft.called
    assert not mock_trainer.called

@patch('os.path.exists')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
@patch('lightning.pytorch.Trainer')
@patch('pytorch_forecasting.TemporalFusionTransformer.from_dataset')
def test_aipw_and_saving(mock_tft, mock_trainer, mock_read_csv, mock_to_csv, mock_exists, dummy_safe_data, dummy_causal_sub_data, dummy_tft_model):
    mock_exists.return_value = False

    mock_model_instance = MagicMock()

    # We need predict to return a specific set of predictions
    # T=1 and T=0 preds
    # Preds should be a tensor of shape (40, 1)

    mock_idx = pd.DataFrame({
        'COWcode': ['100']*20 + ['200']*20,
        'time_idx': list(range(1990, 2010)) * 2,
    })
    mock_pred = torch.tensor([[5.0] if i % 2 == 0 else [10.0] for i in range(40)]) # dummy values

    mock_model_instance.predict.return_value = (mock_pred, mock_idx)
    mock_tft.return_value = mock_model_instance

    target = 'fraser_bmp_score'

    run_causal_analysis(target, dummy_safe_data, dummy_tft_model, dummy_causal_sub_data, n_bootstrap=5)

    # verify that the results were saved
    # We expect 2 calls to to_csv:
    # 1. mediation_data_fraser_bmp_score.csv
    # 2. mediation_master_summary.csv
    assert mock_to_csv.call_count == 2

    call_args = mock_to_csv.call_args_list
    filenames = [call[0][0] for call in call_args]

    assert f"mediation_data_{target}.csv" in filenames
    assert "mediation_master_summary.csv" in filenames

@patch('os.path.exists')
@patch('pandas.DataFrame.to_csv')
@patch('pandas.read_csv')
@patch('lightning.pytorch.Trainer')
@patch('pytorch_forecasting.TemporalFusionTransformer.from_dataset')
@patch('builtins.print')
def test_insufficient_aipw_data(mock_print, mock_tft, mock_trainer, mock_read_csv, mock_to_csv, mock_exists, dummy_safe_data, dummy_causal_sub_data, dummy_tft_model):
    mock_exists.return_value = False

    # Make propensity_score mostly missing so dropna leaves < 10 rows
    dummy_causal_sub_data.loc[:35, 'propensity_score'] = np.nan

    mock_model_instance = MagicMock()
    mock_idx = pd.DataFrame({'COWcode': ['100']*40, 'time_idx': list(range(1990, 2010))*2})
    mock_pred = torch.zeros((40, 1))
    mock_model_instance.predict.return_value = (mock_pred, mock_idx)
    mock_tft.return_value = mock_model_instance

    target = 'fraser_bmp_score'
    run_causal_analysis(target, dummy_safe_data, dummy_tft_model, dummy_causal_sub_data, n_bootstrap=2)

    mock_print.assert_any_call("Not enough data for AIPW.")

    # Should not save if not enough data
    assert not mock_to_csv.called
