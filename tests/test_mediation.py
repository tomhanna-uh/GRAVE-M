import pytest
import pandas as pd
import numpy as np
import sys
import os
import json
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Extract the run_mediation_analysis function from the notebook
def get_mediation_function():
    with open('tft_alba_02272026_2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_lines = []
    found_func = False

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for line in source:
                if line.startswith('def run_mediation_analysis('):
                    found_func = True

                # We need to collect the whole function.
                # Simplest way is to just find the cell with the function and return it
                if found_func:
                    return ''.join(source)
    return None

func_code = get_mediation_function()

# create a dummy local dict to execute just the function definition
local_vars = {
    'pd': pd,
    'np': np,
    'LinearRegression': LinearRegression,
    'resample': resample
}
if func_code:
    # Filter out IPython magic
    filtered_lines = [line for line in func_code.split('\n') if not line.strip().startswith(('!', '%'))]

    # We only want the function definition, not the execution below it
    func_lines = []
    in_func = False
    for line in filtered_lines:
        if line.startswith('def run_mediation_analysis('):
            in_func = True
            func_lines.append(line)
        elif in_func:
            if line.strip() == '' or line.startswith(' ') or line.startswith('\t'):
                func_lines.append(line)
            else:
                break

    exec('\n'.join(func_lines), globals(), local_vars)

run_mediation_analysis = local_vars['run_mediation_analysis']

def test_extract():
    """Verify that the mediation function was successfully extracted and is callable."""
    assert callable(run_mediation_analysis)

def test_mediation_analysis_perfect_mediation():
    """
    Test a scenario where mediation is perfect.
    T -> M (a = 1.0)
    M -> Y (b = 1.0)
    T -> Y (c' = 0.0)
    Covariates have no effect.
    """
    np.random.seed(42)
    n = 1000
    T = np.random.normal(0, 1, n)
    X1 = np.random.normal(0, 1, n)

    # M = 1.0*T + 0.0*X1 + noise
    M = 1.0 * T + np.random.normal(0, 0.1, n)

    # Y = 0.0*T + 1.0*M + 0.0*X1 + noise
    Y = 1.0 * M + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({'T': T, 'M': M, 'Y': Y, 'X1': X1})

    results, _ = run_mediation_analysis(df, 'T', 'M', 'Y', ['X1'], n_boot=10)

    # Check estimates
    assert abs(results['a']['estimate'] - 1.0) < 0.1
    assert abs(results['b']['estimate'] - 1.0) < 0.1
    assert abs(results['indirect']['estimate'] - 1.0) < 0.1
    assert abs(results['c_prime']['estimate'] - 0.0) < 0.1
    assert abs(results['total']['estimate'] - 1.0) < 0.1

def test_mediation_analysis_no_mediation():
    """
    Test a scenario where there is no mediation.
    T -> M (a = 0.0)
    M -> Y (b = 0.0)
    T -> Y (c' = 1.0)
    """
    np.random.seed(42)
    n = 1000
    T = np.random.normal(0, 1, n)
    X1 = np.random.normal(0, 1, n)

    # M = 0.0*T + noise
    M = np.random.normal(0, 0.1, n)

    # Y = 1.0*T + 0.0*M + noise
    Y = 1.0 * T + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({'T': T, 'M': M, 'Y': Y, 'X1': X1})

    results, _ = run_mediation_analysis(df, 'T', 'M', 'Y', ['X1'], n_boot=10)

    assert abs(results['a']['estimate'] - 0.0) < 0.1
    assert abs(results['b']['estimate'] - 0.0) < 0.1
    assert abs(results['indirect']['estimate'] - 0.0) < 0.1
    assert abs(results['c_prime']['estimate'] - 1.0) < 0.1
    assert abs(results['total']['estimate'] - 1.0) < 0.1

def test_mediation_analysis_partial_mediation():
    """
    Test a scenario with partial mediation.
    T -> M (a = 0.5)
    M -> Y (b = 0.5)
    T -> Y (c' = 0.5)
    """
    np.random.seed(42)
    n = 1000
    T = np.random.normal(0, 1, n)
    X1 = np.random.normal(0, 1, n)

    # M = 0.5*T + noise
    M = 0.5 * T + np.random.normal(0, 0.1, n)

    # Y = 0.5*T + 0.5*M + noise
    Y = 0.5 * T + 0.5 * M + np.random.normal(0, 0.1, n)

    df = pd.DataFrame({'T': T, 'M': M, 'Y': Y, 'X1': X1})

    results, _ = run_mediation_analysis(df, 'T', 'M', 'Y', ['X1'], n_boot=10)

    assert abs(results['a']['estimate'] - 0.5) < 0.1
    assert abs(results['b']['estimate'] - 0.5) < 0.1
    assert abs(results['indirect']['estimate'] - 0.25) < 0.1
    assert abs(results['c_prime']['estimate'] - 0.5) < 0.1
    assert abs(results['total']['estimate'] - 0.75) < 0.1

def test_mediation_analysis_error_handling():
    """
    Test error handling for missing columns
    """
    np.random.seed(42)
    n = 100
    T = np.random.normal(0, 1, n)
    df = pd.DataFrame({'T': T})

    # run_mediation_analysis returns (None, None) when columns are missing
    results, boot_results = run_mediation_analysis(df, 'T', 'M', 'Y', ['X1'], n_boot=10)
    assert results is None
    assert boot_results is None

def test_mediation_analysis_all_nan():
    """
    Test when data becomes empty after dropping NaNs
    """
    df = pd.DataFrame({
        'T': [np.nan, np.nan],
        'M': [1, np.nan],
        'Y': [np.nan, 2],
        'X1': [3, 4]
    })

    # run_mediation_analysis returns (None, None) when not enough data points
    results, boot_results = run_mediation_analysis(df, 'T', 'M', 'Y', ['X1'], n_boot=10)
    assert results is None
    assert boot_results is None
