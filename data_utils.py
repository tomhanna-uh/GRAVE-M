import os
import sys
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def master_data_prep(input_filename="GRAVE_M_Master_Dataset_Final_v3.csv"):
    """
    Master data preparation function.
    Consolidates data loading, filtering, and MICE imputation.
    """
    print("--- PHASE 1: LOADING & FILTERING ---")

    file_path = None
    if os.path.exists(input_filename):
        file_path = input_filename
    else:
        for root, dirs, current_files in os.walk(os.getcwd()):
            if input_filename in current_files:
                file_path = os.path.join(root, input_filename)
                break

    if file_path is None:
        if 'google.colab' in sys.modules:
            from google.colab import files
            print(f"Please upload '{input_filename}' now.")
            uploaded = files.upload()
            if len(uploaded) > 0:
                file_path = list(uploaded.keys())[0]
            else:
                raise FileNotFoundError("No file uploaded.")
        else:
            raise FileNotFoundError(f"File {input_filename} not found.")

    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    def sanitize_id(x):
        try: return str(int(float(x)))
        except: return str(x)

    if 'COWcode' in df.columns:
        df['COWcode'] = df['COWcode'].apply(sanitize_id)
    if 'year' in df.columns:
        df['year'] = df['year'].astype(int)

    DROP_NAMES = ["Hong Kong", "Palestine", "Gaza", "West Bank", "Zanzibar", "Somaliland"]
    drop_pattern = '|'.join(DROP_NAMES)
    initial_len = len(df)
    if 'country_name' in df.columns:
        df = df[~df['country_name'].str.contains(drop_pattern, case=False, na=False)].copy()
        print(f"Dropped {initial_len - len(df)} rows matching non-sovereign names.")

    if 'year' in df.columns:
        df = df[(df['year'] >= 1990) & (df['year'] <= 2015)].reset_index(drop=True)
        df['time_idx'] = df['year']
        print(f"Filtered to 1990-2015: {len(df)} rows")

    print("--- PHASE 2: MISSING VALUE IMPUTATION (MICE) ---")
    target_cols = ["fraser_bmp_score", "gini_disp", "resource_rents", "unified_gdp_pc", "unified_pop", "log_oil_gas_wealth"]
    target_cols = [c for c in target_cols if c in df.columns]

    predictor_cols = ["year", "v2x_libdem", "is_petro_state", "unified_corruption"]
    predictor_cols = [c for c in predictor_cols if c in df.columns]

    impute_subset = df[target_cols + predictor_cols].copy()

    imputer = IterativeImputer(max_iter=20, random_state=42, sample_posterior=True)
    imputed_data = imputer.fit_transform(impute_subset)
    imputed_df = pd.DataFrame(imputed_data, columns=impute_subset.columns)

    for col in target_cols:
        missing_n = df[col].isna().sum()
        df[col] = imputed_df[col]
        print(f"  - {col}: Filled {missing_n} missing values.")

    print("--- PHASE 3: SAFETY TRANSFORMS ---")
    cols_to_fix = target_cols + ["v2x_libdem", "unified_corruption", 'trade_export_hhi', 'exp_dep_usa']
    for col in cols_to_fix:
        if col in df.columns and 'COWcode' in df.columns:
            df[col] = df.groupby('COWcode')[col].ffill().bfill().fillna(0.0)

    for col in ["unified_gdp_pc", "unified_pop"]:
        if col in df.columns and (df[col] < 0).any():
            df[col] = df[col].clip(lower=0.0)

    if "unified_gdp_pc" in df.columns:
        df["log_gdp_pc"] = np.log1p(df["unified_gdp_pc"])
    if "unified_pop" in df.columns:
        df["log_pop"] = np.log1p(df["unified_pop"])

    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(df[col].max())

    if "fraser_bmp_score" in df.columns:
        upper_bmp = df["fraser_bmp_score"].quantile(0.99)
        df["fraser_bmp_score"] = df["fraser_bmp_score"].clip(upper=upper_bmp)

    print("--- PHASE 4: CATEGORICAL CASTING ---")
    categorical_vars = [
        "is_petro_state", "is_aut_episode", "is_dem_episode", "alba_member",
        "mid_count_total", "mid_high_fatality_event",
        "is_leftist_leader", "is_rightist_leader",
        "gli_leader_ideology_num", "mid_max_fatality_cat", "mid_max_hostility"
    ]

    for var in categorical_vars:
        if var in df.columns:
            df[var] = df[var].apply(
                lambda x: str(int(float(x))) if pd.notnull(x) and str(x).replace('.','').isdigit() else str(x)
            )
            df[var] = df[var].replace({'nan': '0', 'NaN': '0', '<NA>': '0'})
            df[var] = df[var].astype("category")

    print(f"✓ Data Prep Complete. Shape: {df.shape}")
    return df
