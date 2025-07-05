import os
from datetime import datetime

# --- Dosya Yolları ve Temel Ayarlar ---
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "matches_optimized.db"
LOG_FILE = "match_prediction_log_PRO_FINAL_v2.txt"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
AG_MODELS_BASE_PATH = f"AutogluonModels_ProFinal/run_{TIMESTAMP}/"
ELO_FİLE_PATH = "elos_latest.pkl"
CSV_OUT = "autogluon_predictions_PRO_FINAL_V35_v2.csv"

# --- Ana Hedef Pazar ---
MAIN_TARGET_MARKET = 'Over_2_5'

# --- Model ve Veri Ayarları ---
DATA_LOAD_START_DATE = '2020-01-01'
LEAGUE_COL_NAME = 'lleague'
RANDOM_SEED = 42

# --- AutoGluon Ayarları ---
AG_TIME_LIMIT_L1 = 180 # Reduced for faster testing
AG_TIME_LIMIT_L2 = 90  # Reduced for faster testing
AG_PRESETS = 'medium_quality'

CUSTOM_HYPERPARAMETERS = {
    'GBM': {'extra_trees': True},
    'CAT': {},
    'XGB': {},
    'NN_TORCH': {},
}

# --- Model Pazarları Tanımları ---
MODEL_CONFIGS = {
    'Over_1_5': {'problem_type': 'binary', 'classes': {1: 'Over', 0: 'Under'}, 'odds_col': 'c_fto15_16'},
    'Over_2_5': {'problem_type': 'binary', 'classes': {1: 'Over', 0: 'Under'}, 'odds_col': 'c_fto25_16'},
    'Over_3_5': {'problem_type': 'binary', 'classes': {1: 'Over', 0: 'Under'}, 'odds_col': 'c_fto35_16'},
    'KG_Var': {'problem_type': 'binary', 'classes': {1: 'Yes', 0: 'No'}, 'odds_col': 'c_btts1_16'},
    'MS_1X2': {'problem_type': 'multiclass', 'classes': {'MS1': 'MS1', 'MSX': 'MSX', 'MS2': 'MS2'}, 'odds_col': None}
}

# --- Backtester Ayarları ---
BACKTEST_START_DATE = '2024-01-01'
KELLY_FRACTION = 0.3
MAX_KELLY_STAKE_PERCENTAGE = 0.05 # Kasanın %5'inden fazla bahis yapma
