import json
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import sqlite3
import os
import re
import logging
import pickle
import random
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import brier_score_loss
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
from autogluon.tabular import TabularPredictor
from sklearn.isotonic import IsotonicRegression
try:
    from betacal import BetaCalibration
except ImportError:
    BetaCalibration = None

from config import LEAGUE_COL_NAME, RANDOM_SEED



def parse_score(df):
    logging.info("Skorlar ('score' sütunu) ayrıştırılıyor...")
    if 'score' not in df.columns or df['score'].isnull().all():
        df['home_goals'], df['away_goals'], df['total_goals'] = np.nan, np.nan, np.nan
        return df
    df['score'] = df['score'].replace('0:0', np.nan)
    clean_scores = df['score'].str.replace(r'[^0-9:]', '', regex=True)
    score_parts = clean_scores.str.split(':', expand=True)
    df['home_goals'] = pd.to_numeric(score_parts[0], errors='coerce')
    df['away_goals'] = pd.to_numeric(score_parts[1], errors='coerce')
    df['total_goals'] = df['home_goals'] + df['away_goals']
    return df

def generate_odds_features(df):
    logging.info("Oran bazlı özellikler üretiliyor...")
    df_out = df.copy()
    new_features_data = {}
    new_feature_names = []

    markets = {'1x2_16':['c_ft1_16','c_ftx_16','c_ft2_16'],'ou25_16':['c_fto25_16','c_ftu25_16'],'btts_16':['c_btts1_16','c_btts2_16']}
    all_odds_cols = [col for cols in markets.values() for col in cols if col in df_out.columns]

    for col in all_odds_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        prob_col = f'prob_{col}'
        new_features_data[prob_col] = 1 / df_out[col]
        new_feature_names.append(prob_col)

        open_col = col.replace('c_', 'o_')
        if open_col in df_out.columns:
            df_out[open_col] = pd.to_numeric(df_out[open_col], errors='coerce')
            move_col = f'move_{col}'
            new_features_data[move_col] = df_out[col] - df_out[open_col]
            move_sign_col = f'move_sign_{col}'
            new_features_data[move_sign_col] = np.sign(new_features_data[move_col])
            new_feature_names.extend([move_col, move_sign_col])

    for market_name, market_cols in markets.items():
        prob_cols = [f'prob_{c}' for c in market_cols if f'prob_{c}' in new_features_data]
        if len(prob_cols) == len(market_cols):
            vig_col = f'vig_{market_name}'
            new_features_data[vig_col] = pd.DataFrame(new_features_data)[prob_cols].sum(axis=1, min_count=len(prob_cols))
            new_feature_names.append(vig_col)
            valid_vig_mask = new_features_data[vig_col].notna() & (new_features_data[vig_col] > 0)
            for prob_col in prob_cols:
                fair_prob_col = f'fair_{prob_col}'
                new_features_data[fair_prob_col] = pd.Series(np.nan, index=df_out.index)
                new_features_data[fair_prob_col].loc[valid_vig_mask] = new_features_data[prob_col].loc[valid_vig_mask] / new_features_data[vig_col].loc[valid_vig_mask]
                new_feature_names.append(fair_prob_col)

    # Ana hedef pazar için 'edge' özelliğini hesapla
    from config import MAIN_TARGET_MARKET, MODEL_CONFIGS
    main_target_config = MODEL_CONFIGS[MAIN_TARGET_MARKET]
    main_odds_col = main_target_config['odds_col']
    main_prob_col = f'prob_{main_odds_col}'
    main_fair_prob_col = f'fair_{main_prob_col}'

    if main_fair_prob_col in new_features_data and main_odds_col in df_out.columns:
        new_features_data['edge'] = (new_features_data[main_fair_prob_col] * df_out[main_odds_col]) - 1
        new_feature_names.append('edge')

    # Tüm yeni özellikleri tek seferde birleştir
    df_out = pd.concat([df_out, pd.DataFrame(new_features_data, index=df_out.index)], axis=1)
    df_out[new_feature_names] = df_out[new_feature_names].fillna(0)

    return df_out, list(set(new_feature_names))

def get_robust_calibrator(raw_p, y):
    logging.info("Sağlam kalibratör seçimi yapılıyor (Öncelik: Beta > Isotonic > Sigmoid)...")
    if BetaCalibration:
        try:
            calibrator = BetaCalibration().fit(raw_p, y)
            logging.info("Öncelikli kalibratör 'BetaCalibration' başarıyla seçildi.")
            return calibrator
        except Exception as e:
            logging.warning(f"BetaCalibration sırasında bir hata oluştu: {e}")
    try:
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(raw_p, y)
        logging.info("İkincil kalibratör 'IsotonicRegression' başarıyla seçildi.")
        return calibrator
    except Exception as e:
        logging.warning(f"IsotonicRegression sırasında bir hata oluştu: {e}")
    from sklearn.linear_model import LogisticRegression
    s = LogisticRegression(solver='lbfgs').fit(raw_p.reshape(-1,1), y)
    class SigmoidCalibrator:
        def predict(self, p):
            return s.predict_proba(p.reshape(-1,1))[:,1]
    logging.info("Garanti fallback kalibratörü 'SigmoidCalibrator' seçildi.")
    return SigmoidCalibrator()

def apply_vig_filter(df, vig_threshold=1.20):
    logging.info(f"Aşırı vig filtresi uygulanıyor (vig > {vig_threshold})...")
    original_rows = len(df)
    vig_cols = [col for col in df.columns if 'vig_' in col]
    for col in vig_cols:
        df = df[(df[col].isna()) | (df[col] <= vig_threshold)]
    rows_dropped = original_rows - len(df)
    if rows_dropped > 0:
        logging.info(f"{rows_dropped} satır aşırı vig nedeniyle elendi.")
    return df

def add_time_decay_weight(df, lookback_days=365, decay_lambda=0.002):
    logging.info(f"Zamanla azalan ağırlık özelliği hesaplanıyor...")
    df_out = df.copy()
    latest_date = df_out.loc[df_out['total_goals'].notna(), 'dateth'].max() if 'total_goals' in df_out.columns and df_out['total_goals'].notna().any() else df_out['dateth'].max()
    df_out['days_diff'] = (latest_date - df_out['dateth']).dt.days
    mask = (df_out['days_diff'] >= 0) & (df_out['days_diff'] <= lookback_days)
    df_out['time_decay_weight'] = 0.0
    df_out.loc[mask, 'time_decay_weight'] = np.exp(-decay_lambda * df_out.loc[mask, 'days_diff'])
    df_out = df_out.drop(columns=['days_diff'])
    return df_out, ['time_decay_weight']

def get_dynamic_k_factor(elo_rating, games_played, base_k=30, max_k=50, min_k=10):
    # K-factor based on ELO rating (higher ELO, lower K)
    if elo_rating < 1800:
        k_elo = max_k
    elif elo_rating > 2400:
        k_elo = min_k
    else:
        # Linear interpolation between 1800 and 2400
        k_elo = max_k - (elo_rating - 1800) * (max_k - min_k) / (2400 - 1800)

    # K-factor based on games played (more games, lower K)
    # This is a simple inverse relationship, can be made more sophisticated
    k_games = base_k * (1 / (1 + games_played / 30)) # Adjust 30 for desired decay rate

    # Combine both factors (e.g., average or weighted average)
    return (k_elo + k_games) / 2

def calculate_elo_ratings(df, initial_elos=None, form_window=5):
    logging.info(f"Elo ve Form Elo özellikleri hesaplanıyor...")
    elos = {} if initial_elos is None else initial_elos.copy()
    team_games_played = {team: 0 for team in pd.concat([df['home'], df['away']]).unique()} # Her takım için oynanan maç sayısını takip et
    df_sorted = df.sort_values('dateth').copy()
    home_elos, away_elos = [], []
    for index, row in df_sorted.iterrows():
        h, a = row['home'], row['away']
        h_elo, a_elo = elos.get(h, 1500), elos.get(a, 1500)

        # Dinamik K-faktörlerini hesapla
        k_h = get_dynamic_k_factor(h_elo, team_games_played.get(h, 0))
        k_a = get_dynamic_k_factor(a_elo, team_games_played.get(a, 0))

        home_elos.append(h_elo); away_elos.append(a_elo)
        if pd.notna(row['home_goals']):
            e_home = 1 / (1 + 10**((a_elo - h_elo) / 400))
            s_home = 0.5 if row['home_goals'] == row['away_goals'] else (1.0 if row['home_goals'] > row['away_goals'] else 0.0)
            elos[h] = h_elo + k_h * (s_home - e_home)
            elos[a] = a_elo + k_a * ((1 - s_home) - (1 - e_home))
            
            # Maç sayısını güncelle
            team_games_played[h] = team_games_played.get(h, 0) + 1
            team_games_played[a] = team_games_played.get(a, 0) + 1
    df_sorted['home_elo'], df_sorted['away_elo'] = home_elos, away_elos
    df_sorted['elo_farki'] = df_sorted['home_elo'] - df_sorted['away_elo']
    
    team_elos_timeline = pd.concat([
        df_sorted[['matchid', 'dateth', 'home', 'home_elo']].rename(columns={'home': 'team', 'home_elo': 'elo'}),
        df_sorted[['matchid', 'dateth', 'away', 'away_elo']].rename(columns={'away': 'team', 'away_elo': 'elo'})
    ]).sort_values(['team', 'dateth'])
    team_elos_timeline['form_elo_change'] = team_elos_timeline.groupby('team')['elo'].diff(periods=form_window).fillna(0)
    
    df_sorted = df_sorted.merge(team_elos_timeline[['matchid', 'team', 'form_elo_change']].rename(columns={'team': 'home', 'form_elo_change': 'home_form_elo_change'}), on=['matchid', 'home'], how='left')
    df_sorted = df_sorted.merge(team_elos_timeline[['matchid', 'team', 'form_elo_change']].rename(columns={'team': 'away', 'form_elo_change': 'away_form_elo_change'}), on=['matchid', 'away'], how='left')
    
    new_cols = ['home_elo', 'away_elo', 'elo_farki', 'home_form_elo_change', 'away_form_elo_change']
    return df.merge(df_sorted[['matchid'] + new_cols], on='matchid', how='left'), elos, new_cols

def calculate_relative_odds_features(df):
    logging.info("Lige göre göreceli oran özellikleri (veri sızıntısı önlenmiş) hesaplanıyor...")
    df_out = df.sort_values('dateth').copy()
    odds_cols = [c for c in df.columns if c.startswith('c_') and '_16' in c]
    
    df_out[odds_cols] = df_out[odds_cols].apply(pd.to_numeric, errors='coerce')

    # Yeni özellikler için bir liste oluştur
    features_to_concat = []
    new_feature_names = []
    
    for col in odds_cols:
        avg_col = f"league_avg_{col}"
        # Doğrudan df_out üzerinde transform uygula
        df_out[avg_col] = df_out.groupby(LEAGUE_COL_NAME)[col].transform(lambda x: x.shift(1).expanding(min_periods=20).mean())
        
        rel_col = f"rel_{col}"
        df_out[rel_col] = df_out[col] - df_out[avg_col]
        new_feature_names.append(rel_col)
        
    # Geçici league_avg sütunlarını kaldır
    df_out.drop(columns=[c for c in df_out.columns if 'league_avg_' in c], inplace=True)
    df_out[new_feature_names] = df_out[new_feature_names].fillna(0)

    return df_out, new_feature_names

def calculate_rolling_features(df):
    logging.info("Dinamik zaman serisi (rolling) özellikleri hesaplanıyor...")
    df_out = df.sort_values('dateth').copy()
    
    team_stats_base = pd.concat([
        df_out[['dateth', 'home', 'home_goals', 'away_goals']].rename(columns={'home': 'team', 'home_goals': 'goals_scored', 'away_goals': 'goals_conceded'}),
        df_out[['dateth', 'away', 'home_goals', 'away_goals']].rename(columns={'away': 'team', 'away_goals': 'goals_scored', 'home_goals': 'goals_conceded'})
    ], ignore_index=True).sort_values(['team', 'dateth'])
    
    team_stats_base['goal_diff'] = team_stats_base['goals_scored'] - team_stats_base['goals_conceded']
    team_stats_base['is_win'] = (team_stats_base['goals_scored'] > team_stats_base['goals_conceded']).astype(int)
    team_stats_base['is_draw'] = (team_stats_base['goals_scored'] == team_stats_base['goals_conceded']).astype(int)
    team_stats_base['is_loss'] = (team_stats_base['goals_scored'] < team_stats_base['goals_conceded']).astype(int)
    
    windows = [3, 5, 10]
    stats_to_roll = ['goals_scored', 'goals_conceded', 'goal_diff', 'is_win', 'is_draw', 'is_loss']
    
    rolling_operations = {'mean': 'avg', 'std': 'std', 'skew': 'skew', 'kurt': 'kurt', 'sum': 'sum'}

    grouped = team_stats_base.groupby('team')
    
    new_rolling_features = []
    for stat in stats_to_roll:
        for w in windows:
            for op_func, op_name in rolling_operations.items():
                if (op_func == 'sum' and stat not in ['is_win', 'is_draw', 'is_loss']) or \
                   (op_func != 'sum' and stat in ['is_win', 'is_draw', 'is_loss']):
                    continue

                col_name = f'roll_{w}_{op_name}_{stat}'
                team_stats_base[col_name] = grouped[stat].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).__getattribute__(op_func)())
                new_rolling_features.append(col_name)

    team_stats_base.drop(columns=['goals_scored', 'goals_conceded', 'goal_diff', 'is_win', 'is_draw', 'is_loss'], inplace=True)
    
    # Yeni özellik DataFrame'lerini oluştur
    home_features_df = team_stats_base.rename(columns={'team': 'home'})
    home_features_df.columns = ['dateth', 'home'] + [f'{c}_h' for c in new_rolling_features]

    away_features_df = team_stats_base.rename(columns={'team': 'away'})
    away_features_df.columns = ['dateth', 'away'] + [f'{c}_a' for c in new_rolling_features]

    # Ana DataFrame'e birleştir
    df_out = df_out.merge(home_features_df, on=['dateth', 'home'], how='left')
    df_out = df_out.merge(away_features_df, on=['dateth', 'away'], how='left')

    final_rolling_features = []
    for f_name in new_rolling_features:
        h_feat = f_name + '_h'
        a_feat = f_name + '_a'
        diff_feat = f_name + '_diff'
        df_out[diff_feat] = df_out[h_feat] - df_out[a_feat]
        final_rolling_features.extend([h_feat, a_feat, diff_feat])

    df_out.fillna(0, inplace=True)
    logging.info(f"{len(final_rolling_features)} adet rolling özellik oluşturuldu.")
    return df_out, final_rolling_features

def calculate_poisson_features(df_train, df_predict):
    logging.info("Poisson gol beklentisi özellikleri hesaplanıyor...")
    df_train_copy, df_predict_copy = df_train.copy(), df_predict.copy()
    
    # Ağırlıklı eğitim verisi setini oluştur
    train_hist = df_train_copy[df_train_copy['time_decay_weight'] > 0].copy()
    if train_hist.empty:
        logging.warning("Poisson özellikleri için yeterli geçmiş veri bulunamadı.")
        return df_train, df_predict, []

    # Zaman ağırlıklı genel gol ortalamalarını hesapla
    avg_home_goals = np.average(train_hist['home_goals'], weights=train_hist['time_decay_weight'])
    avg_away_goals = np.average(train_hist['away_goals'], weights=train_hist['time_decay_weight'])

    # Gerekli güç sütunlarının varlığını kontrol et
    required_strength_cols = [
        'home_attack_strength', 'home_defense_strength', 
        'away_attack_strength', 'away_defense_strength'
    ]
    if not all(col in df_train_copy.columns for col in required_strength_cols):
        logging.error("Poisson hesaplaması için gerekli takım gücü sütunları bulunamadı. Lütfen önce `calculate_team_strength_ratings` fonksiyonunun çalıştığından emin olun.")
        # Hata durumunda orijinal DataFrame'leri ve boş bir özellik listesini döndür
        return df_train, df_predict, []

    def apply_lambda(df):
        # NaN değerleri, ortalama gücü temsil eden 1.0 ile doldur
        df[required_strength_cols] = df[required_strength_cols].fillna(1.0)
        
        # Lambda değerlerini hesapla
        df['lambda_home'] = df['home_attack_strength'] * df['away_defense_strength'] * avg_home_goals
        df['lambda_away'] = df['away_attack_strength'] * df['home_defense_strength'] * avg_away_goals
        df['poisson_ratio'] = df['lambda_home'] / (df['lambda_away'] + 1e-6) # Sıfıra bölme hatasını önle
        return df

    # Hem eğitim hem de tahmin setine uygula
    df_train_updated = apply_lambda(df_train_copy)
    df_predict_updated = apply_lambda(df_predict_copy)
    
    poisson_feature_names = ['lambda_home', 'lambda_away', 'poisson_ratio']
    
    return df_train_updated, df_predict_updated, poisson_feature_names

def calculate_h2h_features(df):
    logging.info("Head-to-Head (H2H) özellikleri hesaplanıyor...")
    df_out = df.sort_values('dateth').copy()
    h2h_features = []

    # Takım çifti için benzersiz tanımlayıcı oluştur
    df_out['team_pair'] = df_out.apply(lambda row: tuple(sorted([row['home'], row['away']])), axis=1)

    # H2H hesaplamaları için gerekli veriyi hazırla
    h2h_data = df_out[['dateth', 'team_pair', 'home', 'away', 'home_goals', 'away_goals']].copy()
    h2h_data['home_win'] = (h2h_data['home_goals'] > h2h_data['away_goals']).astype(int)
    h2h_data['away_win'] = (h2h_data['away_goals'] > h2h_data['home_goals']).astype(int)
    h2h_data['draw'] = (h2h_data['home_goals'] == h2h_data['away_goals']).astype(int)
    h2h_data['h2h_goal_diff'] = h2h_data['home_goals'] - h2h_data['away_goals']

    # Her takım çifti için rolling H2H istatistikleri hesapla
    grouped_h2h = h2h_data.groupby('team_pair')

    h2h_stats_to_roll = ['home_win', 'away_win', 'draw', 'h2h_goal_diff']
    h2h_windows = [3, 5] # H2H için daha kısa pencereler

    h2h_features_list = []
    for stat in h2h_stats_to_roll:
        for w in h2h_windows:
            avg_col = f'h2h_roll_{w}_avg_{stat}'
            h2h_data[avg_col] = grouped_h2h[stat].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
            h2h_features_list.append(avg_col)

    # H2H özelliklerini ana DataFrame'e geri birleştir (veri sızıntısını önlemek için merge_asof kullan)
    df_out = pd.merge_asof(df_out, h2h_data[['dateth', 'team_pair'] + h2h_features_list], 
                           on='dateth', by='team_pair', direction='backward')

    df_out.drop(columns=['team_pair'], inplace=True)
    df_out[h2h_features_list] = df_out[h2h_features_list].fillna(0) # Yeni H2H özellikleri için NaN değerleri doldur

    logging.info(f"{len(h2h_features_list)} adet H2H özellik oluşturuldu.")
    return df_out, h2h_features_list

def calculate_league_features(df):
    logging.info("Lig karakteristik özellikleri hesaplanıyor...")
    df_out = df.sort_values('dateth').copy()
    new_league_features = []

    # Maç sonuçlarını kolayca toplamak için yeni sütunlar
    df_out['is_home_win'] = (df_out['home_goals'] > df_out['away_goals']).astype(int)
    df_out['is_away_win'] = (df_out['away_goals'] > df_out['home_goals']).astype(int)
    df_out['is_draw'] = (df_out['home_goals'] == df_out['away_goals']).astype(int)
    df_out['is_btts'] = ((df_out['home_goals'] > 0) & (df_out['away_goals'] > 0)).astype(int)
    df_out['is_over_2_5'] = (df_out['total_goals'] > 2.5).astype(int) # Örnek olarak 2.5 üstü

    # Yuvarlanacak/genişletilecek metrikler
    metrics = {
        'total_goals': ['mean', 'std'],
        'is_home_win': ['mean'], # Ev sahibi galibiyet oranı
        'is_away_win': ['mean'], # Deplasman galibiyet oranı
        'is_draw': ['mean'],     # Beraberlik oranı
        'is_btts': ['mean'],     # KG Var oranı
        'is_over_2_5': ['mean']  # 2.5 Üstü oranı
    }

    grouped_by_league = df_out.groupby(LEAGUE_COL_NAME)

    for metric, ops in metrics.items():
        for op in ops:
            col_name_prefix = f'league_{metric}_{op}'
            # Uzun vadeli lig karakteristikleri için genişleyen ortalama/std
            df_out[f'{col_name_prefix}_exp'] = grouped_by_league[metric].transform(lambda x: x.shift(1).expanding(min_periods=10).agg(op))
            new_league_features.append(f'{col_name_prefix}_exp')

            # Yakın zamandaki lig formu için yuvarlanan ortalama/std (örn. son 20 maç)
            df_out[f'{col_name_prefix}_roll20'] = grouped_by_league[metric].transform(lambda x: x.shift(1).rolling(window=20, min_periods=5).agg(op))
            new_league_features.append(f'{col_name_prefix}_roll20')

    # Yeni ligler veya erken tarih için NaN değerleri doldur.
    # Şimdilik 0 ile dolduruyoruz, ancak daha sonra global ortalamalarla doldurma düşünülebilir.
    df_out[new_league_features] = df_out[new_league_features].fillna(0)

    logging.info(f"{len(new_league_features)} adet lig karakteristik özelliği oluşturuldu.")
    return df_out, new_league_features

def calculate_league_features(df):
    logging.info("Lig karakteristik özellikleri hesaplanıyor...")
    df_out = df.sort_values('dateth').copy()
    new_league_features = []

    # Maç sonuçlarını kolayca toplamak için yeni sütunlar
    df_out['is_home_win'] = (df_out['home_goals'] > df_out['away_goals']).astype(int)
    df_out['is_away_win'] = (df_out['away_goals'] > df_out['home_goals']).astype(int)
    df_out['is_draw'] = (df_out['home_goals'] == df_out['away_goals']).astype(int)
    df_out['is_btts'] = ((df_out['home_goals'] > 0) & (df_out['away_goals'] > 0)).astype(int)
    df_out['is_over_2_5'] = (df_out['total_goals'] > 2.5).astype(int) # Örnek olarak 2.5 üstü

    # Yuvarlanacak/genişletilecek metrikler
    metrics = {
        'total_goals': ['mean', 'std'],
        'is_home_win': ['mean'], # Ev sahibi galibiyet oranı
        'is_away_win': ['mean'], # Deplasman galibiyet oranı
        'is_draw': ['mean'],     # Beraberlik oranı
        'is_btts': ['mean'],     # KG Var oranı
        'is_over_2_5': ['mean']  # 2.5 Üstü oranı
    }

    grouped_by_league = df_out.groupby(LEAGUE_COL_NAME)

    for metric, ops in metrics.items():
        for op in ops:
            col_name_prefix = f'league_{metric}_{op}'
            # Uzun vadeli lig karakteristikleri için genişleyen ortalama/std
            df_out[f'{col_name_prefix}_exp'] = grouped_by_league[metric].transform(lambda x: x.shift(1).expanding(min_periods=10).agg(op))
            new_league_features.append(f'{col_name_prefix}_exp')

            # Yakın zamandaki lig formu için yuvarlanan ortalama/std (örn. son 20 maç)
            df_out[f'{col_name_prefix}_roll20'] = grouped_by_league[metric].transform(lambda x: x.shift(1).rolling(window=20, min_periods=5).agg(op))
            new_league_features.append(f'{col_name_prefix}_roll20')

    # Yeni ligler veya erken tarih için NaN değerleri doldur.
    # Şimdilik 0 ile dolduruyoruz, ancak daha sonra global ortalamalarla doldurma düşünülebilir.
    df_out[new_league_features] = df_out[new_league_features].fillna(0)

    logging.info(f"{len(new_league_features)} adet lig karakteristik özelliği oluşturuldu.")
    return df_out, new_league_features

def calculate_team_strength_ratings(df, window=5):
    logging.info(f"Takım hücum ve savunma güçleri hesaplanıyor (son {window} maç)...")
    df_out = df.copy()
    
    # Her takımın oynadığı maçları tek bir listede topla
    team_matches = pd.concat([
        df_out[['dateth', 'home', 'home_goals', 'away_goals']].rename(columns={'home': 'team', 'home_goals': 'goals_scored', 'away_goals': 'goals_conceded'}),
        df_out[['dateth', 'away', 'away_goals', 'home_goals']].rename(columns={'away': 'team', 'away_goals': 'goals_scored', 'home_goals': 'goals_conceded'})
    ]).sort_values(['team', 'dateth'])

    # Yuvarlanan ortalamaları hesapla
    team_matches_agg = team_matches.groupby(['dateth', 'team']).agg(
        goals_scored=('goals_scored', 'mean'),
        goals_conceded=('goals_conceded', 'mean')
    ).reset_index()

    team_matches_agg['rolling_goals_scored'] = team_matches_agg.groupby('team')['goals_scored'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
    team_matches_agg['rolling_goals_conceded'] = team_matches_agg.groupby('team')['goals_conceded'].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())

    # Ortalama golleri hesapla (lig ortalaması gibi düşünebiliriz)
    overall_avg_goals_scored = team_matches_agg['goals_scored'].mean()
    overall_avg_goals_conceded = team_matches_agg['goals_conceded'].mean()

    # Hücum ve savunma güçlerini hesapla
    team_matches_agg['attack_strength'] = team_matches_agg['rolling_goals_scored'] / overall_avg_goals_scored
    team_matches_agg['defense_strength'] = team_matches_agg['rolling_goals_conceded'] / overall_avg_goals_conceded

    # Ana DataFrame'e geri birleştir
    # Home takım için güçleri ekle
    home_strength_df = team_matches_agg[['dateth', 'team', 'attack_strength', 'defense_strength']].rename(columns={
        'team': 'home',
        'attack_strength': 'home_attack_strength',
        'defense_strength': 'home_defense_strength'
    })
    df_out = df_out.merge(home_strength_df, on=['dateth', 'home'], how='left')

    # Away takım için güçleri ekle
    away_strength_df = team_matches_agg[['dateth', 'team', 'attack_strength', 'defense_strength']].rename(columns={
        'team': 'away',
        'attack_strength': 'away_attack_strength',
        'defense_strength': 'away_defense_strength'
    })
    df_out = df_out.merge(away_strength_df, on=['dateth', 'away'], how='left')

    # NaN değerleri doldur (yeni takımlar veya yeterli geçmişi olmayanlar için)
    new_cols = ['home_attack_strength', 'home_defense_strength', 'away_attack_strength', 'away_defense_strength']
    df_out[new_cols] = df_out[new_cols].fillna(1.0) # Ortalama güç olarak 1.0 kullan

    logging.info(f"{len(new_cols)} adet takım gücü özelliği oluşturuldu.")
    return df_out, new_cols

def train_team_embeddings(df_train, min_matches=10):
    logging.info("Word2Vec ve SVD embedding modelleri eğitiliyor...")
    fitters = {}
    team_counts = pd.concat([df_train['home'], df_train['away']]).value_counts()
    valid_teams = team_counts[team_counts >= min_matches].index
    df_filtered = df_train[df_train['home'].isin(valid_teams) & df_train['away'].isin(valid_teams)]
    
    dynamic_window = min(10, int(df_filtered.groupby('home').size().mean() / 2)) if not df_filtered.empty else 5
    logging.info(f"Word2Vec için dinamik pencere boyutu: {dynamic_window}")

    sentences = pd.concat([df_filtered.groupby('home')['away'].apply(list), df_filtered.groupby('away')['home'].apply(list)]).groupby(level=0).sum().tolist()
    if sentences:
        w2v_model = Word2Vec(sentences, vector_size=20, window=dynamic_window, min_count=min_matches, workers=-1, seed=RANDOM_SEED)
        fitters['w2v'] = {team: w2v_model.wv[team] for team in w2v_model.wv.index_to_key}
        fitters['w2v_vector_size'] = w2v_model.wv.vector_size
        fitters['w2v_mean'] = np.mean(list(fitters['w2v'].values()), axis=0)
    
    team_stats = pd.concat([
        df_train.groupby('home').agg(avg_gs_h=('home_goals','mean'),std_gs_h=('home_goals','std'),avg_gc_h=('away_goals','mean'),std_gc_h=('away_goals','std')),
        df_train.groupby('away').agg(avg_gs_a=('away_goals','mean'),std_gs_a=('away_goals','std'),avg_gc_a=('home_goals','mean'),std_gc_a=('home_goals','std'))
    ], axis=1).fillna(0)
    
    if not team_stats.empty:
        stats_scaled = StandardScaler(with_mean=False).fit_transform(team_stats)
        size = min(15, stats_scaled.shape[1] - 1)
        if size > 0:
            svd_model = TruncatedSVD(n_components=size, random_state=RANDOM_SEED).fit(stats_scaled)
            fitters['svd_team_vecs'] = dict(zip(team_stats.index, svd_model.transform(stats_scaled)))
            fitters['svd_vector_size'] = size
            fitters['svd_mean'] = np.mean(list(fitters['svd_team_vecs'].values()), axis=0)
    return fitters

def apply_quality_filters(df, min_league_matches=50, start_week=4):
    logging.info("Veri kalitesi filtreleri uygulanıyor...")
    original_rows = len(df)
    df['match_week'] = df.groupby([LEAGUE_COL_NAME, df['dateth'].dt.year])['dateth'].rank(method='dense').astype(int)
    df = df[df['match_week'] > start_week].copy()
    logging.info(f"Sezon başı filtresi ({start_week}. hafta sonrası) {original_rows - len(df)} satırı eledi.")
    
    original_rows = len(df)
    train_df = df[df['total_goals'].notna()]
    if not train_df.empty:
        league_counts = train_df[LEAGUE_COL_NAME].value_counts()
        valid_leagues = league_counts[league_counts >= min_league_matches].index
        df = df[df[LEAGUE_COL_NAME].isin(valid_leagues)].copy()
        logging.info(f"Minimum {min_league_matches} maç filtresi {original_rows - len(df)} satırı eledi. {len(valid_leagues)} lig kaldı.")
    return df

def apply_team_embeddings(df, fitters):
    logging.info(f"{len(df)} satıra embeddingler uygulanıyor...")
    df_out = df.copy()
    for emb_type in ['w2v', 'svd']:
        emb_dict = fitters.get(f'{emb_type}_team_vecs' if emb_type == 'svd' else emb_type, {})
        emb_size = fitters.get(f'{emb_type}_vector_size', 0)
        mean_emb = fitters.get(f'{emb_type}_mean', np.zeros(emb_size)) # Ortalama embedding veya sıfır vektör

        if emb_size > 0 and (emb_dict or mean_emb.any()): # emb_dict boş olsa bile mean_emb varsa devam et
            for team_role in ['h', 'a']:
                team_col = "home" if team_role == "h" else "away"
                # Takım embedding'i yoksa ortalama embedding'i kullan
                emb_list = df_out[team_col].map(emb_dict).apply(lambda v: v if isinstance(v, np.ndarray) else mean_emb)
                emb_df = pd.DataFrame(emb_list.to_list(), index=df_out.index).add_prefix(f'{team_role}_{emb_type}_vec_')
                df_out = pd.concat([df_out, emb_df], axis=1)
    return df_out

def add_interaction_features(df):
    logging.info("Etkileşim özellikleri hesaplanıyor...")
    df_out = df.copy()
    new_interaction_features = []

    # elo_farki * edge
    if 'elo_farki' in df_out.columns and 'edge' in df_out.columns:
        df_out['elo_farki_x_edge'] = df_out['elo_farki'] * df_out['edge']
        new_interaction_features.append('elo_farki_x_edge')

    # home_attack_strength * away_defense_strength
    if 'home_attack_strength' in df_out.columns and 'away_defense_strength' in df_out.columns:
        df_out['home_att_x_away_def'] = df_out['home_attack_strength'] * df_out['away_defense_strength']
        new_interaction_features.append('home_att_x_away_def')

    # away_attack_strength * home_defense_strength
    if 'away_attack_strength' in df_out.columns and 'home_defense_strength' in df_out.columns:
        df_out['away_att_x_home_def'] = df_out['away_attack_strength'] * df_out['home_defense_strength']
        new_interaction_features.append('away_att_x_home_def')

    # home_elo * home_attack_strength
    if 'home_elo' in df_out.columns and 'home_attack_strength' in df_out.columns:
        df_out['home_elo_x_home_att'] = df_out['home_elo'] * df_out['home_attack_strength']
        new_interaction_features.append('home_elo_x_home_att')

    # away_elo * away_attack_strength
    if 'away_elo' in df_out.columns and 'away_attack_strength' in df_out.columns:
        df_out['away_elo_x_away_att'] = df_out['away_elo'] * df_out['away_attack_strength']
        new_interaction_features.append('away_elo_x_away_att')

    # home_elo * home_defense_strength
    if 'home_elo' in df_out.columns and 'home_defense_strength' in df_out.columns:
        df_out['home_elo_x_home_def'] = df_out['home_elo'] * df_out['home_defense_strength']
        new_interaction_features.append('home_elo_x_home_def')

    # away_elo * away_defense_strength
    if 'away_elo' in df_out.columns and 'away_defense_strength' in df_out.columns:
        df_out['away_elo_x_away_def'] = df_out['away_elo'] * df_out['away_defense_strength']
        new_interaction_features.append('away_elo_x_away_def')

    # poisson_ratio * edge
    if 'poisson_ratio' in df_out.columns and 'edge' in df_out.columns:
        df_out['poisson_ratio_x_edge'] = df_out['poisson_ratio'] * df_out['edge']
        new_interaction_features.append('poisson_ratio_x_edge')

    # roll_win_diff (ev ve deplasman yuvarlanan galibiyet oranları farkı)
    # Bu özellik, calculate_rolling_features içinde 'is_win' için 'diff' özelliği olarak zaten oluşturulmuş olabilir.
    # Eğer yoksa, burada manuel olarak oluşturulur.
    if 'roll_3_sum_is_win_h' in df_out.columns and 'roll_3_sum_is_win_a' in df_out.columns:
        df_out['roll_3_sum_is_win_diff'] = df_out['roll_3_sum_is_win_h'] - df_out['roll_3_sum_is_win_a']
        new_interaction_features.append('roll_3_sum_is_win_diff')
    if 'roll_5_sum_is_win_h' in df_out.columns and 'roll_5_sum_is_win_a' in df_out.columns:
        df_out['roll_5_sum_is_win_diff'] = df_out['roll_5_sum_is_win_h'] - df_out['roll_5_sum_is_win_a']
        new_interaction_features.append('roll_5_sum_is_win_diff')
    if 'roll_10_sum_is_win_h' in df_out.columns and 'roll_10_sum_is_win_a' in df_out.columns:
        df_out['roll_10_sum_is_win_diff'] = df_out['roll_10_sum_is_win_h'] - df_out['roll_10_sum_is_win_a']
        new_interaction_features.append('roll_10_sum_is_win_diff')

    df_out[new_interaction_features] = df_out[new_interaction_features].fillna(0)
    logging.info(f"{len(new_interaction_features)} adet etkileşim özelliği oluşturuldu.")
    return df_out, new_interaction_features

def calc_league_dynamic_thresholds(df: pd.DataFrame, league_col: str, prob_col: str, target_col: str, odds_col: str, min_bets: int = 15, default_thr: float = 0.75) -> dict:
    """
    Her lig için ve global olarak, doğrudan karı maksimize eden en uygun bahis eşiğini hesaplar.
    """
    logging.info("Doğrudan karı maksimize eden dinamik eşikler hesaplanıyor...")
    df = df[[league_col, prob_col, target_col, odds_col]].copy()
    df[[prob_col, target_col, odds_col]] = df[[prob_col, target_col, odds_col]].apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)

    def find_best_threshold(data, scope="Global"):
        best_profit, best_thr = -np.inf, None
        for thr in np.arange(0.50, 0.96, 0.01):
            bets = data[data[prob_col] >= thr]
            num_bets = len(bets)
            if num_bets >= min_bets:
                profit = np.where(bets[target_col] == 1, bets[odds_col] - 1, -1).sum()
                
                if profit > best_profit:
                    best_profit, best_thr = profit, thr
        
        # Sadece pozitif kâr getiren bir eşik bulunduysa onu döndür
        if best_profit > 0:
            logging.info(f"Kapsam: {scope} -> En iyi eşik: {best_thr:.2f} (Kar: {best_profit:.2f})")
            return round(best_thr, 2)
        logging.info(f"Kapsam: {scope} -> Kârlı bir eşik bulunamadı.")
        return None

    # 1. Tüm veri setini kullanarak sağlam bir GLOBAL varsayılan eşik hesapla
    global_default_threshold = find_best_threshold(df)
    if global_default_threshold is None:
        global_default_threshold = default_thr
        logging.warning(f"Global olarak kârlı bir eşik bulunamadı. Sabit varsayılan ({default_thr}) kullanılacak.")

    # 2. Her lig için özel eşik hesapla
    league_thresholds = {}
    for lg, g in df.groupby(league_col):
        if len(g) < min_bets * 2:
            continue
        league_thr = find_best_threshold(g, scope=lg)
        if league_thr is not None:
            league_thresholds[lg] = league_thr
            
    # 3. Nihai sözlüğe GLOBAL varsaylanı ekle
    league_thresholds['DEFAULT'] = global_default_threshold
    
    logging.info(f"Dinamik eşikler tamamlandı: {len(league_thresholds)-1} lige özel eşik + DEFAULT={league_thresholds['DEFAULT']}")
    return league_thresholds

def prepare_data_for_ag(df):
    df_copy = df.copy()
    df_copy[df_copy.select_dtypes(['object']).columns] = df_copy.select_dtypes(['object']).astype('category')
    return df_copy
