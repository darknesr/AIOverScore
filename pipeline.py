import logging
import pandas as pd
import numpy as np
import sqlite3
import os
import pickle
import shutil # BU SATIRI EKLEYİN
from datetime import datetime
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder

from .utils import (
    calculate_league_power_ranking,
    calculate_volatility_features,
    parse_score,
    generate_odds_features,
    apply_vig_filter,
    add_time_decay_weight,
    calculate_elo_ratings,
    calculate_relative_odds_features,
    calculate_rolling_features,
    calculate_poisson_features,
    calculate_h2h_features,
    calculate_league_features,
    train_team_embeddings,
    apply_team_embeddings,
    get_robust_calibrator,
    calc_league_dynamic_thresholds,
    prepare_data_for_ag,
    calculate_team_strength_ratings,
    add_interaction_features, # Yeni eklenen import
    select_features_by_importance, # YENİ
    profit_scorer
)
from config import LEAGUE_COL_NAME, RANDOM_SEED, MAIN_TARGET_MARKET

def load_and_prepare_data(DB_PATH, DATA_LOAD_START_DATE, MODEL_CONFIGS):
    logging.info("### 1. VERİ YÜKLEME VE İLK HAZIRLIK ###")
    with sqlite3.connect(DB_PATH) as conn:
        all_db_columns = pd.read_sql_query("PRAGMA table_info('matches')", conn)['name'].tolist()
        safe_base_columns = ['matchid', 'dateth', 'league', 'lleague', 'home', 'away', 'score']
        odds_columns = [col for col in all_db_columns if col.startswith(('c_', 'o_'))]
        query = f"SELECT {', '.join(list(set(safe_base_columns + odds_columns)))} FROM matches WHERE dateth >= ?"
        df = pd.read_sql_query(query, conn, params=(DATA_LOAD_START_DATE,))
    
    df["dateth"] = pd.to_datetime(df["dateth"], errors='coerce')
    df[LEAGUE_COL_NAME] = df[LEAGUE_COL_NAME].fillna('Diğer')
    df = df.dropna(subset=['dateth', 'home', 'away', LEAGUE_COL_NAME]).sort_values('dateth').reset_index(drop=True)
    df = parse_score(df)

    # Bellek optimizasyonu: Sayısal sütunları daha küçük tiplere dönüştür
    def downcast_dataframe_dtypes(df):
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                # Sadece değer aralığı izin veriyorsa int32'ye dönüştür
                if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype('int32')
        return df

    df = downcast_dataframe_dtypes(df)
    
    for target, config in MODEL_CONFIGS.items():
        if 'Over' in target:
            goals = float(target.split('_')[1].replace(',', '.'))
            df[f'target_{target}'] = (df['total_goals'] > goals).astype(int)
        elif 'KG_Var' in target:
            df[f'target_{target}'] = ((df['home_goals'] > 0) & (df['away_goals'] > 0)).astype(int)
        elif 'MS_1X2' in target:
            df[f'target_{target}'] = np.select([df['home_goals'] > df['away_goals'], df['home_goals'] < df['away_goals']], ['MS1', 'MS2'], default='MSX')
    return df

def engineer_features(df, ELO_FİLE_PATH):
    logging.info("### 2. ÖZELLİK MÜHENDİSLİĞİ (GELİŞTİRİLMİŞ) ###")
    df, odds_features = generate_odds_features(df)
    df = apply_vig_filter(df, vig_threshold=1.20)
    
    # Elo özelliklerini en başta hesapla
    if os.path.exists(ELO_FİLE_PATH):
        with open(ELO_FİLE_PATH, 'rb') as f:
            initial_elos = pickle.load(f)
        logging.info(f"{len(initial_elos)} takım için başlangıç Elo değerleri '{ELO_FİLE_PATH}' dosyasından yüklendi.")
    else:
        initial_elos = None
        logging.info(f"Başlangıç Elo dosyası '{ELO_FİLE_PATH}' bulunamadı, tüm takımlar için 1500'den başlanacak.")
    
    df, updated_elos, elo_features = calculate_elo_ratings(df, initial_elos=initial_elos)

    df, rolling_features = calculate_rolling_features(df)
    df, rel_odds_features = calculate_relative_odds_features(df)
    df, h2h_features = calculate_h2h_features(df)
    df, league_features = calculate_league_features(df)
    df, team_strength_features = calculate_team_strength_ratings(df)

    df, volatility_features = calculate_volatility_features(df)
    df, league_elo_features = calculate_league_power_ranking(df)

    # Yeni: Oran Değişimi (Odds Movement) Özellikleri
    odds_movement_features = []
    o_cols = {c.replace('o_', '') for c in df.columns if c.startswith('o_')}
    c_cols = {c.replace('c_', '') for c in df.columns if c.startswith('c_')}
    common_odds = o_cols.intersection(c_cols)
    for suffix in common_odds:
        feature_name = f'odds_move_{suffix}'
        df[feature_name] = df[f'c_{suffix}'] - df[f'o_{suffix}']
        odds_movement_features.append(feature_name)
    logging.info(f"{len(odds_movement_features)} adet oran değişimi özelliği eklendi.")
    
    df, decay_features = add_time_decay_weight(df)

    # Tüm özellik mühendisliği tamamlandıktan sonra df_train ve df_predict oluştur
    TODAY = pd.to_datetime(datetime.now().date())
    df_train = df[(df['dateth'] < TODAY) & (df['total_goals'].notna())].copy()
    df_predict = df[df['dateth'] >= TODAY].copy()
    logging.info(f"Veri ayrıştırıldı: {len(df_train)} eğitim, {len(df_predict)} tahmin.")
    if df_train.empty: raise ValueError("Eğitim için geçmiş maç verisi bulunamadı.")

    df_train, df_predict, poisson_features = calculate_poisson_features(df_train, df_predict)
    fitters = train_team_embeddings(df_train)
    df_train = apply_team_embeddings(df_train, fitters)
    df_predict = apply_team_embeddings(df_predict, fitters)

    df_train, interaction_features = add_interaction_features(df_train)
    df_predict, _ = add_interaction_features(df_predict)

    vec_features = [c for c in df_train.columns if 'vec' in c]
    team_id_features = [c for c in df_train.columns if 'team_id' in c]
    
    # Tüm özellikleri birleştir
    ALL_FEATURES = list(set(
        vec_features + 
        odds_features + 
        rel_odds_features + 
        elo_features + 
        decay_features + 
        poisson_features + 
        team_id_features +
        rolling_features +
        h2h_features +
        league_features +
        team_strength_features +
        interaction_features + # Yeni eklenen etkileşim özellikleri
        odds_movement_features + # Yeni eklenen oran değişimi özellikleri
        ['edge'] + 
        volatility_features +
        league_elo_features # YENİ
    ))
    
    logging.info(f"Toplam {len(ALL_FEATURES)} adet özellik ile model eğitilecek.")
    
    logging.info(f"Özellik mühendisliği sonrası tahmin edilecek maç sayısı: {len(df_predict)}")
    return df_train, df_predict, ALL_FEATURES, fitters, updated_elos

def run_stacking_workflow(df_train, df_predict, ALL_FEATURES, AG_MODELS_BASE_PATH, AG_PRESETS, AG_TIME_LIMIT_L1, AG_TIME_LIMIT_L2, CUSTOM_HYPERPARAMETERS, MODEL_CONFIGS):
    logging.info("### 3. STACKING MODEL EĞİTİMİ VE TAHMİN (GELİŞTİRİLMİŞ) ###")
    os.makedirs(AG_MODELS_BASE_PATH, exist_ok=True)
    meta_features_train = pd.DataFrame(index=df_train.index)
    meta_features_predict = pd.DataFrame(index=df_predict.index)

    # FAZ 1: L1 MODELLER
    for model_name, config in MODEL_CONFIGS.items():
        target_col, problem_type = f"target_{model_name}", config['problem_type']
        if target_col not in df_train.columns: continue

        feature_cols = list(dict.fromkeys(ALL_FEATURES + ['time_decay_weight']))
        train_data_ag = df_train[feature_cols + [target_col]].dropna(subset=feature_cols + [target_col])

        if len(train_data_ag) < 100: continue

        # YENİ VE DOĞRU KOD
        # Probleme göre doğru değerlendirme metriğini seç
        if 'binary' in problem_type:
            eval_metric = 'roc_auc'
        elif 'multiclass' in problem_type:
            # 'log_loss' çok sınıflı problemler için iyi bir başlangıç metriğidir
            eval_metric = 'log_loss' 
        else:
            # Diğer problem tipleri için varsayılan
            eval_metric = 'auto'

        logging.info(f"'{model_name}' modeli için '{eval_metric}' metriği kullanılacak.")
        predictor_l1 = TabularPredictor(
            label=target_col, 
            problem_type=problem_type, 
            path=os.path.join(AG_MODELS_BASE_PATH, f"L1_Global_{model_name}"), 
            eval_metric=eval_metric
        )
        predictor_l1.fit(prepare_data_for_ag(train_data_ag), presets=AG_PRESETS, time_limit=AG_TIME_LIMIT_L1, hyperparameters=CUSTOM_HYPERPARAMETERS, ag_args_fit={'random_seed': RANDOM_SEED, 'sample_weight': 'time_decay_weight'})

        preds_train_oof = predictor_l1.predict_proba(prepare_data_for_ag(train_data_ag.drop(columns=[target_col])))
        preds_predict = predictor_l1.predict_proba(prepare_data_for_ag(df_predict[feature_cols]))

        preds_train_oof.columns = [f'prob_L1G_{model_name}_{c}' for c in preds_train_oof.columns]
        preds_predict.columns = [f'prob_L1G_{model_name}_{c}' for c in preds_predict.columns]

        meta_features_train = meta_features_train.join(preds_train_oof)
        meta_features_predict = meta_features_predict.join(preds_predict)

    # --- YENİ ADIM: ÖZELLİK SEÇİMİ ---
    logging.info("### Meta-Model için Özellik Seçimi Yapılıyor... ###")
    meta_target_col = f"target_{MAIN_TARGET_MARKET}"

    # Özellik seçimi için eğitim setini hazırla
    fs_train_df = pd.concat([meta_features_train, df_train[[meta_target_col]]], axis=1).dropna()

    if len(fs_train_df) > 50:
        # En önemli 75 özelliği seç
        selected_meta_features = select_features_by_importance(
            predictor=None, # utils'deki fonksiyon kendi predictor'ını oluşturacak
            X=fs_train_df.drop(columns=[meta_target_col]),
            y=fs_train_df[meta_target_col],
            feature_count=75,
            problem_type='binary'
        )
    else:
        logging.warning("Özellik seçimi için yetersiz veri, tüm meta özellikler kullanılacak.")
        selected_meta_features = meta_features_train.columns.tolist()

    # FAZ 2: META-MODEL (L2) EĞİTİMİ (KÂRA ODAKLI)
    logging.info(f"### FAZ 2: Meta-Model (L2) Eğitimi ({len(selected_meta_features)} özellik ile)... ###")
    main_odds_col = MODEL_CONFIGS[MAIN_TARGET_MARKET]['odds_col']

    # YENİ VE DOĞRU KOD
    # Sadece df_train'de gerçekten var olan ek özellikleri seç
    l2_extra_features = [c for c in ['elo_farki', f'fair_prob_{main_odds_col}', f'rel_{main_odds_col}', 'edge'] if c in df_train.columns]

    # Meta-Modelin kullanacağı nihai özellik listesi (meta-özellikler + ek özellikler)
    l2_final_features = selected_meta_features + l2_extra_features

    # df_train'den SADECE ona ait olan sütunları seçerek bir liste oluştur
    df_train_cols_to_select = [meta_target_col, main_odds_col] + l2_extra_features

    # Şimdi verileri doğru yerlerden alarak birleştir:
    # 1. L1 modellerinin ürettiği meta özellikleri al
    # 2. df_train'den sadece yukarıda belirlediğimiz sütunları al
    meta_train_df_full = pd.concat([meta_features_train, df_train[df_train_cols_to_select]], axis=1)

    # Birleştirme sonrası, eksik veri içeren satırları temizle
    meta_train_df_full.dropna(subset=l2_final_features, inplace=True)

    if len(meta_train_df_full) < 50:
        logging.error("Meta-modeli eğitmek için yeterli veri üretilemedi.")
        return None, None, None, None

    # L2 modelini eğitmeye başlamadan önce eski kalıntıları temizle
    l2_model_path = os.path.join(AG_MODELS_BASE_PATH, "L2_Meta_Model")
    if os.path.exists(l2_model_path):
        logging.warning(f"Eski L2 model klasörü bulundu: '{l2_model_path}'. Temizleniyor...")
        shutil.rmtree(l2_model_path)

    # Modeli, en iyi olasılık tahminlerini yapması için standart ve güvenilir bir metrikle eğitiyoruz.
    # Kârlılık, olasılıklar elde edildikten sonra dinamik eşik belirleme adımında optimize edilecek.
    predictor_l2 = TabularPredictor(
        label=meta_target_col, 
        problem_type='binary', 
        path=l2_model_path, 
        eval_metric='roc_auc'
    )

    # fit() fonksiyonuna sadece temel argümanları veriyoruz.
    fit_kwargs = {
        'presets': AG_PRESETS,
        'time_limit': AG_TIME_LIMIT_L2,
        'ag_args_fit': {'random_seed': RANDOM_SEED},
        'calibrate': False, # Kalibrasyonu biz kendimiz yapıyoruz
    }

    # Modeli eğit
    meta_train_for_fit = prepare_data_for_ag(meta_train_df_full[l2_final_features + [meta_target_col]])
    predictor_l2.fit(meta_train_for_fit, **fit_kwargs)

    # Tahminleri yap
    # YENİ VE DOĞRU KOD
    # Tahmin verisini birleştirirken, df_predict'ten SADECE ona ait olan
    # 'l2_extra_features' sütunlarını alıyoruz.
    meta_predict_df_full = pd.concat([meta_features_predict, df_predict[l2_extra_features]], axis=1)
    meta_predict_df_full = meta_predict_df_full.reindex(columns=meta_train_for_fit.drop(columns=[meta_target_col]).columns).fillna(0.5)

    raw_pred = predictor_l2.predict_proba(prepare_data_for_ag(meta_predict_df_full))[1]
    raw_val = predictor_l2.predict_proba(meta_train_for_fit.drop(columns=[meta_target_col]))[1]

    # Kalibrasyon ve sonuçların hazırlanması (mevcut kod gibi devam)
    best_cal = get_robust_calibrator(raw_val.values, meta_train_for_fit[meta_target_col].values)
    final_stacked_probas_predict = best_cal.predict(raw_pred.values) 
    calibrated_train_probas_backtest = best_cal.predict(raw_val.values) 

    prob_col_name = f'stacked_prob_{MAIN_TARGET_MARKET.lower()}'
    train_results_for_thresholding = df_train.loc[raw_val.index].copy()
    train_results_for_thresholding[prob_col_name] = calibrated_train_probas_backtest

    # league_thr = calc_league_dynamic_thresholds(train_results_for_thresholding, league_col=LEAGUE_COL_NAME, prob_col=prob_col_name, target_col=meta_target_col, odds_col=main_odds_col)

    # YENİ VE ÇİFT STRATEJİLİ HALİ

    # --- EŞİK HESAPLAMA (İKİ FARKLI STRATEJİ İLE) ---

    # 1. Strateji: 'conservative' (Bankocu)
    # Yüksek isabet oranı hedefler, sadece modelin çok emin olduğu maçları seçer.
    conservative_thresholds = calc_league_dynamic_thresholds(
        train_results_for_thresholding, 
        league_col=LEAGUE_COL_NAME, 
        prob_col=prob_col_name, 
        target_col=meta_target_col, 
        odds_col=main_odds_col,
        strategy='conservative' # Strateji: Muhafazakar
    )

    # 2. Strateji: 'max_profit' (Fırsatçı)
    # Toplam kârı maksimize etmeye odaklanır, 'değer' (value) olan riskli maçları bulabilir.
    profit_thresholds = calc_league_dynamic_thresholds(
        train_results_for_thresholding, 
        league_col=LEAGUE_COL_NAME, 
        prob_col=prob_col_name, 
        target_col=meta_target_col, 
        odds_col=main_odds_col,
        strategy='max_profit' # Strateji: Kâr Maksimizasyonu
    )

    # 3. Strateji: 'advanced_profit' (Dengeli Usta)
    balanced_thresholds = calc_league_dynamic_thresholds(
        train_results_for_thresholding, 
        league_col=LEAGUE_COL_NAME, 
        prob_col=prob_col_name, 
        target_col=meta_target_col, 
        odds_col=main_odds_col,
        strategy='advanced_profit' # YENİ DENGELİ STRATEJİ
    )

    # --- SONUÇLARI HAZIRLAMA (İKİ EŞİK DEĞERİYLE) ---
    results_df = df_predict.loc[meta_predict_df_full.index, ['matchid', 'league', LEAGUE_COL_NAME, 'home', 'away', 'dateth']].copy()
    prob_col_name = f'stacked_prob_{MAIN_TARGET_MARKET.lower()}'
    results_df[prob_col_name] = final_stacked_probas_predict
    
    # İki farklı stratejinin eşiklerini ayrı sütunlara ekle
    results_df['guven_esigi_garanti'] = results_df[LEAGUE_COL_NAME].map(conservative_thresholds).fillna(conservative_thresholds['DEFAULT'])
    results_df['guven_esigi_riskli'] = results_df[LEAGUE_COL_NAME].map(profit_thresholds).fillna(profit_thresholds['DEFAULT'])
    results_df['guven_esigi_dengeli'] = results_df[LEAGUE_COL_NAME].map(balanced_thresholds).fillna(balanced_thresholds['DEFAULT'])

    results_df['edge'] = (results_df[prob_col_name] * df_predict[main_odds_col]) - 1
    results_df['quality_score'] = (results_df[prob_col_name] * 0.70) + (results_df['edge'].clip(lower=0) * 0.30)


    # Back-test için de benzer bir yapı kurulabilir ama şimdilik tahmin odaklı gidelim.
    backtest_df = df_train.loc[raw_val.index, ['matchid', 'dateth', 'league', LEAGUE_COL_NAME]].copy()
    backtest_df[prob_col_name] = calibrated_train_probas_backtest
    # Back-test için de benzer bir yapı kurulabilir ama şimdilik tahmin odaklı gidelim.
    backtest_df = df_train.loc[raw_val.index, ['matchid', 'dateth', 'league', LEAGUE_COL_NAME]].copy()
    prob_col_name = f'stacked_prob_{MAIN_TARGET_MARKET.lower()}' # prob_col_name'i tekrar tanımla
    backtest_df[prob_col_name] = calibrated_train_probas_backtest
    backtest_df['guven_esigi'] = backtest_df[LEAGUE_COL_NAME].map(profit_thresholds).fillna(profit_thresholds['DEFAULT'])

    # --- SONUÇLARI HAZIRLAMA (İKİ EŞİK DEĞERİYLE) ---
    results_df = df_predict.loc[meta_predict_df_full.index, ['matchid', 'league', LEAGUE_COL_NAME, 'home', 'away', 'dateth']].copy()
    results_df[prob_col_name] = final_stacked_probas_predict

    # İki farklı stratejinin eşiklerini ayrı sütunlara ekle
    results_df['guven_esigi_garanti'] = results_df[LEAGUE_COL_NAME].map(conservative_thresholds).fillna(conservative_thresholds['DEFAULT'])
    results_df['guven_esigi_riskli'] = results_df[LEAGUE_COL_NAME].map(profit_thresholds).fillna(profit_thresholds['DEFAULT'])

    backtest_df = df_train.loc[raw_val.index, ['matchid', 'dateth', 'league', LEAGUE_COL_NAME, main_odds_col]].copy()
    backtest_df[prob_col_name] = calibrated_train_probas_backtest
    backtest_df['guven_esigi'] = backtest_df[LEAGUE_COL_NAME].map(profit_thresholds).fillna(profit_thresholds['DEFAULT'])

    return results_df, backtest_df, best_cal, profit_thresholds
