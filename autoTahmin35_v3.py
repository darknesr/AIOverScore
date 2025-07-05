import logging
import os
import json
import pickle
import pandas as pd
import shutil

from src.pipeline import load_and_prepare_data, engineer_features, run_stacking_workflow
from backtester import run_backtest_simulation # Modüler backtester import ediliyor
from config import (
    DB_PATH, LOG_FILE, TIMESTAMP, AG_MODELS_BASE_PATH, ELO_FİLE_PATH, CSV_OUT,
    MAIN_TARGET_MARKET, DATA_LOAD_START_DATE, LEAGUE_COL_NAME, RANDOM_SEED,
    AG_TIME_LIMIT_L1, AG_TIME_LIMIT_L2, AG_PRESETS, CUSTOM_HYPERPARAMETERS, MODEL_CONFIGS
)

# --- Logging ve Tohumlama ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(LOG_FILE, mode='w'), logging.StreamHandler()])
# np.random.seed(RANDOM_SEED) # Artık utils içinde ayarlanıyor
# random.seed(RANDOM_SEED) # Artık utils içinde ayarlanıyor
logging.info(f"Rastgelelik tohumu (seed) {RANDOM_SEED} olarak ayarlandı.")
logging.info(f"Ana hedef pazar: {MAIN_TARGET_MARKET}")


from db_manager import save_predictions_to_db

if __name__ == "__main__":
    try:
        logging.info("### Gelişmiş V35 (Düzeltilmiş ve İyileştirilmiş) İş Akışı Başlatıldı ###")
        
        output_dir = f"training_run_{TIMESTAMP}"
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Tüm çıktılar '{output_dir}' klasörüne kaydedilecek.")

        full_df = load_and_prepare_data(DB_PATH, DATA_LOAD_START_DATE, MODEL_CONFIGS)

        logging.info(f"Veritabanından {len(full_df)} maç yüklendi.")
        if not full_df.empty:
            # full_df'in 'dateth' sütununun datetime nesneleri içerdiğinden emin olalım
            full_df['dateth'] = pd.to_datetime(full_df['dateth'])
            logging.info(f"Veri tarih aralığı: {full_df['dateth'].min()} - {full_df['dateth'].max()}")
            # Zaman damgasını datetime.now() ile karşılaştır
            future_matches = full_df[full_df['dateth'] >= pd.to_datetime("now").normalize()]
            logging.info(f"Bugün ve sonrası için {len(future_matches)} maç bulundu.")

        df_train, df_predict, ALL_FEATURES, fitters, updated_elos = engineer_features(full_df, ELO_FİLE_PATH)

        if df_predict.empty:
            logging.warning("Tahmin edilecek yeni maç bulunamadı. İş akışı sonlandırılıyor.")
        else:
            results, backtest, calibrator, thresholds = run_stacking_workflow(
                df_train, df_predict, ALL_FEATURES, AG_MODELS_BASE_PATH, AG_PRESETS, 
                AG_TIME_LIMIT_L1, AG_TIME_LIMIT_L2, CUSTOM_HYPERPARAMETERS, MODEL_CONFIGS
            )
            
            if results.empty:
                logging.warning("Tahmin sonuçları boş. Veritabanına kaydedilecek bir şey yok.")
            else:
                # Tahminleri veritabanına kaydet
                save_predictions_to_db(results)

            logging.info("### Back-test simülasyonları çalıştırılıyor... ###")
            if backtest is not None and not backtest.empty:
                run_backtest_simulation(backtest, full_df, LEAGUE_COL_NAME, strategy='fixed')
                run_backtest_simulation(backtest, full_df, LEAGUE_COL_NAME, strategy='kelly')
            else:
                logging.warning("Backtest için veri bulunamadı, bu adım atlanıyor.")

            logging.info("Eğitim sonrası dosyalar kaydediliyor...")
            files_to_save = {
                "elos_latest.pkl": updated_elos,
                "fitters.pkl": fitters,
                "best_calibrator.pkl": calibrator,
                "league_thresholds.json": thresholds
            }

            for fname, obj in files_to_save.items():
                versioned_path = os.path.join(output_dir, fname)
                if ".json" in fname:
                    with open(versioned_path, 'w') as f: json.dump(obj, f, indent=4)
                    with open(fname, 'w') as f: json.dump(obj, f, indent=4)
                else: # Pickle dosyaları
                    with open(versioned_path, 'wb') as f: pickle.dump(obj, f)
                    with open(fname, 'wb') as f: pickle.dump(obj, f)

            logging.info(f"Tüm gerekli dosyalar predict.py için hazırlandı ve '{output_dir}' klasörüne yedeklendi.")

    except Exception as e:
        logging.critical("İş akışında beklenmedik bir hata oluştu!", exc_info=True)
        logging.critical(f"Log dosyası: {LOG_FILE}")

    finally:
        # Model klasörünü silerek disk alanı aç
        try:
            # AG_MODELS_BASE_PATH'ın dolu ve geçerli bir yol olduğundan emin ol
            if AG_MODELS_BASE_PATH and os.path.exists(AG_MODELS_BASE_PATH):
                logging.info(f"Geçici model klasörü '{AG_MODELS_BASE_PATH}' siliniyor...")
                shutil.rmtree(AG_MODELS_BASE_PATH)
                logging.info("Model klasörü başarıyla silindi.")
        except Exception as e:
            logging.error(f"Model klasörü silinirken bir hata oluştu: {e}")
        
        logging.info("### Tüm İş Akışı Tamamlandı ###")