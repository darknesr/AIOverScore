import sqlite3
import logging
from config import DB_PATH, TIMESTAMP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def save_predictions_to_db(predictions_df):
    """Tahminleri veritabanına akıllı bir şekilde kaydeder."""
    TABLE_NAME = "predictions"
    MODEL_VERSION = f"autoTahmin35_v3_{TIMESTAMP}"

    if predictions_df is None or predictions_df.empty:
        logging.warning("Kaydedilecek tahmin bulunamadı. Veritabanı işlemi atlanıyor.")
        return

    # Gerekli sütunları seç ve yeniden adlandır
    df_to_save = predictions_df[[
        'matchid', 'dateth', 'league', 'home', 'away',
        'stacked_prob_over_2_5', 'guven_esigi'
    ]].rename(columns={
        'stacked_prob_over_2_5': 'predicted_probability',
        'guven_esigi': 'threshold'
    })

    df_to_save['model_version'] = MODEL_VERSION

    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            cursor = conn.cursor()

            # 1. Yeni tahminleri geçici bir tabloya yükle
            df_to_save.to_sql('temp_predictions', conn, if_exists='replace', index=False)
            logging.info(f"{len(df_to_save)} yeni tahmin geçici tabloya yüklendi.")

            # 2. Mevcut aktif tahminleri pasif yap (sadece güncellenecek olanları)
            update_query = f"""
            UPDATE {TABLE_NAME}
            SET is_active = 0
            WHERE is_active = 1 AND matchid IN (SELECT matchid FROM temp_predictions);
            """
            cursor.execute(update_query)
            logging.info(f"{cursor.rowcount} adet eski tahmin pasif olarak işaretlendi.")

            # 3. Yeni tahminleri ana tabloya ekle
            insert_query = f"""
            INSERT INTO {TABLE_NAME} (
                matchid, dateth, league, home, away, 
                predicted_probability, threshold, model_version, is_active
            )
            SELECT 
                matchid, dateth, league, home, away, 
                predicted_probability, threshold, model_version, 1
            FROM temp_predictions;
            """
            cursor.execute(insert_query)
            logging.info(f"{cursor.rowcount} adet yeni tahmin veritabanına aktif olarak eklendi.")

            # 4. Geçici tabloyu sil
            cursor.execute("DROP TABLE temp_predictions")
            
            conn.commit()
            logging.info("Veritabanı kayıt işlemleri başarıyla tamamlandı.")

    except sqlite3.Error as e:
        logging.error(f"Veritabanı kaydı sırasında hata: {e}")
        conn.rollback()
