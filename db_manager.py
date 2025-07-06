import sqlite3
import logging
from config import DB_PATH, TIMESTAMP

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# YENİ VE ÇİFT EŞİKLİ HALİ

def save_predictions_to_db(predictions_df):
    """
    Tahminleri (iki farklı eşik stratejisiyle) veritabanına akıllı bir şekilde kaydeder.
    """
    TABLE_NAME = "predictions"
    MODEL_VERSION = f"autoTahmin35_v3_{TIMESTAMP}"

    if predictions_df is None or predictions_df.empty:
        logging.warning("Kaydedilecek tahmin bulunamadı. Veritabanı işlemi atlanıyor.")
        return

    # Ana tahmin olasılığı sütununun adını dinamik olarak bul
    prob_col = next((col for col in predictions_df.columns if 'stacked_prob' in col), None)
    if not prob_col:
        logging.error("Tahmin olasılığı sütunu ('stacked_prob_...') bulunamadı!")
        return
        
    # Gerekli sütunları seç ve veritabanı için yeniden adlandır
    # Artık 'threshold' yerine 'threshold_conservative' ve 'threshold_profit' kullanacağız
    df_to_save = predictions_df[[
        'matchid', 'dateth', 'league', 'home', 'away',
        prob_col, 'guven_esigi_garanti', 'guven_esigi_riskli'
    ]].rename(columns={
        prob_col: 'predicted_probability',
        'guven_esigi_garanti': 'threshold_conservative',
        'guven_esigi_riskli': 'threshold_profit'
    })

    df_to_save['model_version'] = MODEL_VERSION

    try:
        with sqlite3.connect(DB_PATH, timeout=10) as conn:
            cursor = conn.cursor()

            # Geçici tabloya yeni tahminleri yükle
            df_to_save.to_sql('temp_predictions', conn, if_exists='replace', index=False)
            logging.info(f"{len(df_to_save)} yeni tahmin geçici tabloya yüklendi.")

            # Mevcut aktif tahminleri pasif yap
            update_query = f"""
            UPDATE {TABLE_NAME}
            SET is_active = 0
            WHERE is_active = 1 AND matchid IN (SELECT matchid FROM temp_predictions);
            """
            cursor.execute(update_query)
            logging.info(f"{cursor.rowcount} adet eski tahmin pasif olarak işaretlendi.")
            
            # Ana tabloya yeni tahminleri ekle (yeni sütunlarla)
            # NOT: Eğer 'predictions' tablosunda bu sütunlar yoksa, tabloyu silip yeniden oluşturmanız
            # veya ALTER TABLE ile sütunları eklemeniz gerekebilir.
            insert_query = f"""
            INSERT INTO {TABLE_NAME} (
                matchid, dateth, league, home, away, 
                predicted_probability, threshold_conservative, threshold_profit, 
                model_version, is_active
            )
            SELECT 
                matchid, dateth, league, home, away, 
                predicted_probability, threshold_conservative, threshold_profit,
                model_version, 1
            FROM temp_predictions;
            """
            cursor.execute(insert_query)
            logging.info(f"{cursor.rowcount} adet yeni tahmin veritabanına aktif olarak eklendi.")

            # Geçici tabloyu sil
            cursor.execute("DROP TABLE temp_predictions")
            
            conn.commit()
            logging.info("Veritabanı kayıt işlemleri başarıyla tamamlandı.")

    except sqlite3.Error as e:
        # Eğer hata "no such column" ise, tabloyu güncelleme uyarısı ver
        if "no such column" in str(e):
            logging.error(f"Veritabanı hatası: {e}")
            logging.error("LÜTFEN DİKKAT: 'predictions' tablosunda 'threshold_conservative' veya 'threshold_profit' sütunları eksik olabilir.")
            logging.error("Çözüm için veritabanı dosyasını silip yeniden oluşturabilir veya 'ALTER TABLE' komutuyla sütunları ekleyebilirsiniz.")
        else:
            logging.error(f"Veritabanı kaydı sırasında hata: {e}")
        if conn:
            conn.rollback()
