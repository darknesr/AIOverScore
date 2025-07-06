import logging
import pandas as pd
import numpy as np
from config import BACKTEST_START_DATE, KELLY_FRACTION, MAX_KELLY_STAKE_PERCENTAGE

def run_backtest_simulation(
    results_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    LEAGUE_COL_NAME: str,
    target_col: str = 'target_Over_2_5',
    odds_col: str   = 'c_fto25_16', # Payout için hala kapanış oranını kullanıyoruz
    strategy: str   = 'fixed',
    initial_bankroll: float = 1000.0,
    transaction_cost: float = 0.005 # %0.5 işlem maliyeti
):
    logging.info(f"--- BACK-TEST SİMÜLASYONU BAŞLATILIYOR (Strateji: {strategy.upper()}) ---")
    cols_needed = ['matchid', 'dateth', LEAGUE_COL_NAME, target_col, odds_col]
    hist = historical_df[historical_df[target_col].notna()][cols_needed].dropna() # Sadece sonuçları bilinenleri al
    merge_cols = ['matchid', 'stacked_prob_over_2_5', 'guven_esigi']
    sim_results_df = results_df[merge_cols]
    backtest_data = hist.merge(sim_results_df, on='matchid', how='inner')
    backtest_data = backtest_data[backtest_data['dateth'] >= pd.to_datetime(BACKTEST_START_DATE)].copy()
    backtest_data.sort_values('dateth', inplace=True)
    prob_col = 'stacked_prob_over_2_5'
    if backtest_data.empty or odds_col not in backtest_data.columns:
        logging.warning("Back-test periyodu için veri veya oran kolonu bulunamadı.")
        return
    backtest_data[target_col] = pd.to_numeric(backtest_data[target_col], errors='coerce')
    backtest_data[odds_col] = pd.to_numeric(backtest_data[odds_col], errors='coerce')
    backtest_data['edge'] = backtest_data[prob_col] * backtest_data[odds_col] - 1
    if 'guven_esigi' in backtest_data.columns:
        logging.info("Lige özel güven eşikleri bahis kararında kullanılıyor.")
        backtest_data['is_bet_placed'] = (backtest_data['edge'] > 0) & (backtest_data[prob_col] > backtest_data['guven_esigi'])
    else:
        backtest_data['is_bet_placed'] = backtest_data['edge'] > 0
    bets_df = backtest_data[backtest_data['is_bet_placed']].copy()
    if bets_df.empty:
        logging.info("Back-test periyodunda koşullara uygun hiç bahis bulunamadı.")
        return
    bankroll = initial_bankroll
    bets_df.loc[:, 'pnl'] = 0.0
    bets_df.loc[:, 'stake'] = 0.0
    if strategy == 'fixed':
        stake_amount = 1.0
        bets_df.loc[:, 'stake'] = stake_amount
        bets_df.loc[:, 'pnl'] = np.where(bets_df[target_col] == 1, stake_amount * (bets_df[odds_col] - 1) - (stake_amount * transaction_cost), -stake_amount)
    elif strategy == 'kelly':
        for i, row in bets_df.iterrows():
            # Kasa kontrolü: inf veya nan ise veya sıfırın altındaysa dur
            if not np.isfinite(bankroll) or bankroll <= 0: 
                logging.warning(f"Kasa değeri geçersiz hale geldi ({bankroll}). Simülasyon durduruluyor.")
                break

            odds = row[odds_col]
            if pd.isna(odds) or (odds - 1) < 1e-9: continue
            edge = row['edge']
            frac = edge / (odds - 1)
            if frac <= 0: continue
            
            # Kelly fraksiyonunu ve maksimum bahis miktarını uygula
            stake_frac = min(KELLY_FRACTION, frac)
            stake = bankroll * stake_frac
            
            # Mutlak bahis miktarını sınırlama
            max_stake_abs = initial_bankroll * MAX_KELLY_STAKE_PERCENTAGE
            stake = min(stake, max_stake_abs)

            pnl = -stake
            if row[target_col] == 1:
                pnl = stake * (odds - 1)
            
            pnl -= stake * transaction_cost # Her bahisten işlem maliyetini düş

            bankroll += pnl
            bets_df.loc[i, 'stake'] = stake
            bets_df.loc[i, 'pnl'] = pnl

    num_bets = len(bets_df)
    total_staked = bets_df['stake'].sum()
    total_profit = bets_df['pnl'].sum()
    
    # ROI hesaplamasında sıfıra bölme hatasını önle
    roi = (total_profit / total_staked) * 100 if total_staked > 0 and np.isfinite(total_staked) else 0
    
    wins = bets_df[bets_df['pnl'] > 0]
    num_wins = len(wins)
    win_rate = (num_wins / num_bets) * 100 if num_bets > 0 else 0
    avg_odds_won = wins[odds_col].mean() if num_wins > 0 else 0
    avg_odds_all = bets_df[odds_col].mean()

    logging.info(f"--- BACK-TEST SONUÇLARI (Strateji: {strategy.upper()}) ---")
    logging.info(f"Toplam Bahis Sayısı: {num_bets}")
    logging.info(f"Kazanan Bahis Sayısı: {num_wins} ({win_rate:.2f}%)")
    logging.info(f"Ortalama Bahis Oranı: {avg_odds_all:.2f}")
    logging.info(f"Ortalama Kazanan Oran: {avg_odds_won:.2f}")
    logging.info(f"Toplam Yatırılan Miktar: {total_staked:.2f} birim")
    logging.info(f"Net Kar/Zarar: {total_profit:.2f} birim")
    logging.info(f"Yatırım Getirisi (ROI/Yield): {roi:.2f}%")
    if strategy == 'kelly':
        # Kasa büyümesi hesaplamasında sıfıra bölme ve nan/inf kontrolü
        growth = ((bankroll - initial_bankroll) / initial_bankroll) * 100 if initial_bankroll > 0 and np.isfinite(bankroll) else 0
        logging.info(f"Başlangıç Kasası: {initial_bankroll:.2f} birim")
        logging.info(f"Bitiş Kasası: {bankroll:.2f} birim")
        logging.info(f"Kasa Büyümesi: {growth:.2f}%")
    logging.info("----------------------------------------------------------")