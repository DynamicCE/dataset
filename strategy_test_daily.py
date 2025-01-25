import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

class IBBVCTLStrategy:
    def __init__(self, 
                 initial_equity=100000,
                 order_size_pct=0.5,
                 pyramiding=2,
                 commission=0.001,  # Günlük işlemler için düşük komisyon
                 atr_length=14,     # Günlük veriler için kısa ATR
                 atr_multiplier_sl=2.5,  # Daha yakın stop loss
                 atr_multiplier_tp=2.0,  # Daha yakın take profit
                 overbought_oversold_length=2,
                 max_volatility=40.0,  # Daha yüksek volatilite limiti
                 base_position_size=0.5,  # Daha küçük pozisyon büyüklüğü
                 atr_position_divider=15.0,
                 min_position_size=0.1,
                 trend_length=20,    # Daha kısa trend periyodu
                 min_volume_multiplier=1.2):  # Daha düşük hacim gerekliliği
        
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.order_size_pct = order_size_pct
        self.pyramiding = pyramiding
        self.commission = commission
        self.atr_length = atr_length
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        self.overbought_oversold_length = overbought_oversold_length
        self.max_volatility = max_volatility
        self.base_position_size = base_position_size
        self.atr_position_divider = atr_position_divider
        self.min_position_size = min_position_size
        self.trend_length = trend_length
        self.min_volume_multiplier = min_volume_multiplier
        
        self.positions = []
        self.trades = []
        self.equity_curve = []
        self.max_equity = initial_equity
        self.max_drawdown = 0

    # ... (diğer metodlar aynı kalacak)

def main():
    # CSV dosyasını oku
    print("Veriler okunuyor...")
    df = pd.read_csv('Ethereum_26.01.2024-25.01.2025_historical_data_coinmarketcap.csv', sep=';')
    
    # timestamp sütununu datetime'a çevir ve index olarak ayarla
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Veri yapısını kontrol et
    print("\nVeri Yapısı:")
    print(df.head())
    print("\nSütunlar:", df.columns.tolist())
    print("\nVeri Boyutu:", df.shape)
    print("\nVeri Aralığı:")
    print("Başlangıç:", df.index.min())
    print("Bitiş:", df.index.max())
    
    # Stratejiyi başlat
    strategy = IBBVCTLStrategy(
        initial_equity=100000,
        order_size_pct=0.5,
        pyramiding=2,
        commission=0.001,  # Düşük komisyon
        atr_length=14,     # Kısa ATR
        atr_multiplier_sl=2.5,  # Yakın stop loss
        atr_multiplier_tp=2.0,  # Yakın take profit
        overbought_oversold_length=2,
        max_volatility=40.0,  # Yüksek volatilite limiti
        base_position_size=0.5,  # Küçük pozisyon büyüklüğü
        atr_position_divider=15.0,
        min_position_size=0.1,
        trend_length=20,    # Kısa trend periyodu
        min_volume_multiplier=1.2  # Düşük hacim gerekliliği
    )
    
    # Backtest
    print("\nBacktest yapılıyor...")
    stats = strategy.backtest(df)
    
    # Sonuçları yazdır
    print("\nStrateji Test Sonuçları:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Equity curve grafiğini çiz ve kaydet
    os.makedirs('analysis/results', exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy.equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Zaman')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/results/equity_curve.png')
    plt.close()
    
    # Trade listesini kaydet
    trades_df = pd.DataFrame(strategy.trades)
    trades_df.to_csv('analysis/results/trades.csv', index=False)
    print("\nSonuçlar 'analysis/results' klasörüne kaydedildi.")

if __name__ == "__main__":
    main() 