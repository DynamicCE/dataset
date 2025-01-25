from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import pytz

def get_solana_historical_data():
    # Binance client oluştur (API key olmadan da çalışır)
    client = Client()
    
    # Başlangıç ve bitiş tarihlerini ayarla (5 yıl öncesinden bugüne)
    end_date = datetime.now(pytz.UTC)
    start_date = end_date - timedelta(days=5*365)  # 5 yıl
    
    # Solana-USDT paritesi için 5dk'lık mum verilerini çek
    klines = client.get_historical_klines(
        symbol="SOLUSDT",
        interval=Client.KLINE_INTERVAL_5MINUTE,
        start_str=start_date.strftime("%d %b %Y %H:%M:%S"),
        end_str=end_date.strftime("%d %b %Y %H:%M:%S")
    )
    
    # Verileri DataFrame'e dönüştür
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Timestamp'i datetime'a çevir
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Gerekli sütunları seç ve float'a çevir
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    # CSV olarak kaydet
    filename = f"data/SOLUSDT_5min_{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"Veriler {filename} dosyasına kaydedildi")
    return df

if __name__ == "__main__":
    get_solana_historical_data() 