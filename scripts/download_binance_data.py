from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time

def download_klines(symbol='BTCUSDT', interval='5m', start_date=None, end_date=None):
    """
    Binance'dan belirtilen sembol için kline verilerini indirir
    
    Args:
        symbol (str): Trading pair (örn: 'BTCUSDT')
        interval (str): Kline aralığı (örn: '5m')
        start_date (datetime): Başlangıç tarihi
        end_date (datetime): Bitiş tarihi
    """
    # Binance client başlatma (API key olmadan da çalışır)
    client = Client()
    
    # Tarihleri ms cinsinden dönüştürme
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    # Veriyi çekme
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_ts,
        end_str=end_ts
    )
    
    # DataFrame oluşturma
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 
        'volume', 'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    
    # Gereksiz kolonları silme
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Timestamp'i datetime'a çevirme
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Float dönüşümü
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    return df

def main():
    # Tarih aralığını belirleme
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # Trading pair'i belirleme
    symbol = 'BTCUSDT'
    
    try:
        # Veriyi indirme
        df = download_klines(
            symbol=symbol,
            interval='5m',
            start_date=start_date,
            end_date=end_date
        )
        
        # CSV olarak kaydetme
        filename = f'data/{symbol}_5m_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        df.to_csv(filename, index=False)
        print(f'Veri başarıyla indirildi: {filename}')
        
    except Exception as e:
        print(f'Hata oluştu: {str(e)}')

if __name__ == '__main__':
    main() 