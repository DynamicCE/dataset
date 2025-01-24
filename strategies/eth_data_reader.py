import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

def read_eth_data(file_path):
    """Tekil dosya okuma fonksiyonu"""
    try:
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Desteklenmeyen dosya formatı. Lütfen Parquet, CSV veya Excel dosyası kullanın.")
        
        # Tarih sütunu varsa datetime formatına çevirme
        date_columns = df.select_dtypes(include=['datetime64[ns]']).columns
        if len(date_columns) == 0 and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        return df
    except Exception as e:
        print(f"Hata: {file_path} dosyası okunurken bir sorun oluştu - {str(e)}")
        return None

def read_all_eth_data(directory_path="Arşiv"):
    """Tüm parquet dosyalarını okuyup birleştirir"""
    # Parquet dosyalarını bul
    parquet_files = glob.glob(os.path.join(directory_path, "*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"{directory_path} klasöründe parquet dosyası bulunamadı.")
    
    # Tüm dosyaları oku ve birleştir
    dfs = []
    for file in sorted(parquet_files):
        df = read_eth_data(file)
        if df is not None:
            dfs.append(df)
            print(f"{os.path.basename(file)} okundu. Şekil: {df.shape}")
    
    if not dfs:
        raise ValueError("Hiçbir dosya okunamadı!")
    
    # Dataframe'leri birleştir
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Duplike kayıtları temizle
    combined_df = combined_df.drop_duplicates()
    
    # Tarihe göre sırala
    date_columns = combined_df.select_dtypes(include=['datetime64[ns]']).columns
    if len(date_columns) > 0:
        combined_df = combined_df.sort_values(by=date_columns[0])
    
    return combined_df

def get_hourly_candles(df):
    """Dakikalık verilerden saatlik mum çubuğu verilerini oluşturur"""
    # timestamp sütununu index olarak ayarla
    df = df.set_index('timestamp')
    
    # Saatlik yeniden örnekleme
    hourly_df = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'MA_20': 'last',
        'MA_50': 'last',
        'MA_200': 'last',
        'RSI': 'last',
        '%K': 'last',
        '%D': 'last',
        'ADX': 'last',
        'ATR': 'last',
        'Trendline': 'last',
        'MACD': 'last',
        'Signal': 'last',
        'Histogram': 'last',
        'BL_Upper': 'last',
        'BL_Lower': 'last',
        'MN_Upper': 'last',
        'MN_Lower': 'last'
    }).dropna()
    
    # Index'i sütun olarak geri al
    hourly_df = hourly_df.reset_index()
    
    return hourly_df

def save_data(df, file_path):
    """Verileri parquet formatında kaydeder"""
    try:
        # Dizin yoksa oluştur
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Verileri kaydet
        df.to_parquet(file_path, index=False)
        print(f"Veriler başarıyla kaydedildi: {file_path}")
        
        # Dosya boyutunu göster
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB cinsinden
        print(f"Dosya boyutu: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Hata: Veriler kaydedilirken bir sorun oluştu - {str(e)}")

def preview_eth_data(df, timeframe="1min"):
    """Veri önizleme fonksiyonu"""
    print(f"\n=== {timeframe} Veri Seti Özeti ===")
    print(f"Toplam Kayıt Sayısı: {df.shape[0]:,}")
    print(f"Sütun Sayısı: {df.shape[1]}")
    
    print("\n=== Sütunlar ve Veri Tipleri ===")
    print(df.dtypes)
    
    print("\n=== İlk 5 Kayıt ===")
    print(df.head())
    
    print("\n=== Temel İstatistikler ===")
    print(df.describe())
    
    # Tarih aralığını göster
    date_columns = df.select_dtypes(include=['datetime64[ns]']).columns
    if len(date_columns) > 0:
        date_col = date_columns[0]
        print(f"\n=== Tarih Aralığı ===")
        print(f"Başlangıç: {df[date_col].min()}")
        print(f"Bitiş: {df[date_col].max()}")
        print(f"Toplam Gün: {(df[date_col].max() - df[date_col].min()).days}")

if __name__ == "__main__":
    # Çıktı dizini
    output_dir = "processed_data"
    
    # Tüm ETH verilerini oku
    print("\n=== VERİLER OKUNUYOR ===")
    df = read_all_eth_data()
    
    # Dakikalık verileri kaydet
    print("\n=== DAKİKALIK VERİLER HAZIRLANIYOR ===")
    preview_eth_data(df, "1min")
    save_data(df, os.path.join(output_dir, "eth_1min.parquet"))
    
    # Saatlik mum çubuğu verilerini oluştur ve kaydet
    print("\n=== SAATLİK VERİLER HAZIRLANIYOR ===")
    hourly_df = get_hourly_candles(df)
    preview_eth_data(hourly_df, "1hour")
    save_data(hourly_df, os.path.join(output_dir, "eth_1hour.parquet")) 