import pandas as pd
import requests
import csv
from datetime import datetime

# Binance API endpoint
url = "https://api.binance.com/api/v3/klines"

# Parametreler
symbol = "SOLUSDT"  # Solana için trading pair
interval = "5m"     # 5 dakikalık veri
limit = 1000        # Max 1000 veri çekebilirsiniz

# Binance API'ye istek gönder
response = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit})
data = response.json()

# Veriyi DataFrame'e dönüştür
columns = ["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Asset Volume", 
           "Number of Trades", "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore"]
df = pd.DataFrame(data, columns=columns)

# Zaman damgalarını okunabilir formata çevir
df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

# CSV olarak kaydet
df.to_csv("solana_5m_data.csv", index=False)
print("Veri başarıyla kaydedildi: solana_5m_data.csv")