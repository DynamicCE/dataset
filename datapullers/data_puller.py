import requests
import csv
import time
from datetime import datetime, timedelta

def binance_verisi_cek(symbol, interval, start_date, end_date):
    """
    Binance API'den belirli bir zaman aralığı için mum verisi çeker.

    Args:
        symbol (str): Sembol çifti (örn. "SOLUSDT").
        interval (str): Mum aralığı (örn. "5m", "1h", "1d").
        start_date (str): Başlangıç tarihi (YYYY-MM-DD formatında).
        end_date (str): Bitiş tarihi (YYYY-MM-DD formatında).

    Returns:
        list: Mum verisi listesi. Her mum şu formatta bir listedir:
              [Open Time, Open, High, Low, Close, Volume, Close Time, Quote Asset Volume, Number of Trades, Taker buy base asset volume, Taker buy quote asset volume, Ignore]
              Başarısız olursa None döner.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    limit = 1000  # Binance API limiti, maksimum 1000 mum alabiliriz bir seferde

    mum_verisi = []

    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "limit": limit
        }
        if (start_ts + limit * interval_ms(interval)) < end_ts:
             params["endTime"] = start_ts + limit * interval_ms(interval) -1
        else:
            params["endTime"] = end_ts


        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if not data: # eğer veri yoksa döngüden çık
                break
            mum_verisi.extend(data)
            start_ts = int(data[-1][6]) + 1 # Son mumun kapanış zamanından sonraki milisaniye başlasın
            time.sleep(0.1) # API limitlerini aşmamak için küçük bir bekleme
        else:
            print(f"Hata kodu: {response.status_code}")
            print(response.json()) # Hata detaylarını yazdır
            return None # Hata durumunda None dön
    return mum_verisi

def interval_ms(interval):
    """
    Mum aralığını milisaniyeye çevirir. Sadece 5 dakikalık aralık için.
    """
    if interval == "5m":
        return 5 * 60 * 1000
    else:
        raise ValueError("Sadece 5 dakikalık aralık destekleniyor.")


def csv_kaydet(data, filename="sol_5dk_verisi.csv"):
    """
    Mum verisini CSV dosyasına kaydeder.

    Args:
        data (list): Mum verisi listesi.
        filename (str): CSV dosya adı.
    """
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Başlık satırı
        csv_writer.writerow(["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Asset Volume", "Number of Trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
        csv_writer.writerows(data)
    print(f"Veri '{filename}' dosyasına kaydedildi.")

if __name__ == "__main__":
    symbol = "SOLUSDT"
    interval = "5m"
    end_date = datetime.now().strftime("%Y-%m-%d") # Bugünün tarihi
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d") # Bir yıl öncesi

    print(f"{symbol} için {interval} aralığında, {start_date} - {end_date} tarihleri arası veri çekiliyor...")
    mum_data = binance_verisi_cek(symbol, interval, start_date, end_date)

    if mum_data:
        csv_kaydet(mum_data, filename=f"{symbol}_{interval}_yillik_veri.csv")
    else:
        print("Veri çekme işlemi başarısız oldu.")