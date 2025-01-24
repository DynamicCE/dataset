# Crypto Trading Strategy Analysis

Inside Bar ve Volume bazlı stratejilerin kripto varlıklar üzerinde analizi.

## Project Structure

```
.
├── data/               # Ham ve işlenmiş veriler
├── strategies/         # Trading stratejileri
├── analysis/          # Analiz sonuçları
│   ├── patterns/      # Tekrar eden yapılar
│   ├── statistics/    # İstatistiksel analizler
│   └── results/       # Strateji test sonuçları
└── docs/              # Dokümantasyon
```

## Gereksinimler

```bash
pip install -r requirements.txt
```

## Kullanım

```python
python strategy_tester.py
```

## Özellikler

- Saatlik ve dakikalık ETH verilerini işleme
- IBBVCTL (Inside Bar Breakout Volume Change Trend Line) stratejisi implementasyonu
- Overbought/Oversold ve Divergence filtreleri
- Detaylı backtest sonuçları
- Equity eğrisi görselleştirme

## Strateji Parametreleri

- Order Size: 50% of equity
- Pyramiding: 2
- Commission: 0.015
- ATR Length: 20
- ATR Multiplier for Stop Loss: 3.5
- ATR Multiplier for Take Profit: 2
- Length for Overbought/Oversold: 2
- Initial Equity: 100,000 USD

## Çıktılar

Strateji testi sonucunda şu bilgiler üretilir:

- Net kar/zarar
- Maksimum drawdown
- Profit factor
- Toplam işlem sayısı
- Kazançlı işlem yüzdesi
- Sharpe ve Sortino oranları
- Margin calls (varsa)

## Lisans

MIT
