# Ethereum Trading Strategy Backtester

Bu proje, Inside Bar + HV with Donchian Trailing stratejisini Ethereum üzerinde test etmek için geliştirilmiştir.

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

1. Ethereum verilerinizi CSV formatında projenin ana dizinine yerleştirin
2. Aşağıdaki komutu çalıştırın:

```bash
python strategy_runner.py
```

## Sonuçlar

Backtest sonuçları `analysis/results/` dizininde oluşturulacaktır:

- `equity_curve.png`: Sermaye eğrisi grafiği
- `performance_metrics.csv`: Performans metrikleri

## Strateji Detayları

- Inside Bar pattern tespiti
- Tarihsel Volatilite (HV) filtresi
- Donchian Channel bazlı trailing stop
- %100 pozisyon büyüklüğü
- %0.2 komisyon oranı
