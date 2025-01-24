import pandas as pd
import numpy as np
from eth_data_reader import read_all_eth_data, get_hourly_candles
import matplotlib.pyplot as plt
from datetime import datetime
import os

class IBBVCTLStrategy:
    def __init__(self, 
                 initial_equity=100000,
                 order_size_pct=0.5,
                 pyramiding=2,
                 commission=0.015,
                 atr_length=20,
                 atr_multiplier_sl=3.5,
                 atr_multiplier_tp=2.0,
                 overbought_oversold_length=2,
                 max_volatility=30.0,  # Maksimum volatilite limiti
                 base_position_size=0.75,  # Temel pozisyon büyüklüğü
                 atr_position_divider=20.0,  # ATR pozisyon bölen
                 min_position_size=0.1,
                 trend_length=200,  # Trend tespiti için MA uzunluğu
                 min_volume_multiplier=1.5):  # Minimum hacim çarpanı
        
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
        
    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_volatility(self, df, length=10):
        # Pinescript ile aynı formül
        returns = np.log(df['close'] / df['close'].shift(1))
        annual = 365
        per = 1  # Saatlik veriler için 1
        hv = 100 * returns.rolling(window=length).std() * np.sqrt(annual / per)
        return hv
    
    def is_inside_bar(self, df, i):
        if i < 1:
            return False
        return (df['high'].iloc[i] < df['high'].iloc[i-1] and 
                df['low'].iloc[i] > df['low'].iloc[i-1])
    
    def calculate_overbought_oversold(self, df):
        # Pinescript ile aynı formül
        ys1 = (df['high'] + df['low'] + df['close'] * 2) / 4
        rk3 = ys1.ewm(span=self.overbought_oversold_length, adjust=False).mean()
        rk4 = ys1.rolling(window=self.overbought_oversold_length).std()
        rk5 = (ys1 - rk3) * 100 / rk4
        rk6 = rk5.ewm(span=self.overbought_oversold_length, adjust=False).mean()
        up = rk6.ewm(span=self.overbought_oversold_length, adjust=False).mean()
        down = up.ewm(span=self.overbought_oversold_length, adjust=False).mean()
        return up, down
    
    def calculate_rsi(self, df, period=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def calculate_stochastic(self, df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return k.rolling(window=3).mean()
    
    def calculate_cci(self, df, period=14):
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mean_deviation)
    
    def calculate_obv(self, df):
        close_diff = df['close'].diff()
        # Volume verisini düzelt
        volume = pd.to_numeric(df['volume'].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
        obv = pd.Series(0, index=df.index)
        
        for i in range(1, len(df)):
            if close_diff.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close_diff.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def check_divergence(self, df, i, indicator, is_positive=True, lookback=14):
        if i < lookback:
            return False
            
        if is_positive:
            # Son crossover noktasını bul
            for j in range(i-1, i-lookback, -1):
                if j < 1:
                    return False
                # Indicator lowest değerini geçiyor mu?
                indicator_lowest = indicator.iloc[j-lookback:j].min()
                if (indicator.iloc[j-1] <= indicator_lowest and 
                    indicator.iloc[j] > indicator_lowest):
                    # İlk crossover noktası
                    first_cross_price = df['close'].iloc[j]
                    first_cross_indicator = indicator.iloc[j]
                    
                    # İkinci crossover noktasını ara
                    for k in range(j-1, j-lookback, -1):
                        if k < 1:
                            return False
                        if (indicator.iloc[k-1] <= indicator_lowest and 
                            indicator.iloc[k] > indicator_lowest):
                            # İkinci crossover noktası
                            second_cross_price = df['close'].iloc[k]
                            second_cross_indicator = indicator.iloc[k]
                            
                            # Divergence kontrolü
                            return (first_cross_price < second_cross_price and 
                                   first_cross_indicator > second_cross_indicator)
                    break
        else:
            # Son crossunder noktasını bul
            for j in range(i-1, i-lookback, -1):
                if j < 1:
                    return False
                # Indicator highest değerini geçiyor mu?
                indicator_highest = indicator.iloc[j-lookback:j].max()
                if (indicator.iloc[j-1] >= indicator_highest and 
                    indicator.iloc[j] < indicator_highest):
                    # İlk crossunder noktası
                    first_cross_price = df['close'].iloc[j]
                    first_cross_indicator = indicator.iloc[j]
                    
                    # İkinci crossunder noktasını ara
                    for k in range(j-1, j-lookback, -1):
                        if k < 1:
                            return False
                        if (indicator.iloc[k-1] >= indicator_highest and 
                            indicator.iloc[k] < indicator_highest):
                            # İkinci crossunder noktası
                            second_cross_price = df['close'].iloc[k]
                            second_cross_indicator = indicator.iloc[k]
                            
                            # Divergence kontrolü
                            return (first_cross_price > second_cross_price and 
                                   first_cross_indicator < second_cross_indicator)
                    break
        
        return False
    
    def check_crossover(self, series1, series2, i):
        if i < 1:
            return False
        return series1.iloc[i-1] <= series2.iloc[i-1] and series1.iloc[i] > series2.iloc[i]
    
    def check_crossunder(self, series1, series2, i):
        if i < 1:
            return False
        return series1.iloc[i-1] >= series2.iloc[i-1] and series1.iloc[i] < series2.iloc[i]
    
    def calculate_trend(self, df):
        """Trend yönünü hesapla"""
        ma = df['close'].rolling(window=self.trend_length).mean()
        trend = pd.Series(index=df.index)
        trend[df['close'] > ma] = 1  # Yükselen trend
        trend[df['close'] < ma] = -1  # Düşen trend
        return trend
        
    def calculate_dynamic_tp_multiplier(self, trend, volatility):
        """Trend ve volatiliteye göre dinamik TP çarpanı hesapla"""
        base_multiplier = self.atr_multiplier_tp
        
        # Trend yönünde TP'yi uzat
        if trend == 1:  # Yükselen trend
            base_multiplier *= 1.2
        elif trend == -1:  # Düşen trend
            base_multiplier *= 0.8
            
        # Volatiliteye göre ayarla
        if volatility > 25:
            base_multiplier *= 0.8  # Yüksek volatilitede TP'yi yakınlaştır
        elif volatility < 15:
            base_multiplier *= 1.2  # Düşük volatilitede TP'yi uzaklaştır
            
        return base_multiplier
        
    def calculate_dynamic_stop_multiplier(self, trend, volatility):
        """Trend ve volatiliteye göre dinamik stop çarpanı hesapla"""
        base_multiplier = self.atr_multiplier_sl
        
        # Trend yönünde SL'yi uzat
        if trend == 1:  # Yükselen trend
            base_multiplier *= 0.9  # Long pozisyonlarda SL'yi yakınlaştır
        elif trend == -1:  # Düşen trend
            base_multiplier *= 1.1  # Short pozisyonlarda SL'yi uzaklaştır
            
        # Volatiliteye göre ayarla
        if volatility > 25:
            base_multiplier *= 1.2
        elif volatility > 20:
            base_multiplier *= 1.1
            
        return base_multiplier
        
    def calculate_position_size(self, price, atr, volatility):
        """ATR ve volatiliteye göre pozisyon büyüklüğü hesapla"""
        # ATR bazlı pozisyon büyüklüğü
        atr_based_size = self.base_position_size * (1 / (atr / self.atr_position_divider))
        
        # Volatilite bazlı azaltma
        vol_factor = max(0.2, 1 - (volatility / self.max_volatility))
        
        # Final pozisyon büyüklüğü
        position_size = min(self.base_position_size, atr_based_size) * vol_factor
        
        # Minimum değerden küçük olmasın
        position_size = max(position_size, self.min_position_size)
        
        return position_size
        
    def is_valid_trading_hour(self, timestamp):
        """İşlem saati kontrolü"""
        hour = timestamp.hour
        # 01:00-02:00 arası işlem yapma
        return hour not in [1, 2]
    
    def is_volume_sufficient(self, df, i):
        """Hacim yeterli mi kontrol et"""
        # Volume verisini düzelt
        volume = pd.to_numeric(df['volume'].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
        avg_volume = volume.rolling(window=20).mean()
        return volume.iloc[i] > avg_volume.iloc[i] * self.min_volume_multiplier
    
    def backtest(self, df):
        df = df.copy()
        
        # Teknik göstergeler
        df['atr'] = self.calculate_atr(df, self.atr_length)
        df['volatility'] = self.calculate_volatility(df)
        df['volatility_increase'] = (df['volatility'] - df['volatility'].shift(1)) / df['volatility'].shift(1) >= 0.27
        df['trend'] = self.calculate_trend(df)
        
        df['rsi'] = self.calculate_rsi(df)
        df['macd'], df['signal'] = self.calculate_macd(df)
        df['stoch'] = self.calculate_stochastic(df)
        df['cci'] = self.calculate_cci(df)
        df['obv'] = self.calculate_obv(df)
        
        up, down = self.calculate_overbought_oversold(df)
        df['up'] = up
        df['down'] = down
        
        self.equity_curve = [self.initial_equity]
        current_position = None
        open_trades = []
        
        inside_bar_count = 0
        breakout_count = 0
        breakdown_count = 0
        
        # Inside bar ve sinyal takibi için değişkenler
        last_inside_bar = {'high': None, 'low': None, 'index': None}
        breakout_occurred = False
        breakdown_occurred = False
        time_limit = 10  # Inside bar sonrası sinyal için zaman limiti
        
        # İstatistik değişkenleri
        self.long_stops = []
        self.long_tps = []
        self.short_stops = []
        self.short_tps = []
        self.stop_times = []
        self.tp_times = []
        
        for i in range(1, len(df)):
            current_bar = {
                'high': df['high'].iloc[i],
                'low': df['low'].iloc[i],
                'close': df['close'].iloc[i]
            }
            
            current_time = df.index[i]
            current_volatility = df['volatility'].iloc[i]
            current_atr = df['atr'].iloc[i]
            
            # Volatilite kontrolü
            if current_volatility > self.max_volatility:
                continue
                
            # Saat kontrolü
            if not self.is_valid_trading_hour(current_time):
                continue
            
            # Hacim kontrolü ekle
            if not self.is_volume_sufficient(df, i):
                continue
            
            # Overbought/Oversold sinyalleri
            buy_signal_oos = self.check_crossover(df['up'], df['down'], i)
            sell_signal_oos = self.check_crossunder(df['up'], df['down'], i)
            
            # Divergence kontrolleri
            positive_divergence = (
                self.check_divergence(df, i, df['stoch'], True)
            )
            
            negative_divergence = (
                self.check_divergence(df, i, df['stoch'], False)
            )
            
            # Inside bar kontrolü
            if self.is_inside_bar(df, i):
                inside_bar_count += 1
                last_inside_bar = {
                    'high': df['high'].iloc[i-1],
                    'low': df['low'].iloc[i-1],
                    'index': i
                }
                breakout_occurred = False
                breakdown_occurred = False
            
            # Sinyal kontrolleri
            if last_inside_bar['high'] is not None:
                within_time_limit = (i - last_inside_bar['index']) <= time_limit
                
                if within_time_limit:
                    # Pozisyon büyüklüğü hesapla
                    position_size_pct = self.calculate_position_size(
                        current_bar['close'],
                        current_atr,
                        current_volatility
                    )
                    
                    # Trend bazlı stop ve TP çarpanları
                    current_trend = df['trend'].iloc[i]
                    dynamic_stop_multiplier = self.calculate_dynamic_stop_multiplier(current_trend, current_volatility)
                    dynamic_tp_multiplier = self.calculate_dynamic_tp_multiplier(current_trend, current_volatility)
                    
                    # Breakout kontrolü
                    if (not breakdown_occurred and 
                        current_bar['close'] > last_inside_bar['high'] and 
                        df['volatility_increase'].iloc[i] and
                        not negative_divergence and
                        not sell_signal_oos and
                        (current_trend == 1 or current_volatility < 20)):  # Trend filtresi
                        
                        breakout_count += 1
                        breakout_occurred = True
                        
                        entry_price = current_bar['close']
                        position_size = (self.equity * position_size_pct) / entry_price
                        stop_loss = entry_price - current_atr * dynamic_stop_multiplier
                        take_profit = entry_price + current_atr * dynamic_tp_multiplier
                        
                        trade = {
                            'type': 'long',
                            'entry_price': entry_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': df.index[i]
                        }
                        if current_position is None or len(open_trades) < self.pyramiding:
                            open_trades.append(trade)
                    
                    # Breakdown kontrolü
                    elif (not breakout_occurred and 
                          current_bar['close'] < last_inside_bar['low'] and 
                          df['volatility_increase'].iloc[i] and
                          not positive_divergence and
                          not buy_signal_oos and
                          (current_trend == -1 or current_volatility < 20)):  # Trend filtresi
                        
                        breakdown_count += 1
                        breakdown_occurred = True
                        
                        entry_price = current_bar['close']
                        position_size = (self.equity * position_size_pct) / entry_price
                        stop_loss = entry_price + current_atr * dynamic_stop_multiplier
                        take_profit = entry_price - current_atr * dynamic_tp_multiplier
                        
                        trade = {
                            'type': 'short',
                            'entry_price': entry_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'entry_time': df.index[i]
                        }
                        if current_position is None or len(open_trades) < self.pyramiding:
                            open_trades.append(trade)
                
                if not within_time_limit or breakout_occurred or breakdown_occurred:
                    last_inside_bar = {'high': None, 'low': None, 'index': None}
            
            # Açık pozisyonları kontrol et
            for trade in open_trades[:]:
                current_price = current_bar['close']
                pnl = 0
                
                if trade['type'] == 'long':
                    if current_price <= trade['stop_loss']:
                        pnl = (current_price - trade['entry_price']) * trade['size']
                        pnl -= abs(pnl) * self.commission
                        self.equity += pnl
                        trade['exit_price'] = current_price
                        trade['exit_time'] = df.index[i]
                        trade['pnl'] = pnl
                        trade['exit_type'] = 'stop'
                        self.trades.append(trade)
                        self.long_stops.append({
                            'entry_price': trade['entry_price'],
                            'stop_price': trade['stop_loss'],
                            'entry_time': trade['entry_time'],
                            'exit_time': trade['exit_time'],
                            'atr': current_atr,
                            'volatility': current_volatility
                        })
                        self.stop_times.append(df.index[i].hour)
                        open_trades.remove(trade)
                    elif current_price >= trade['take_profit']:
                        pnl = (current_price - trade['entry_price']) * trade['size']
                        pnl -= abs(pnl) * self.commission
                        self.equity += pnl
                        trade['exit_price'] = current_price
                        trade['exit_time'] = df.index[i]
                        trade['pnl'] = pnl
                        trade['exit_type'] = 'tp'
                        self.trades.append(trade)
                        self.long_tps.append({
                            'entry_price': trade['entry_price'],
                            'tp_price': trade['take_profit'],
                            'entry_time': trade['entry_time'],
                            'exit_time': trade['exit_time'],
                            'atr': current_atr,
                            'volatility': current_volatility
                        })
                        self.tp_times.append(df.index[i].hour)
                        open_trades.remove(trade)
                
                elif trade['type'] == 'short':
                    if current_price >= trade['stop_loss']:
                        pnl = (trade['entry_price'] - current_price) * trade['size']
                        pnl -= abs(pnl) * self.commission
                        self.equity += pnl
                        trade['exit_price'] = current_price
                        trade['exit_time'] = df.index[i]
                        trade['pnl'] = pnl
                        trade['exit_type'] = 'stop'
                        self.trades.append(trade)
                        self.short_stops.append({
                            'entry_price': trade['entry_price'],
                            'stop_price': trade['stop_loss'],
                            'entry_time': trade['entry_time'],
                            'exit_time': trade['exit_time'],
                            'atr': current_atr,
                            'volatility': current_volatility
                        })
                        self.stop_times.append(df.index[i].hour)
                        open_trades.remove(trade)
                    elif current_price <= trade['take_profit']:
                        pnl = (trade['entry_price'] - current_price) * trade['size']
                        pnl -= abs(pnl) * self.commission
                        self.equity += pnl
                        trade['exit_price'] = current_price
                        trade['exit_time'] = df.index[i]
                        trade['pnl'] = pnl
                        trade['exit_type'] = 'tp'
                        self.trades.append(trade)
                        self.short_tps.append({
                            'entry_price': trade['entry_price'],
                            'tp_price': trade['take_profit'],
                            'entry_time': trade['entry_time'],
                            'exit_time': trade['exit_time'],
                            'atr': current_atr,
                            'volatility': current_volatility
                        })
                        self.tp_times.append(df.index[i].hour)
                        open_trades.remove(trade)
            
            self.equity_curve.append(self.equity)
            self.max_equity = max(self.max_equity, self.equity)
            self.max_drawdown = max(self.max_drawdown, (self.max_equity - self.equity) / self.max_equity)
        
        # Debug bilgilerini yazdır
        print(f"\nToplam Inside Bar Sayısı: {inside_bar_count}")
        print(f"Toplam Breakout Sayısı: {breakout_count}")
        print(f"Toplam Breakdown Sayısı: {breakdown_count}")
        print(f"Toplam İşlem Sayısı: {len(self.trades)}")
        
        # Stop ve TP istatistiklerini yazdır
        total_stops = len(self.long_stops) + len(self.short_stops)
        total_tps = len(self.long_tps) + len(self.short_tps)
        
        print("\nStop/TP İstatistikleri:")
        print(f"Toplam Stop Sayısı: {total_stops}")
        print(f"Toplam TP Sayısı: {total_tps}")
        print(f"Stop/TP Oranı: {total_stops/(total_stops+total_tps)*100:.2f}%")
        
        print("\nLong Pozisyonlar:")
        print(f"Stop Sayısı: {len(self.long_stops)}")
        print(f"TP Sayısı: {len(self.long_tps)}")
        
        print("\nShort Pozisyonlar:")
        print(f"Stop Sayısı: {len(self.short_stops)}")
        print(f"TP Sayısı: {len(self.short_tps)}")
        
        # Stop zamanları analizi
        if self.stop_times:
            stop_hours = pd.Series(self.stop_times).value_counts().sort_index()
            print("\nEn Çok Stop Olunan Saatler:")
            for hour, count in stop_hours.nlargest(5).items():
                print(f"Saat {hour:02d}:00 - {count} stop")
        
        # ATR ve Volatilite analizi
        if self.long_stops or self.short_stops:
            all_stops = self.long_stops + self.short_stops
            avg_atr = sum(stop['atr'] for stop in all_stops) / len(all_stops)
            avg_vol = sum(stop['volatility'] for stop in all_stops) / len(all_stops)
            print(f"\nStop Olduğumuzda Ortalama ATR: {avg_atr:.2f}")
            print(f"Stop Olduğumuzda Ortalama Volatilite: {avg_vol:.2f}%")
    
    def get_statistics(self):
        if not self.trades:
            return {
                'Net Profit': 0,
                'Max Drawdown (%)': 0,
                'Profit Factor': 0,
                'Total Trades': 0,
                'Profitable Trades (%)': 0,
                'Sharpe Ratio': 0,
                'Sortino Ratio': 0,
                'Final Equity': self.initial_equity
            }
        
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['pnl'] > 0])
        total_profit = sum(t['pnl'] for t in self.trades)
        
        # Sharpe ve Sortino Oranları
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        excess_returns = returns - 0.02/365  # Risk-free rate (günlük)
        sharpe_ratio = np.sqrt(365) * excess_returns.mean() / excess_returns.std()
        
        negative_returns = excess_returns[excess_returns < 0]
        sortino_ratio = np.sqrt(365) * excess_returns.mean() / negative_returns.std() if len(negative_returns) > 0 else 0
        
        profit_factor = abs(sum(t['pnl'] for t in self.trades if t['pnl'] > 0)) / abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0)) if sum(t['pnl'] for t in self.trades if t['pnl'] < 0) != 0 else float('inf')
        
        stats = {
            'Net Profit': total_profit,
            'Max Drawdown (%)': self.max_drawdown * 100,
            'Profit Factor': profit_factor,
            'Total Trades': total_trades,
            'Profitable Trades (%)': (profitable_trades / total_trades) * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Final Equity': self.equity
        }
        
        return stats

def main():
    # Veriyi oku
    print("Veriler okunuyor...")
    hourly_df = pd.read_parquet("processed_data/eth_1hour.parquet")
    
    # Tüm veriyi kullan
    filtered_df = hourly_df.copy()
    filtered_df.set_index('timestamp', inplace=True)
    
    # Debug: Veri yapısını kontrol et
    print("\nVeri Yapısı:")
    print(filtered_df.head())
    print("\nSütunlar:", filtered_df.columns.tolist())
    print("\nVeri Boyutu:", filtered_df.shape)
    print("\nVeri Aralığı:")
    print("Başlangıç:", filtered_df.index.min())
    print("Bitiş:", filtered_df.index.max())
    print("\nEksik Değerler:")
    print(filtered_df.isnull().sum())
    
    # Stratejiyi başlat
    strategy = IBBVCTLStrategy(
        initial_equity=100000,
        order_size_pct=0.5,
        pyramiding=2,
        commission=0.015,
        atr_length=20,
        atr_multiplier_sl=3.5,
        atr_multiplier_tp=2.0,
        overbought_oversold_length=2,
        max_volatility=30.0,
        base_position_size=0.75,
        atr_position_divider=20.0,
        min_position_size=0.1,
        trend_length=200,
        min_volume_multiplier=1.5
    )
    
    # Backtest
    print("Backtest yapılıyor...")
    strategy.backtest(filtered_df)
    
    # Sonuçları al ve kaydet
    stats = strategy.get_statistics()
    
    # Sonuçları dosyaya kaydet
    results_dir = "processed_data"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "backtest_results.txt"), "w") as f:
        f.write("=== IBBVCTL Strateji Test Sonuçları ===\n")
        f.write(f"Test Periyodu: {filtered_df.index.min().strftime('%Y-%m-%d')} - {filtered_df.index.max().strftime('%Y-%m-%d')}\n\n")
        
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Equity eğrisini çiz ve kaydet
    plt.figure(figsize=(12, 6))
    plt.plot(strategy.equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Bar Sayısı')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "equity_curve.png"))
    plt.close()
    
    print(f"\nSonuçlar {results_dir} klasörüne kaydedildi.")
    print("\nStrateji Test Sonuçları:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()
