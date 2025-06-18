# 技术指标计算模块
from AlgorithmImports import *
import numpy as np

# 条件导入talib - 本地测试时可能不可用，但QuantConnect平台支持
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: talib not available in local environment, using fallback methods")

from config import TechnicalConfig

class TechnicalIndicatorCalculator:
    """技术指标计算器 - 负责各种技术指标的计算"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.tech_config = TechnicalConfig()
        
    def calculate_technical_indicators(self, prices, volumes=None):
        """计算技术指标"""
        indicators = {}
        
        try:
            # 数据预处理和验证
            if len(prices) < 50:
                self.algorithm.log_debug(f"Insufficient data for technical indicators: {len(prices)} < 50", log_type='data')
                return None
            
            # 确保价格数据没有NaN或异常值
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)) or np.any(prices <= 0):
                self.algorithm.log_debug("Invalid price data detected in technical indicators", log_type='data')
                return None
            
            # 价格相关指标 - 添加异常处理
            if TALIB_AVAILABLE:
                try:
                    indicators['sma_20'] = talib.SMA(prices, timeperiod=20)
                    indicators['sma_50'] = talib.SMA(prices, timeperiod=50)
                    indicators['ema_12'] = talib.EMA(prices, timeperiod=12)
                    indicators['ema_26'] = talib.EMA(prices, timeperiod=26)
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating moving averages: {e}", log_type='data')
                    # 提供备用计算
                    indicators['sma_20'] = self._safe_moving_average(prices, 20)
                    indicators['sma_50'] = self._safe_moving_average(prices, 50)
                    indicators['ema_12'] = self._safe_ema(prices, 12)
                    indicators['ema_26'] = self._safe_ema(prices, 26)
            else:
                # 直接使用备用计算
                indicators['sma_20'] = self._safe_moving_average(prices, 20)
                indicators['sma_50'] = self._safe_moving_average(prices, 50)
                indicators['ema_12'] = self._safe_ema(prices, 12)
                indicators['ema_26'] = self._safe_ema(prices, 26)
            
            # RSI - 增强异常处理
            if TALIB_AVAILABLE:
                try:
                    rsi = talib.RSI(prices, timeperiod=self.tech_config.TECHNICAL_INDICATORS['rsi_period'])
                    # 处理RSI的边界情况
                    if rsi is not None:
                        rsi = np.nan_to_num(rsi, nan=50.0)  # RSI的NaN用50填充（中性）
                        rsi = np.clip(rsi, 0, 100)  # 确保RSI在0-100范围内
                        indicators['rsi'] = rsi
                    else:
                        indicators['rsi'] = self._safe_rsi(prices, self.tech_config.TECHNICAL_INDICATORS['rsi_period'])
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating RSI: {e}", log_type='data')
                    indicators['rsi'] = self._safe_rsi(prices, self.tech_config.TECHNICAL_INDICATORS['rsi_period'])
            else:
                indicators['rsi'] = self._safe_rsi(prices, self.tech_config.TECHNICAL_INDICATORS['rsi_period'])
            
            # MACD - 增强异常处理
            if TALIB_AVAILABLE:
                try:
                    macd, macd_signal, macd_hist = talib.MACD(prices, 
                                                             fastperiod=12, 
                                                             slowperiod=26, 
                                                             signalperiod=9)
                    indicators['macd'] = np.nan_to_num(macd, nan=0.0)
                    indicators['macd_signal'] = np.nan_to_num(macd_signal, nan=0.0)
                    indicators['macd_hist'] = np.nan_to_num(macd_hist, nan=0.0)
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating MACD: {e}", log_type='data')
                    # 简化的MACD计算
                    ema_12 = self._safe_ema(prices, 12)
                    ema_26 = self._safe_ema(prices, 26)
                    macd = ema_12 - ema_26
                    indicators['macd'] = np.nan_to_num(macd, nan=0.0)
                    indicators['macd_signal'] = self._safe_ema(macd, 9)
                    indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            else:
                # 简化的MACD计算
                ema_12 = self._safe_ema(prices, 12)
                ema_26 = self._safe_ema(prices, 26)
                macd = ema_12 - ema_26
                indicators['macd'] = np.nan_to_num(macd, nan=0.0)
                indicators['macd_signal'] = self._safe_ema(macd, 9)
                indicators['macd_hist'] = indicators['macd'] - indicators['macd_signal']
            
            # 布林带 - 增强异常处理
            if TALIB_AVAILABLE:
                try:
                    bb_upper, bb_middle, bb_lower = talib.BBANDS(prices, 
                                                               timeperiod=20, 
                                                               nbdevup=2, 
                                                               nbdevdn=2)
                    indicators['bb_upper'] = np.nan_to_num(bb_upper, nan=prices)
                    indicators['bb_middle'] = np.nan_to_num(bb_middle, nan=prices)
                    indicators['bb_lower'] = np.nan_to_num(bb_lower, nan=prices)
                    
                    # 布林带宽度
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    indicators['bb_width'] = np.nan_to_num(bb_width, nan=0.1)
                    
                    # 布林带位置
                    bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
                    indicators['bb_position'] = np.nan_to_num(bb_position, nan=0.5)
                    
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating Bollinger Bands: {e}", log_type='data')
                    # 简化的布林带计算
                    sma_20 = self._safe_moving_average(prices, 20)
                    rolling_std = self._safe_rolling_std(prices, 20)
                    indicators['bb_upper'] = sma_20 + 2 * rolling_std
                    indicators['bb_middle'] = sma_20
                    indicators['bb_lower'] = sma_20 - 2 * rolling_std
                    indicators['bb_width'] = np.full_like(prices, 0.1)
                    indicators['bb_position'] = np.full_like(prices, 0.5)
            else:
                # 简化的布林带计算
                sma_20 = self._safe_moving_average(prices, 20)
                rolling_std = self._safe_rolling_std(prices, 20)
                indicators['bb_upper'] = sma_20 + 2 * rolling_std
                indicators['bb_middle'] = sma_20
                indicators['bb_lower'] = sma_20 - 2 * rolling_std
                indicators['bb_width'] = np.full_like(prices, 0.1)
                indicators['bb_position'] = np.full_like(prices, 0.5)
            
            # 成交量指标（如果有成交量数据）
            if volumes is not None and len(volumes) == len(prices):
                try:
                    # 成交量移动平均
                    indicators['volume_sma'] = self._safe_moving_average(volumes, 20)
                    
                    # 成交量比率
                    volume_ratio = volumes / indicators['volume_sma']
                    indicators['volume_ratio'] = np.nan_to_num(volume_ratio, nan=1.0)
                    
                    # OBV (On Balance Volume)
                    obv = self._calculate_obv(prices, volumes)
                    indicators['obv'] = np.nan_to_num(obv, nan=0.0)
                    
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating volume indicators: {e}", log_type='data')
                    indicators['volume_sma'] = np.full_like(volumes, np.mean(volumes))
                    indicators['volume_ratio'] = np.full_like(volumes, 1.0)
                    indicators['obv'] = np.zeros_like(volumes)
            else:
                # 无成交量数据时的默认值
                indicators['volume_sma'] = np.ones_like(prices)
                indicators['volume_ratio'] = np.ones_like(prices)
                indicators['obv'] = np.zeros_like(prices)
            
            # ATR (Average True Range) - 增强异常处理
            if TALIB_AVAILABLE:
                try:
                    # 构造高低收价格数据
                    high_prices = prices * 1.02  # 假设高价比收盘价高2%
                    low_prices = prices * 0.98   # 假设低价比收盘价低2%
                    
                    atr = talib.ATR(high_prices, low_prices, prices, timeperiod=14)
                    indicators['atr'] = np.nan_to_num(atr, nan=np.std(prices))
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating ATR: {e}", log_type='data')
                    indicators['atr'] = self._safe_atr(prices, 14)
            else:
                indicators['atr'] = self._safe_atr(prices, 14)
            
            # 动量指标
            if TALIB_AVAILABLE:
                try:
                    momentum = talib.MOM(prices, timeperiod=10)
                    indicators['momentum'] = np.nan_to_num(momentum, nan=0.0)
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating momentum: {e}", log_type='data')
                    # 简单动量计算
                    momentum = np.zeros_like(prices)
                    momentum[10:] = prices[10:] - prices[:-10]
                    indicators['momentum'] = momentum
            else:
                # 简单动量计算
                momentum = np.zeros_like(prices)
                momentum[10:] = prices[10:] - prices[:-10]
                indicators['momentum'] = momentum
            
            # ROC (Rate of Change)
            if TALIB_AVAILABLE:
                try:
                    roc = talib.ROC(prices, timeperiod=10)
                    indicators['roc'] = np.nan_to_num(roc, nan=0.0)
                except Exception as e:
                    self.algorithm.log_debug(f"Error calculating ROC: {e}", log_type='data')
                    # 简单ROC计算
                    roc = np.zeros_like(prices)
                    roc[10:] = (prices[10:] - prices[:-10]) / prices[:-10] * 100
                    indicators['roc'] = roc
            else:
                # 简单ROC计算
                roc = np.zeros_like(prices)
                roc[10:] = (prices[10:] - prices[:-10]) / prices[:-10] * 100
                indicators['roc'] = roc
            
            return indicators
            
        except Exception as e:
            self.algorithm.log_debug(f"Critical error in technical indicators calculation: {e}", log_type='data')
            return None
    
    def _safe_moving_average(self, prices, window):
        """安全的移动平均计算"""
        try:
            if len(prices) < window:
                return np.full_like(prices, np.mean(prices))
            
            result = np.full_like(prices, np.nan)
            for i in range(window-1, len(prices)):
                result[i] = np.mean(prices[i-window+1:i+1])
            
            # 前面的NaN用第一个有效值填充
            first_valid = result[window-1]
            result[:window-1] = first_valid
            
            return result
        except:
            return np.full_like(prices, np.mean(prices))
    
    def _safe_ema(self, prices, window):
        """安全的指数移动平均计算"""
        try:
            alpha = 2.0 / (window + 1.0)
            result = np.zeros_like(prices)
            result[0] = prices[0]
            
            for i in range(1, len(prices)):
                result[i] = alpha * prices[i] + (1 - alpha) * result[i-1]
            
            return result
        except:
            return np.full_like(prices, np.mean(prices))
    
    def _safe_rsi(self, prices, window):
        """安全的RSI计算"""
        try:
            if len(prices) < window + 1:
                return np.full_like(prices, 50.0)
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gains = np.zeros_like(prices)
            avg_losses = np.zeros_like(prices)
            
            # 初始平均
            avg_gains[window] = np.mean(gains[:window])
            avg_losses[window] = np.mean(losses[:window])
            
            # 后续计算
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (window - 1) + gains[i-1]) / window
                avg_losses[i] = (avg_losses[i-1] * (window - 1) + losses[i-1]) / window
            
            rs = avg_gains / (avg_losses + 1e-10)  # 避免除零
            rsi = 100 - (100 / (1 + rs))
            
            # 前面的值用50填充
            rsi[:window] = 50.0
            
            return rsi
        except:
            return np.full_like(prices, 50.0)
    
    def _safe_rolling_std(self, prices, window):
        """安全的滚动标准差计算"""
        try:
            if len(prices) < window:
                return np.full_like(prices, np.std(prices))
            
            result = np.full_like(prices, np.nan)
            for i in range(window-1, len(prices)):
                result[i] = np.std(prices[i-window+1:i+1])
            
            # 前面的NaN用第一个有效值填充
            first_valid = result[window-1]
            result[:window-1] = first_valid
            
            return result
        except:
            return np.full_like(prices, np.std(prices))
    
    def _safe_atr(self, prices, window):
        """安全的ATR计算"""
        try:
            # 简化的ATR计算，使用价格变化的标准差
            if len(prices) < window:
                return np.full_like(prices, np.std(prices))
            
            returns = np.abs(np.diff(prices))
            result = np.full_like(prices, np.nan)
            
            for i in range(window, len(prices)):
                result[i] = np.mean(returns[i-window:i])
            
            # 填充前面的值
            first_valid = result[window]
            result[:window] = first_valid
            
            return result
        except:
            return np.full_like(prices, np.std(prices) * 0.1)
    
    def _calculate_obv(self, prices, volumes):
        """计算OBV指标"""
        try:
            obv = np.zeros_like(prices)
            obv[0] = volumes[0]
            
            for i in range(1, len(prices)):
                if prices[i] > prices[i-1]:
                    obv[i] = obv[i-1] + volumes[i]
                elif prices[i] < prices[i-1]:
                    obv[i] = obv[i-1] - volumes[i]
                else:
                    obv[i] = obv[i-1]
            
            return obv
        except:
            return np.zeros_like(prices)

class EnhancedTechnicalCalculator:
    """增强技术指标计算器 - 负责高级技术指标计算"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.tech_config = TechnicalConfig()
    
    def calculate_enhanced_technical_indicators(self, prices, volumes=None):
        """计算增强技术指标"""
        enhanced_indicators = {}
        
        try:
            if len(prices) < 50:
                return None
            
            # 1. 价格动量指标
            enhanced_indicators.update(self._calculate_momentum_indicators(prices))
            
            # 2. 波动率指标
            enhanced_indicators.update(self._calculate_volatility_indicators(prices))
            
            # 3. 趋势强度指标
            enhanced_indicators.update(self._calculate_trend_indicators(prices))
            
            # 4. 成交量指标（如果有数据）
            if volumes is not None:
                enhanced_indicators.update(self._calculate_volume_indicators(prices, volumes))
            
            # 5. 市场结构指标
            enhanced_indicators.update(self._calculate_market_structure_indicators(prices))
            
            return enhanced_indicators
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in enhanced technical indicators: {e}", log_type='data')
            return {}
    
    def _calculate_momentum_indicators(self, prices):
        """计算动量指标"""
        indicators = {}
        
        try:
            # 多周期ROC
            for period in [5, 10, 20]:
                roc = np.zeros_like(prices)
                if len(prices) > period:
                    roc[period:] = (prices[period:] - prices[:-period]) / prices[:-period] * 100
                indicators[f'roc_{period}'] = roc
            
            # 价格加速度
            if len(prices) > 2:
                velocity = np.diff(prices)
                acceleration = np.diff(velocity)
                acc_padded = np.concatenate([[0, 0], acceleration])
                indicators['price_acceleration'] = acc_padded
            else:
                indicators['price_acceleration'] = np.zeros_like(prices)
            
            return indicators
        except:
            return {}
    
    def _calculate_volatility_indicators(self, prices):
        """计算波动率指标"""
        indicators = {}
        
        try:
            # 历史波动率
            returns = np.diff(prices) / prices[:-1]
            returns_padded = np.concatenate([[0], returns])
            
            # 20日波动率
            rolling_vol = np.zeros_like(prices)
            for i in range(20, len(prices)):
                rolling_vol[i] = np.std(returns_padded[i-19:i+1]) * np.sqrt(252)
            
            # 填充前面的值
            if len(prices) >= 20:
                rolling_vol[:20] = rolling_vol[20]
            
            indicators['volatility_20d'] = rolling_vol
            
            # 波动率比率
            if len(prices) > 40:
                vol_short = np.zeros_like(prices)
                vol_long = np.zeros_like(prices)
                
                for i in range(10, len(prices)):
                    vol_short[i] = np.std(returns_padded[i-9:i+1])
                for i in range(30, len(prices)):
                    vol_long[i] = np.std(returns_padded[i-29:i+1])
                
                vol_ratio = vol_short / (vol_long + 1e-10)
                indicators['volatility_ratio'] = vol_ratio
            else:
                indicators['volatility_ratio'] = np.ones_like(prices)
            
            return indicators
        except:
            return {}
    
    def _calculate_trend_indicators(self, prices):
        """计算趋势指标"""
        indicators = {}
        
        try:
            # 趋势强度
            if len(prices) > 20:
                trend_strength = np.zeros_like(prices)
                for i in range(20, len(prices)):
                    # 线性回归斜率作为趋势强度
                    x = np.arange(20)
                    y = prices[i-19:i+1]
                    slope = np.polyfit(x, y, 1)[0]
                    trend_strength[i] = slope / prices[i]  # 标准化
                
                indicators['trend_strength'] = trend_strength
            else:
                indicators['trend_strength'] = np.zeros_like(prices)
            
            # 价格位置（在最近高低点中的位置）
            if len(prices) > 20:
                price_position = np.zeros_like(prices)
                for i in range(20, len(prices)):
                    recent_high = np.max(prices[i-19:i+1])
                    recent_low = np.min(prices[i-19:i+1])
                    if recent_high > recent_low:
                        price_position[i] = (prices[i] - recent_low) / (recent_high - recent_low)
                    else:
                        price_position[i] = 0.5
                
                indicators['price_position'] = price_position
            else:
                indicators['price_position'] = np.full_like(prices, 0.5)
            
            return indicators
        except:
            return {}
    
    def _calculate_volume_indicators(self, prices, volumes):
        """计算成交量指标"""
        indicators = {}
        
        try:
            # 成交量价格趋势 (VPT)
            vpt = np.zeros_like(prices)
            if len(prices) > 1:
                for i in range(1, len(prices)):
                    price_change = (prices[i] - prices[i-1]) / prices[i-1]
                    vpt[i] = vpt[i-1] + volumes[i] * price_change
            
            indicators['vpt'] = vpt
            
            # 成交量震荡器
            vol_osc = np.zeros_like(volumes)
            if len(volumes) > 20:
                for i in range(20, len(volumes)):
                    vol_short = np.mean(volumes[i-9:i+1])
                    vol_long = np.mean(volumes[i-19:i+1])
                    if vol_long > 0:
                        vol_osc[i] = (vol_short - vol_long) / vol_long * 100
            
            indicators['volume_oscillator'] = vol_osc
            
            return indicators
        except:
            return {}
    
    def _calculate_market_structure_indicators(self, prices):
        """计算市场结构指标"""
        indicators = {}
        
        try:
            # 支撑阻力强度
            if len(prices) > 50:
                support_resistance = np.zeros_like(prices)
                for i in range(25, len(prices)-25):
                    window = prices[i-25:i+26]
                    current_price = prices[i]
                    
                    # 计算当前价格在窗口中的分位数
                    percentile = np.percentile(window, 
                                             np.searchsorted(np.sort(window), current_price) / len(window) * 100)
                    support_resistance[i] = percentile
                
                indicators['support_resistance'] = support_resistance
            else:
                indicators['support_resistance'] = np.full_like(prices, 50.0)
            
            # 价格分形维度（简化版）
            if len(prices) > 30:
                fractal_dim = np.zeros_like(prices)
                for i in range(15, len(prices)-15):
                    window = prices[i-15:i+16]
                    # 简化的分形维度计算
                    price_range = np.max(window) - np.min(window)
                    if price_range > 0:
                        fractal_dim[i] = np.log(len(window)) / np.log(len(window) * price_range / np.sum(np.abs(np.diff(window))))
                    else:
                        fractal_dim[i] = 1.0
                
                indicators['fractal_dimension'] = fractal_dim
            else:
                indicators['fractal_dimension'] = np.ones_like(prices)
            
            return indicators
        except:
            return {} 