# 特征工程模块
from AlgorithmImports import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from config import AlgorithmConfig
from technical_indicators import TechnicalIndicatorCalculator, EnhancedTechnicalCalculator

class FeatureEngineer:
    """特征工程器 - 负责特征矩阵创建和数据清理"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.scalers = {}
        self.tech_calculator = TechnicalIndicatorCalculator(algorithm_instance)
        self.enhanced_calculator = EnhancedTechnicalCalculator(algorithm_instance)
        self.fundamental_cache = {}
        self.macro_cache = {}
        self.use_fixed_24_features = False
        
    def create_feature_matrix(self, prices, volumes=None, symbol=None):
        """创建特征矩阵"""
        try:
            if self.use_fixed_24_features:
                return self._create_fixed_24_feature_matrix(prices, volumes, symbol)
            else:
                return self._create_comprehensive_feature_matrix(prices, volumes, symbol)
        except Exception as e:
            self.algorithm.log_debug(f"Error creating feature matrix for {symbol}: {e}", log_type='data')
            return None
    
    def _create_fixed_24_feature_matrix(self, prices, volumes=None, symbol=None):
        """创建固定24特征的特征矩阵"""
        try:
            if len(prices) < 126:
                self.algorithm.log_debug(f"Insufficient data for 24-feature matrix: {len(prices)} < 126", log_type='data')
                return None
            
            # 计算技术指标
            indicators = self.tech_calculator.calculate_technical_indicators(prices, volumes)
            if indicators is None:
                self.algorithm.log_debug(f"Failed to calculate technical indicators for {symbol}", log_type='data')
                return None
            
            # 选择最后126个数据点
            data_length = 126
            start_idx = len(prices) - data_length
            
            # 提取24个特征
            features = []
            
            # 1-4: 移动平均线
            features.append(indicators['sma_20'][start_idx:])
            features.append(indicators['sma_50'][start_idx:])
            features.append(indicators['ema_12'][start_idx:])
            features.append(indicators['ema_26'][start_idx:])
            
            # 5: RSI
            features.append(indicators['rsi'][start_idx:])
            
            # 6-8: MACD系列
            features.append(indicators['macd'][start_idx:])
            features.append(indicators['macd_signal'][start_idx:])
            features.append(indicators['macd_hist'][start_idx:])
            
            # 9-12: 布林带
            features.append(indicators['bb_upper'][start_idx:])
            features.append(indicators['bb_middle'][start_idx:])
            features.append(indicators['bb_lower'][start_idx:])
            features.append(indicators['bb_width'][start_idx:])
            
            # 13: ATR
            features.append(indicators['atr'][start_idx:])
            
            # 14-15: 动量指标
            features.append(indicators['momentum'][start_idx:])
            features.append(indicators['roc'][start_idx:])
            
            # 16-18: 成交量指标
            features.append(indicators['volume_sma'][start_idx:])
            features.append(indicators['volume_ratio'][start_idx:])
            features.append(indicators['obv'][start_idx:])
            
            # 19-24: 价格相关特征
            price_slice = prices[start_idx:]
            features.append(price_slice)  # 19: 原始价格
            
            # 20: 价格变化率
            price_change = np.zeros_like(price_slice)
            price_change[1:] = (price_slice[1:] - price_slice[:-1]) / price_slice[:-1]
            features.append(price_change)
            
            # 21: 累积收益
            cumulative_return = np.cumprod(1 + price_change) - 1
            features.append(cumulative_return)
            
            # 22: 波动率
            rolling_volatility = np.zeros_like(price_slice)
            for i in range(20, len(price_slice)):
                rolling_volatility[i] = np.std(price_change[i-19:i+1])
            rolling_volatility[:20] = rolling_volatility[20] if len(price_slice) > 20 else 0.01
            features.append(rolling_volatility)
            
            # 23: 最高价位置
            highest_position = np.zeros_like(price_slice)
            for i in range(len(price_slice)):
                lookback = min(i+1, 20)
                recent_high = np.max(price_slice[i-lookback+1:i+1])
                highest_position[i] = price_slice[i] / recent_high if recent_high > 0 else 1.0
            features.append(highest_position)
            
            # 24: 最低价位置
            lowest_position = np.zeros_like(price_slice)
            for i in range(len(price_slice)):
                lookback = min(i+1, 20)
                recent_low = np.min(price_slice[i-lookback+1:i+1])
                lowest_position[i] = price_slice[i] / recent_low if recent_low > 0 else 1.0
            features.append(lowest_position)
            
            # 转置矩阵：(126, 24)
            feature_matrix = np.column_stack(features)
            
            self.algorithm.log_debug(f"Created 24-feature matrix for {symbol}: shape {feature_matrix.shape}", log_type='data')
            
            # 数据清理
            cleaned_matrix = self.clean_feature_matrix(feature_matrix, symbol)
            
            return cleaned_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error creating 24-feature matrix for {symbol}: {e}", log_type='data')
            return None
    
    def _create_comprehensive_feature_matrix(self, prices, volumes=None, symbol=None):
        """创建综合特征矩阵（原始版本）"""
        try:
            if len(prices) < 100:
                self.algorithm.log_debug(f"Insufficient data for feature matrix: {len(prices)} < 100", log_type='data')
                return None
            
            # 计算技术指标
            indicators = self.tech_calculator.calculate_technical_indicators(prices, volumes)
            if indicators is None:
                return None
            
            # 计算增强技术指标
            enhanced_indicators = self.enhanced_calculator.calculate_enhanced_technical_indicators(prices, volumes)
            
            # 计算基本面特征
            fundamental_features = self.calculate_fundamental_features(symbol, len(prices))
            
            # 计算宏观特征
            macro_features = self.calculate_macro_features(len(prices))
            
            # 计算微观结构特征
            microstructure_features = self.calculate_microstructure_features(prices, volumes)
            
            # 组合所有特征
            all_features = []
            
            # 添加技术指标
            for key, value in indicators.items():
                if value is not None and len(value) == len(prices):
                    all_features.append(value)
            
            # 添加增强指标
            for key, value in enhanced_indicators.items():
                if value is not None and len(value) == len(prices):
                    all_features.append(value)
            
            # 添加基本面特征
            if fundamental_features:
                for key, value in fundamental_features.items():
                    if value is not None and len(value) == len(prices):
                        all_features.append(value)
            
            # 添加宏观特征
            if macro_features:
                for key, value in macro_features.items():
                    if value is not None and len(value) == len(prices):
                        all_features.append(value)
            
            # 添加微观结构特征
            if microstructure_features:
                for key, value in microstructure_features.items():
                    if value is not None and len(value) == len(prices):
                        all_features.append(value)
            
            if len(all_features) == 0:
                self.algorithm.log_debug(f"No valid features generated for {symbol}", log_type='data')
                return None
            
            # 转置为特征矩阵
            feature_matrix = np.column_stack(all_features)
            
            self.algorithm.log_debug(f"Created comprehensive feature matrix for {symbol}: shape {feature_matrix.shape}", log_type='data')
            
            # 数据清理
            cleaned_matrix = self.clean_feature_matrix(feature_matrix, symbol)
            
            return cleaned_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error creating comprehensive feature matrix for {symbol}: {e}", log_type='data')
            return None
    
    def enable_fixed_24_features(self, enable=True):
        """启用/禁用固定24特征模式"""
        self.use_fixed_24_features = enable
        self.algorithm.log_debug(f"Fixed 24-feature mode: {'enabled' if enable else 'disabled'}", log_type='data')
    
    def clean_feature_matrix(self, feature_matrix, symbol=None):
        """清理特征矩阵 - 增强数据质量控制"""
        try:
            if feature_matrix is None:
                return None
            
            original_shape = feature_matrix.shape
            
            # 1. 检查NaN值
            nan_count = np.isnan(feature_matrix).sum()
            if nan_count > 0:
                nan_ratio = nan_count / feature_matrix.size
                self.algorithm.log_debug(f"Found {nan_count} NaN values ({nan_ratio:.2%}) in feature matrix for {symbol}", log_type='data')
                
                # 如果NaN比例过高，返回None
                if nan_ratio > self.config.DATA_QUALITY_CONFIG['max_missing_ratio']:
                    self.algorithm.log_debug(f"Too many NaN values ({nan_ratio:.2%}) for {symbol}, rejecting", log_type='data')
                    return None
                
                # 尝试智能填充NaN值
                if self.config.DATA_QUALITY_CONFIG.get('enable_nan_interpolation', True):
                    feature_matrix = self._intelligent_nan_filling(feature_matrix, symbol)
                    if feature_matrix is None:
                        return None
            
            # 2. 检查无穷值
            inf_count = np.isinf(feature_matrix).sum()
            if inf_count > 0:
                self.algorithm.log_debug(f"Found {inf_count} infinite values in feature matrix for {symbol}, clipping", log_type='data')
                feature_matrix = np.clip(feature_matrix, -1e10, 1e10)
            
            # 3. 异常值检测和处理
            if self.config.DATA_QUALITY_CONFIG.get('outlier_detection_method') == 'iqr':
                feature_matrix = self._handle_outliers_iqr(feature_matrix, symbol)
            
            # 4. 数据一致性检查
            if self.config.DATA_QUALITY_CONFIG.get('data_consistency_check', True):
                if not self._validate_feature_consistency(feature_matrix, symbol):
                    return None
            
            # 5. 最终验证
            final_nan_count = np.isnan(feature_matrix).sum()
            if final_nan_count > 0:
                self.algorithm.log_debug(f"Still have {final_nan_count} NaN values after cleaning for {symbol}", log_type='data')
                return None
            
            self.algorithm.log_debug(f"Successfully cleaned feature matrix for {symbol}: {original_shape} -> {feature_matrix.shape}", log_type='data')
            return feature_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error cleaning feature matrix for {symbol}: {e}", log_type='data')
            return None
    
    def _intelligent_nan_filling(self, feature_matrix, symbol=None):
        """智能NaN值填充"""
        try:
            if feature_matrix is None:
                return None
            
            method = self.config.DATA_QUALITY_CONFIG.get('nan_interpolation_method', 'linear')
            max_consecutive = self.config.DATA_QUALITY_CONFIG.get('max_consecutive_nans', 10)
            
            for col in range(feature_matrix.shape[1]):
                column_data = feature_matrix[:, col]
                nan_mask = np.isnan(column_data)
                
                if not np.any(nan_mask):
                    continue
                
                # 检查连续NaN长度
                consecutive_nans = self._count_consecutive_nans(nan_mask)
                if consecutive_nans > max_consecutive:
                    self.algorithm.log_debug(f"Too many consecutive NaN values ({consecutive_nans}) in column {col} for {symbol}", log_type='data')
                    return None
                
                # 根据方法填充
                if method == 'linear':
                    # 线性插值
                    valid_indices = np.where(~nan_mask)[0]
                    if len(valid_indices) >= 2:
                        feature_matrix[:, col] = np.interp(
                            np.arange(len(column_data)),
                            valid_indices,
                            column_data[valid_indices]
                        )
                    else:
                        # 不足两个有效值，用均值填充
                        if len(valid_indices) > 0:
                            feature_matrix[:, col] = np.nan_to_num(column_data, nan=np.mean(column_data[valid_indices]))
                        else:
                            feature_matrix[:, col] = np.zeros_like(column_data)
            
            return feature_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in intelligent NaN filling for {symbol}: {e}", log_type='data')
            return None
    
    def _count_consecutive_nans(self, nan_mask):
        """计算最大连续NaN数量"""
        max_consecutive = 0
        current_consecutive = 0
        
        for is_nan in nan_mask:
            if is_nan:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _handle_outliers_iqr(self, feature_matrix, symbol):
        """使用IQR方法处理异常值"""
        try:
            threshold = self.config.DATA_QUALITY_CONFIG.get('outlier_threshold', 2.5)
            
            for col in range(feature_matrix.shape[1]):
                column_data = feature_matrix[:, col]
                
                Q1 = np.percentile(column_data, 25)
                Q3 = np.percentile(column_data, 75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # 裁剪异常值
                feature_matrix[:, col] = np.clip(column_data, lower_bound, upper_bound)
            
            return feature_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error handling outliers for {symbol}: {e}", log_type='data')
            return feature_matrix
    
    def _validate_feature_consistency(self, feature_matrix, symbol):
        """验证特征一致性"""
        try:
            # 检查特征矩阵形状
            if feature_matrix.shape[0] == 0 or feature_matrix.shape[1] == 0:
                self.algorithm.log_debug(f"Empty feature matrix for {symbol}", log_type='data')
                return False
            
            # 检查数值范围
            if np.any(np.abs(feature_matrix) > 1e15):
                self.algorithm.log_debug(f"Extreme values detected in feature matrix for {symbol}", log_type='data')
                return False
            
            # 检查方差
            feature_vars = np.var(feature_matrix, axis=0)
            zero_var_count = np.sum(feature_vars == 0)
            if zero_var_count > feature_matrix.shape[1] * 0.5:  # 超过50%的特征方差为0
                self.algorithm.log_debug(f"Too many zero-variance features ({zero_var_count}) for {symbol}", log_type='data')
                return False
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"Error validating feature consistency for {symbol}: {e}", log_type='data')
            return False
    
    def calculate_fundamental_features(self, symbol, data_length):
        """计算基本面特征"""
        try:
            # 简化的基本面特征（由于QuantConnect限制）
            fundamental_features = {}
            
            # 使用缓存避免重复计算
            cache_key = f"{symbol}_{data_length}"
            if cache_key in self.fundamental_cache:
                return self.fundamental_cache[cache_key]
            
            # 模拟基本面数据（实际应用中应从数据源获取）
            pe_ratio = np.full(data_length, 15.0)  # 默认P/E比率
            pb_ratio = np.full(data_length, 2.0)   # 默认P/B比率
            debt_ratio = np.full(data_length, 0.3) # 默认负债比率
            
            fundamental_features['pe_ratio'] = pe_ratio
            fundamental_features['pb_ratio'] = pb_ratio
            fundamental_features['debt_ratio'] = debt_ratio
            
            # 缓存结果
            self.fundamental_cache[cache_key] = fundamental_features
            
            return fundamental_features
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating fundamental features for {symbol}: {e}", log_type='data')
            return {}
    
    def calculate_macro_features(self, data_length):
        """计算宏观经济特征"""
        try:
            # 简化的宏观特征
            macro_features = {}
            
            # 使用缓存
            cache_key = f"macro_{data_length}"
            if cache_key in self.macro_cache:
                return self.macro_cache[cache_key]
            
            # 模拟宏观数据
            interest_rate = np.full(data_length, 0.05)    # 利率
            inflation_rate = np.full(data_length, 0.02)   # 通胀率
            gdp_growth = np.full(data_length, 0.03)       # GDP增长率
            
            macro_features['interest_rate'] = interest_rate
            macro_features['inflation_rate'] = inflation_rate
            macro_features['gdp_growth'] = gdp_growth
            
            # 缓存结果
            self.macro_cache[cache_key] = macro_features
            
            return macro_features
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating macro features: {e}", log_type='data')
            return {}
    
    def calculate_microstructure_features(self, prices, volumes):
        """计算微观结构特征"""
        try:
            if volumes is None or len(volumes) != len(prices):
                return {}
            
            microstructure_features = {}
            
            # 价格影响
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                volume_changes = np.diff(volumes)
                
                # 填充第一个值
                returns = np.concatenate([[0], returns])
                volume_changes = np.concatenate([[0], volume_changes])
                
                # 成交量-收益相关性
                price_impact = np.zeros_like(prices)
                for i in range(20, len(prices)):
                    corr = np.corrcoef(returns[i-19:i+1], volume_changes[i-19:i+1])[0, 1]
                    price_impact[i] = corr if not np.isnan(corr) else 0
                
                microstructure_features['price_impact'] = price_impact
            
            # 买卖压力（简化版）
            if len(prices) > 1:
                buying_pressure = np.zeros_like(prices)
                for i in range(1, len(prices)):
                    if prices[i] > prices[i-1]:
                        buying_pressure[i] = volumes[i]
                    elif prices[i] < prices[i-1]:
                        buying_pressure[i] = -volumes[i]
                
                microstructure_features['buying_pressure'] = buying_pressure
            
            return microstructure_features
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating microstructure features: {e}", log_type='data')
            return {}

class DataScaler:
    """数据缩放器 - 负责特征缩放和标准化"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.scalers = {}
        self.config = AlgorithmConfig()
    
    def scale_data(self, data, symbol, fit=True):
        """缩放数据"""
        try:
            if data is None or len(data) == 0:
                return None
            
            # 确保数据是2D数组
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # 获取或创建缩放器
            if symbol not in self.scalers or fit:
                # 使用RobustScaler，对异常值更鲁棒
                self.scalers[symbol] = RobustScaler()
                
                if fit:
                    try:
                        self.scalers[symbol].fit(data)
                        self.algorithm.log_debug(f"Fitted scaler for {symbol} with data shape {data.shape}", log_type='data')
                    except Exception as fit_error:
                        self.algorithm.log_debug(f"Error fitting scaler for {symbol}: {fit_error}", log_type='data')
                        return data  # 返回原始数据
            
            # 应用缩放
            try:
                scaled_data = self.scalers[symbol].transform(data)
                self.algorithm.log_debug(f"Scaled data for {symbol}: {data.shape} -> {scaled_data.shape}", log_type='data')
                return scaled_data
            except Exception as transform_error:
                self.algorithm.log_debug(f"Error transforming data for {symbol}: {transform_error}", log_type='data')
                return data  # 返回原始数据
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in scale_data for {symbol}: {e}", log_type='data')
            return data
    
    def inverse_scale_predictions(self, predictions, symbol):
        """逆缩放预测结果"""
        try:
            if predictions is None or symbol not in self.scalers:
                return predictions
            
            # 确保预测数据是2D数组
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            
            try:
                # 逆变换
                original_scale_predictions = self.scalers[symbol].inverse_transform(predictions)
                
                # 如果原始是1D，返回1D
                if original_scale_predictions.shape[1] == 1:
                    original_scale_predictions = original_scale_predictions.flatten()
                
                self.algorithm.log_debug(f"Inverse scaled predictions for {symbol}", log_type='data')
                return original_scale_predictions
                
            except Exception as inverse_error:
                self.algorithm.log_debug(f"Error inverse scaling predictions for {symbol}: {inverse_error}", log_type='data')
                return predictions
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in inverse_scale_predictions for {symbol}: {e}", log_type='data')
            return predictions 