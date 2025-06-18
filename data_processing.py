# 数据处理主模块 - 重构版本
from AlgorithmImports import *
import numpy as np
from config import AlgorithmConfig
from data_acquisition import DataAcquisition, DataValidator
from technical_indicators import TechnicalIndicatorCalculator, EnhancedTechnicalCalculator
from feature_engineering import FeatureEngineer, DataScaler

class DataProcessor:
    """数据处理器 - 主要协调器，整合各个子模块"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
        # 初始化子模块
        self.data_acquisition = DataAcquisition(algorithm_instance)
        self.feature_engineer = FeatureEngineer(algorithm_instance)
        self.data_scaler = DataScaler(algorithm_instance)
        self.validator = DataValidator()
        
        # 启用固定24特征模式以避免文件过大
        self.feature_engineer.enable_fixed_24_features(True)
        
    def get_current_price_robust(self, symbol):
        """获取当前价格 - 委托给数据获取模块"""
        return self.data_acquisition.get_current_price_robust(symbol)
    
    def validate_symbol_data_availability(self, symbols):
        """验证股票数据可用性 - 委托给数据获取模块"""
        return self.data_acquisition.validate_symbol_data_availability(symbols)
    
    def get_historical_data(self, symbol, days=None):
        """获取历史数据 - 委托给数据获取模块"""
        return self.data_acquisition.get_historical_data(symbol, days)
    
    def create_feature_matrix(self, prices, volumes=None, symbol=None):
        """创建特征矩阵 - 委托给特征工程模块"""
        return self.feature_engineer.create_feature_matrix(prices, volumes, symbol)
    
    def clean_feature_matrix(self, feature_matrix, symbol=None):
        """清理特征矩阵 - 委托给特征工程模块"""
        return self.feature_engineer.clean_feature_matrix(feature_matrix, symbol)
    
    def scale_data(self, data, symbol, fit=True):
        """缩放数据 - 委托给数据缩放模块"""
        return self.data_scaler.scale_data(data, symbol, fit)
    
    def inverse_scale_predictions(self, predictions, symbol):
        """逆缩放预测 - 委托给数据缩放模块"""
        return self.data_scaler.inverse_scale_predictions(predictions, symbol)
    
    def create_multi_horizon_sequences(self, data, seq_length, horizons):
        """创建多时间范围序列"""
        try:
            if data is None or len(data) < seq_length + max(horizons):
                return None, None
            
            X_sequences = []
            y_sequences = []
            
            for horizon in horizons:
                X_horizon = []
                y_horizon = []
                
                for i in range(len(data) - seq_length - horizon + 1):
                    X_horizon.append(data[i:(i + seq_length)])
                    y_horizon.append(data[i + seq_length + horizon - 1])
                
                if len(X_horizon) > 0:
                    X_sequences.append(np.array(X_horizon))
                    y_sequences.append(np.array(y_horizon))
            
            return X_sequences, y_sequences
            
        except Exception as e:
            self.algorithm.log_debug(f"Error creating multi-horizon sequences: {e}", log_type='data')
            return None, None
    
    def detect_outliers(self, data, method='iqr', threshold=1.5):
        """检测异常值"""
        try:
            if data is None or len(data) == 0:
                return np.array([])
            
            if method == 'iqr':
                Q1 = np.percentile(data, 25)
                Q3 = np.percentile(data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                return np.where(outliers)[0]
            
            return np.array([])
            
        except Exception as e:
            self.algorithm.log_debug(f"Error detecting outliers: {e}", log_type='data')
            return np.array([])
    
    def clean_data(self, data, symbol):
        """清理数据"""
        try:
            if data is None or len(data) == 0:
                return None
            
            # 检查数据质量
            is_valid, message = self.validator.check_data_quality(data, symbol)
            if not is_valid:
                self.algorithm.log_debug(f"Data quality check failed for {symbol}: {message}", log_type='data')
                return None
            
            # 检测和处理异常值
            outlier_indices = self.detect_outliers(data)
            if len(outlier_indices) > 0:
                self.algorithm.log_debug(f"Found {len(outlier_indices)} outliers for {symbol}", log_type='data')
                # 用中位数替换异常值
                median_value = np.median(data)
                data[outlier_indices] = median_value
            
            return data
            
        except Exception as e:
            self.algorithm.log_debug(f"Error cleaning data for {symbol}: {e}", log_type='data')
            return data
    
    def calculate_market_regime(self, prices):
        """计算市场状态"""
        try:
            if len(prices) < 50:
                return np.full_like(prices, 1)  # 默认正常状态
            
            # 计算波动率
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            rolling_vol = np.zeros_like(prices)
            for i in range(20, len(prices)):
                rolling_vol[i] = np.std(returns[i-19:i+1])
            
            # 填充前面的值
            if len(prices) >= 20:
                rolling_vol[:20] = rolling_vol[20]
            
            # 定义市场状态
            # 0: 低波动率, 1: 正常波动率, 2: 高波动率
            vol_threshold_low = np.percentile(rolling_vol[rolling_vol > 0], 33)
            vol_threshold_high = np.percentile(rolling_vol[rolling_vol > 0], 67)
            
            regime = np.ones_like(prices)
            regime[rolling_vol < vol_threshold_low] = 0
            regime[rolling_vol > vol_threshold_high] = 2
            
            return regime
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating market regime: {e}", log_type='data')
            return np.full_like(prices, 1)
    
    def enable_fixed_24_features(self, enable=True):
        """启用/禁用固定24特征模式"""
        self.feature_engineer.enable_fixed_24_features(enable)
        self.algorithm.log_debug(f"DataProcessor: Fixed 24-feature mode {'enabled' if enable else 'disabled'}", log_type='data')
    
    def get_feature_count(self):
        """获取当前特征数量"""
        if self.feature_engineer.use_fixed_24_features:
            return 24
        else:
            return "variable"  # 综合特征数量不固定 