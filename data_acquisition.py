# 数据获取模块
from AlgorithmImports import *
import numpy as np
from config import AlgorithmConfig

class DataAcquisition:
    """数据获取器 - 负责历史数据获取和价格数据验证"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def get_current_price_robust(self, symbol):
        """
        多层备用方案获取当前价格
        Level 1: data_slice (最快)
        Level 2: Securities集合 (备用)
        Level 3: History API (最后备用)
        """
        symbol_str = str(symbol)
        
        try:
            # Level 1: 尝试从data_slice获取
            if hasattr(self.algorithm, 'data_slice') and self.algorithm.data_slice is not None:
                if self.algorithm.data_slice.ContainsKey(symbol):
                    symbol_data = self.algorithm.data_slice[symbol]
                    if hasattr(symbol_data, 'Price') and symbol_data.Price > 0:
                        self.algorithm.log_debug(f"Price for {symbol_str} from data_slice: {symbol_data.Price}", log_type="data")
                        return symbol_data.Price
                        
            # Level 2: 尝试从Securities集合获取
            if symbol in self.algorithm.Securities:
                security = self.algorithm.Securities[symbol]
                if hasattr(security, 'Price') and security.Price > 0:
                    self.algorithm.log_debug(f"Price for {symbol_str} from Securities: {security.Price}", log_type="data")
                    return security.Price
                elif hasattr(security, 'Close') and security.Close > 0:
                    self.algorithm.log_debug(f"Price for {symbol_str} from Securities.Close: {security.Close}", log_type="data")
                    return security.Close
                    
            # Level 3: 尝试从History API获取最近价格
            try:
                recent_history = self.algorithm.History(symbol, 1, Resolution.Daily)
                history_list = list(recent_history)
                if history_list and len(history_list) > 0:
                    last_bar = history_list[-1]
                    if hasattr(last_bar, 'Close') and last_bar.Close > 0:
                        self.algorithm.log_debug(f"Price for {symbol_str} from History: {last_bar.Close}", log_type="data")
                        return last_bar.Close
            except Exception as history_error:
                self.algorithm.log_debug(f"History API failed for {symbol_str}: {history_error}", log_type="data")
                
            # 所有方法都失败
            self.algorithm.log_debug(f"Failed to get current price for {symbol_str} using all methods", log_type="data")
            return None
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in get_current_price_robust for {symbol_str}: {e}", log_type="data")
            return None
    
    def validate_symbol_data_availability(self, symbols):
        """
        验证一组股票的数据可用性
        返回有效股票列表和价格字典
        """
        valid_symbols = []
        current_prices = {}
        
        for symbol in symbols:
            price = self.get_current_price_robust(symbol)
            if price is not None and price > 0:
                valid_symbols.append(symbol)
                current_prices[symbol] = price
            else:
                self.algorithm.log_debug(f"Symbol {symbol} excluded due to data unavailability", log_type="data")
                
        self.algorithm.log_debug(f"Data validation: {len(valid_symbols)}/{len(symbols)} symbols have valid data", log_type="data")
        return valid_symbols, current_prices
    
    def get_historical_data(self, symbol, days=None):
        """获取历史价格数据，默认800天"""
        if days is None:
            days = 800
        try:
            self.algorithm.log_debug(f"Getting historical data for {symbol}, requesting {days} days", log_type='data')
            
            try:
                history = self.algorithm.History(symbol, days, Resolution.DAILY)
            except Exception as history_error:
                self.algorithm.log_debug(f"Error calling History API for {symbol}: {history_error}", log_type='data')
                return None
            
            # 将枚举对象转换为列表
            try:
                history_list = list(history)
                self.algorithm.log_debug(f"Retrieved {len(history_list)} data points for {symbol}", log_type='data')
            except Exception as list_error:
                self.algorithm.log_debug(f"Error converting history to list for {symbol}: {list_error}", log_type='data')
                return None
                
            if len(history_list) == 0:
                self.algorithm.log_debug(f"No historical data available for {symbol}", log_type='data')
                return None
            
            try:
                prices = np.array([x.Close for x in history_list])
                volumes = np.array([x.Volume for x in history_list])
                
                self.algorithm.log_debug(f"Extracted prices for {symbol}:", log_type='data')
                self.algorithm.log_debug(f"  Price range: {prices.min():.2f} - {prices.max():.2f}", log_type='data')
                self.algorithm.log_debug(f"  Price count: {len(prices)}", log_type='data')
                
            except Exception as extract_error:
                self.algorithm.log_debug(f"Error extracting prices for {symbol}: {extract_error}", log_type='data')
                return None
            
            # 数据质量检查
            if self._validate_price_data(prices, symbol):
                self.algorithm.log_debug(f"Price data validation passed for {symbol}", log_type='data')
                return prices
            else:
                self.algorithm.log_debug(f"Price data validation failed for {symbol}", log_type='data')
                return None
                
        except Exception as e:
            self.algorithm.log_debug(f"CRITICAL ERROR getting historical data for {symbol}: {str(e)}", log_type='data')
            self.algorithm.log_debug(f"Historical data error type: {type(e).__name__}", log_type='data')
            import traceback
            self.algorithm.log_debug(f"Historical data error traceback: {traceback.format_exc()}", log_type='data')
            return None
    
    def _validate_price_data(self, prices, symbol):
        """验证价格数据质量"""
        if len(prices) == 0:
            self.algorithm.log_debug(f"Empty price data for {symbol}", log_type='data')
            return False
            
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            self.algorithm.log_debug(f"Invalid price data (NaN/Inf) for {symbol}", log_type='data')
            return False
            
        if np.any(prices <= 0):
            self.algorithm.log_debug(f"Non-positive prices found for {symbol}", log_type='data')
            return False
            
        # 检查异常波动
        returns = np.diff(prices) / prices[:-1]
        if np.any(np.abs(returns) > 0.5):  # 单日涨跌幅超过50%
            self.algorithm.log_debug(f"Extreme price movements detected for {symbol}", log_type='data')
            return False
            
        return True

class DataValidator:
    """数据验证器 - 负责数据质量检查和验证"""
    
    @staticmethod
    def validate_sequences(X, y):
        """验证序列数据"""
        if X is None or y is None:
            return False
        
        if len(X) == 0 or len(y) == 0:
            return False
        
        if len(X) != len(y):
            return False
        
        # 检查NaN值
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            return False
        
        return True
    
    @staticmethod
    def check_data_quality(data, symbol):
        """检查数据质量"""
        if data is None or len(data) == 0:
            return False, "Empty data"
        
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            nan_ratio = nan_count / len(data)
            if nan_ratio > 0.1:  # 超过10%的NaN值
                return False, f"Too many NaN values: {nan_ratio:.2%}"
        
        inf_count = np.sum(np.isinf(data))
        if inf_count > 0:
            return False, f"Contains {inf_count} infinite values"
        
        return True, "Data quality OK" 