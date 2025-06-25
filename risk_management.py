# 风险管理模块 - QuantConnect兼容版
from AlgorithmImports import *
import numpy as np
import pandas as pd
from config import AlgorithmConfig
from collections import defaultdict

class VIXMonitor:
    """VIX监控器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self._vix_history = []
        self._last_vix_value = 20.0
        self._max_history_length = 20
        self.current_risk_state = "NORMAL"
        # 添加日期检查变量
        self._last_update_date = None
        
    def update_vix_data(self, current_time):
        """更新VIX数据并返回风险状态 - 每天只更新一次"""
        try:
            # 检查是否今天已经更新过VIX数据
            current_date = current_time.date()
            if self._last_update_date == current_date:
                # 今天已经更新过，直接返回当前状态
                return {
                    "mode": self.current_risk_state,
                    "vix_value": self._last_vix_value,
                    "data_source": "缓存数据（今日已更新）"
                }
            
            # 开始新的一天的VIX数据更新
            self.algorithm.log_debug(f"开始{current_date}的VIX数据更新", log_type="risk")
            
            # 直接从Securities获取VIX价格 - 参考vix.py的方法
            if hasattr(self.algorithm, 'vix_symbol') and self.algorithm.vix_symbol is not None:
                if self.algorithm.Securities.ContainsKey(self.algorithm.vix_symbol):
                    current_vix = self.algorithm.Securities[self.algorithm.vix_symbol].Price
                    if current_vix > 0:
                        self._last_vix_value = float(current_vix)
                        self.algorithm.log_debug(f"获取到VIX数据: {self._last_vix_value:.2f}", log_type="risk")
                    else:
                        self.algorithm.log_debug("VIX价格无效，使用上次值", log_type="risk")
                        # 保持上次的值不变
                else:
                    self.algorithm.log_debug("VIX数据不可用，使用上次值", log_type="risk")
            else:
                self.algorithm.log_debug("VIX Symbol未设置，使用默认值20.0", log_type="risk")
                self._last_vix_value = 20.0
            
            # 更新历史记录和计算风险状态
            self._update_history(self._last_vix_value)
            vix_change_rate = self._calculate_change_rate()
            risk_state = self._analyze_risk_state(self._last_vix_value, vix_change_rate)
            
            self.current_risk_state = risk_state.get("mode", "NORMAL")
            risk_state["data_source"] = "真实VIX数据（今日首次更新）"
            risk_state["vix_value"] = self._last_vix_value
            
            # 更新日期记录
            self._last_update_date = current_date
            
            self.algorithm.log_debug(f"VIX监控更新: VIX={self._last_vix_value:.2f}, 模式={self.current_risk_state}", log_type="risk")
            
            return risk_state
        except Exception as e:
            self.algorithm.log_debug(f"VIX数据更新失败: {e}，使用默认风险状态", log_type="risk")
            return self._get_default_risk_state()

    def _update_history(self, vix_value):
        self._vix_history.append(vix_value)
        if len(self._vix_history) > self._max_history_length:
            self._vix_history = self._vix_history[-self._max_history_length:]

    def _calculate_change_rate(self):
        if len(self._vix_history) < 2:
            return 0.0
        current_vix = self._vix_history[-1]
        previous_vix = self._vix_history[-2]
        if previous_vix == 0:
            return 0.0
        return (current_vix - previous_vix) / previous_vix

    def _analyze_risk_state(self, current_vix, vix_change_rate):
        vix_config = self.config.RISK_CONFIG
        risk_state = {
            "mode": "NORMAL",
            "vix_level": "medium",
            "rapid_rise": False,
            "extreme_level": False,
            "recommended_action": "正常交易"
        }
        
        try:
            if current_vix >= vix_config['vix_extreme_level']:
                risk_state.update({
                    "mode": "EXTREME",
                    "vix_level": "extreme",
                    "extreme_level": True,
                    "recommended_action": "极端防御"
                })
            elif current_vix >= vix_config['vix_normalization_threshold']:
                risk_state.update({
                    "mode": "DEFENSE",
                    "vix_level": "high",
                    "recommended_action": "防御模式"
                })
            elif current_vix <= 15:
                risk_state["vix_level"] = "low"
            
            if vix_change_rate >= vix_config['vix_rapid_rise_threshold']:
                risk_state["rapid_rise"] = True
                if risk_state["mode"] == "NORMAL":
                    risk_state["mode"] = "DEFENSE"
            
            return risk_state
        except:
            return self._get_default_risk_state()

    def _get_default_risk_state(self):
        return {
            "mode": self.current_risk_state,
            "vix_level": "medium",
            "rapid_rise": False,
            "extreme_level": False,
            "recommended_action": "正常交易",
            "vix_value": self._last_vix_value,
            "data_source": "缓存数据（异常情况）"
        }

    def is_defense_mode_active(self):
        return self.current_risk_state in ["DEFENSE", "EXTREME"]

    def is_extreme_mode_active(self):
        return self.current_risk_state == "EXTREME"

class HedgingManager:
    """对冲管理器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def calculate_hedging_allocation(self, vix_risk_state, regular_symbols):
        try:
            hedging_config = self.config.HEDGING_CONFIG
            base_hedge_ratio = hedging_config.get('base_hedge_ratio', 0.0)
            mode = vix_risk_state.get("mode", "NORMAL")
            
            if mode == "EXTREME":
                hedge_ratio = min(0.3, base_hedge_ratio * 3)
            elif mode == "DEFENSE":
                hedge_ratio = min(0.15, base_hedge_ratio * 2)
            else:
                hedge_ratio = base_hedge_ratio
            
            hedging_allocations = {}
            if hedge_ratio > 0:
                sqqq_symbol = Symbol.Create("SQQQ", SecurityType.EQUITY, Market.USA)
                if self._is_tradable(sqqq_symbol):
                    hedging_allocations[sqqq_symbol] = hedge_ratio
            
            return hedging_allocations
        except:
            return {}

    def _is_tradable(self, symbol):
        try:
            if symbol not in self.algorithm.Securities:
                self.algorithm.AddEquity(symbol.Value, Resolution.MINUTE)
            history = self.algorithm.History(symbol, 5, Resolution.DAILY)
            return len(list(history)) > 0
        except:
            return False

    def integrate_hedging_with_portfolio(self, regular_weights, regular_symbols, vix_risk_state):
        try:
            hedging_allocations = self.calculate_hedging_allocation(vix_risk_state, regular_symbols)
            if not hedging_allocations:
                return regular_weights, regular_symbols
            
            total_hedge_weight = sum(hedging_allocations.values())
            adjustment_factor = max(0.1, 1.0 - total_hedge_weight)
            
            adjusted_weights = {k: v * adjustment_factor for k, v in regular_weights.items()}
            final_symbols = list(regular_symbols)
            
            for symbol, weight in hedging_allocations.items():
                adjusted_weights[symbol] = weight
                if symbol not in final_symbols:
                    final_symbols.append(symbol)
            
            return adjusted_weights, final_symbols
        except:
            return regular_weights, regular_symbols

class ConcentrationLimiter:
    """集中度限制器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
    
    def apply_concentration_limits(self, weights, symbols):
        try:
            limited_weights = self._limit_individual_weights(weights)
            sector_limited = self._limit_sector_concentration(limited_weights, symbols)
            return self._renormalize_weights(sector_limited)
        except:
            return self._renormalize_weights(weights)
    
    def _limit_individual_weights(self, weights):
        max_weight = self.config.RISK_CONFIG.get('max_single_position', 0.15)
        limited_weights = {}
        excess_weight = 0.0
        valid_symbols = []
        
        for symbol, weight in weights.items():
            if weight > max_weight:
                limited_weights[symbol] = max_weight
                excess_weight += weight - max_weight
            else:
                limited_weights[symbol] = weight
                valid_symbols.append(symbol)
        
        if excess_weight > 0 and valid_symbols:
            redistribution = excess_weight / len(valid_symbols)
            for symbol in valid_symbols:
                new_weight = limited_weights[symbol] + redistribution
                limited_weights[symbol] = min(new_weight, max_weight)
        
        return limited_weights
    
    def _limit_sector_concentration(self, weights, symbols):
        try:
            sector_mapping = {
                'AAPL': '科技', 'MSFT': '科技', 'GOOGL': '科技', 'AMZN': '科技',
                'TSLA': '科技', 'META': '科技', 'NVDA': '科技', 'NFLX': '科技',
                'JPM': '金融', 'BAC': '金融', 'WFC': '金融', 'GS': '金融',
                'JNJ': '医疗', 'PFE': '医疗', 'UNH': '医疗', 'ABBV': '医疗',
                'XOM': '能源', 'CVX': '能源', 'COP': '能源',
                'WMT': '消费', 'PG': '消费', 'KO': '消费', 'PEP': '消费'
            }
            
            sector_weights = {}
            max_sector_weight = self.config.RISK_CONFIG.get('max_sector_concentration', 0.40)
            
            for symbol, weight in weights.items():
                sector = sector_mapping.get(symbol.Value, '其他')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            adjusted_weights = weights.copy()
            for sector, total_weight in sector_weights.items():
                if total_weight > max_sector_weight:
                    sector_symbols = [s for s in symbols if sector_mapping.get(s.Value, '其他') == sector]
                    reduction_factor = max_sector_weight / total_weight
                    for symbol in sector_symbols:
                        if symbol in adjusted_weights:
                            adjusted_weights[symbol] *= reduction_factor
            
            return adjusted_weights
        except:
            return weights
    
    def _renormalize_weights(self, weights):
        total_weight = sum(weights.values())
        if total_weight > 0:
            return {symbol: weight / total_weight for symbol, weight in weights.items()}
        equal_weight = 1.0 / len(weights) if weights else 0
        return {symbol: equal_weight for symbol in weights.keys()}

class DrawdownMonitor:
    """回撤监控器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.peak_value = 0
        self.portfolio_values = []
        self.last_alert_time = None  # 添加上次警报时间
    
    def update_portfolio_value(self, current_value):
        try:
            self.portfolio_values.append(current_value)
            if current_value > self.peak_value:
                self.peak_value = current_value
            
            if len(self.portfolio_values) > 252:
                self.portfolio_values = self.portfolio_values[-252:]
            
            if self.peak_value > 0:
                current_drawdown = (self.peak_value - current_value) / self.peak_value
                self._check_drawdown_alert(current_drawdown)
                return current_drawdown
            return 0.0
        except:
            return 0.0
    
    def _check_drawdown_alert(self, current_drawdown):
        threshold = self.config.RISK_CONFIG.get('max_drawdown_threshold', 0.15)
        if current_drawdown > threshold:
            # 限制警报频率：每小时最多一次
            current_time = self.algorithm.Time
            if (self.last_alert_time is None or 
                (current_time - self.last_alert_time).total_seconds() >= 3600):
                self.algorithm.log_debug(f"回撤警报: {current_drawdown:.2%}", log_type="risk")
                self.last_alert_time = current_time

class VolatilityMonitor:
    """波动率监控器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.daily_returns = []
    
    def update_return(self, daily_return):
        try:
            self.daily_returns.append(daily_return)
            if len(self.daily_returns) > 60:
                self.daily_returns = self.daily_returns[-60:]
            
            if len(self.daily_returns) >= 20:
                volatility = np.std(self.daily_returns) * np.sqrt(252)
                threshold = self.config.RISK_CONFIG.get('volatility_threshold', 0.25)
                if volatility > threshold:
                    self.algorithm.log_debug(f"波动率警报: {volatility:.2%}", log_type="risk")
                return volatility
            return 0.0
        except:
            return 0.0

class RiskManager:
    """风险管理器 - 主控制器"""
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.risk_metrics_history = []
        
        # 初始化子模块
        self.vix_monitor = VIXMonitor(algorithm_instance)
        self.hedging_manager = HedgingManager(algorithm_instance)
        self.concentration_limiter = ConcentrationLimiter(algorithm_instance)
        self.drawdown_monitor = DrawdownMonitor(algorithm_instance)
        self.volatility_monitor = VolatilityMonitor(algorithm_instance)
        
    def apply_risk_controls(self, expected_returns, symbols):
        """应用风险控制"""
        try:
            # VIX风险监控
            vix_risk_state = self.vix_monitor.update_vix_data(self.algorithm.Time)
            self.algorithm.log_debug(f"[VIX状态] {vix_risk_state.get('mode', 'NORMAL')}", log_type="risk")
            
            # 流动性筛选
            liquid_symbols = self._filter_by_liquidity(symbols)
            
            # 波动率筛选
            volatility_filtered = self._filter_by_volatility(liquid_symbols)
            
            # 相关性筛选
            correlation_filtered = self._filter_by_correlation(volatility_filtered)
            
            # VIX防御筛选
            vix_filtered = self._apply_vix_filter(correlation_filtered, vix_risk_state)
            
            # 调整收益
            risk_adjusted_returns = self._adjust_returns_for_risk(expected_returns, vix_filtered, vix_risk_state)
            
            # 最终检查
            final_symbols = self._final_check(vix_filtered)
            
            # 保存状态
            self.algorithm._vix_risk_state = vix_risk_state
            
            return risk_adjusted_returns, final_symbols
        except Exception as e:
            self.algorithm.log_debug(f"风控错误: {str(e)}", log_type="risk")
            return expected_returns, symbols
    
    def _filter_by_liquidity(self, symbols):
        filtered = []
        for symbol in symbols:
            try:
                history = self.algorithm.History(symbol, 20, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) >= 10:
                    volumes = [x.Volume for x in history_list]
                    avg_volume = np.mean(volumes)
                    if avg_volume >= self.config.RISK_CONFIG['liquidity_min_volume']:
                        filtered.append(symbol)
            except:
                continue
        return filtered
    
    def _filter_by_volatility(self, symbols):
        filtered = []
        for symbol in symbols:
            try:
                history = self.algorithm.History(symbol, 60, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) >= 30:
                    prices = np.array([x.Close for x in history_list])
                    returns = np.diff(prices) / prices[:-1]
                    volatility = np.std(returns) * np.sqrt(252)
                    if volatility <= self.config.RISK_CONFIG['volatility_threshold']:
                        filtered.append(symbol)
            except:
                continue
        return filtered
    
    def _filter_by_correlation(self, symbols):
        if len(symbols) <= 1:
            return symbols
        try:
            returns_data = {}
            for symbol in symbols:
                history = self.algorithm.History(symbol, 60, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) >= 30:
                    prices = np.array([x.Close for x in history_list])
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return symbols
            
            symbols_list = list(returns_data.keys())
            return symbols_list
        except:
            return symbols
    
    def _apply_vix_filter(self, symbols, vix_risk_state):
        mode = vix_risk_state.get("mode", "NORMAL")
        if mode == "EXTREME":
            defensive_stocks = ["JNJ", "PG", "KO", "WMT", "PFE"]
            filtered = [s for s in symbols if s.Value in defensive_stocks]
            return filtered[:3] if filtered else symbols[:3]
        elif mode == "DEFENSE":
            target_count = max(3, int(len(symbols) * 0.7))
            return symbols[:target_count]
        return symbols
    
    def _adjust_returns_for_risk(self, expected_returns, symbols, vix_risk_state):
        try:
            adjusted = expected_returns.copy()
            mode = vix_risk_state.get("mode", "NORMAL")
            
            risk_factor = 1.0
            if mode == "EXTREME":
                risk_factor = 0.3
            elif mode == "DEFENSE":
                risk_factor = 0.7
            
            for symbol in symbols:
                if symbol in adjusted:
                    adjusted[symbol] *= risk_factor
            
            return adjusted
        except:
            return expected_returns
    
    def _final_check(self, symbols):
        valid = []
        for symbol in symbols:
            if symbol in self.algorithm.Securities:
                if self.algorithm.Securities[symbol].Price > 0:
                    valid.append(symbol)
        return valid if valid else symbols[:3] 