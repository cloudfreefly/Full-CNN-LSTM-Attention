# 多元化强制执行器
from AlgorithmImports import *
import numpy as np

class DiversificationEnforcer:
    """多元化强制执行器 - 确保投资组合始终保持多元化"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        
    def enforce_diversification(self, weights, symbols, expected_returns=None):
        """强制执行多元化策略"""
        try:
            if len(weights) == 0 or len(symbols) == 0:
                self.algorithm.log_debug(f"输入为空，跳过多元化强制执行", log_type="diversification")
                return weights, symbols
            
            # 获取风险控制开关
            try:
                risk_switches = getattr(self.algorithm, 'config', None)
                if risk_switches and hasattr(risk_switches, 'RISK_CONTROL_SWITCHES'):
                    risk_switches = risk_switches.RISK_CONTROL_SWITCHES
                else:
                    # 如果配置不存在，使用默认值
                    risk_switches = {
                        'enable_diversification_enforcer': True,
                        'enable_risk_logging': True,
                        'enable_detailed_risk_analysis': True
                    }
            except Exception:
                # 如果配置访问失败，使用默认值
                risk_switches = {
                    'enable_diversification_enforcer': True,
                    'enable_risk_logging': True,
                    'enable_detailed_risk_analysis': True
                }
            
            # 检查是否启用分散化强制执行
            if not risk_switches.get('enable_diversification_enforcer', True):
                if risk_switches.get('enable_risk_logging', True):
                    self.algorithm.log_debug(f"分散化强制执行已禁用", log_type="diversification")
                return weights, symbols
            
            # 1. 检查当前多元化状态
            diversification_metrics = self._calculate_diversification_metrics(weights, symbols)
            
            if risk_switches.get('enable_detailed_risk_analysis', True):
                self._log_diversification_metrics(diversification_metrics, "当前")
            
            # 2. 如果多元化不足，强制改进
            if diversification_metrics['is_under_diversified']:
                if risk_switches.get('enable_risk_logging', True):
                    self.algorithm.log_debug(f"多元化不足({diversification_metrics['n_stocks']}只股票)，强制改进", log_type="diversification")
                
                weights, symbols = self._force_diversification(weights, symbols, expected_returns)
                
                # 3. 验证结果
                final_metrics = self._calculate_diversification_metrics(weights, symbols)
                if risk_switches.get('enable_risk_logging', True):
                    self.algorithm.log_debug(f"强制多元化完成: {final_metrics['n_stocks']}只股票", log_type="diversification")
                
                if risk_switches.get('enable_detailed_risk_analysis', True):
                    self._log_diversification_metrics(final_metrics, "改进后")
            else:
                if risk_switches.get('enable_risk_logging', True):
                    self.algorithm.log_debug(f"当前多元化状态良好，无需强制改进", log_type="diversification")
            
            return weights, symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"多元化强制执行错误: {e}", log_type='diversification')
            return weights, symbols
    
    def _calculate_diversification_metrics(self, weights, symbols):
        """计算多元化指标"""
        metrics = {
            'n_stocks': len(symbols),
            'max_weight': np.max(weights) if len(weights) > 0 else 0,
            'min_weight': np.min(weights) if len(weights) > 0 else 0,
            'weight_std': np.std(weights) if len(weights) > 0 else 0,
            'herfindahl_index': np.sum(weights ** 2) if len(weights) > 0 else 0,
            'effective_stocks': 1 / np.sum(weights ** 2) if len(weights) > 0 and np.sum(weights ** 2) > 0 else 0,
            'concentration_ratio': 0,
            'is_under_diversified': False
        }
        
        # 计算集中度比率（前3大持仓的权重之和）
        if len(weights) > 0:
            sorted_weights = np.sort(weights)[::-1]  # 降序排列
            top_3_weights = sorted_weights[:min(3, len(sorted_weights))]
            metrics['concentration_ratio'] = np.sum(top_3_weights)
        
        # 判断是否多元化不足
        metrics['is_under_diversified'] = (
            metrics['n_stocks'] < 4 or  # 少于4只股票
            metrics['max_weight'] > 0.5 or  # 单股权重超过50%
            metrics['concentration_ratio'] > 0.8 or  # 前3大持仓超过80%
            metrics['effective_stocks'] < 3  # 有效股票数少于3只
        )
        
        return metrics
    
    def _log_diversification_metrics(self, metrics, prefix=""):
        """记录多元化指标"""
        prefix_str = f"{prefix} " if prefix else ""
        self.algorithm.log_debug(f"{prefix_str}多元化指标:", log_type="diversification")
        self.algorithm.log_debug(f"  股票数量: {metrics['n_stocks']}", log_type="diversification")
        self.algorithm.log_debug(f"  最大权重: {metrics['max_weight']:.2%}", log_type="diversification")
        self.algorithm.log_debug(f"  权重标准差: {metrics['weight_std']:.3f}", log_type="diversification")
        self.algorithm.log_debug(f"  集中度比率(前3): {metrics['concentration_ratio']:.2%}", log_type="diversification")
        self.algorithm.log_debug(f"  有效股票数: {metrics['effective_stocks']:.1f}", log_type="diversification")
        self.algorithm.log_debug(f"  多元化不足: {'是' if metrics['is_under_diversified'] else '否'}", log_type="diversification")
    
    def _force_diversification(self, weights, symbols, expected_returns=None):
        """强制多元化"""
        try:
            # 目标参数
            min_stocks = 4
            max_weight = 0.30  # 单股最大权重30%
            target_stocks = min(8, len(symbols))  # 目标8只股票
            
            # 策略1：如果股票数量不足，扩展股票池
            if len(symbols) < min_stocks:
                self.algorithm.log_debug(f"股票数量不足({len(symbols)} < {min_stocks})，尝试扩展", log_type="diversification")
                # 这里需要从更大的股票池中选择，暂时使用当前的
                target_stocks = max(min_stocks, len(symbols))
            
            # 策略2：重新分配权重，确保多元化
            new_weights = self._redistribute_weights_for_diversification(
                weights, symbols, target_stocks, max_weight, expected_returns
            )
            
            # 策略3：如果仍然不够多元化，强制等权重分配
            final_metrics = self._calculate_diversification_metrics(new_weights, symbols)
            if final_metrics['is_under_diversified']:
                self.algorithm.log_debug(f"仍然多元化不足，采用强制等权重策略", log_type="diversification")
                new_weights = self._apply_equal_weight_fallback(symbols, target_stocks)
            
            return new_weights, symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"强制多元化过程错误: {e}", log_type='diversification')
            # 失败时返回等权重
            n = len(symbols)
            return np.ones(n) / n if n > 0 else np.array([]), symbols
    
    def _redistribute_weights_for_diversification(self, weights, symbols, target_stocks, max_weight, expected_returns):
        """重新分配权重以实现多元化"""
        try:
            n_stocks = len(symbols)
            
            if n_stocks == 0:
                return np.array([])
            
            # 1. 限制最大权重
            capped_weights = np.minimum(weights, max_weight)
            
            # 2. 重新分配被削减的权重
            excess_weight = np.sum(weights) - np.sum(capped_weights)
            if excess_weight > 0:
                # 将多余权重均匀分配给所有股票
                additional_weight = excess_weight / n_stocks
                capped_weights += additional_weight
                # 再次确保不超过最大权重
                capped_weights = np.minimum(capped_weights, max_weight)
            
            # 3. 应用多元化偏好
            diversification_strength = 0.4  # 40%的等权重偏向
            equal_weights = np.ones(n_stocks) / n_stocks
            diversified_weights = (1 - diversification_strength) * capped_weights + diversification_strength * equal_weights
            
            # 4. 归一化
            diversified_weights = diversified_weights / np.sum(diversified_weights)
            
            # 5. 确保最小权重
            min_weight = 0.03  # 3%最小权重
            diversified_weights = np.maximum(diversified_weights, min_weight)
            
            # 6. 最终归一化
            diversified_weights = diversified_weights / np.sum(diversified_weights)
            
            self.algorithm.log_debug(f"权重重分配完成: 最大权重={np.max(diversified_weights):.2%}", log_type="diversification")
            
            return diversified_weights
            
        except Exception as e:
            self.algorithm.log_debug(f"权重重分配错误: {e}", log_type='diversification')
            return weights
    
    def _apply_equal_weight_fallback(self, symbols, target_stocks):
        """应用等权重备用策略"""
        n_stocks = min(target_stocks, len(symbols))
        
        if n_stocks == 0:
            return np.array([])
        
        # 简单等权重分配
        equal_weights = np.ones(n_stocks) / n_stocks
        
        # 如果原始股票数量更多，只保留前n_stocks只
        if len(symbols) > n_stocks:
            # 可以根据预期收益或其他指标选择，这里简化为前n_stocks只
            equal_weights = np.concatenate([equal_weights, np.zeros(len(symbols) - n_stocks)])
        
        self.algorithm.log_debug(f"应用等权重备用策略: {n_stocks}只股票，每只权重={1/n_stocks:.2%}", log_type="diversification")
        
        return equal_weights
    
    def check_portfolio_diversification(self, portfolio_value):
        """检查当前投资组合的多元化状态"""
        try:
            current_holdings = []
            current_weights = []
            total_invested = 0
            
            # 获取当前持仓
            for holding in self.algorithm.Portfolio.Values:
                if holding.Invested:
                    current_holdings.append(str(holding.Symbol))
                    current_weights.append(holding.HoldingsValue)
                    total_invested += holding.HoldingsValue
            
            if total_invested > 0:
                # 转换为权重
                weight_array = np.array(current_weights) / total_invested
                
                # 计算多元化指标
                metrics = self._calculate_diversification_metrics(weight_array, current_holdings)
                
                # 记录状态
                self.algorithm.log_debug("=== 当前投资组合多元化检查 ===", log_type="diversification")
                self._log_diversification_metrics(metrics)
                
                return metrics
            
        except Exception as e:
            self.algorithm.log_debug(f"投资组合多元化检查错误: {e}", log_type='diversification')
        
        return None 