# 杠杆管理模块
from AlgorithmImports import *
import numpy as np
from config import AlgorithmConfig

class LeverageManager:
    """杠杆管理器 - 支持150%最高持仓"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
        # 杠杆状态跟踪
        self._current_leverage_ratio = 1.0
        self._target_leverage_ratio = 1.0
        self._last_leverage_update = None
        self._leverage_history = []
        
        # 风险状态缓存
        self._current_vix = None
        self._current_volatility = None
        self._risk_level = 'medium'
    
    def update_leverage_parameters(self):
        """更新杠杆管理参数配置"""
        try:
            # 重新加载配置
            self.config = AlgorithmConfig()
            
            # 重新计算目标杠杆比例
            self._target_leverage_ratio = self.calculate_target_leverage_ratio()
            
            self.algorithm.log_debug("杠杆管理参数已更新", log_type="leverage")
            
        except Exception as e:
            self.algorithm.log_debug(f"更新杠杆参数失败: {e}", log_type="leverage")
        
    def calculate_target_leverage_ratio(self, market_data=None):
        """
        计算目标杠杆比例
        根据市场风险状况动态调整杠杆水平
        """
        try:
            leverage_config = self.config.LEVERAGE_CONFIG
            
            if not leverage_config.get('enable_leverage', False):
                return 1.0
            
            # 获取市场风险指标
            vix_level = self._get_vix_level()
            volatility_level = self._get_volatility_level()
            drawdown_level = self._get_current_drawdown()
            
            # 风险评估
            risk_level = self._assess_risk_level(vix_level, volatility_level, drawdown_level)
            
            # 根据风险水平确定杠杆比例
            target_leverage = self._get_leverage_by_risk_level(risk_level)
            
            # 平滑调整，避免频繁变动
            smooth_factor = leverage_config.get('leverage_smooth_factor', 0.3)
            min_change = leverage_config.get('min_leverage_change', 0.05)
            
            if self._current_leverage_ratio > 0:
                leverage_change = target_leverage - self._current_leverage_ratio
                if abs(leverage_change) > min_change:
                    # 应用平滑因子
                    adjusted_leverage = self._current_leverage_ratio + leverage_change * smooth_factor
                    target_leverage = adjusted_leverage
                else:
                    # 变化太小，保持当前杠杆
                    target_leverage = self._current_leverage_ratio
            
            # 确保在合理范围内
            max_leverage = leverage_config.get('max_leverage_ratio', 1.5)
            target_leverage = max(0.5, min(max_leverage, target_leverage))
            
            self._target_leverage_ratio = target_leverage
            self._risk_level = risk_level
            
            self.algorithm.log_debug(f"杠杆计算: 风险等级={risk_level}, 目标杠杆={target_leverage:.2f}x", 
                                    log_type="risk")
            
            return target_leverage
            
        except Exception as e:
            self.algorithm.log_debug(f"杠杆计算错误: {e}", log_type="risk")
            return 1.0
    
    def _get_vix_level(self):
        """获取VIX水平"""
        try:
            if hasattr(self.algorithm, 'vix_monitor'):
                return getattr(self.algorithm.vix_monitor, '_current_vix', 20)
            return 20  # 默认中等水平
        except:
            return 20
    
    def _get_volatility_level(self):
        """获取波动率水平"""
        try:
            # 从风险管理器获取当前波动率
            if hasattr(self.algorithm, 'risk_manager'):
                return getattr(self.algorithm.risk_manager, '_current_volatility', 0.15)
            return 0.15  # 默认中等波动率
        except:
            return 0.15
    
    def _get_current_drawdown(self):
        """获取当前回撤水平"""
        try:
            if hasattr(self.algorithm, 'drawdown_monitor'):
                return abs(getattr(self.algorithm.drawdown_monitor, '_current_drawdown', 0.0))
            return 0.0
        except:
            return 0.0
    
    def _assess_risk_level(self, vix_level, volatility_level, drawdown_level):
        """评估市场风险等级"""
        leverage_config = self.config.LEVERAGE_CONFIG
        
        # VIX风险评估
        if vix_level < leverage_config.get('low_risk_vix_threshold', 18):
            vix_risk = 'low'
        elif vix_level < leverage_config.get('medium_risk_vix_threshold', 25):
            vix_risk = 'medium'
        elif vix_level < leverage_config.get('high_risk_vix_threshold', 35):
            vix_risk = 'high'
        else:
            vix_risk = 'extreme'
        
        # 波动率风险评估
        if volatility_level < leverage_config.get('low_risk_volatility_threshold', 0.12):
            vol_risk = 'low'
        elif volatility_level < leverage_config.get('medium_risk_volatility_threshold', 0.20):
            vol_risk = 'medium'
        elif volatility_level < leverage_config.get('high_risk_volatility_threshold', 0.30):
            vol_risk = 'high'
        else:
            vol_risk = 'extreme'
        
        # 回撤风险评估
        if drawdown_level < 0.05:
            dd_risk = 'low'
        elif drawdown_level < 0.10:
            dd_risk = 'medium'
        elif drawdown_level < 0.15:
            dd_risk = 'high'
        else:
            dd_risk = 'extreme'
        
        # 综合风险评估（取最高风险等级）
        risk_levels = ['low', 'medium', 'high', 'extreme']
        overall_risk = max([vix_risk, vol_risk, dd_risk], key=lambda x: risk_levels.index(x))
        
        self.algorithm.log_debug(f"风险评估: VIX={vix_risk}, 波动率={vol_risk}, 回撤={dd_risk}, "
                               f"综合={overall_risk}", log_type="risk")
        
        return overall_risk
    
    def _get_leverage_by_risk_level(self, risk_level):
        """根据风险等级获取杠杆比例"""
        leverage_config = self.config.LEVERAGE_CONFIG
        
        leverage_map = {
            'low': leverage_config.get('low_risk_leverage_ratio', 1.5),
            'medium': leverage_config.get('medium_risk_leverage_ratio', 1.2),
            'high': leverage_config.get('high_risk_leverage_ratio', 0.8),
            'extreme': leverage_config.get('extreme_risk_leverage_ratio', 0.5)
        }
        
        return leverage_map.get(risk_level, 1.0)
    
    def update_leverage_ratio(self):
        """更新当前杠杆比例"""
        try:
            target_leverage = self.calculate_target_leverage_ratio()
            
            # 检查是否需要更新
            if self._should_update_leverage():
                self._current_leverage_ratio = target_leverage
                self._last_leverage_update = self.algorithm.Time
                
                # 记录杠杆变化
                self._leverage_history.append({
                    'time': self.algorithm.Time,
                    'leverage_ratio': target_leverage,
                    'risk_level': self._risk_level
                })
                
                # 限制历史记录长度
                if len(self._leverage_history) > 100:
                    self._leverage_history = self._leverage_history[-100:]
                
                self.algorithm.log_debug(f"杠杆更新: {self._current_leverage_ratio:.2f}x "
                                       f"(风险等级: {self._risk_level})", log_type="risk")
                
                return True
            
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"杠杆更新错误: {e}", log_type="risk")
            return False
    
    def _should_update_leverage(self):
        """判断是否应该更新杠杆比例"""
        leverage_config = self.config.LEVERAGE_CONFIG
        update_frequency = leverage_config.get('leverage_adjustment_frequency', 1)
        
        # 检查更新频率
        if self._last_leverage_update:
            days_since_update = (self.algorithm.Time - self._last_leverage_update).days
            if days_since_update < update_frequency:
                return False
        
        # 检查杠杆变化幅度
        min_change = leverage_config.get('min_leverage_change', 0.05)
        leverage_change = abs(self._target_leverage_ratio - self._current_leverage_ratio)
        
        return leverage_change >= min_change
    
    def get_current_leverage_ratio(self):
        """
        获取当前杠杆比例
        返回: float - 当前杠杆比例（1.0=无杠杆，1.3=130%仓位）
        """
        try:
            leverage_config = self.config.LEVERAGE_CONFIG
            
            if not leverage_config.get('enable_leverage', False):
                return 1.0
            
            # 获取最新的目标杠杆比例
            if self._target_leverage_ratio is None:
                self._target_leverage_ratio = self.calculate_target_leverage_ratio()
            
            # 返回当前目标杠杆比例
            current_ratio = self._target_leverage_ratio
            
            self.algorithm.log_debug(f"当前杠杆比例: {current_ratio:.2f}", log_type="leverage")
            
            return current_ratio
            
        except Exception as e:
            self.algorithm.log_debug(f"获取杠杆比例错误: {str(e)}", log_type="error")
            return 1.0
    
    def get_target_leverage_ratio(self):
        """获取目标杠杆比例"""
        return self._target_leverage_ratio
    
    def get_leverage_status(self):
        """获取杠杆状态信息"""
        return {
            'current_leverage': self._current_leverage_ratio,
            'target_leverage': self._target_leverage_ratio,
            'risk_level': self._risk_level,
            'last_update': self._last_leverage_update,
            'equity_ratio': self._current_leverage_ratio,  # 股票仓位比例等于杠杆比例
            'cash_ratio': 1.0 - self._current_leverage_ratio  # 现金比例（可能为负）
        }
    
    def apply_leverage_to_weights(self, weights, symbols):
        """将杠杆比例应用到权重中"""
        try:
            if weights is None or len(weights) == 0:
                return weights, symbols
            
            current_leverage = self.get_current_leverage_ratio()
            
            # 将权重按杠杆比例缩放
            leveraged_weights = weights * current_leverage
            
            self.algorithm.log_debug(f"应用杠杆: {current_leverage:.2f}x, "
                                   f"权重总和: {np.sum(weights):.4f} -> {np.sum(leveraged_weights):.4f}", 
                                   log_type="portfolio")
            
            return leveraged_weights, symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"杠杆应用错误: {e}", log_type="portfolio")
            return weights, symbols
    
    def check_margin_requirements(self):
        """检查保证金要求"""
        try:
            leverage_config = self.config.LEVERAGE_CONFIG
            portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            cash = float(self.algorithm.Portfolio.Cash)
            
            # 计算当前杠杆使用情况
            if portfolio_value > 0:
                cash_ratio = cash / portfolio_value
                equity_ratio = 1.0 - cash_ratio
                
                # 检查是否接近保证金追缴线
                margin_call_threshold = -0.40  # 借款40%时触发
                liquidation_threshold = -0.45  # 借款45%时强制平仓
                
                if cash_ratio <= liquidation_threshold:
                    self.algorithm.log_debug("⚠️ 触发强制平仓线！立即减仓", log_type="risk")
                    return 'liquidation'
                elif cash_ratio <= margin_call_threshold:
                    self.algorithm.log_debug("⚠️ 触发保证金追缴线！需要减仓", log_type="risk")
                    return 'margin_call'
                elif cash_ratio <= -0.30:
                    self.algorithm.log_debug("⚠️ 杠杆使用率较高，注意风险", log_type="risk")
                    return 'warning'
                
            return 'normal'
            
        except Exception as e:
            self.algorithm.log_debug(f"保证金检查错误: {e}", log_type="risk")
            return 'error'
    
    def get_leverage_statistics(self):
        """获取杠杆使用统计"""
        if not self._leverage_history:
            return {}
        
        recent_history = self._leverage_history[-30:]  # 最近30次记录
        
        leverage_ratios = [h['leverage_ratio'] for h in recent_history]
        
        return {
            'avg_leverage': np.mean(leverage_ratios),
            'max_leverage': np.max(leverage_ratios),
            'min_leverage': np.min(leverage_ratios),
            'current_leverage': self._current_leverage_ratio,
            'leverage_changes': len(self._leverage_history),
            'risk_distribution': self._get_risk_distribution()
        }
    
    def _get_risk_distribution(self):
        """获取风险等级分布"""
        if not self._leverage_history:
            return {}
        
        recent_history = self._leverage_history[-50:]
        risk_levels = [h['risk_level'] for h in recent_history]
        
        distribution = {}
        for level in ['low', 'medium', 'high', 'extreme']:
            distribution[level] = risk_levels.count(level) / len(risk_levels)
        
        return distribution 