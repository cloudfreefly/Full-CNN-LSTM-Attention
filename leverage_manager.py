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
            # *** Alert Black Bat杠杆管理器执行验证 ***
            self.algorithm.log_debug("Alert Black Bat杠杆管理器已执行", log_type="risk")
            
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
            
            # === 极端风险快速降杠杆机制 ===
            is_extreme_situation = (
                risk_level == 'extreme' or 
                vix_level >= 22 or 
                drawdown_level >= 0.08 or
                volatility_level >= 0.30
            )
            
            if is_extreme_situation:
                # 极端情况下直接使用目标杠杆，不平滑
                self.algorithm.log_debug(f"极端风险状况，立即调整杠杆: {self._current_leverage_ratio:.2f} -> {target_leverage:.2f}", 
                                       log_type="risk")
            else:
                # 平滑调整，避免频繁变动
                smooth_factor = leverage_config.get('leverage_smooth_factor', 0.6)  # 从0.3提高到0.6
                min_change = leverage_config.get('min_leverage_change', 0.03)  # 从0.05降低到0.03
                
                if self._current_leverage_ratio > 0:
                    leverage_change = target_leverage - self._current_leverage_ratio
                    if abs(leverage_change) > min_change:
                        # 根据风险等级动态调整平滑因子
                        if risk_level in ['high', 'extreme']:
                            # 高风险时快速响应
                            dynamic_smooth_factor = min(0.8, smooth_factor * 1.5)
                        elif abs(leverage_change) > 0.15:
                            # 大幅度变化时提高响应速度
                            dynamic_smooth_factor = min(0.8, smooth_factor * 1.3)
                        else:
                            dynamic_smooth_factor = smooth_factor
                        
                        # 应用动态平滑因子
                        adjusted_leverage = self._current_leverage_ratio + leverage_change * dynamic_smooth_factor
                        target_leverage = adjusted_leverage
                        self.algorithm.log_debug(f"平滑调整杠杆: {self._current_leverage_ratio:.2f} -> {target_leverage:.2f} (因子:{dynamic_smooth_factor:.1f})", 
                                               log_type="risk")
                    else:
                        # 变化太小，保持当前杠杆
                        target_leverage = self._current_leverage_ratio
                        self.algorithm.log_debug(f"杠杆变化过小，保持当前值: {target_leverage:.2f}", log_type="risk")
            
            # 确保在合理范围内
            max_leverage = leverage_config.get('max_leverage_ratio', 1.5)
            target_leverage = max(0.5, min(max_leverage, target_leverage))
            
            self._target_leverage_ratio = target_leverage
            self._risk_level = risk_level
            
            self.algorithm.log_debug(f"杠杆计算完成: 风险等级={risk_level}, 目标杠杆={target_leverage:.2f}x, 极端状况={is_extreme_situation}", 
                                    log_type="risk")
            
            return target_leverage
            
        except Exception as e:
            self.algorithm.log_debug(f"杠杆计算错误: {e}", log_type="risk")
            return 1.0
    
    def _get_vix_level(self):
        """获取VIX水平"""
        try:
            if hasattr(self.algorithm, 'vix_monitor'):
                # 正确的属性名是_last_vix_value，不是_current_vix
                vix_value = getattr(self.algorithm.vix_monitor, '_last_vix_value', None)
                if vix_value is not None and vix_value > 0:
                    self.algorithm.log_debug(f"从VIX监控器获取到VIX值: {vix_value:.2f}", log_type="risk")
                    return vix_value
                else:
                    self.algorithm.log_debug("VIX监控器中无有效VIX值，使用默认值20", log_type="risk")
                    return 20
            else:
                self.algorithm.log_debug("算法中无VIX监控器，使用默认VIX值20", log_type="risk")
                return 20  # 默认中等水平
        except Exception as e:
            self.algorithm.log_debug(f"获取VIX水平错误: {e}，使用默认值20", log_type="risk")
            return 20
    
    def _get_volatility_level(self):
        """获取波动率水平"""
        try:
            # 从风险管理器的波动率监控器获取实际波动率
            if hasattr(self.algorithm, 'risk_manager') and hasattr(self.algorithm.risk_manager, 'volatility_monitor'):
                volatility_monitor = self.algorithm.risk_manager.volatility_monitor
                
                # 首先尝试计算当前波动率
                if hasattr(self.algorithm, 'previous_portfolio_value'):
                    current_value = self.algorithm.Portfolio.TotalPortfolioValue
                    daily_return = (current_value - self.algorithm.previous_portfolio_value) / self.algorithm.previous_portfolio_value
                    current_volatility = volatility_monitor.update_return(daily_return)
                    
                    if current_volatility > 0:
                        self.algorithm.log_debug(f"获取到实际波动率: {current_volatility:.3f}", log_type="risk")
                        return current_volatility
                
                # 如果有历史收益率数据，直接计算波动率
                if len(volatility_monitor.daily_returns) >= 20:
                    import numpy as np
                    volatility = np.std(volatility_monitor.daily_returns) * np.sqrt(252)
                    self.algorithm.log_debug(f"通过历史收益率计算波动率: {volatility:.3f}", log_type="risk")
                    return volatility
                
                self.algorithm.log_debug("波动率监控器中数据不足，使用默认值", log_type="risk")
            else:
                self.algorithm.log_debug("风险管理器或波动率监控器不可用，使用默认值", log_type="risk")
            
            return 0.15  # 默认中等波动率
        except Exception as e:
            self.algorithm.log_debug(f"获取波动率失败: {e}，使用默认值", log_type="risk")
            return 0.15
    
    def _get_current_drawdown(self):
        """获取当前回撤水平"""
        try:
            # 从风险管理器的回撤监控器获取实际回撤
            if hasattr(self.algorithm, 'risk_manager') and hasattr(self.algorithm.risk_manager, 'drawdown_monitor'):
                drawdown_monitor = self.algorithm.risk_manager.drawdown_monitor
                
                # 更新当前投资组合价值并获取回撤
                current_value = self.algorithm.Portfolio.TotalPortfolioValue
                current_drawdown = drawdown_monitor.update_portfolio_value(current_value)
                
                if current_drawdown is not None:
                    self.algorithm.log_debug(f"获取到实际回撤: {current_drawdown:.3f}", log_type="risk")
                    return abs(current_drawdown)
                
                self.algorithm.log_debug("回撤监控器返回None，使用默认值", log_type="risk")
            else:
                self.algorithm.log_debug("风险管理器或回撤监控器不可用，使用默认值", log_type="risk")
            
            return 0.0
        except Exception as e:
            self.algorithm.log_debug(f"获取回撤失败: {e}，使用默认值", log_type="risk")
            return 0.0
    
    def _assess_risk_level(self, vix_level, volatility_level, drawdown_level):
        """评估市场风险等级 - Alert Black Bat分析优化"""
        leverage_config = self.config.LEVERAGE_CONFIG
        
        # VIX风险评估 - 使用Alert Black Bat优化的阈值
        if vix_level < leverage_config.get('low_risk_vix_threshold', 18):
            vix_risk = 'low'
        elif vix_level < leverage_config.get('medium_risk_vix_threshold', 22):  # 从25降至22
            vix_risk = 'medium'
        elif vix_level < leverage_config.get('high_risk_vix_threshold', 30):   # 从35降至30
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
        
        # === Alert Black Bat分析：优化回撤风险评估阈值 ===
        if drawdown_level < 0.05:      # 5%以下为低风险
            dd_risk = 'low'
        elif drawdown_level < 0.08:    # 5-8%为中等风险
            dd_risk = 'medium'
        elif drawdown_level < 0.12:    # 8-12%为高风险
            dd_risk = 'high'
        else:                          # 12%以上为极端风险
            dd_risk = 'extreme'
        
        # === Alert Black Bat分析：动态杠杆调整逻辑 ===
        enable_drawdown_adjustment = leverage_config.get('enable_drawdown_based_adjustment', False)
        
        if enable_drawdown_adjustment:
            drawdown_thresholds = leverage_config.get('drawdown_thresholds', {})
            
            for threshold in sorted(drawdown_thresholds.keys(), reverse=True):
                if drawdown_level >= threshold:
                    # 直接基于回撤设置杠杆，优先级最高
                    target_leverage = drawdown_thresholds[threshold]
                    self.algorithm.log_debug(f"Alert Black Bat杠杆: 回撤{drawdown_level:.1%} -> {target_leverage}x", 
                                           log_type="risk")
                    # 设置一个特殊标记，表示这是基于回撤的强制调整
                    self._drawdown_forced_leverage = target_leverage
                    break
        
        # 综合风险评估（取最高风险等级）
        risk_levels = ['low', 'medium', 'high', 'extreme']
        overall_risk = max([vix_risk, vol_risk, dd_risk], key=lambda x: risk_levels.index(x))
        
        self.algorithm.log_debug(f"风险评估: VIX={vix_risk}({vix_level:.1f}), 波动率={vol_risk}({volatility_level:.1%}), "
                               f"回撤={dd_risk}({drawdown_level:.1%}), 综合={overall_risk}", log_type="risk")
        
        return overall_risk
    
    def _get_leverage_by_risk_level(self, risk_level):
        """根据风险等级获取杠杆比例 - Alert Black Bat分析优化"""
        leverage_config = self.config.LEVERAGE_CONFIG
        
        # === Alert Black Bat分析：优先检查强制杠杆调整 ===
        if hasattr(self, '_drawdown_forced_leverage'):
            forced_leverage = self._drawdown_forced_leverage
            # 清除标记
            delattr(self, '_drawdown_forced_leverage')
            self.algorithm.log_debug(f"使用Alert Black Bat强制杠杆: {forced_leverage}x", log_type="risk")
            return forced_leverage
        
        # === 检查紧急去杠杆模式 ===
        emergency_config = leverage_config.get('emergency_deleveraging', {})
        if emergency_config.get('enable_emergency_mode', False):
            current_drawdown = self._get_current_drawdown()
            trigger_drawdown = emergency_config.get('trigger_drawdown', 0.10)
            
            if current_drawdown >= trigger_drawdown:
                emergency_leverage = emergency_config.get('emergency_target_leverage', 0.6)
                self.algorithm.log_debug(f"紧急去杠杆模式: 回撤{current_drawdown:.1%}触发，目标杠杆{emergency_leverage}x", 
                                       log_type="risk")
                return emergency_leverage
        
        # 常规风险等级杠杆映射
        leverage_map = {
            'low': leverage_config.get('low_risk_leverage_ratio', 1.5),
            'medium': leverage_config.get('medium_risk_leverage_ratio', 1.2),
            'high': leverage_config.get('high_risk_leverage_ratio', 0.8),
            'extreme': leverage_config.get('extreme_risk_leverage_ratio', 0.5)
        }
        
        return leverage_map.get(risk_level, 1.0)
    
    def update_leverage_ratio(self):
        """更新杠杆比例"""
        try:
            self.algorithm.log_debug("update_leverage_ratio方法已调用", log_type="risk")  # 添加调试日志
            
            # 计算新的目标杠杆比例
            target_leverage = self.calculate_target_leverage_ratio()
            
            # 检查是否需要更新
            if self._should_update_leverage(target_leverage):
                self._current_leverage_ratio = target_leverage
                self._last_leverage_update = self.algorithm.Time
                
                # 记录杠杆历史
                self._leverage_history.append({
                    'timestamp': self.algorithm.Time,
                    'leverage_ratio': target_leverage,
                    'risk_level': self._risk_level
                })
                
                # 限制历史记录长度
                if len(self._leverage_history) > 100:
                    self._leverage_history = self._leverage_history[-50:]
                
                self.algorithm.log_debug(f"杠杆已更新: {target_leverage:.2f}x", log_type="risk")
            else:
                self.algorithm.log_debug(f"杠杆无需更新，保持: {self._current_leverage_ratio:.2f}x", log_type="risk")
                
        except Exception as e:
            self.algorithm.log_debug(f"杠杆更新失败: {e}", log_type="risk")
    
    def _should_update_leverage(self, target_leverage):
        """判断是否应该更新杠杆比例"""
        leverage_config = self.config.LEVERAGE_CONFIG
        
        # 获取当前风险状态，检查是否为极端情况
        vix_level = self._get_vix_level()
        drawdown_level = self._get_current_drawdown()
        volatility_level = self._get_volatility_level()
        
        is_extreme_situation = (
            self._risk_level == 'extreme' or 
            vix_level >= 22 or 
            drawdown_level >= 0.08 or
            volatility_level >= 0.30
        )
        
        # 极端情况下强制更新，绕过所有限制
        if is_extreme_situation:
            self.algorithm.log_debug(f"极端风险状况，强制更新杠杆: {self._current_leverage_ratio:.2f} -> {target_leverage:.2f}", 
                                   log_type="risk")
            return True
        
        # 高风险情况下放宽限制
        if hasattr(self, '_risk_level') and self._risk_level == 'high':
            # 高风险时减少频率限制和变化阈值
            min_change = leverage_config.get('high_risk_min_change', 0.02)  # 高风险时2%变化即更新
            leverage_change = abs(target_leverage - self._current_leverage_ratio)
            
            if leverage_change >= min_change:
                self.algorithm.log_debug(f"高风险状况，杠杆变化达到阈值: {leverage_change:.3f} >= {min_change}", log_type="risk")
                return True
        
        # 检查更新频率（调整为小时级别，而非天级别）
        if self._last_leverage_update:
            hours_since_update = (self.algorithm.Time - self._last_leverage_update).total_seconds() / 3600
            min_hours = leverage_config.get('leverage_adjustment_frequency_hours', 2)  # 最少2小时间隔
            
            if hours_since_update < min_hours:
                # 但如果变化幅度很大，仍然允许更新
                leverage_change = abs(target_leverage - self._current_leverage_ratio)
                if leverage_change >= 0.1:  # 10%以上的大变化
                    self.algorithm.log_debug(f"大幅杠杆变化，绕过频率限制: {leverage_change:.3f}", log_type="risk")
                    return True
                else:
                    self.algorithm.log_debug(f"杠杆更新频率限制: 距上次更新{hours_since_update:.1f}小时 < {min_hours}小时", log_type="risk")
                    return False
        
        # 检查杠杆变化幅度（降低阈值）
        min_change = leverage_config.get('min_leverage_change', 0.03)  # 从0.05降低到0.03
        leverage_change = abs(target_leverage - self._current_leverage_ratio)
        
        if leverage_change >= min_change:
            self.algorithm.log_debug(f"杠杆变化达到阈值: {leverage_change:.3f} >= {min_change}", log_type="risk")
            return True
        else:
            self.algorithm.log_debug(f"杠杆变化不足: {leverage_change:.3f} < {min_change}", log_type="risk")
            return False
    
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