# 自动优化管理器 - 与QuantConnect集成
from AlgorithmImports import *
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from parameter_optimizer import ParameterOptimizationManager, OptimizationConfig
from config import AlgorithmConfig

class QuantConnectOptimizationManager:
    """QuantConnect自动优化管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.optimization_history = []
        self.current_best_config = None
        self.current_best_score = float('-inf')
        
        # 优化设置
        self.optimization_frequency = timedelta(days=30)  # 每月优化一次
        self.last_optimization_date = None
        
        # 性能跟踪
        self.performance_baseline = {}
        self.optimization_enabled = True
        
    def initialize_optimization(self):
        """初始化优化系统"""
        try:
            # 加载历史优化结果
            self._load_optimization_history()
            
            # 设置初始基准
            self._set_performance_baseline()
            
            self.algorithm.Debug("自动优化系统初始化完成")
            
        except Exception as e:
            self.algorithm.Debug(f"优化系统初始化失败: {e}")
            self.optimization_enabled = False
    
    def should_run_optimization(self) -> bool:
        """判断是否应该运行优化"""
        if not self.optimization_enabled:
            return False
            
        # 检查优化频率
        if self.last_optimization_date is None:
            return True
            
        time_since_last = self.algorithm.Time - self.last_optimization_date
        
        # 如果性能显著下降，提前触发优化
        if self._performance_degraded():
            self.algorithm.Debug("检测到性能下降，触发优化")
            return True
            
        return time_since_last >= self.optimization_frequency
    
    def run_parameter_optimization(self):
        """运行参数优化"""
        if not self.should_run_optimization():
            return
            
        try:
            self.algorithm.Debug("=" * 60)
            self.algorithm.Debug("开始运行参数优化...")
            
            # 获取当前性能指标作为基准
            current_performance = self._get_current_performance()
            self._log_current_performance(current_performance)
            
            # 检查性能是否下降
            performance_degraded = self._performance_degraded()
            if performance_degraded:
                self._log_performance_degradation()
            
            # 定义优化参数空间
            parameter_space = self._get_optimization_parameter_space()
            self._log_optimization_parameters(parameter_space)
            
            # 运行优化
            optimization_results = self._run_optimization_batch(parameter_space)
            
            # 分析结果
            best_config = self._analyze_optimization_results(optimization_results)
            
            # 应用最佳配置
            if best_config:
                self._log_optimization_results(best_config, optimization_results)
                self._apply_best_configuration(best_config)
            else:
                self.algorithm.Debug("未找到更好的参数配置，保持当前设置")
                
            # 记录优化历史
            self._record_optimization_result(current_performance, best_config)
            
            self.last_optimization_date = self.algorithm.Time
            self.algorithm.Debug("参数优化完成")
            self.algorithm.Debug("=" * 60)
            
        except Exception as e:
            self.algorithm.Debug(f"参数优化失败: {e}")
    
    def _get_optimization_parameter_space(self) -> Dict:
        """获取当前适用的参数空间"""
        # 基于市场状态调整参数空间
        vix_level = self._get_current_vix_level()
        market_volatility = self._get_market_volatility()
        
        if vix_level > 30:  # 高波动市场
            return {
                'max_drawdown': [0.08, 0.10, 0.12, 0.15],
                'volatility_threshold': [0.20, 0.25, 0.30, 0.35],
                'vix_extreme_level': [25, 30, 35],
                'max_leverage_ratio': [0.8, 1.0, 1.2],
                'defensive_max_cash_ratio': [0.3, 0.4, 0.5]
            }
        elif vix_level < 18:  # 低波动市场
            return {
                'max_drawdown': [0.05, 0.08, 0.10],
                'volatility_threshold': [0.15, 0.20, 0.25],
                'max_leverage_ratio': [1.2, 1.5, 1.8],
                'target_portfolio_size': [10, 12, 15],
                'max_weight': [0.10, 0.12, 0.15]
            }
        else:  # 正常市场
            return {
                'max_drawdown': [0.06, 0.08, 0.10, 0.12],
                'volatility_threshold': [0.18, 0.22, 0.25, 0.28],
                'max_leverage_ratio': [1.0, 1.2, 1.5],
                'target_portfolio_size': [8, 10, 12, 15],
                'rebalance_tolerance': [0.003, 0.005, 0.010]
            }
    
    def _run_optimization_batch(self, parameter_space: Dict) -> List[Dict]:
        """运行优化批次"""
        results = []
        
        # 使用较少的参数组合以节省计算资源
        optimization_methods = ['random_search']  # 在线环境使用随机搜索
        
        for method in optimization_methods:
            try:
                # 创建优化管理器
                optimizer = ParameterOptimizationManager(self.config)
                
                # 运行优化（较少迭代次数）
                result = optimizer.run_optimization(
                    parameter_space=parameter_space,
                    optimization_method=method,
                    objective_function='sharpe_ratio',
                    n_iterations=20  # 减少迭代次数以适应在线环境
                )
                
                results.append(result)
                
            except Exception as e:
                self.algorithm.Debug(f"优化方法 {method} 失败: {e}")
                continue
        
        return results
    
    def _analyze_optimization_results(self, results: List[Dict]) -> Optional[Dict]:
        """分析优化结果"""
        if not results:
            return None
            
        best_result = None
        best_score = float('-inf')
        
        for result in results:
            if result.get('best_score', float('-inf')) > best_score:
                best_score = result['best_score']
                best_result = result
        
        if best_result and best_score > self.current_best_score:
            self.current_best_score = best_score
            return best_result['best_params']
        
        return None
    
    def _apply_best_configuration(self, best_config: Dict):
        """应用最佳配置"""
        try:
            # 更新配置
            for param_name, param_value in best_config.items():
                self._update_algorithm_parameter(param_name, param_value)
            
            self.current_best_config = best_config
            self.algorithm.Debug(f"应用新配置: {best_config}")
            
        except Exception as e:
            self.algorithm.Debug(f"应用配置失败: {e}")
    
    def _update_algorithm_parameter(self, param_name: str, param_value: Any):
        """更新算法参数"""
        try:
            # 映射参数到配置类
            if param_name == 'max_drawdown':
                self.config.RISK_CONFIG['max_drawdown'] = param_value
            elif param_name == 'volatility_threshold':
                self.config.RISK_CONFIG['volatility_threshold'] = param_value
            elif param_name == 'max_leverage_ratio':
                self.config.LEVERAGE_CONFIG['max_leverage_ratio'] = param_value
            elif param_name == 'target_portfolio_size':
                self.config.PORTFOLIO_CONFIG['target_portfolio_size'] = param_value
            elif param_name == 'max_weight':
                self.config.PORTFOLIO_CONFIG['max_weight'] = param_value
            elif param_name == 'rebalance_tolerance':
                self.config.PORTFOLIO_CONFIG['rebalance_tolerance'] = param_value
            elif param_name == 'vix_extreme_level':
                self.config.RISK_CONFIG['vix_extreme_level'] = param_value
            elif param_name == 'defensive_max_cash_ratio':
                self.config.PORTFOLIO_CONFIG['defensive_max_cash_ratio'] = param_value
            
            # 通知相关模块参数已更新
            self._notify_parameter_update(param_name, param_value)
            
        except Exception as e:
            self.algorithm.Debug(f"更新参数 {param_name} 失败: {e}")
    
    def _notify_parameter_update(self, param_name: str, param_value: Any):
        """通知相关模块参数更新"""
        try:
            # 通知风险管理器 - 直接更新配置
            if hasattr(self.algorithm, 'risk_manager'):
                self.algorithm.risk_manager.config = self.config
                if hasattr(self.algorithm.risk_manager, 'vix_monitor') and hasattr(self.algorithm.risk_manager.vix_monitor, 'config'):
                    self.algorithm.risk_manager.vix_monitor.config = self.config
            
            # 通知投资组合优化器
            if hasattr(self.algorithm, 'portfolio_optimizer'):
                if hasattr(self.algorithm.portfolio_optimizer, 'update_portfolio_parameters'):
                    self.algorithm.portfolio_optimizer.update_portfolio_parameters()
                else:
                    # 备用方法：直接更新配置
                    self.algorithm.portfolio_optimizer.config = self.config
            
            # 通知杠杆管理器
            if hasattr(self.algorithm, 'leverage_manager'):
                if hasattr(self.algorithm.leverage_manager, 'update_leverage_parameters'):
                    self.algorithm.leverage_manager.update_leverage_parameters()
                else:
                    # 备用方法：直接更新配置
                    self.algorithm.leverage_manager.config = self.config
                
        except Exception as e:
            self.algorithm.Debug(f"通知参数更新失败: {e}")
    
    def _get_current_performance(self) -> Dict:
        """获取当前性能指标"""
        try:
            portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
            cash_value = float(self.algorithm.Portfolio.Cash)
            
            # 计算收益率
            if hasattr(self.algorithm, '_initial_portfolio_value'):
                total_return = (portfolio_value / self.algorithm._initial_portfolio_value - 1) * 100
            else:
                total_return = 0
            
            # 计算当前杠杆率
            invested_value = portfolio_value - cash_value
            leverage_ratio = invested_value / portfolio_value if portfolio_value > 0 else 0
            
            # 计算最大回撤
            max_drawdown = self._calculate_current_drawdown()
            
            # 计算夏普比率（简化版）
            sharpe_ratio = self._calculate_simple_sharpe_ratio()
            
            # 获取其他性能指标
            return {
                'portfolio_value': portfolio_value,
                'total_return': total_return,
                'cash_ratio': cash_value / portfolio_value if portfolio_value > 0 else 0,
                'leverage_ratio': leverage_ratio,
                'num_holdings': len([h for h in self.algorithm.Portfolio.Values if h.Invested]),
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'timestamp': self.algorithm.Time.isoformat()
            }
            
        except Exception as e:
            self.algorithm.Debug(f"获取性能指标失败: {e}")
            return {}
    
    def _performance_degraded(self) -> bool:
        """检查性能是否显著下降"""
        try:
            current_perf = self._get_current_performance()
            
            if not self.performance_baseline or not current_perf:
                return False
            
            # 多维度性能退化检测
            baseline_return = self.performance_baseline.get('total_return', 0)
            current_return = current_perf.get('total_return', 0)
            return_degradation = current_return - baseline_return
            
            baseline_drawdown = self.performance_baseline.get('max_drawdown', 0)
            current_drawdown = current_perf.get('max_drawdown', 0)
            drawdown_increase = current_drawdown - baseline_drawdown
            
            baseline_sharpe = self.performance_baseline.get('sharpe_ratio', 0)
            current_sharpe = current_perf.get('sharpe_ratio', 0)
            sharpe_degradation = baseline_sharpe - current_sharpe
            
            # 综合判断性能是否退化
            degraded = False
            degradation_reasons = []
            
            if return_degradation < -5.0:
                degraded = True
                degradation_reasons.append(f"收益率下降{abs(return_degradation):.2f}%")
            
            if drawdown_increase > 3.0:
                degraded = True
                degradation_reasons.append(f"最大回撤增加{drawdown_increase:.2f}%")
            
            if sharpe_degradation > 0.5:
                degraded = True
                degradation_reasons.append(f"夏普比率下降{sharpe_degradation:.3f}")
            
            if degraded and degradation_reasons:
                self.algorithm.Debug(f"性能退化原因: {', '.join(degradation_reasons)}")
            
            return degraded
            
        except Exception:
            return False
    
    def _set_performance_baseline(self):
        """设置性能基准"""
        self.performance_baseline = self._get_current_performance()
    
    def _get_current_vix_level(self) -> float:
        """获取当前VIX水平"""
        try:
            if hasattr(self.algorithm, 'vix_monitor'):
                return float(self.algorithm.vix_monitor.current_vix)
            return 20.0  # 默认值
        except:
            return 20.0
    
    def _get_market_volatility(self) -> float:
        """获取市场波动率"""
        try:
            # 计算SPY的20日波动率
            spy_history = self.algorithm.History("SPY", 20, Resolution.DAILY)
            if len(spy_history) >= 20:
                prices = [bar.Close for bar in spy_history]
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                volatility = np.std(returns) * np.sqrt(252)
                return float(volatility)
            return 0.15  # 默认值
        except:
            return 0.15
    
    def _record_optimization_result(self, baseline_performance: Dict, best_config: Dict):
        """记录优化结果"""
        record = {
            'timestamp': self.algorithm.Time.isoformat(),
            'baseline_performance': baseline_performance,
            'optimized_config': best_config,
            'market_conditions': {
                'vix_level': self._get_current_vix_level(),
                'volatility': self._get_market_volatility()
            }
        }
        
        self.optimization_history.append(record)
        
        # 保存到ObjectStore（如果可用）
        try:
            self.algorithm.ObjectStore.Save("optimization_history", json.dumps(self.optimization_history))
        except:
            pass
    
    def _load_optimization_history(self):
        """加载优化历史"""
        try:
            if self.algorithm.ObjectStore.ContainsKey("optimization_history"):
                history_json = self.algorithm.ObjectStore.Read("optimization_history")
                self.optimization_history = json.loads(history_json)
                self.algorithm.Debug(f"加载了 {len(self.optimization_history)} 条优化历史记录")
        except:
            self.optimization_history = []
    
    def get_optimization_summary(self) -> str:
        """获取优化摘要"""
        if not self.optimization_history:
            return "暂无优化历史"
        
        summary = f"优化历史记录: {len(self.optimization_history)} 次\n"
        summary += f"当前最佳配置: {self.current_best_config}\n"
        summary += f"当前最佳评分: {self.current_best_score:.4f}\n"
        summary += f"上次优化时间: {self.last_optimization_date}\n"
        
        return summary
    
    def _log_current_performance(self, performance: Dict):
        """记录当前性能状态"""
        self.algorithm.Debug("当前性能状态:")
        self.algorithm.Debug(f"  组合价值: ${performance.get('portfolio_value', 0):,.2f}")
        self.algorithm.Debug(f"  总收益率: {performance.get('total_return', 0):.2f}%")
        self.algorithm.Debug(f"  现金比例: {performance.get('cash_ratio', 0):.2f}%")
        self.algorithm.Debug(f"  杠杆比例: {performance.get('leverage_ratio', 0):.2f}")
        self.algorithm.Debug(f"  持仓数量: {performance.get('num_holdings', 0)}")
        self.algorithm.Debug(f"  最大回撤: {performance.get('max_drawdown', 0):.2f}%")
        self.algorithm.Debug(f"  夏普比率: {performance.get('sharpe_ratio', 0):.3f}")
    
    def _log_performance_degradation(self):
        """记录性能下降情况"""
        if not self.performance_baseline:
            return
            
        current_perf = self._get_current_performance()
        baseline_return = self.performance_baseline.get('total_return', 0)
        current_return = current_perf.get('total_return', 0)
        degradation = current_return - baseline_return
        
        self.algorithm.Debug("检测到性能下降:")
        self.algorithm.Debug(f"  基准收益率: {baseline_return:.2f}%")
        self.algorithm.Debug(f"  当前收益率: {current_return:.2f}%")
        self.algorithm.Debug(f"  性能下降: {degradation:.2f}%")
        
        # 分析可能的原因
        baseline_drawdown = self.performance_baseline.get('max_drawdown', 0)
        current_drawdown = current_perf.get('max_drawdown', 0)
        
        if current_drawdown > baseline_drawdown + 2:
            self.algorithm.Debug(f"  回撤增加: {current_drawdown - baseline_drawdown:.2f}%")
        
        baseline_leverage = self.performance_baseline.get('leverage_ratio', 0)
        current_leverage = current_perf.get('leverage_ratio', 0)
        
        if abs(current_leverage - baseline_leverage) > 0.1:
            self.algorithm.Debug(f"  杠杆变化: {baseline_leverage:.2f} -> {current_leverage:.2f}")
    
    def _log_optimization_parameters(self, parameter_space: Dict):
        """记录优化参数空间"""
        vix_level = self._get_current_vix_level()
        market_vol = self._get_market_volatility()
        
        self.algorithm.Debug("优化参数设置:")
        self.algorithm.Debug(f"  VIX水平: {vix_level:.2f}")
        self.algorithm.Debug(f"  市场波动率: {market_vol:.2f}%")
        self.algorithm.Debug("  参数空间:")
        
        for param_name, param_values in parameter_space.items():
            self.algorithm.Debug(f"    {param_name}: {param_values}")
    
    def _log_optimization_results(self, best_config: Dict, all_results: List[Dict]):
        """记录优化结果"""
        self.algorithm.Debug("优化结果:")
        self.algorithm.Debug(f"  测试配置数量: {len(all_results)}")
        
        if all_results:
            scores = [r.get('best_score', 0) for r in all_results if 'best_score' in r]
            if scores:
                self.algorithm.Debug(f"  最高评分: {max(scores):.4f}")
                self.algorithm.Debug(f"  最低评分: {min(scores):.4f}")
                self.algorithm.Debug(f"  平均评分: {sum(scores)/len(scores):.4f}")
        
        self.algorithm.Debug("  最佳参数配置:")
        for param_name, param_value in best_config.items():
            old_value = self._get_current_parameter_value(param_name)
            if old_value is not None and old_value != param_value:
                self.algorithm.Debug(f"    {param_name}: {old_value} -> {param_value}")
            else:
                self.algorithm.Debug(f"    {param_name}: {param_value}")
    
    def _get_current_parameter_value(self, param_name: str):
        """获取当前参数值"""
        try:
            if param_name == 'max_drawdown':
                return self.config.RISK_CONFIG.get('max_drawdown')
            elif param_name == 'volatility_threshold':
                return self.config.RISK_CONFIG.get('volatility_threshold')
            elif param_name == 'max_leverage_ratio':
                return self.config.LEVERAGE_CONFIG.get('max_leverage_ratio')
            elif param_name == 'target_portfolio_size':
                return self.config.PORTFOLIO_CONFIG.get('target_portfolio_size')
            elif param_name == 'max_weight':
                return self.config.PORTFOLIO_CONFIG.get('max_weight')
            elif param_name == 'rebalance_tolerance':
                return self.config.PORTFOLIO_CONFIG.get('rebalance_tolerance')
            elif param_name == 'vix_extreme_level':
                return self.config.RISK_CONFIG.get('vix_extreme_level')
            elif param_name == 'defensive_max_cash_ratio':
                return self.config.PORTFOLIO_CONFIG.get('defensive_max_cash_ratio')
        except:
            pass
        return None
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前最大回撤"""
        try:
            if hasattr(self.algorithm, 'drawdown_monitor'):
                return self.algorithm.drawdown_monitor.current_drawdown * 100
            return 0.0
        except:
            return 0.0
    
    def _calculate_simple_sharpe_ratio(self) -> float:
        """计算简化的夏普比率"""
        try:
            if hasattr(self.algorithm, '_daily_returns') and len(self.algorithm._daily_returns) > 10:
                returns = self.algorithm._daily_returns[-252:]  # 最近一年
                if len(returns) > 10:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    if std_return > 0:
                        return (mean_return * 252) / (std_return * np.sqrt(252))
            return 0.0
        except:
            return 0.0

class OptimizationScheduler:
    """优化调度器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.optimization_manager = QuantConnectOptimizationManager(algorithm_instance)
        
        # 调度设置
        self.optimization_enabled = True
        self.weekend_optimization = True
        self.market_close_optimization = True
        
    def initialize(self):
        """初始化调度器"""
        self.optimization_manager.initialize_optimization()
        
        # 调度周末优化
        if self.weekend_optimization:
            self.algorithm.Schedule.On(
                self.algorithm.DateRules.WeekEnd(),
                self.algorithm.TimeRules.At(2, 0),  # 周末凌晨2点
                self._scheduled_optimization
            )
        
        # 调度市场收盘后优化
        if self.market_close_optimization:
            self.algorithm.Schedule.On(
                self.algorithm.DateRules.MonthEnd(),
                self.algorithm.TimeRules.AfterMarketClose("SPY", 60),  # 收盘后1小时
                self._scheduled_optimization
            )
    
    def _scheduled_optimization(self):
        """调度的优化任务"""
        if self.optimization_enabled:
            try:
                self.algorithm.Debug("执行调度的参数优化...")
                self.optimization_manager.run_parameter_optimization()
            except Exception as e:
                self.algorithm.Debug(f"调度优化失败: {e}")
    
    def enable_optimization(self):
        """启用优化"""
        self.optimization_enabled = True
        self.algorithm.Debug("自动优化已启用")
    
    def disable_optimization(self):
        """禁用优化"""
        self.optimization_enabled = False
        self.algorithm.Debug("自动优化已禁用")
    
    def force_optimization(self):
        """强制执行优化"""
        try:
            self.algorithm.Debug("强制执行参数优化...")
            self.optimization_manager.run_parameter_optimization()
        except Exception as e:
            self.algorithm.Debug(f"强制优化失败: {e}")

# 使用示例
class OptimizationIntegrationExample:
    """优化集成示例"""
    
    @staticmethod
    def integrate_with_main_algorithm(algorithm_instance):
        """与主算法集成"""
        
        # 在Initialize中添加
        algorithm_instance.optimization_scheduler = OptimizationScheduler(algorithm_instance)
        algorithm_instance.optimization_scheduler.initialize()
        
        # 在OnData中可以检查是否需要优化
        # if algorithm_instance.optimization_scheduler.optimization_manager.should_run_optimization():
        #     algorithm_instance.optimization_scheduler.force_optimization()
        
        algorithm_instance.Debug("自动优化系统已集成到主算法")

# 配置优化参数的辅助类
class OptimizationConfigHelper:
    """优化配置辅助类"""
    
    @staticmethod
    def create_conservative_parameter_space() -> Dict:
        """创建保守的参数空间"""
        return {
            'max_drawdown': [0.05, 0.08, 0.10],
            'volatility_threshold': [0.15, 0.20, 0.25],
            'max_leverage_ratio': [1.0, 1.2, 1.3],
            'target_portfolio_size': [8, 10, 12],
            'rebalance_tolerance': [0.005, 0.010, 0.015]
        }
    
    @staticmethod
    def create_aggressive_parameter_space() -> Dict:
        """创建激进的参数空间"""
        return {
            'max_drawdown': [0.10, 0.15, 0.20],
            'volatility_threshold': [0.25, 0.30, 0.35],
            'max_leverage_ratio': [1.5, 1.8, 2.0],
            'target_portfolio_size': [12, 15, 18],
            'rebalance_tolerance': [0.003, 0.005, 0.008]
        }
    
    @staticmethod
    def create_adaptive_parameter_space(vix_level: float, market_vol: float) -> Dict:
        """创建自适应参数空间"""
        if vix_level > 30 or market_vol > 0.25:
            return OptimizationConfigHelper.create_conservative_parameter_space()
        elif vix_level < 18 and market_vol < 0.15:
            return OptimizationConfigHelper.create_aggressive_parameter_space()
        else:
            # 混合参数空间
            return {
                'max_drawdown': [0.06, 0.08, 0.10, 0.12],
                'volatility_threshold': [0.18, 0.22, 0.25, 0.28],
                'max_leverage_ratio': [1.0, 1.2, 1.5, 1.7],
                'target_portfolio_size': [8, 10, 12, 15],
                'rebalance_tolerance': [0.003, 0.005, 0.010, 0.015]
            } 