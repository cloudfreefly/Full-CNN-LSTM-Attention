# 参数优化模块 - 自动回测和参数调优
from AlgorithmImports import *
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import itertools
import random
from scipy.optimize import minimize
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

class OptimizationMetrics:
    """优化指标计算类"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """计算夏普比率"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """计算最大回撤"""
        if len(equity_curve) == 0:
            return 0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(np.min(drawdown))
    
    @staticmethod
    def calculate_calmar_ratio(returns, max_drawdown):
        """计算卡尔玛比率"""
        annual_return = (1 + np.mean(returns)) ** 252 - 1
        return annual_return / max_drawdown if max_drawdown > 0 else 0
    
    @staticmethod
    def calculate_win_rate(returns):
        """计算胜率"""
        if len(returns) == 0:
            return 0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def calculate_profit_factor(returns):
        """计算盈利因子"""
        gains = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(losses) == 0:
            return float('inf')
        if len(gains) == 0:
            return 0
            
        return abs(np.sum(gains)) / abs(np.sum(losses))
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.02):
        """计算索丁诺比率"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_deviation = np.std(downside_returns)
        return np.sqrt(252) * np.mean(excess_returns) / downside_deviation if downside_deviation > 0 else 0

class BaseOptimizer(ABC):
    """参数优化基类"""
    
    def __init__(self, algorithm_config, optimization_config):
        self.algorithm_config = algorithm_config
        self.optimization_config = optimization_config
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
        
    @abstractmethod
    def optimize(self, parameter_space: Dict, objective_function: str = 'sharpe_ratio') -> Dict:
        """执行优化"""
        pass
    
    def evaluate_parameters(self, params: Dict) -> Dict:
        """评估单组参数"""
        try:
            # 运行回测
            backtest_results = self._run_simulated_backtest(params)
            
            # 计算评估指标
            metrics = self._calculate_metrics(backtest_results)
            
            # 记录结果
            result = {
                'parameters': params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result)
            
            return result
            
        except Exception as e:
            return {
                'parameters': params,
                'metrics': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }
    
    def _run_simulated_backtest(self, params: Dict) -> Dict:
        """运行模拟回测 - 专注于回撤期间的表现"""
        # 生成模拟的回测结果
        np.random.seed(hash(str(params)) % 2**32)
        
        # 基于参数调整性能
        days = 252 * 2  # 2年
        
        # 改进的基础收益率设置（更现实的市场表现）
        base_daily_return = 0.0008  # 提高基础日收益率
        base_annual_volatility = 0.15  # 基础年化波动率
        
        # 参数影响收益和风险的逻辑改进
        volatility_threshold = params.get('volatility_threshold', 0.20)
        max_drawdown = params.get('max_drawdown', 0.10)
        portfolio_size = params.get('target_portfolio_size', 10)
        leverage_ratio = params.get('max_leverage_ratio', 1.0)
        
        # 更合理的参数影响模型
        # 波动率阈值越低，收益越稳定但可能较低
        volatility_factor = 0.20 / volatility_threshold  # 反比关系
        # 最大回撤限制越严格，风险调整后收益越好
        drawdown_factor = 0.10 / max_drawdown  # 反比关系
        # 投资组合规模适中时表现最佳
        size_factor = 1.0 - abs(portfolio_size - 10) * 0.02
        # 适度杠杆可以提升收益
        leverage_factor = min(leverage_ratio, 1.5) * 0.8
        
        # 综合调整因子
        performance_multiplier = volatility_factor * drawdown_factor * size_factor * leverage_factor
        
        # 调整后的收益和风险参数
        adjusted_daily_return = base_daily_return * performance_multiplier
        adjusted_daily_volatility = (base_annual_volatility / np.sqrt(252)) * (2.0 - volatility_factor)
        
        # 生成收益序列
        returns = np.random.normal(adjusted_daily_return, adjusted_daily_volatility, days)
        
        # 模拟市场回撤期间的表现
        # 添加周期性的回撤事件来测试参数在困难时期的表现
        self._simulate_drawdown_periods(returns, params)
        
        # 计算权益曲线
        equity_curve = np.cumprod(1 + returns) * 100000
        
        # 分析回撤期间的表现
        drawdown_performance = self._analyze_drawdown_performance(equity_curve, returns)
        
        return {
            'returns': returns,
            'equity_curve': equity_curve,
            'final_value': equity_curve[-1],
            'performance_multiplier': performance_multiplier,
            'drawdown_performance': drawdown_performance  # 新增：回撤期间表现分析
        }
    
    def _simulate_drawdown_periods(self, returns: np.ndarray, params: Dict):
        """模拟回撤期间，测试参数的应对能力"""
        days = len(returns)
        
        # 模拟3-4次回撤事件
        num_drawdowns = np.random.randint(3, 5)
        
        for _ in range(num_drawdowns):
            # 随机选择回撤开始时间
            start_day = np.random.randint(0, days - 30)
            # 回撤持续时间：10-30天
            duration = np.random.randint(10, 31)
            end_day = min(start_day + duration, days)
            
            # 回撤严重程度：5%-20%
            drawdown_severity = np.random.uniform(0.05, 0.20)
            
            # 根据参数调整回撤期间的表现
            max_drawdown_param = params.get('max_drawdown', 0.10)
            volatility_threshold = params.get('volatility_threshold', 0.20)
            
            # 参数越保守，回撤期间表现越好
            recovery_factor = max_drawdown_param / 0.10  # 回撤限制越严格，恢复越快
            volatility_control = 0.20 / volatility_threshold  # 波动率控制越好，回撤越小
            
            # 在回撤期间调整收益率
            for day in range(start_day, end_day):
                # 基础负收益
                base_negative_return = -drawdown_severity / duration
                
                # 根据参数调整
                adjusted_return = base_negative_return * (2.0 - recovery_factor) * (2.0 - volatility_control)
                
                # 添加一些随机性
                noise = np.random.normal(0, 0.002)
                returns[day] = adjusted_return + noise
            
            # 回撤后的恢复期
            recovery_days = min(duration // 2, days - end_day)
            for day in range(end_day, end_day + recovery_days):
                # 恢复期的正收益，参数越好恢复越快
                recovery_return = (drawdown_severity / recovery_days) * recovery_factor
                returns[day] = max(returns[day], recovery_return * 0.5)
    
    def _analyze_drawdown_performance(self, equity_curve: np.ndarray, returns: np.ndarray) -> Dict:
        """分析回撤期间的表现"""
        # 识别回撤期间
        peak = equity_curve[0]
        drawdown_periods = []
        current_drawdown_start = None
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                # 新高点，结束当前回撤期
                if current_drawdown_start is not None:
                    drawdown_periods.append({
                        'start': current_drawdown_start,
                        'end': i,
                        'peak': peak,
                        'trough': min(equity_curve[current_drawdown_start:i+1]),
                        'recovery_days': i - current_drawdown_start
                    })
                    current_drawdown_start = None
                peak = value
            else:
                # 下跌，开始或继续回撤
                if current_drawdown_start is None:
                    current_drawdown_start = i
        
        # 计算回撤期间的关键指标
        if drawdown_periods:
            avg_recovery_days = np.mean([dd['recovery_days'] for dd in drawdown_periods])
            max_drawdown_depth = max([(dd['peak'] - dd['trough']) / dd['peak'] for dd in drawdown_periods])
            drawdown_frequency = len(drawdown_periods) / (len(equity_curve) / 252)  # 年化频率
            
            # 计算回撤期间的胜率
            drawdown_win_rates = []
            for dd in drawdown_periods:
                period_returns = returns[dd['start']:dd['end']]
                if len(period_returns) > 0:
                    win_rate = sum(1 for r in period_returns if r > 0) / len(period_returns)
                    drawdown_win_rates.append(win_rate)
            
            avg_drawdown_win_rate = np.mean(drawdown_win_rates) if drawdown_win_rates else 0
            
            return {
                'num_drawdowns': len(drawdown_periods),
                'avg_recovery_days': avg_recovery_days,
                'max_drawdown_depth': max_drawdown_depth,
                'drawdown_frequency': drawdown_frequency,
                'avg_drawdown_win_rate': avg_drawdown_win_rate,
                'drawdown_resilience_score': self._calculate_resilience_score(drawdown_periods)
            }
        else:
            return {
                'num_drawdowns': 0,
                'avg_recovery_days': 0,
                'max_drawdown_depth': 0,
                'drawdown_frequency': 0,
                'avg_drawdown_win_rate': 0,
                'drawdown_resilience_score': 1.0
            }
    
    def _calculate_resilience_score(self, drawdown_periods: List[Dict]) -> float:
        """计算回撤恢复力评分"""
        if not drawdown_periods:
            return 1.0
        
        scores = []
        for dd in drawdown_periods:
            # 恢复速度得分（恢复越快得分越高）
            recovery_score = max(0, 1.0 - dd['recovery_days'] / 60)  # 60天内恢复为满分
            
            # 回撤深度得分（回撤越小得分越高）
            depth = (dd['peak'] - dd['trough']) / dd['peak']
            depth_score = max(0, 1.0 - depth / 0.20)  # 20%以内回撤为满分
            
            # 综合得分
            combined_score = (recovery_score + depth_score) / 2
            scores.append(combined_score)
        
        return np.mean(scores)
    
    def _calculate_metrics(self, backtest_results: Dict) -> Dict:
        """计算性能指标 - 重点关注回撤期间表现"""
        returns = backtest_results['returns']
        equity_curve = backtest_results['equity_curve']
        drawdown_performance = backtest_results.get('drawdown_performance', {})
        
        # 基础性能指标
        basic_metrics = {
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) * 100,
            'annual_return': ((equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1) * 100,
            'sharpe_ratio': OptimizationMetrics.calculate_sharpe_ratio(returns),
            'max_drawdown': OptimizationMetrics.calculate_max_drawdown(equity_curve) * 100,
            'calmar_ratio': OptimizationMetrics.calculate_calmar_ratio(returns, OptimizationMetrics.calculate_max_drawdown(equity_curve)),
            'win_rate': OptimizationMetrics.calculate_win_rate(returns) * 100,
            'profit_factor': OptimizationMetrics.calculate_profit_factor(returns),
            'sortino_ratio': OptimizationMetrics.calculate_sortino_ratio(returns),
            'volatility': np.std(returns) * np.sqrt(252) * 100
        }
        
        # 回撤期间专门指标
        drawdown_metrics = {
            'drawdown_recovery_days': drawdown_performance.get('avg_recovery_days', 30),
            'drawdown_resilience_score': drawdown_performance.get('drawdown_resilience_score', 0.5),
            'drawdown_win_rate': drawdown_performance.get('avg_drawdown_win_rate', 0.3) * 100,
            'drawdown_frequency': drawdown_performance.get('drawdown_frequency', 2.0),
            'max_drawdown_depth': drawdown_performance.get('max_drawdown_depth', 0.15) * 100,
            'num_drawdowns': drawdown_performance.get('num_drawdowns', 4)
        }
        
        # 计算综合回撤导向评分
        drawdown_focused_score = self._calculate_drawdown_focused_score(basic_metrics, drawdown_metrics)
        
        # 合并所有指标
        all_metrics = {**basic_metrics, **drawdown_metrics}
        all_metrics['drawdown_focused_score'] = drawdown_focused_score
        
        return all_metrics
    
    def _calculate_drawdown_focused_score(self, basic_metrics: Dict, drawdown_metrics: Dict) -> float:
        """计算专注于回撤期间表现的综合评分"""
        
        # 基础性能权重（30%）
        sharpe_ratio = max(basic_metrics.get('sharpe_ratio', 0), -2)  # 限制最低值
        calmar_ratio = max(basic_metrics.get('calmar_ratio', 0), -1)
        basic_score = (sharpe_ratio * 0.6 + calmar_ratio * 0.4) * 0.3
        
        # 回撤控制权重（40%）
        max_drawdown = basic_metrics.get('max_drawdown', 15)
        drawdown_control_score = max(0, (15 - max_drawdown) / 15) * 0.4  # 回撤越小得分越高
        
        # 回撤恢复能力权重（30%）
        resilience_score = drawdown_metrics.get('drawdown_resilience_score', 0.5)
        recovery_days = drawdown_metrics.get('drawdown_recovery_days', 30)
        recovery_speed_score = max(0, (60 - recovery_days) / 60)  # 恢复越快得分越高
        drawdown_win_rate = drawdown_metrics.get('drawdown_win_rate', 30) / 100
        
        recovery_score = (resilience_score * 0.4 + recovery_speed_score * 0.3 + drawdown_win_rate * 0.3) * 0.3
        
        # 综合评分
        total_score = basic_score + drawdown_control_score + recovery_score
        
        # 确保评分在合理范围内
        return max(-2.0, min(5.0, total_score))

class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, parameter_space: Dict, objective_function: str = 'sharpe_ratio') -> Dict:
        """网格搜索优化"""
        
        # 生成参数组合
        param_combinations = self._generate_grid_combinations(parameter_space)
        
        # 测试每种组合
        for i, params in enumerate(param_combinations):
            result = self.evaluate_parameters(params)
            
            if 'error' not in result['metrics']:
                score = result['metrics'].get(objective_function, float('-inf'))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _generate_grid_combinations(self, parameter_space: Dict) -> List[Dict]:
        """生成网格参数组合"""
        keys = list(parameter_space.keys())
        values = [parameter_space[key] for key in keys]
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations

class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def optimize(self, parameter_space: Dict, objective_function: str = 'sharpe_ratio', n_iterations: int = 100) -> Dict:
        """随机搜索优化"""
        
        for i in range(n_iterations):
            # 随机生成参数
            params = self._generate_random_params(parameter_space)
            
            result = self.evaluate_parameters(params)
            
            if 'error' not in result['metrics']:
                score = result['metrics'].get(objective_function, float('-inf'))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _generate_random_params(self, parameter_space: Dict) -> Dict:
        """生成随机参数"""
        params = {}
        for key, values in parameter_space.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                # 假设是范围 (min, max)
                if isinstance(values[0], int):
                    params[key] = random.randint(values[0], values[1])
                else:
                    params[key] = random.uniform(values[0], values[1])
        return params

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器（简化版）"""
    
    def optimize(self, parameter_space: Dict, objective_function: str = 'sharpe_ratio', n_iterations: int = 50) -> Dict:
        """贝叶斯优化"""
        print(f"开始贝叶斯优化，迭代次数: {n_iterations}")
        
        # 初始随机采样
        n_initial = min(10, n_iterations // 2)
        for i in range(n_initial):
            params = self._generate_random_params(parameter_space)
            result = self.evaluate_parameters(params)
            
            if 'error' not in result['metrics']:
                score = result['metrics'].get(objective_function, float('-inf'))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
        
        # 贝叶斯优化迭代
        for i in range(n_initial, n_iterations):
            # 简化的获取函数：选择最有希望的参数
            params = self._acquisition_function(parameter_space, objective_function)
            print(f"贝叶斯优化 {i+1}/{n_iterations}: {params}")
            
            result = self.evaluate_parameters(params)
            
            if 'error' not in result['metrics']:
                score = result['metrics'].get(objective_function, float('-inf'))
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"发现更好的参数组合! {objective_function}: {score:.4f}")
        
        print(f"贝叶斯优化完成! 最佳参数: {self.best_params}")
        print(f"最佳{objective_function}: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results
        }
    
    def _acquisition_function(self, parameter_space: Dict, objective_function: str) -> Dict:
        """简化的获取函数"""
        # 在实际实现中，这里应该使用高斯过程和获取函数
        # 现在使用简化版本：基于历史结果的智能采样
        
        if len(self.results) < 3:
            return self._generate_random_params(parameter_space)
        
        # 分析最佳参数的特征
        best_results = sorted(self.results, 
                            key=lambda x: x['metrics'].get(objective_function, float('-inf')), 
                            reverse=True)[:3]
        
        # 在最佳参数附近采样
        best_params = best_results[0]['parameters']
        new_params = {}
        
        for key, values in parameter_space.items():
            if key in best_params:
                if isinstance(values, list):
                    # 在最佳值附近选择
                    current_val = best_params[key]
                    if current_val in values:
                        idx = values.index(current_val)
                        # 选择邻近值
                        candidates = []
                        if idx > 0:
                            candidates.append(values[idx - 1])
                        candidates.append(current_val)
                        if idx < len(values) - 1:
                            candidates.append(values[idx + 1])
                        new_params[key] = random.choice(candidates)
                    else:
                        new_params[key] = random.choice(values)
                elif isinstance(values, tuple) and len(values) == 2:
                    # 在最佳值附近的范围内采样
                    current_val = best_params[key]
                    range_size = (values[1] - values[0]) * 0.2  # 20%的范围
                    min_val = max(values[0], current_val - range_size)
                    max_val = min(values[1], current_val + range_size)
                    
                    if isinstance(values[0], int):
                        new_params[key] = random.randint(int(min_val), int(max_val))
                    else:
                        new_params[key] = random.uniform(min_val, max_val)
            else:
                new_params[key] = self._generate_random_params({key: values})[key]
        
        return new_params
    
    def _generate_random_params(self, parameter_space: Dict) -> Dict:
        """生成随机参数"""
        params = {}
        for key, values in parameter_space.items():
            if isinstance(values, list):
                params[key] = random.choice(values)
            elif isinstance(values, tuple) and len(values) == 2:
                if isinstance(values[0], int):
                    params[key] = random.randint(values[0], values[1])
                else:
                    params[key] = random.uniform(values[0], values[1])
        return params

class ParameterOptimizationManager:
    """参数优化管理器"""
    
    def __init__(self, algorithm_config):
        self.algorithm_config = algorithm_config
        self.optimization_results = {}
        
    def run_optimization(self, parameter_space: Dict, 
                        optimization_method: str = 'grid_search',
                        objective_function: str = 'sharpe_ratio',
                        **kwargs) -> Dict:
        """运行参数优化"""
        
        print(f"开始参数优化...")
        print(f"优化方法: {optimization_method}")
        print(f"目标函数: {objective_function}")
        print(f"参数空间: {parameter_space}")
        
        # 选择优化器
        if optimization_method == 'grid_search':
            optimizer = GridSearchOptimizer(self.algorithm_config, kwargs)
        elif optimization_method == 'random_search':
            optimizer = RandomSearchOptimizer(self.algorithm_config, kwargs)
        elif optimization_method == 'bayesian':
            optimizer = BayesianOptimizer(self.algorithm_config, kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {optimization_method}")
        
        # 执行优化
        start_time = time.time()
        results = optimizer.optimize(parameter_space, objective_function, **kwargs)
        end_time = time.time()
        
        # 添加优化统计信息
        results['optimization_time'] = end_time - start_time
        results['optimization_method'] = optimization_method
        results['objective_function'] = objective_function
        results['parameter_space'] = parameter_space
        
        # 保存结果
        self.optimization_results[f"{optimization_method}_{objective_function}"] = results
        
        print(f"优化完成! 用时: {end_time - start_time:.2f}秒")
        
        return results
    
    def compare_optimization_methods(self, parameter_space: Dict, 
                                   objective_function: str = 'sharpe_ratio') -> Dict:
        """比较不同优化方法"""
        
        methods = ['random_search', 'bayesian']  # 网格搜索可能太慢
        comparison_results = {}
        
        for method in methods:
            print(f"\n{'='*50}")
            print(f"测试优化方法: {method}")
            print(f"{'='*50}")
            
            try:
                if method == 'random_search':
                    result = self.run_optimization(parameter_space, method, objective_function, n_iterations=50)
                elif method == 'bayesian':
                    result = self.run_optimization(parameter_space, method, objective_function, n_iterations=30)
                else:
                    result = self.run_optimization(parameter_space, method, objective_function)
                
                comparison_results[method] = result
                
            except Exception as e:
                print(f"方法 {method} 失败: {e}")
                comparison_results[method] = {'error': str(e)}
        
        # 生成比较报告
        report = self._generate_comparison_report(comparison_results, objective_function)
        
        return {
            'comparison_results': comparison_results,
            'report': report
        }
    
    def _generate_comparison_report(self, results: Dict, objective_function: str) -> str:
        """生成比较报告"""
        report = "\n" + "="*60 + "\n"
        report += "参数优化方法比较报告\n"
        report += "="*60 + "\n\n"
        
        for method, result in results.items():
            if 'error' in result:
                report += f"{method}: 失败 - {result['error']}\n"
            else:
                best_score = result.get('best_score', 'N/A')
                optimization_time = result.get('optimization_time', 'N/A')
                n_evaluations = len(result.get('all_results', []))
                
                report += f"{method}:\n"
                report += f"  最佳{objective_function}: {best_score:.4f}\n"
                report += f"  优化时间: {optimization_time:.2f}秒\n"
                report += f"  评估次数: {n_evaluations}\n"
                report += f"  最佳参数: {result.get('best_params', {})}\n\n"
        
        # 推荐最佳方法
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_method = max(valid_results.keys(), 
                            key=lambda x: valid_results[x].get('best_score', float('-inf')))
            report += f"推荐方法: {best_method}\n"
            report += f"推荐原因: 获得了最佳的{objective_function}值\n"
        
        return report
    
    def save_results(self, filename: str = None):
        """保存优化结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # 处理numpy类型的序列化
                serializable_results = self._make_serializable(self.optimization_results)
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存结果失败: {e}")
    
    def _make_serializable(self, obj):
        """使对象可序列化"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj

# 示例优化配置
class OptimizationConfig:
    """优化配置示例"""
    
    # 风险管理参数空间
    RISK_PARAMETER_SPACE = {
        'max_drawdown': [0.05, 0.08, 0.10, 0.12, 0.15],
        'stop_loss_threshold': [-0.03, -0.05, -0.08, -0.10],
        'volatility_threshold': [0.15, 0.20, 0.25, 0.30],
        'vix_extreme_level': [25, 30, 35, 40]
    }
    
    # 投资组合管理参数空间
    PORTFOLIO_PARAMETER_SPACE = {
        'max_weight': [0.08, 0.10, 0.12, 0.15, 0.20],
        'min_weight': [0.03, 0.05, 0.08, 0.10],
        'target_portfolio_size': [8, 10, 12, 15],
        'rebalance_tolerance': [0.003, 0.005, 0.010, 0.020]
    }
    
    # 杠杆设置参数空间
    LEVERAGE_PARAMETER_SPACE = {
        'max_leverage_ratio': [1.0, 1.2, 1.5, 1.8, 2.0],
        'low_risk_leverage_ratio': [1.2, 1.5, 1.8],
        'medium_risk_leverage_ratio': [1.0, 1.2, 1.5],
        'high_risk_leverage_ratio': [0.6, 0.8, 1.0]
    }
    
    @classmethod
    def get_combined_parameter_space(cls) -> Dict:
        """获取组合参数空间"""
        combined = {}
        combined.update(cls.RISK_PARAMETER_SPACE)
        combined.update(cls.PORTFOLIO_PARAMETER_SPACE)
        combined.update(cls.LEVERAGE_PARAMETER_SPACE)
        return combined
    
    @classmethod
    def create_custom_objective(cls, metrics: Dict) -> float:
        """创建自定义目标函数"""
        score = 0
        for metric, weight in cls.OBJECTIVE_WEIGHTS.items():
            if metric in metrics:
                if metric == 'max_drawdown':
                    # 最大回撤越小越好
                    score += weight * (1 / (1 + metrics[metric] / 100))
                else:
                    score += weight * metrics[metric]
        return score

# 使用示例
def run_parameter_optimization_example():
    """运行参数优化示例"""
    from config import AlgorithmConfig
    
    # 创建优化管理器
    optimizer_manager = ParameterOptimizationManager(AlgorithmConfig)
    
    # 定义参数空间
    parameter_space = {
        'max_drawdown': [0.05, 0.08, 0.10, 0.12],
        'volatility_threshold': [0.15, 0.20, 0.25, 0.30],
        'max_weight': [0.08, 0.10, 0.12, 0.15],
        'target_portfolio_size': [8, 10, 12, 15]
    }
    
    # 运行单一优化方法
    print("运行贝叶斯优化...")
    result = optimizer_manager.run_optimization(
        parameter_space=parameter_space,
        optimization_method='bayesian',
        objective_function='sharpe_ratio',
        n_iterations=20
    )
    
    print("\n最佳参数:")
    print(result['best_params'])
    print(f"\n最佳夏普比率: {result['best_score']:.4f}")
    
    # 比较多种优化方法
    print("\n\n比较不同优化方法...")
    comparison = optimizer_manager.compare_optimization_methods(
        parameter_space=parameter_space,
        objective_function='sharpe_ratio'
    )
    
    print(comparison['report'])
    
    # 保存结果
    optimizer_manager.save_results()

if __name__ == "__main__":
    run_parameter_optimization_example() 