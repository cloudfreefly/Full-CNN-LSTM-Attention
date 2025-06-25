# 投资组合优化策略模块
from AlgorithmImports import *
import numpy as np
from scipy.optimize import minimize

class OptimizationStrategies:
    """投资组合优化策略集合"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
    
    def mean_variance_optimization(self, expected_returns, covariance_matrix):
        """均值方差优化"""
        try:
            n = len(expected_returns)
            
            # 目标函数：最大化夏普比率
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # 避免除零错误
                if portfolio_volatility == 0:
                    return -portfolio_return
                
                sharpe_ratio = portfolio_return / portfolio_volatility
                return -sharpe_ratio  # 最小化负夏普比率
            
            # 约束条件
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # 权重和为1
            )
            
            # 边界条件 - 放宽以允许更大的权重差异
            bounds = tuple((0.02, 0.25) for _ in range(n))  # 每只股票权重2-25%，强制最小权重避免零权重
            
            # 初始猜测
            x0 = np.ones(n) / n
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                self.algorithm.log_debug(f"均值方差优化失败: {result.message}")
                return None
                
        except Exception as e:
            self.algorithm.log_debug(f"均值方差优化错误: {str(e)}")
            return None
    
    def risk_parity_optimization(self, expected_returns, covariance_matrix):
        """风险平价优化"""
        try:
            n = len(expected_returns)
            
            # 风险平价目标函数
            def objective(weights):
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                marginal_contrib = np.dot(covariance_matrix, weights)
                contrib = weights * marginal_contrib
                
                # 计算风险贡献的方差
                target_contrib = portfolio_variance / n
                risk_contrib_diff = contrib - target_contrib
                return np.sum(risk_contrib_diff ** 2)
            
            # 约束条件
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            )
            
            # 边界条件
            bounds = tuple((0.01, 0.5) for _ in range(n))
            
            # 初始猜测
            x0 = np.ones(n) / n
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                self.algorithm.log_debug(f"风险平价优化失败: {result.message}")
                return None
                
        except Exception as e:
            self.algorithm.log_debug(f"风险平价优化错误: {str(e)}")
            return None
    
    def maximum_diversification_optimization(self, expected_returns, covariance_matrix):
        """最大分散化优化"""
        try:
            n = len(expected_returns)
            
            # 最大分散化目标函数
            def objective(weights):
                # 计算投资组合方差
                portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                # 计算加权平均波动率
                individual_volatilities = np.sqrt(np.diag(covariance_matrix))
                weighted_avg_volatility = np.dot(weights, individual_volatilities)
                
                # 分散化比率 = 加权平均波动率 / 投资组合波动率
                if portfolio_volatility == 0:
                    return -weighted_avg_volatility
                
                diversification_ratio = weighted_avg_volatility / portfolio_volatility
                return -diversification_ratio  # 最小化负分散化比率
            
            # 约束条件
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            )
            
            # 边界条件
            bounds = tuple((0, 0.5) for _ in range(n))
            
            # 初始猜测
            x0 = np.ones(n) / n
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                self.algorithm.log_debug(f"最大分散化优化失败: {result.message}")
                return None
                
        except Exception as e:
            self.algorithm.log_debug(f"最大分散化优化错误: {str(e)}")
            return None
    
    def minimum_variance_optimization(self, expected_returns, covariance_matrix):
        """最小方差优化"""
        try:
            n = len(expected_returns)
            
            # 最小方差目标函数
            def objective(weights):
                return np.dot(weights.T, np.dot(covariance_matrix, weights))
            
            # 约束条件
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            )
            
            # 边界条件
            bounds = tuple((0, 0.4) for _ in range(n))
            
            # 初始猜测
            x0 = np.ones(n) / n
            
            # 优化
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                self.algorithm.log_debug(f"最小方差优化失败: {result.message}")
                return None
                
        except Exception as e:
            self.algorithm.log_debug(f"最小方差优化错误: {str(e)}")
            return None
    
    def evaluate_portfolio_quality(self, weights, expected_returns, covariance_matrix):
        """评估投资组合质量"""
        try:
            if weights is None or len(weights) == 0:
                return -np.inf
            
            # 计算预期收益
            portfolio_return = np.dot(weights, expected_returns)
            
            # 计算风险
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # 计算夏普比率
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # 计算分散化程度
            concentration = np.sum(weights ** 2)  # Herfindahl指数
            diversification_score = 1 - concentration
            
            # 综合评分
            quality_score = sharpe_ratio * 0.6 + diversification_score * 0.4
            
            return quality_score
            
        except Exception as e:
            self.algorithm.log_debug(f"投资组合质量评估错误: {str(e)}")
            return -np.inf 