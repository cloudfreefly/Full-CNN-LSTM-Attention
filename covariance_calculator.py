# 协方差矩阵计算模块
from AlgorithmImports import *
import numpy as np
import pandas as pd

class CovarianceCalculator:
    """协方差矩阵计算器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        
    def calculate_covariance_matrix(self, symbols):
        """计算协方差矩阵"""
        try:
            self.algorithm.log_debug(f"开始计算协方差矩阵，股票数量: {len(symbols)}", log_type="portfolio")
            
            # 获取历史收益率
            returns_data = self._get_historical_returns(symbols)
            
            if returns_data is None or returns_data.empty:
                self.algorithm.log_debug("无法获取历史收益率数据", log_type="portfolio")
                return np.eye(len(symbols)) * 0.01  # 返回单位矩阵
            
            # 计算协方差矩阵
            cov_matrix = returns_data.cov().values
            
            # 清理和验证协方差矩阵
            cov_matrix = self._clean_covariance_matrix(cov_matrix)
            
            return cov_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"协方差矩阵计算失败: {str(e)}", log_type="portfolio")
            return np.eye(len(symbols)) * 0.01
    
    def _get_historical_returns(self, symbols):
        """获取历史收益率数据"""
        try:
            # 获取历史价格数据（最近63个交易日）
            history_days = 63
            
            # 获取价格历史
            try:
                history = self.algorithm.History(symbols, history_days, Resolution.Daily)
                if history.empty:
                    self.algorithm.log_debug("历史数据为空", log_type="portfolio")
                    return None
            except Exception as e:
                self.algorithm.log_debug(f"获取历史数据失败: {str(e)}", log_type="portfolio")
                return None
            
            # 将历史数据转换为DataFrame
            price_data = {}
            for symbol in symbols:
                try:
                    symbol_data = history.loc[symbol]
                    if not symbol_data.empty:
                        price_data[str(symbol)] = symbol_data['close']
                except:
                    continue
            
            if not price_data:
                self.algorithm.log_debug("没有可用的价格数据", log_type="portfolio")
                return None
            
            price_df = pd.DataFrame(price_data)
            
            # 计算日收益率
            returns_df = price_df.pct_change().dropna()
            
            # 验证返回数据
            for column in returns_df.columns:
                returns_df[column] = self._validate_returns(returns_df[column], column)
            
            return returns_df
            
        except Exception as e:
            self.algorithm.log_debug(f"获取历史收益率失败: {str(e)}", log_type="portfolio")
            return None
    
    def _validate_returns(self, returns, symbol):
        """验证和清理收益率数据"""
        # 移除异常值
        returns = returns.replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0)
        
        # 限制极端值
        returns = np.clip(returns, -0.5, 0.5)  # 限制在-50%到50%之间
        
        return returns
    
    def _clean_covariance_matrix(self, cov_matrix):
        """清理协方差矩阵"""
        try:
            # 确保矩阵是数值型
            cov_matrix = np.array(cov_matrix, dtype=float)
            
            # 替换无效值
            cov_matrix = np.nan_to_num(cov_matrix, nan=0.0001, posinf=0.1, neginf=-0.1)
            
            # 确保矩阵是正定的
            eigenvals = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvals <= 0):
                # 添加对角线正则化
                regularization = 0.001
                cov_matrix += np.eye(cov_matrix.shape[0]) * regularization
                self.algorithm.log_debug("协方差矩阵已正则化", log_type="portfolio")
            
            return cov_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"协方差矩阵清理失败: {str(e)}", log_type="portfolio")
            # 返回单位矩阵作为备用
            n = cov_matrix.shape[0] if hasattr(cov_matrix, 'shape') else len(cov_matrix)
            return np.eye(n) * 0.01 