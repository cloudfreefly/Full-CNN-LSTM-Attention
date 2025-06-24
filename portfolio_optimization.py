# 投资组合优化模块
from AlgorithmImports import *
import numpy as np
import pandas as pd

from config import AlgorithmConfig
from risk_management import ConcentrationLimiter
from covariance_calculator import CovarianceCalculator
from smart_rebalancer import SmartRebalancer
from optimization_strategies import OptimizationStrategies

class PortfolioOptimizer:
    """投资组合优化器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.concentration_limiter = ConcentrationLimiter(algorithm_instance)
        
        # 初始化子模块
        self.covariance_calculator = CovarianceCalculator(algorithm_instance)
        self.smart_rebalancer = SmartRebalancer(algorithm_instance)
        self.optimization_strategies = OptimizationStrategies(algorithm_instance)
        
        # 记录决策因素
        self._last_decision_factors = {}
        self._last_equity_ratio = 0.0  # 恢复：初始化为0，等待动态计算
        self._last_panic_score = 0
    
    def update_portfolio_parameters(self):
        """更新投资组合参数配置"""
        try:
            # 重新加载配置
            self.config = AlgorithmConfig()
            
            # 通知子模块更新配置
            if hasattr(self.concentration_limiter, 'config'):
                self.concentration_limiter.config = self.config
            if hasattr(self.covariance_calculator, 'config'):
                self.covariance_calculator.config = self.config
            if hasattr(self.smart_rebalancer, 'config'):
                self.smart_rebalancer.config = self.config
            if hasattr(self.optimization_strategies, 'config'):
                self.optimization_strategies.config = self.config
            
            self.algorithm.log_debug("投资组合优化参数已更新", log_type="portfolio")
            
        except Exception as e:
            self.algorithm.log_debug(f"更新投资组合参数失败: {e}", log_type="portfolio")
        
    def optimize_portfolio(self, expected_returns, 
                         covariance_matrix, 
                         symbols):
        """优化投资组合权重"""
        try:
            # 检查是否禁用复杂优化，直接使用等权重
            if self.config.PORTFOLIO_CONFIG.get('disable_optimization', False):
                self.algorithm.log_debug(f"直接使用等权重策略，跳过复杂优化", log_type="portfolio")
                target_size = self.config.PORTFOLIO_CONFIG['target_portfolio_size']
                n_symbols = min(target_size, len(symbols))
                
                # 选择前N只股票（可以根据预期收益排序）
                if isinstance(expected_returns, (list, np.ndarray)) and len(expected_returns) == len(symbols):
                    # 按预期收益排序选择股票
                    symbol_returns = list(zip(symbols, expected_returns))
                    sorted_symbol_returns = sorted(symbol_returns, key=lambda x: x[1], reverse=True)
                    selected_symbols = [s for s, r in sorted_symbol_returns[:n_symbols]]
                else:
                    # 选择前n_symbols只
                    selected_symbols = symbols[:n_symbols]
                
                # 等权重分配
                equal_weights = np.ones(n_symbols) / n_symbols
                
                self.algorithm.log_debug(f"等权重组合: {n_symbols}只股票，每只{1/n_symbols:.1%}", log_type="portfolio")
                for symbol in selected_symbols:
                    self.algorithm.log_debug(f"  {symbol}: {1/n_symbols:.1%}", log_type="portfolio")
                
                return equal_weights, selected_symbols
            
            # 简化日志
            n = len(symbols)
            self._all_negative_scenario = False
            
            if n == 0:
                return np.array([]), []
            
            if n == 1:
                return np.array([1.0]), symbols
            
            # 验证输入数据
            try:
                validation_result = self._validate_optimization_inputs(expected_returns, covariance_matrix, n)
                if not validation_result:
                    equal_weights = np.ones(n) / n
                    return equal_weights, symbols
            except Exception as validation_error:
                return np.ones(n) / n, symbols
            
            # 应用防御策略：处理负期望收益的标的
            self.algorithm.log_debug("Step 2: Applying defensive strategy...", log_type="portfolio")
            try:
                filtered_returns, filtered_cov, filtered_symbols = self._apply_defensive_strategy(
                    expected_returns, covariance_matrix, symbols
                )
                self.algorithm.log_debug("Defensive strategy result:", log_type="portfolio")
                self.algorithm.log_debug(f"  Original symbols: {len(symbols)}", log_type="portfolio")
                self.algorithm.log_debug(f"  Filtered symbols: {len(filtered_symbols)}", log_type="portfolio")
                self.algorithm.log_debug(f"  Filtered returns shape: {filtered_returns.shape if hasattr(filtered_returns, 'shape') else len(filtered_returns)}", log_type="portfolio")
                
            except Exception as defensive_error:
                self.algorithm.log_debug(f"Error in defensive strategy: {defensive_error}", log_type="portfolio")
                # 使用原始数据
                filtered_returns, filtered_cov, filtered_symbols = expected_returns, covariance_matrix, symbols
            
            # 如果所有标的都被过滤掉，返回空组合
            if len(filtered_symbols) == 0:
                self.algorithm.log_debug("⚠️  All symbols filtered out due to negative returns - holding cash", log_type="portfolio")
                return np.array([]), []
            
            # 尝试多种优化方法（使用过滤后的数据）
            self.algorithm.log_debug("Step 3: Trying multiple optimization methods...", log_type="portfolio")
            optimization_methods = [
                ("Mean-Variance", self.optimization_strategies.mean_variance_optimization),
                ("Risk-Parity", self.optimization_strategies.risk_parity_optimization),
                ("Max-Diversification", self.optimization_strategies.maximum_diversification_optimization)
            ]
            
            best_weights = None
            best_score = -np.inf
            method_results = []
            
            for method_name, method_func in optimization_methods:
                try:
                    self.algorithm.log_debug(f"Trying {method_name} optimization...", log_type="portfolio")
                    weights = method_func(filtered_returns, filtered_cov)
                    
                    if weights is not None:
                        self.algorithm.log_debug(f"{method_name} returned weights: {weights}", log_type="portfolio")
                        score = self.optimization_strategies.evaluate_portfolio_quality(weights, filtered_returns, filtered_cov)
                        method_results.append(f"{method_name}: {score:.4f}")
                        self.algorithm.log_debug(f"{method_name} score: {score:.4f}", log_type="portfolio")
                        
                        if score > best_score:
                            best_weights = weights
                            best_score = score
                            self.algorithm.log_debug(f"✓ {method_name} optimization succeeded (new best score: {score:.4f})", log_type="portfolio")
                        else:
                            self.algorithm.log_debug(f"○ {method_name} optimization completed (score: {score:.4f})", log_type="portfolio")
                    else:
                        method_results.append(f"{method_name}: Failed")
                        self.algorithm.log_debug(f"✗ {method_name} optimization returned None", log_type="portfolio")
                        
                except Exception as method_error:
                    method_results.append(f"{method_name}: Error")
                    self.algorithm.log_debug(f"✗ {method_name} optimization failed: {method_error}", log_type="portfolio")
                    import traceback
                    self.algorithm.log_debug(f"{method_name} error traceback: {traceback.format_exc()}", log_type="portfolio")
                    continue
            
            # 记录所有方法的结果
            self.algorithm.log_debug(f"Optimization results: {', '.join(method_results)}", log_type="portfolio")
            
            if best_weights is None:
                self.algorithm.log_debug("All optimization methods failed, using equal weights", log_type="portfolio")
                best_weights = np.ones(len(filtered_symbols)) / len(filtered_symbols)
                best_score = 0.0
            
            # 应用约束和限制
            self.algorithm.log_debug("Step 4: Applying constraints...", log_type="portfolio")
            try:
                constrained_weights = self._apply_constraints(best_weights, filtered_symbols)
                self.algorithm.log_debug("Constraints applied:", log_type="portfolio")
                self.algorithm.log_debug(f"  Before: {best_weights}", log_type="portfolio")
                self.algorithm.log_debug(f"  After: {constrained_weights}", log_type="portfolio")
            except Exception as constraint_error:
                self.algorithm.log_debug(f"Error applying constraints: {constraint_error}", log_type="portfolio")
                constrained_weights = best_weights
            
            # 最终筛选
            self.algorithm.log_debug("Step 5: Final screening...", log_type="portfolio")
            try:
                final_weights, final_symbols = self._final_screening(constrained_weights, filtered_symbols)
                self.algorithm.log_debug("Final screening result:", log_type="portfolio")
                self.algorithm.log_debug(f"  Weights: {final_weights}", log_type="portfolio")
                self.algorithm.log_debug(f"  Symbols: {final_symbols}", log_type="portfolio")
            except Exception as screening_error:
                self.algorithm.log_debug(f"Error in final screening: {screening_error}", log_type="portfolio")
                final_weights, final_symbols = constrained_weights, filtered_symbols
            
            self.algorithm.log_debug(f"=== Portfolio optimization completed ===", log_type="portfolio")
            self.algorithm.log_debug("Results summary:", log_type="portfolio")
            self.algorithm.log_debug(f"  Best method score: {best_score:.4f}", log_type="portfolio")
            self.algorithm.log_debug(f"  Final symbols count: {len(final_symbols)}", log_type="portfolio")
            self.algorithm.log_debug(f"  Final weights sum: {np.sum(final_weights) if final_weights is not None and len(final_weights) > 0 else 'N/A'}", log_type="portfolio")
            
            self.algorithm.log_debug(f"[优化器] 收到预期收益: {expected_returns}", log_type="portfolio")
            self.algorithm.log_debug(f"[优化器] 收到协方差矩阵: {covariance_matrix}", log_type="portfolio")
            self.algorithm.log_debug(f"[优化器] 收到可用股票列表: {symbols}", log_type="portfolio")
            self.algorithm.log_debug(f"[优化器] 输出目标权重: {final_weights}", log_type="portfolio")
            
            # 修复NumPy数组布尔值判断问题
            weights_empty = (final_weights is None or 
                           (hasattr(final_weights, '__len__') and len(final_weights) == 0) or
                           (hasattr(final_weights, 'size') and final_weights.size == 0))
            weights_all_zero = False
            if not weights_empty:
                try:
                    weights_all_zero = np.all(np.array(final_weights) == 0)
                except:
                    weights_all_zero = all(w == 0 for w in final_weights)
            
            if weights_empty or weights_all_zero:
                self.algorithm.log_debug('[优化器] 优化失败，全部权重为0或无权重', log_type="portfolio")
            if hasattr(self, '_last_decision_factors'):
                self.algorithm.log_debug(f"[优化器] 决策因子: {self._last_decision_factors}", log_type="portfolio")
            
            return final_weights, final_symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"CRITICAL ERROR in portfolio optimization: {str(e)}", log_type="portfolio")
            self.algorithm.log_debug(f"Portfolio optimization error type: {type(e).__name__}", log_type="portfolio")
            import traceback
            self.algorithm.log_debug(f"Portfolio optimization error traceback: {traceback.format_exc()}", log_type="portfolio")
            
            n = len(symbols)
            if n > 0:
                equal_weights = np.ones(n) / n
                self.algorithm.log_debug(f"Returning emergency equal weights: {equal_weights}", log_type="portfolio")
                return equal_weights, symbols
            else:
                self.algorithm.log_debug("Returning empty arrays due to critical error", log_type="portfolio")
                return np.array([]), []
    
    def _validate_optimization_inputs(self, expected_returns, 
                                    covariance_matrix, n):
        """验证优化输入数据"""
        if len(expected_returns) != n or covariance_matrix.shape != (n, n):
            return False
        
        if np.any(np.isnan(expected_returns)) or np.any(np.isinf(expected_returns)):
            return False
        
        if np.any(np.isnan(covariance_matrix)) or np.any(np.isinf(covariance_matrix)):
            return False
        
        try:
            eigenvals = np.linalg.eigvals(covariance_matrix)
            if np.any(eigenvals <= 0):
                return False
        except:
            return False
        
        return True
    
    def _apply_defensive_strategy(self, expected_returns, covariance_matrix, symbols):
        """应用防御策略：处理负期望收益的标的"""
        try:
            # 检查输入数据的一致性
            if len(expected_returns) != len(symbols) or covariance_matrix.shape[0] != len(symbols):
                self.algorithm.log_debug(f"Input size mismatch: returns={len(expected_returns)}, symbols={len(symbols)}, cov={covariance_matrix.shape}", log_type="portfolio")
                return expected_returns, covariance_matrix, symbols
            
            # 计算市场状态指标
            avg_expected_return = np.mean(expected_returns)
            negative_ratio = np.sum(expected_returns < 0) / len(expected_returns)
            volatility_estimate = np.sqrt(np.mean(np.diag(covariance_matrix))) if covariance_matrix.shape[0] > 0 else 0
            
            self.algorithm.log_debug("Market condition analysis:", log_type="portfolio")
            self.algorithm.log_debug(f"  Average expected return: {avg_expected_return:.4f}", log_type="portfolio")
            self.algorithm.log_debug(f"  Negative return ratio: {negative_ratio:.2%}", log_type="portfolio")
            self.algorithm.log_debug(f"  Estimated volatility: {volatility_estimate:.4f}", log_type="portfolio")
            
            # 动态判断是否需要防御策略（更宽松的触发条件）
            should_apply_defense = False
            defense_reason = ""
            
            # 条件1：所有收益都为负（原有逻辑）
            if np.all(expected_returns < 0):
                should_apply_defense = True
                defense_reason = "全负收益"
            
            # 条件2：负收益标的比例过高 - 提高阈值
            elif negative_ratio >= 0.8:  # 80%以上标的为负收益才触发
                should_apply_defense = True
                defense_reason = f"负收益比例过高({negative_ratio:.1%})"
            
            # 条件3：平均预期收益为负且幅度很大 - 降低敏感度
            elif avg_expected_return < -0.03:  # 平均预期收益低于-3%才触发
                should_apply_defense = True
                defense_reason = f"平均收益过低({avg_expected_return:.2%})"
            
            # 条件4：市场波动率极高 - 提高阈值
            elif volatility_estimate > 0.35:  # 年化波动率35%以上才触发
                should_apply_defense = True
                defense_reason = f"波动率极高({volatility_estimate:.2%})"
            
            # 条件5：基于算法实例的额外风险指标
            try:
                # 检查回撤指标
                if hasattr(self.algorithm, 'drawdown_monitor'):
                    current_value = self.algorithm.Portfolio.TotalPortfolioValue
                    drawdown_metrics = self.algorithm.drawdown_monitor.update_portfolio_value(current_value)
                    current_drawdown = drawdown_metrics.get('current_drawdown', 0)
                    
                    if current_drawdown > 0.05:  # 回撤超过5%就触发防御（从15%大幅降低）
                        should_apply_defense = True
                        defense_reason += f" + 回撤过大({current_drawdown:.1%})"
                        
                        # 高回撤时强制激活对冲策略
                        if current_drawdown > 0.20:  # 回撤 > 20%时强制对冲
                            self._force_hedging_activation = True
                            self.algorithm.log_debug(f"高回撤触发强制对冲: {current_drawdown:.1%}", log_type="portfolio")
                        else:
                            self._force_hedging_activation = False
                
                # 检查波动率指标
                if hasattr(self.algorithm, 'volatility_monitor') and hasattr(self.algorithm, 'previous_portfolio_value'):
                    current_value = self.algorithm.Portfolio.TotalPortfolioValue
                    daily_return = (current_value - self.algorithm.previous_portfolio_value) / self.algorithm.previous_portfolio_value
                    volatility_metrics = self.algorithm.volatility_monitor.update_return(daily_return)
                    
                    if volatility_metrics.get('volatility_alert', False):
                        should_apply_defense = True
                        defense_reason += " + 波动率警告"
                        
            except Exception as risk_check_error:
                self.algorithm.log_debug(f"Error checking additional risk metrics: {risk_check_error}", log_type="portfolio")
            
            # 如果不需要应用防御策略，仍然要进行常规筛选
            if not should_apply_defense:
                self.algorithm.log_debug("市场状况良好，使用常规策略但仍会动态调整现金比例", log_type="portfolio")
                # 保存平均预期收益，用于动态现金管理
                self.algorithm._last_avg_expected_return = avg_expected_return
                self.algorithm._market_volatility = volatility_estimate
                self.algorithm._negative_ratio = negative_ratio
                return expected_returns, covariance_matrix, symbols
            
            # 应用防御策略
            self.algorithm.log_debug(f"触发防御策略: {defense_reason}", log_type="portfolio")
            
            # 防御策略参数
            min_return_threshold = self.config.PORTFOLIO_CONFIG.get('moderate_loss_threshold', -0.02)  # -2%
            max_negative_positions = self.config.PORTFOLIO_CONFIG.get('min_positive_symbols', 3)
            
            # 创建筛选条件
            positive_returns_mask = expected_returns >= 0
            acceptable_negative_mask = expected_returns >= min_return_threshold
            
            # 如果有正收益的标的，优先选择它们
            if np.any(positive_returns_mask):
                # 选择所有正收益标的
                selected_mask = positive_returns_mask.copy()
                
                # 如果正收益标的数量太少，可以加入一些较好的负收益标的
                if np.sum(selected_mask) < 3 and np.any(acceptable_negative_mask & ~positive_returns_mask):
                    # 优先选择防御性股票
                    defensive_priorities = {
                        'GLD': 10,      # 黄金ETF - 最高优先级
                        'SPY': 8,       # 大盘ETF
                        'LLY': 7,       # 医药股相对防御
                        'MSFT': 6,      # 大型科技股
                        'AAPL': 6,      # 大型科技股
                        'AMZN': 5,      # 大型科技股但波动更大
                    }
                    
                    # 在可接受的负收益标的中选择最好的几个，优先考虑防御性
                    negative_candidates_indices = np.where(acceptable_negative_mask & ~positive_returns_mask)[0]
                    if len(negative_candidates_indices) > 0:
                        # 计算综合得分：收益率 + 防御性得分
                        candidate_scores = []
                        for idx in negative_candidates_indices:
                            symbol_str = str(symbols[idx])
                            return_score = expected_returns[idx]  # 负值，越接近0越好
                            defensive_score = defensive_priorities.get(symbol_str, 3) / 10.0  # 归一化到0-1
                            combined_score = return_score + defensive_score  # 综合得分
                            candidate_scores.append((idx, combined_score))
                        
                        # 按综合得分排序（从高到低）
                        candidate_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # 最多添加几个负收益标的来补充组合
                        needed_count = min(max_negative_positions, 5 - np.sum(selected_mask), len(candidate_scores))
                        for i in range(needed_count):
                            idx = candidate_scores[i][0]
                            selected_mask[idx] = True
                        
                        self.algorithm.log_debug(f"Added {needed_count} defensive negative return assets", log_type="portfolio")
            
            else:
                # 如果没有正收益标的，选择损失最小的几个
                if np.any(acceptable_negative_mask):
                    selected_mask = acceptable_negative_mask.copy()
                    self.algorithm.log_debug("No positive returns available, using acceptable negative returns", log_type="portfolio")
                else:
                    # 极端情况：选择损失最小的前几个标的
                    sorted_indices = np.argsort(-expected_returns)  # 从高到低排序
                    selected_mask = np.zeros(len(expected_returns), dtype=bool)
                    num_to_select = min(5, len(expected_returns))  # 最多选择5个
                    selected_mask[sorted_indices[:num_to_select]] = True
                    self.algorithm.log_debug(f"Extreme case: selected top {num_to_select} assets with smallest losses", log_type="portfolio")
            
            # 标记需要应用防御性现金管理
            self._all_negative_scenario = True
            self._defense_reason = defense_reason
            
            # 应用筛选
            if np.any(selected_mask):
                filtered_returns = expected_returns[selected_mask]
                
                # 处理协方差矩阵的索引 - 检查是 DataFrame 还是 numpy 数组
                if hasattr(covariance_matrix, 'iloc'):
                    # 如果是 pandas DataFrame，使用 iloc 进行索引
                    filtered_cov = covariance_matrix.iloc[selected_mask, selected_mask]
                else:
                    # 如果是 numpy 数组，使用 np.ix_ 进行索引
                    filtered_cov = covariance_matrix[np.ix_(selected_mask, selected_mask)]
                
                filtered_symbols = [symbols[i] for i in range(len(symbols)) if selected_mask[i]]
                
                # 记录筛选结果
                positive_count = np.sum(filtered_returns >= 0)
                negative_count = np.sum(filtered_returns < 0)
                avg_return = np.mean(filtered_returns)
                
                # 保存关键指标，用于防御性现金管理
                self.algorithm._last_avg_expected_return = avg_return
                self.algorithm._market_volatility = volatility_estimate
                self.algorithm._negative_ratio = negative_ratio
                
                self.algorithm.log_debug("Defensive strategy applied:", log_type="portfolio")
                self.algorithm.log_debug(f"  Original assets: {len(symbols)}", log_type="portfolio")
                self.algorithm.log_debug(f"  Filtered assets: {len(filtered_symbols)}", log_type="portfolio")
                self.algorithm.log_debug(f"  Positive returns: {positive_count}", log_type="portfolio")
                self.algorithm.log_debug(f"  Negative returns: {negative_count}", log_type="portfolio")
                self.algorithm.log_debug(f"  Average expected return: {avg_return:.4f}", log_type="portfolio")
                
                return filtered_returns, filtered_cov, filtered_symbols
            
            else:
                # 如果筛选后没有标的剩余，返回原始数据但发出警告
                self.algorithm.log_debug("⚠️  Defensive strategy filtered out all assets - using original data", log_type="portfolio")
                return expected_returns, covariance_matrix, symbols
                
        except Exception as e:
            self.algorithm.log_debug(f"Error in defensive strategy: {e}", log_type="portfolio")
            # 出错时返回原始数据
            return expected_returns, covariance_matrix, symbols
    

    
    def _apply_constraints(self, weights, symbols):
        """应用各种约束条件 - 确保多元化"""
        # 1. 应用基本权重限制
        min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
        max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
        
        # 先确保权重在合理范围内
        constrained_weights = np.clip(weights, min_weight, max_weight)
        
        # 2. 应用集中度限制 - 但不要过度分散
        constrained_weights = self.concentration_limiter.apply_concentration_limits(constrained_weights, symbols)
        
        # 3. 确保没有权重过小（可能被集中度限制器删除）
        constrained_weights = np.maximum(constrained_weights, min_weight)
        
        # 4. 重新归一化以确保总和正确
        total_weight = np.sum(constrained_weights)
        if total_weight > 0:
            constrained_weights = constrained_weights / total_weight
            
            # 5. 动态计算目标股票仓位比例 - Alert Black Bat分析
            target_equity_ratio = self._calculate_dynamic_equity_ratio(weights, symbols)
            
            # 6. 应用股票仓位比例 - 修复：确保权重正确记录和应用
            self.algorithm.log_debug(f"应用股票仓位比例: {target_equity_ratio:.2%}", log_type="portfolio")
            
            # 正确应用仓位比例：保持股票间相对权重，但缩放总权重
            if target_equity_ratio > 0:
                constrained_weights = constrained_weights * target_equity_ratio
                self.algorithm.log_debug(f"应用仓位比例后总权重: {np.sum(constrained_weights):.2%}", log_type="portfolio")
            else:
                # 如果目标股票仓位为0，清空所有权重
                constrained_weights = np.zeros_like(constrained_weights)
                self.algorithm.log_debug("目标股票仓位为0%，清空所有权重", log_type="portfolio")
        
        # 7. 最终检查权重分布
        n_nonzero = np.sum(constrained_weights > 0.001)  # 计算有效持仓数量
        self.algorithm.log_debug(f"约束后: {n_nonzero}只股票, 总权重: {np.sum(constrained_weights):.2%}", log_type="portfolio")
        
        return constrained_weights

    def _calculate_dynamic_equity_ratio(self, weights, symbols):
        """动态股票仓位计算 - Alert Black Bat分析优化，支持杠杆到130%"""
        try:
            # *** Alert Black Bat代码执行验证 ***
            self.algorithm.log_debug("Alert Black Bat代码已执行", log_type="risk")
            
            # 获取基础配置
            portfolio_config = self.config.PORTFOLIO_CONFIG
            leverage_config = self.config.LEVERAGE_CONFIG
            risk_config = self.config.RISK_CONFIG
            
            # === Alert Black Bat分析：动态现金管理优先级最高 ===
            dynamic_cash_config = risk_config.get('dynamic_cash_management', {})
            enable_dynamic_cash = dynamic_cash_config.get('enable_dynamic_cash', False)
            
            # 关键日志：只在启用时输出
            if enable_dynamic_cash:
                self.algorithm.log_debug("Alert Black Bat动态现金管理已启用", log_type="risk")
            
            if enable_dynamic_cash:
                market_condition = self._assess_market_condition()
                cash_range = self._get_cash_range_by_market_condition(market_condition, dynamic_cash_config)
                
                # 根据市场状况调整现金比例
                if market_condition == 'crisis':
                    # 危机市场：大幅增加现金比例
                    cash_ratio = cash_range[1]  # 使用范围上限
                    equity_ratio = 1.0 - cash_ratio
                    self.algorithm.log_debug(f"Alert Black Bat危机: 现金{cash_ratio:.0%}, 股票{equity_ratio:.0%}", 
                                           log_type="risk")
                elif market_condition == 'volatile':
                    # 波动市场：适度增加现金比例
                    cash_ratio = (cash_range[0] + cash_range[1]) / 2
                    equity_ratio = 1.0 - cash_ratio
                    self.algorithm.log_debug(f"Alert Black Bat波动: 现金{cash_ratio:.0%}, 股票{equity_ratio:.0%}", 
                                           log_type="risk")
                else:
                    # 正常市场：允许轻微杠杆
                    cash_ratio = cash_range[0]  # 使用范围下限（可能为负）
                    equity_ratio = 1.0 - cash_ratio
                    self.algorithm.log_debug(f"Alert Black Bat正常: 现金{cash_ratio:.0%}, 股票{equity_ratio:.0%}", 
                                           log_type="risk")
                
                # 缓存最终的股票仓位比例
                self._last_equity_ratio = equity_ratio
                return equity_ratio
            
            # === 传统模式（如果未启用Alert Black Bat动态现金管理） ===
            # 集成杠杆管理器获取目标杠杆比例
            target_leverage = 1.0  # 默认值
            if hasattr(self.algorithm, 'leverage_manager'):
                target_leverage = self.algorithm.leverage_manager.get_current_leverage_ratio()
                self.algorithm.log_debug(f"获取目标杠杆比例: {target_leverage:.2f}", log_type="leverage")
            
            # 基于杠杆调整仓位范围
            if leverage_config.get('enable_leverage', False):
                # 杠杆模式下的仓位设置
                base_equity_ratio = min(target_leverage, leverage_config.get('max_leverage_ratio', 1.3))
                max_equity_ratio = min(target_leverage, leverage_config.get('max_leverage_ratio', 1.3))
                min_equity_ratio = 0.05  # 最小5%仓位
                
                self.algorithm.log_debug(f"杠杆模式 - 基础仓位:{base_equity_ratio:.2f}, 最高仓位:{max_equity_ratio:.2f}", 
                                       log_type="leverage")
            else:
                # 传统模式 - 限制在100%以下
                base_equity_ratio = 0.98
                max_equity_ratio = 0.98
                min_equity_ratio = 0.05
                
                self.algorithm.log_debug("传统模式 - 最高仓位限制在98%", log_type="portfolio")
            
            # 市场风险评估
            risk_level = self._assess_market_risk()
            self.algorithm.log_debug(f"市场风险等级: {risk_level}", log_type="risk")
            
            # 根据风险水平调整股票仓位比例
            risk_adjustments = {
                'very_low': 0.05,   # 极低风险：+5%
                'low': 0.0,         # 低风险：基准
                'medium': -0.15,    # 中等风险：-15%
                'high': -0.30,      # 高风险：-30%
                'very_high': -0.50  # 极高风险：-50%
            }
            
            risk_adjustment = risk_adjustments.get(risk_level, 0.0)
            equity_ratio = base_equity_ratio + risk_adjustment
            
            # 应用杠杆环境下的约束
            equity_ratio = max(min_equity_ratio, min(max_equity_ratio, equity_ratio))
            
            # 检查是否处于防御模式
            if hasattr(self.algorithm, 'system_monitor'):
                defense_mode = self.algorithm.system_monitor.get_defense_mode()
                if defense_mode in ['moderate_defense', 'aggressive_defense', 'extreme_defense']:
                    # 防御模式下限制杠杆
                    defense_limits = {
                        'moderate_defense': 0.8,
                        'aggressive_defense': 0.6,
                        'extreme_defense': 0.4
                    }
                    max_defense_ratio = defense_limits.get(defense_mode, 0.8)
                    equity_ratio = min(equity_ratio, max_defense_ratio)
                    self.algorithm.log_debug(f"防御模式 {defense_mode} - 仓位限制至 {equity_ratio:.2f}", 
                                           log_type="defense")
            
            self.algorithm.log_debug(f"最终股票仓位比例: {equity_ratio:.2f} (目标杠杆:{target_leverage:.2f})", 
                                   log_type="portfolio")
            
            return equity_ratio
            
        except Exception as e:
            self.algorithm.log_debug(f"动态股票仓位计算错误: {str(e)}", log_type="error")
            # 杠杆模式下的默认值
            return target_leverage if hasattr(self.algorithm, 'leverage_manager') else 0.98
    
    def _final_screening(self, weights, symbols):
        """最终筛选和权重调整 - 确保多元化投资"""
        threshold = self.config.PORTFOLIO_CONFIG['weight_threshold']
        target_size = self.config.PORTFOLIO_CONFIG['target_portfolio_size']
        min_size = self.config.PORTFOLIO_CONFIG['min_portfolio_size']
        max_size = self.config.PORTFOLIO_CONFIG['max_portfolio_size']
        
        # 强制等权重策略优先
        if self.config.PORTFOLIO_CONFIG.get('force_equal_weights', False):
            self.algorithm.log_debug(f"应用强制等权重策略: {target_size}只股票", log_type="portfolio")
            # 选择权重最高的N只股票进行等权重分配
            top_indices = np.argsort(weights)[-target_size:]
            final_symbols = [symbols[i] for i in top_indices]
            final_weights = np.ones(target_size) / target_size
            
            self.algorithm.log_debug(f"等权重组合: {len(final_symbols)}只股票，每只{1/target_size:.1%}", log_type="portfolio")
            for symbol in final_symbols:
                self.algorithm.log_debug(f"  {symbol}: {1/target_size:.1%}", log_type="portfolio")
            
            return final_weights, final_symbols
        
        # 原有逻辑（简化日志）
        valid_mask = weights >= threshold
        n_valid = np.sum(valid_mask)
        
        # 如果筛选结果太少，采用Top-N策略
        if n_valid < min_size:
            top_n = min(target_size, len(weights))
            top_indices = np.argsort(weights)[-top_n:]
            valid_mask = np.zeros(len(weights), dtype=bool)
            valid_mask[top_indices] = True
            n_valid = top_n
            self.algorithm.log_debug(f"筛选不足，采用Top-{n_valid}策略", log_type="portfolio")
        
        # 如果筛选结果太多，限制数量
        elif n_valid > max_size:
            valid_indices = np.where(valid_mask)[0]
            valid_weights = weights[valid_indices]
            sorted_indices = valid_indices[np.argsort(valid_weights)[::-1]]
            
            valid_mask = np.zeros(len(weights), dtype=bool)
            valid_mask[sorted_indices[:max_size]] = True
            n_valid = max_size
            self.algorithm.log_debug(f"限制至{n_valid}只股票", log_type="portfolio")
        
        # 提取最终的股票和权重
        final_symbols = [symbols[i] for i in range(len(symbols)) if valid_mask[i]]
        final_weights = weights[valid_mask]
        
        if len(final_weights) > 0:
            # 重新平衡权重确保多元化
            final_weights = self._rebalance_for_diversification(final_weights, final_symbols)
            
            # 获取目标股票仓位比例
            target_equity_ratio = getattr(self, '_last_equity_ratio', 0.98)
            
            # 杠杆模式权重归一化
            if hasattr(self.algorithm, 'leverage_manager') and self.config.LEVERAGE_CONFIG.get('enable_leverage', False):
                # 杠杆模式：权重总和可以超过1.0
                current_sum = np.sum(final_weights)
                if current_sum > 0:
                    # 按目标股票仓位比例缩放权重
                    final_weights = final_weights * (target_equity_ratio / current_sum)
                    
                    self.algorithm.log_debug(f"杠杆模式权重调整: 目标仓位{target_equity_ratio:.2f}, 权重总和{np.sum(final_weights):.2f}", 
                                           log_type="leverage")
                else:
                    final_weights = np.zeros_like(final_weights)
            else:
                # 传统模式：权重总和归一化为1.0
                final_weights = final_weights / np.sum(final_weights)
                self.algorithm.log_debug("传统模式权重归一化", log_type="portfolio")
            
            self.algorithm.log_debug(f"最终组合: {len(final_symbols)}只股票, 总权重: {np.sum(final_weights):.2f}", log_type="portfolio")
            for symbol, weight in zip(final_symbols, final_weights):
                self.algorithm.log_debug(f"  {symbol}: {weight:.1%}", log_type="portfolio")
        else:
            self.algorithm.log_debug("警告: 没有股票被选中", log_type="portfolio")
        
        return final_weights, final_symbols
    
    def _rebalance_for_diversification(self, weights, symbols):
        """重新平衡权重以提高多元化程度"""
        try:
            min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
            max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
            diversification_pref = self.config.PORTFOLIO_CONFIG['diversification_preference']
            
            # 确保所有权重都在合理范围内
            weights = np.clip(weights, min_weight, max_weight)
            
            # 计算多元化调整
            n_stocks = len(weights)
            if n_stocks > 1:
                # 理想的等权重
                equal_weight = 1.0 / n_stocks
                
                # 混合优化权重和等权重，提高多元化
                adjusted_weights = (1 - diversification_pref) * weights + diversification_pref * equal_weight
                
                # 再次确保权重限制
                adjusted_weights = np.clip(adjusted_weights, min_weight, max_weight)
                
                self.algorithm.log_debug(f"多元化调整: 偏好={diversification_pref:.1%}, 理想等权重={equal_weight:.2%}", log_type="portfolio")
                return adjusted_weights
            
            return weights
            
        except Exception as e:
            self.algorithm.log_debug(f"多元化重平衡错误: {e}", log_type="portfolio")
            return weights
    
    def get_last_decision_factors(self):
        """获取最近一次的决策因素"""
        return self._last_decision_factors.copy()
    
    def validate_position_consistency(self, weights, symbols):
        """验证仓位一致性 - 新增：确保显示与实际执行一致"""
        try:
            if weights is None or len(weights) == 0:
                self.algorithm.log_debug("验证结果: 无权重分配 (全现金策略)", log_type="optimizer")
                return True
            
            weights_sum = np.sum(weights)
            equity_ratio = getattr(self, '_last_equity_ratio', 0.0)
            
            self.algorithm.log_debug("=== 仓位一致性验证 ===", log_type="optimizer")
            self.algorithm.log_debug(f"权重总和: {weights_sum:.4f}", log_type="optimizer")
            self.algorithm.log_debug(f"目标股票仓位: {equity_ratio:.2%}", log_type="optimizer")
            self.algorithm.log_debug(f"股票数量: {len(symbols)}", log_type="optimizer")
            
            # 检查一致性
            consistency_ok = True
            
            if equity_ratio == 0.0 and weights_sum > 0.001:
                self.algorithm.log_debug("⚠️ 不一致: 股票仓位0%但权重总和>0", log_type="optimizer")
                consistency_ok = False
            elif equity_ratio > 0.0 and weights_sum < 0.001:
                self.algorithm.log_debug("⚠️ 不一致: 股票仓位>0%但权重总和≈0", log_type="optimizer")
                consistency_ok = False
            else:
                expected_sum = equity_ratio
                diff = abs(weights_sum - expected_sum)
                if diff > 0.05:  # 5%容忍度
                    self.algorithm.log_debug(f"⚠️ 权重总和偏差过大: 预期{expected_sum:.2%}, 实际{weights_sum:.2%}, 差异{diff:.2%}", log_type="optimizer")
                    consistency_ok = False
            
            if consistency_ok:
                self.algorithm.log_debug("✓ 仓位一致性验证通过", log_type="optimizer")
            else:
                self.algorithm.log_debug("✗ 仓位一致性验证失败", log_type="optimizer")
            
            return consistency_ok
            
        except Exception as e:
            self.algorithm.log_debug(f"仓位一致性验证错误: {e}", log_type="optimizer")
            return False
    
    def _assess_market_risk(self):
        """评估市场风险等级"""
        try:
            # 检查防御模式
            defense_mode = None
            if hasattr(self.algorithm, 'system_monitor'):
                defense_mode = self.algorithm.system_monitor.get_defense_mode()
            
            # 检查回撤水平
            current_drawdown = 0.0
            if hasattr(self.algorithm, 'drawdown_monitor'):
                current_drawdown = abs(self.algorithm.drawdown_monitor.get_current_drawdown())
            
            # 检查VIX风险状态
            vix_risk_state = 'normal'
            if hasattr(self.algorithm, 'vix_monitor'):
                vix_risk_state = self.algorithm.vix_monitor.get_risk_state()
            
            # 整合杠杆管理器的风险评估
            leverage_risk_level = 'medium'
            if hasattr(self.algorithm, 'leverage_manager'):
                leverage_risk_level = getattr(self.algorithm.leverage_manager, '_risk_level', 'medium')
            
            # 综合风险等级评估
            risk_factors = []
            
            # 防御模式风险
            if defense_mode == 'extreme_defense':
                risk_factors.append('very_high')
            elif defense_mode == 'aggressive_defense':
                risk_factors.append('high')
            elif defense_mode == 'moderate_defense':
                risk_factors.append('medium')
            
            # 回撤风险 - Alert Black Bat分析优化
            if current_drawdown > 0.12:
                risk_factors.append('very_high')
            elif current_drawdown > 0.08:
                risk_factors.append('high')
            elif current_drawdown > 0.05:
                risk_factors.append('medium')
            else:
                risk_factors.append('low')
            
            # VIX风险
            if vix_risk_state == 'extreme':
                risk_factors.append('very_high')
            elif vix_risk_state == 'high':
                risk_factors.append('high')
            elif vix_risk_state == 'medium':
                risk_factors.append('medium')
            else:
                risk_factors.append('low')
            
            # 杠杆管理器风险
            if leverage_risk_level == 'extreme':
                risk_factors.append('very_high')
            elif leverage_risk_level == 'high':
                risk_factors.append('high')
            elif leverage_risk_level == 'medium':
                risk_factors.append('medium')
            else:
                risk_factors.append('low')
            
            # 取最高风险等级
            risk_hierarchy = ['very_low', 'low', 'medium', 'high', 'very_high']
            if not risk_factors:
                risk_factors = ['medium']
            
            final_risk = max(risk_factors, key=lambda x: risk_hierarchy.index(x) if x in risk_hierarchy else 2)
            
            self.algorithm.log_debug(f"市场风险评估: {final_risk} (防御:{defense_mode}, 回撤:{current_drawdown:.2%}, "
                                   f"VIX:{vix_risk_state}, 杠杆:{leverage_risk_level})", log_type="risk")
            
            return final_risk
            
        except Exception as e:
            self.algorithm.log_debug(f"市场风险评估错误: {e}", log_type="risk")
            return 'medium'
    
    def _assess_market_condition(self):
        """Alert Black Bat分析：评估市场状况"""
        try:
            # 获取当前回撤
            current_drawdown = 0.0
            if hasattr(self.algorithm, 'drawdown_monitor'):
                current_drawdown = abs(self.algorithm.drawdown_monitor.get_current_drawdown())
            
            # 获取VIX水平
            vix_level = 20
            if hasattr(self.algorithm, 'vix_monitor'):
                vix_level = getattr(self.algorithm.vix_monitor, '_current_vix', 20)
            
            # 根据Alert Black Bat分析结论判断市场状况
            if current_drawdown >= 0.10 or vix_level >= 30:
                condition = 'crisis'      # 危机市场：回撤≥10%或VIX≥30
            elif current_drawdown >= 0.05 or vix_level >= 22:
                condition = 'volatile'    # 波动市场：回撤≥5%或VIX≥22
            else:
                condition = 'normal'      # 正常市场
            
            # 只在非正常状态时输出日志
            if condition != 'normal':
                self.algorithm.log_debug(f"Alert Black Bat: {condition} (回撤{current_drawdown:.1%}, VIX{vix_level:.1f})", log_type="risk")
            
            return condition
                
        except Exception as e:
            self.algorithm.log_debug(f"Alert Black Bat评估错误: {e}", log_type="risk")
            return 'normal'
    
    def _get_cash_range_by_market_condition(self, market_condition, dynamic_cash_config):
        """Alert Black Bat分析：根据市场状况获取现金比例范围"""
        try:
            if market_condition == 'crisis':
                return dynamic_cash_config.get('crisis_market_cash_range', (0.20, 0.40))
            elif market_condition == 'volatile':
                return dynamic_cash_config.get('volatile_market_cash_range', (0.10, 0.20))
            else:
                return dynamic_cash_config.get('normal_market_cash_range', (-0.02, 0.05))
                
        except Exception as e:
            self.algorithm.log_debug(f"现金范围获取错误: {e}", log_type="risk")
            return (0.0, 0.05)  # 默认范围


 