# 风险管理模块
from AlgorithmImports import *
import numpy as np
import pandas as pd
# 简化类型注解以兼容QuantConnect云端
from config import AlgorithmConfig
from collections import defaultdict

class RiskManager:
    """风险管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.risk_metrics_history = []
        
        # 初始化VIX监控器
        self.vix_monitor = VIXMonitor(algorithm_instance)
        
    def apply_risk_controls(self, expected_returns, symbols):
        """应用风险控制"""
        # 1. 打印VIX相关阈值
        vix_cfg = self.config.RISK_CONFIG
        self.algorithm.log_debug(f"[VIX配置] rapid_rise={vix_cfg['vix_rapid_rise_threshold']}, extreme={vix_cfg['vix_extreme_level']}, normalization={vix_cfg['vix_normalization_threshold']}, defense_min_equity={vix_cfg['vix_defense_min_equity']}", log_type="risk")
        
        self.algorithm.log_debug("[风控] 初始股票池: {}".format(list(symbols)), log_type="risk")
        
        # 0. VIX风险状态监控
        vix_risk_state = self.vix_monitor.update_vix_data(self.algorithm.Time)
        self.algorithm.log_debug(f"[VIX状态] {vix_risk_state}", log_type="risk")
        
        # 1. 流动性筛选
        liquid_symbols = self._filter_by_liquidity(symbols)
        self.algorithm.log_debug(f"[风控] 流动性筛选后: {list(liquid_symbols)}", log_type="risk")
        
        # 2. 波动率筛选
        volatility_filtered = self._filter_by_volatility(liquid_symbols, expected_returns)
        self.algorithm.log_debug(f"[风控] 波动率筛选后: {list(volatility_filtered)}", log_type="risk")
        
        # 3. 相关性检查
        correlation_filtered = self._filter_by_correlation(volatility_filtered)
        self.algorithm.log_debug(f"[风控] 相关性筛选后: {list(correlation_filtered)}", log_type="risk")
        
        # 4. 应用波动指标信号筛选
        volatility_signal_filtered = self._apply_volatility_indicator_filter(correlation_filtered)
        self.algorithm.log_debug(f"[风控] 波动指标筛选后: {list(volatility_signal_filtered)}", log_type="risk")
        
        # 5. VIX防御性筛选
        vix_filtered_symbols = self._apply_vix_defensive_filter(volatility_signal_filtered, vix_risk_state)
        self.algorithm.log_debug(f"[风控] VIX防御筛选后: {list(vix_filtered_symbols)}", log_type="risk")
        
        # 6. 调整预期收益（风险调整，包含VIX风险）
        risk_adjusted_returns = self._adjust_returns_for_risk(expected_returns, vix_filtered_symbols, vix_risk_state)
        
        # 7. 最终有效性检查
        final_symbols = self._final_validity_check(vix_filtered_symbols)
        self.algorithm.log_debug(f"[风控] 最终有效性检查后: {list(final_symbols)}", log_type="risk")
        
        # 8. 保存VIX风险状态供组合优化使用
        self.algorithm._vix_risk_state = vix_risk_state
        
        return risk_adjusted_returns, final_symbols
    
    def _filter_by_liquidity(self, symbols):
        """基于流动性过滤股票"""
        liquid_symbols = []
        
        for symbol in symbols:
            try:
                # 获取最近的成交量数据
                history = self.algorithm.History(symbol, 20, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) < 10:
                    continue
                
                volumes = [x.Volume for x in history_list]
                avg_volume = np.mean(volumes)
                
                                # 检查平均成交量是否满足要求
                if avg_volume >= self.config.RISK_CONFIG['liquidity_min_volume']:
                    liquid_symbols.append(symbol)
                else:
                    self.algorithm.log_debug(f"Filtered out {symbol} due to low liquidity: {avg_volume:,.0f}", log_type="risk")
                    
            except Exception as e:
                self.algorithm.log_debug(f"Error checking liquidity for {symbol}: {e}", log_type="risk")
                continue
        
        return liquid_symbols
    
    def _filter_by_volatility(self, symbols, expected_returns):
        """基于波动率过滤股票"""
        volatility_filtered = []
        
        for symbol in symbols:
            try:
                # 计算历史波动率
                history = self.algorithm.History(symbol, 60, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) < 30:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
                
                # 检查波动率是否在合理范围内
                if volatility <= self.config.RISK_CONFIG['volatility_threshold']:
                    volatility_filtered.append(symbol)
                else:
                    self.algorithm.log_debug(f"Filtered out {symbol} due to high volatility: {volatility:.3f}", log_type="risk")
                    
            except Exception as e:
                self.algorithm.log_debug(f"Error calculating volatility for {symbol}: {e}", log_type="risk")
                continue
        
        return volatility_filtered
    
    def _filter_by_correlation(self, symbols):
        """基于相关性过滤股票"""
        if len(symbols) <= 1:
            return symbols
        
        try:
            # 计算相关性矩阵
            correlation_matrix = self._calculate_correlation_matrix(symbols)
            
            if correlation_matrix is None:
                return symbols
            
            # 识别高相关性的股票对
            high_corr_pairs = self._find_high_correlation_pairs(
                correlation_matrix, 
                self.config.RISK_CONFIG['correlation_threshold']
            )
            
            # 从高相关性对中选择保留的股票
            filtered_symbols = self._select_from_correlated_pairs(symbols, high_corr_pairs)
            
            return filtered_symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in correlation filtering: {e}", log_type="risk")
            return symbols
    
    def _calculate_correlation_matrix(self, symbols):
        """计算相关性矩阵"""
        try:
            returns_data = {}
            
            for symbol in symbols:
                history = self.algorithm.History(symbol, 60, Resolution.DAILY)
                history_list = list(history)
                if len(history_list) < 30:
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                returns_data[symbol] = returns
            
            if len(returns_data) < 2:
                return None
            
            # 构建收益率DataFrame
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {}
            
            for symbol, returns in returns_data.items():
                aligned_returns[symbol] = returns[-min_length:]
            
            returns_df = pd.DataFrame(aligned_returns)
            correlation_matrix = returns_df.corr().values
            
            return correlation_matrix
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating correlation matrix: {e}", log_type="risk")
            return None
    
    def _find_high_correlation_pairs(self, correlation_matrix, 
                                   threshold):
        """找到高相关性的股票对"""
        high_corr_pairs = []
        n = correlation_matrix.shape[0]
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(correlation_matrix[i, j]) > threshold:
                    high_corr_pairs.append((i, j))
        
        return high_corr_pairs
    
    def _select_from_correlated_pairs(self, symbols, 
                                    high_corr_pairs):
        """从高相关性对中选择保留的股票"""
        if not high_corr_pairs:
            return symbols
        
        # 简单策略：从每个高相关性对中随机选择一个
        to_remove = set()
        
        for i, j in high_corr_pairs:
            if i not in to_remove and j not in to_remove:
                # 随机选择移除其中一个
                to_remove.add(j)  # 保留索引较小的
        
        filtered_symbols = [symbols[i] for i in range(len(symbols)) if i not in to_remove]
        
        if len(to_remove) > 0:
            removed_symbols = [symbols[i] for i in to_remove]
            self.algorithm.log_debug(f"Removed due to high correlation: {removed_symbols}", log_type="risk")
        
        return filtered_symbols
    
    def _apply_volatility_indicator_filter(self, symbols):
        """应用波动指标信号筛选"""
        try:
            self.algorithm.log_debug("应用波动指标信号筛选...", log_type="risk")
            
            # 检查是否有波动指标分析结果
            if not hasattr(self.algorithm, 'preferred_volatility_indicator'):
                self.algorithm.log_debug("未找到波动指标分析结果，跳过指标筛选", log_type="risk")
                return symbols
            
            preferred_indicator = getattr(self.algorithm, 'preferred_volatility_indicator', '自动选择')
            self.algorithm.log_debug(f"使用推荐的波动指标: {preferred_indicator}", log_type="risk")
            
            filtered_symbols = []
            
            for symbol in symbols:
                try:
                    # 获取历史数据
                    history = self.algorithm.History(symbol, 60, Resolution.DAILY)
                    history_list = list(history)
                    if len(history_list) < 30:
                        self.algorithm.log_debug(f"{symbol} 历史数据不足，跳过波动信号筛选", log_type="risk")
                        filtered_symbols.append(symbol)  # 数据不足时保留
                        continue
                    
                    prices = np.array([x.Close for x in history_list])
                    
                    # 获取波动指标信号
                    volatility_signals = self.algorithm.data_processor.get_volatility_indicator_signals(
                        prices, preferred_indicator=preferred_indicator
                    )
                    
                    if volatility_signals is None:
                        self.algorithm.log_debug(f"{symbol} 无法获取波动指标信号，保留该股票", log_type="risk")
                        filtered_symbols.append(symbol)
                        continue
                    
                    # 根据波动指标信号进行筛选
                    if self._evaluate_volatility_signal_quality(volatility_signals, symbol):
                        filtered_symbols.append(symbol)
                    else:
                        self.algorithm.log_debug(f"根据波动指标筛选掉 {symbol}", log_type="risk")
                    
                except Exception as e:
                    self.algorithm.log_debug(f"处理 {symbol} 的波动指标时出错: {e}", log_type="risk")
                    filtered_symbols.append(symbol)  # 出错时保留
                    continue
            
            self.algorithm.log_debug(f"波动指标筛选结果: {len(symbols)} -> {len(filtered_symbols)}", log_type="risk")
            return filtered_symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"波动指标筛选过程出错: {e}", log_type="risk")
            return symbols  # 出错时返回原始列表
    
    def _apply_vix_defensive_filter(self, symbols, vix_risk_state):
        """应用VIX防御性筛选"""
        try:
            if not vix_risk_state['defense_mode']:
                return symbols
            
            self.algorithm.log_debug("应用VIX防御性筛选...", log_type="risk")
            
            # 在极端模式下，优先保留防御性较强的股票
            if vix_risk_state['extreme_mode']:
                return self._select_defensive_stocks(symbols, vix_risk_state)
            
            # 普通防御模式下的筛选逻辑
            return self._apply_moderate_vix_filter(symbols, vix_risk_state)
            
        except Exception as e:
            self.algorithm.log_debug(f"VIX防御筛选出错: {e}", log_type="risk")
            return symbols
    
    def _select_defensive_stocks(self, symbols, vix_risk_state):
        """在极端VIX模式下选择防御性股票"""
        try:
            # 防御性股票的优先级排序
            defensive_priorities = {
                'GLD': 10,      # 黄金ETF - 最高优先级
                'SPY': 8,       # 大盘ETF
                'LLY': 7,       # 医药股相对防御
                'MSFT': 6,      # 大型科技股
                'AAPL': 6,      # 大型科技股
                'AMZN': 5,      # 大型科技股但波动更大
            }
            
            # 计算每个股票的防御性得分
            stock_scores = []
            for symbol in symbols:
                symbol_str = str(symbol)
                base_score = defensive_priorities.get(symbol_str, 3)  # 默认得分3
                
                # 基于历史波动率调整得分
                try:
                    history = self.algorithm.History(symbol, 30, Resolution.Daily)
                    history_list = list(history)
                    if len(history_list) >= 20:
                        prices = np.array([x.Close for x in history_list])
                        returns = np.diff(prices) / prices[:-1]
                        volatility = np.std(returns) * np.sqrt(252)
                        
                        # 波动率越低，防御性得分越高
                        volatility_score = max(0, 5 - volatility * 10)
                        total_score = base_score + volatility_score
                    else:
                        total_score = base_score
                    
                    stock_scores.append((symbol, total_score))
                    
                except Exception as e:
                    stock_scores.append((symbol, base_score))
            
            # 排序并选择前几名
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            max_stocks = min(3, len(stock_scores))  # 极端模式最多3只股票
            
            selected_symbols = [item[0] for item in stock_scores[:max_stocks]]
            
            self.algorithm.log_debug(f"VIX极端模式选择防御性股票: {[str(s) for s in selected_symbols]}", log_type="risk")
            return selected_symbols
            
        except Exception as e:
            self.algorithm.log_debug(f"选择防御性股票出错: {e}", log_type="risk")
            return symbols[:3]  # 出错时返回前3个
    
    def _apply_moderate_vix_filter(self, symbols, vix_risk_state):
        """应用中等VIX防御筛选"""
        # 在中等防御模式下，过滤掉高波动率股票
        filtered_symbols = []
        
        for symbol in symbols:
            try:
                history = self.algorithm.History(symbol, 20, Resolution.Daily)
                history_list = list(history)
                if len(history_list) < 15:
                    filtered_symbols.append(symbol)  # 数据不足时保留
                    continue
                
                prices = np.array([x.Close for x in history_list])
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns) * np.sqrt(252)
                
                # VIX防御模式下的波动率阈值更严格
                vix_vol_threshold = 0.25  # 比普通的30%更严格
                if volatility <= vix_vol_threshold:
                    filtered_symbols.append(symbol)
                else:
                    self.algorithm.log_debug(f"VIX防御模式过滤高波动股票: {symbol} (波动率: {volatility:.2%})", log_type="risk")
                    
            except Exception as e:
                filtered_symbols.append(symbol)  # 出错时保留
                
        return filtered_symbols
    
    def _evaluate_volatility_signal_quality(self, signals, symbol):
        """评估波动指标信号质量"""
        try:
            # 检查信号的有效性
            if not signals or 'volatility_level' not in signals:
                return True  # 无信号时默认通过
            
            volatility_level = signals['volatility_level']
            if isinstance(volatility_level, np.ndarray):
                current_volatility = volatility_level[-1] if len(volatility_level) > 0 else 0
            else:
                current_volatility = volatility_level
            
            # 波动率过滤条件
            volatility_threshold = self.config.RISK_CONFIG.get('signal_volatility_threshold', 2.0)
            
            # 检查是否处于极端波动状态
            if current_volatility > volatility_threshold:
                self.algorithm.log_debug(f"{symbol} 波动率过高 ({current_volatility:.3f}), 筛选掉", log_type="risk")
                return False
            
            # 检查超买超卖状态
            if 'overbought' in signals and 'oversold' in signals:
                overbought = signals['overbought']
                oversold = signals['oversold']
                
                if isinstance(overbought, np.ndarray):
                    current_overbought = overbought[-1] if len(overbought) > 0 else False
                    current_oversold = oversold[-1] if len(oversold) > 0 else False
                else:
                    current_overbought = overbought
                    current_oversold = oversold
                
                # 避免在极端超买超卖时买入
                if current_overbought:
                    self.algorithm.log_debug(f"{symbol} 处于超买状态，暂时避开", log_type="risk")
                    return False
                
                # 超卖可能是买入机会，但需要谨慎
                if current_oversold:
                    self.algorithm.log_debug(f"{symbol} 处于超卖状态，谨慎考虑", log_type="risk")
                    # 这里可以添加更复杂的逻辑，比如结合趋势强度
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"评估 {symbol} 波动信号时出错: {e}", log_type="risk")
            return True  # 出错时默认通过
    
    def _adjust_returns_for_risk(self, expected_returns, symbols, vix_risk_state=None):
        """基于风险调整预期收益"""
        risk_adjusted_returns = {}
        
        for symbol in symbols:
            if symbol not in expected_returns:
                continue
            
            base_return = expected_returns[symbol]
            
            try:
                # 计算基础风险调整因子
                base_risk_factor = self._calculate_risk_factor(symbol)
                
                # 应用VIX风险调整
                vix_risk_factor = self._calculate_vix_risk_factor(symbol, vix_risk_state)
                
                # 综合风险因子
                combined_risk_factor = base_risk_factor * vix_risk_factor
                
                # 应用风险调整
                adjusted_return = base_return * combined_risk_factor
                risk_adjusted_returns[symbol] = adjusted_return
                
                if vix_risk_state and vix_risk_state['defense_mode']:
                    self.algorithm.log_debug(f"{symbol} VIX risk adjustment: {base_return:.6f} -> {adjusted_return:.6f} (base: {base_risk_factor:.3f}, vix: {vix_risk_factor:.3f})", log_type="risk")
                
            except Exception as e:
                self.algorithm.log_debug(f"Error adjusting return for {symbol}: {e}", log_type="risk")
                risk_adjusted_returns[symbol] = base_return
        
        return risk_adjusted_returns
    
    def _calculate_vix_risk_factor(self, symbol, vix_risk_state):
        """计算VIX风险调整因子（优化版 - 减少过度削减）"""
        if vix_risk_state is None or not vix_risk_state['defense_mode']:
            return 1.0
        
        try:
            symbol_str = str(symbol)
            base_factor = 1.0
            
            # 基于VIX风险等级调整 - 减少削减幅度
            if vix_risk_state['extreme_mode']:
                # 极端模式：适度降低风险偏好，但不过度削减
                if symbol_str in ['GLD', 'SPY']:
                    base_factor = 1.15  # 防御性资产获得适度加成
                elif symbol_str in ['SQQQ']:
                    base_factor = 1.35  # 对冲产品获得较大加成
                else:
                    base_factor = 0.75  # 其他股票适度削减（从0.6提高到0.75）
            
            elif vix_risk_state['defense_mode']:
                # 防御模式：轻微降低风险偏好
                if symbol_str in ['GLD', 'SPY', 'LLY']:
                    base_factor = 1.08  # 防御性资产小幅加成
                elif symbol_str in ['SQQQ']:
                    base_factor = 1.38  # 对冲产品适度加成
                else:
                    base_factor = 0.80  # 其他股票轻微削减（从0.8提高到0.9）
            
            # 基于VIX变化率进一步调整 - 减少调整幅度
            vix_change_rate = vix_risk_state.get('vix_change_rate', 0)
            if vix_change_rate > 0.15:  # 提高触发阈值从0.1到0.15
                if symbol_str not in ['GLD', 'SPY', 'SQQQ']:
                    change_penalty = 1 - min(0.15, vix_change_rate * 0.5)  # 减少惩罚幅度
                    base_factor *= change_penalty
            
            return max(0.5, min(1.5, base_factor))  # 提高最小值从0.3到0.5
            
        except Exception as e:
            self.algorithm.log_debug(f"计算VIX风险因子出错: {e}", log_type="risk")
            return 0.9 if vix_risk_state['defense_mode'] else 1.0  # 提高默认值
    
    def _calculate_risk_factor(self, symbol):
        """计算个股风险调整因子（优化版 - 减少过度削减）"""
        try:
            # 获取历史数据
            history = self.algorithm.History(symbol, 60, Resolution.DAILY)
            history_list = list(history)
            if len(history_list) < 30:
                return 1.0
            
            prices = np.array([x.Close for x in history_list])
            returns = np.diff(prices) / prices[:-1]
            
            # 计算多个风险指标
            volatility = np.std(returns) * np.sqrt(252)
            skewness = self._calculate_skewness(returns)
            max_drawdown = self._calculate_max_drawdown(prices)
            
            # 综合风险评分 (0.7-1.2, 1表示正常风险) - 缩小调整范围
            volatility_score = max(0.7, 1 - (volatility - 0.2) / 0.6)  # 基准从50%调整为20-80%范围
            skewness_score = max(0.8, 1 - abs(skewness) / 4)   # 降低偏度影响，基准从2提高到4
            drawdown_score = max(0.7, 1 - (max_drawdown - 0.1) / 0.4)  # 基准从30%调整为10-50%范围
            
            # 加权平均 - 降低风险因子的影响权重
            risk_factor = (0.4 * volatility_score + 0.3 * drawdown_score + 0.3 * skewness_score)
            
            # 确保在更窄的合理范围内 - 减少极端调整
            risk_factor = max(0.75, min(1.15, risk_factor))
            
            return risk_factor
            
        except Exception as e:
            self.algorithm.log_debug(f"Error calculating risk factor for {symbol}: {e}", log_type="risk")
            return 1.0
    
    def _calculate_skewness(self, returns):
        """计算偏度"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_max_drawdown(self, prices):
        """计算最大回撤"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_drawdown = 0.0
        
        for price in prices:
            if price > peak:
                peak = price
            
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _final_validity_check(self, symbols):
        """最终有效性检查 - 使用多层数据获取方案"""
        valid_symbols = []
        
        for symbol in symbols:
            try:
                # 使用数据处理器的稳健价格获取方法
                current_price = self.algorithm.data_processor.get_current_price_robust(symbol)
                
                if current_price is not None and current_price > 0:
                    valid_symbols.append(symbol)
                    self.algorithm.log_debug(f"Symbol {symbol} validated with price: {current_price}", log_type="risk")
                else:
                    self.algorithm.log_debug(f"Symbol {symbol} excluded - no valid price data", log_type="risk")
                    
            except Exception as e:
                self.algorithm.log_debug(f"Error validating {symbol}: {e}", log_type="risk")
                continue
        
        # 安全检查：如果所有股票都被过滤掉，保留原始列表的一部分
        if len(valid_symbols) == 0 and len(symbols) > 0:
            self.algorithm.log_debug("WARNING: All symbols filtered out, applying emergency protection", log_type="risk")
            # 保留前3只股票作为紧急备用
            emergency_symbols = symbols[:min(3, len(symbols))]
            self.algorithm.log_debug(f"Emergency protection: keeping {[str(s) for s in emergency_symbols]}", log_type="risk")
            return emergency_symbols
        
        self.algorithm.log_debug(f"Final validity check: {len(symbols)} -> {len(valid_symbols)} symbols", log_type="risk")
        return valid_symbols

class ConcentrationLimiter:
    """集中度限制器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def apply_concentration_limits(self, weights, symbols):
        """应用集中度限制 - 促进多元化而非过度限制"""
        if len(weights) == 0:
            return weights
        
        # 1. 单个持仓权重限制 - 但确保不会导致过度集中
        weights = self._limit_individual_weights(weights)
        
        # 2. 促进多元化而不是限制行业集中度
        weights = self._promote_diversification(weights, symbols)
        
        # 3. 轻度重新归一化（不大幅改变权重分布）
        weights = self._gentle_renormalize(weights)
        
        return weights
    
    def _limit_individual_weights(self, weights):
        """限制单个持仓权重 - 温和方式"""
        max_weight = self.config.PORTFOLIO_CONFIG['max_weight']
        min_weight = self.config.PORTFOLIO_CONFIG['min_weight']
        
        # 温和地调整过大的权重，避免过度集中
        excess_weights = np.maximum(0, weights - max_weight)
        total_excess = np.sum(excess_weights)
        
        if total_excess > 0:
            # 限制最大权重
            constrained_weights = np.minimum(weights, max_weight)
            
            # 将超额权重均匀分配给所有股票（而不是只分配给有容量的股票）
            n_stocks = len(weights)
            additional_weight = total_excess / n_stocks
            
            # 确保分配后不会超过最大权重
            final_weights = constrained_weights + additional_weight
            final_weights = np.minimum(final_weights, max_weight)
            
            # 确保不会低于最小权重
            final_weights = np.maximum(final_weights, min_weight)
            
            return final_weights
        
        # 确保所有权重都不低于最小值
        return np.maximum(weights, min_weight)
    
    def _limit_sector_concentration(self, weights, symbols):
        """限制行业集中度（简化版本）"""
        # 简化的行业分类
        sector_mapping = self._get_simple_sector_mapping()
        
        # 计算每个行业的权重
        sector_weights = defaultdict(float)
        symbol_sectors = {}
        
        for i, symbol in enumerate(symbols):
            sector = sector_mapping.get(symbol, 'Other')
            sector_weights[sector] += weights[i]
            symbol_sectors[symbol] = sector
        
        # 检查是否有行业超过限制
        max_sector_weight = self.config.PORTFOLIO_CONFIG['sector_max_weight']
        
        for sector, total_weight in sector_weights.items():
            if total_weight > max_sector_weight:
                # 计算需要削减的权重
                excess = total_weight - max_sector_weight
                sector_symbols = [s for s in symbols if symbol_sectors[s] == sector]
                
                # 按当前权重比例削减
                for i, symbol in enumerate(symbols):
                    if symbol in sector_symbols:
                        reduction_ratio = excess / total_weight
                        weights[i] *= (1 - reduction_ratio)
        
        return weights
    
    def _get_simple_sector_mapping(self):
        """获取简化的行业分类"""
        # 这是一个简化的映射，实际应用中应该使用更准确的行业分类数据
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOG': 'Technology',
            'META': 'Technology',
            'TSLA': 'Technology',
            'NVDA': 'Technology',
            'AVGO': 'Technology',
            'INTC': 'Technology',
            'NFLX': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'SPY': 'ETF',
            'GLD': 'Commodities',
            'LLY': 'Healthcare'
        }
        return sector_mapping
    
    def _promote_diversification(self, weights, symbols):
        """促进多元化投资"""
        try:
            if len(weights) <= 1:
                return weights
            
            # 计算多元化指标
            n_stocks = len(weights)
            ideal_equal_weight = 1.0 / n_stocks
            diversification_pref = self.config.PORTFOLIO_CONFIG.get('diversification_preference', 0.2)
            
            # 混合当前权重和等权重，促进多元化
            diversified_weights = (1 - diversification_pref) * weights + diversification_pref * ideal_equal_weight
            
            return diversified_weights
            
        except Exception as e:
            self.algorithm.log_debug(f"多元化促进错误: {e}", log_type="risk")
            return weights
    
    def _gentle_renormalize(self, weights):
        """温和的权重归一化 - 保持相对权重分布"""
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            # 简单归一化，不添加额外的现金缓冲调整
            normalized_weights = weights / total_weight
            return normalized_weights
        
        return weights
    
    def _renormalize_weights(self, weights):
        """重新归一化权重"""
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            # 保留现金缓冲
            cash_buffer = self.config.PORTFOLIO_CONFIG['cash_buffer']
            target_investment_ratio = 1.0 - cash_buffer
            
            normalized_weights = weights * (target_investment_ratio / total_weight)
            return normalized_weights
        
        return weights

class DrawdownMonitor:
    """回撤监控器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.portfolio_values = []
        self.max_portfolio_value = 0
        
    def update_portfolio_value(self, current_value):
        """更新组合价值并计算回撤指标"""
        self.portfolio_values.append(current_value)
        
        if current_value > self.max_portfolio_value:
            self.max_portfolio_value = current_value
        
        # 计算当前回撤
        current_drawdown = (self.max_portfolio_value - current_value) / self.max_portfolio_value
        
        # 计算历史最大回撤
        max_drawdown = self._calculate_historical_max_drawdown()
        
        # 检查回撤警告
        drawdown_alert = self._check_drawdown_alert(current_drawdown)
        
        metrics = {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'portfolio_value': current_value,
            'peak_value': self.max_portfolio_value,
            'drawdown_alert': drawdown_alert
        }
        
        return metrics
    
    def _calculate_historical_max_drawdown(self):
        """计算历史最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        values = np.array(self.portfolio_values)
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _check_drawdown_alert(self, current_drawdown):
        """检查是否需要回撤警告"""
        max_allowed_drawdown = self.config.RISK_CONFIG['max_drawdown']
        return current_drawdown > max_allowed_drawdown

class VolatilityMonitor:
    """波动率监控器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.returns_history = []
        
    def update_return(self, daily_return):
        """更新日收益率并计算波动率指标"""
        self.returns_history.append(daily_return)
        
        # 保持最近252个交易日的数据
        if len(self.returns_history) > 252:
            self.returns_history = self.returns_history[-252:]
        
        if len(self.returns_history) < 20:
            return {'volatility': 0, 'volatility_alert': False}
        
        # 计算年化波动率
        returns_array = np.array(self.returns_history)
        volatility = np.std(returns_array) * np.sqrt(252)
        
        # 计算滚动波动率（最近20天）
        recent_volatility = np.std(returns_array[-20:]) * np.sqrt(252)
        
        # 波动率警告
        volatility_threshold = AlgorithmConfig.RISK_CONFIG['volatility_threshold']
        volatility_alert = recent_volatility > volatility_threshold
        
        return {
            'annual_volatility': volatility,
            'recent_volatility': recent_volatility,
            'volatility_alert': volatility_alert
        }

class VIXMonitor:
    """VIX波动率指数监控器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.vix_history = []  # VIX历史数据
        self.vix_symbol = "VIX"  # VIX符号
        self._last_vix_value = None
        self._defense_mode_active = False
        self._extreme_mode_active = False
        self._recovery_mode_active = False
        
        # 恢复机制状态跟踪
        self._recovery_start_time = None
        self._recovery_start_equity_ratio = 0.15  # 记录恢复开始时的股票仓位
        self._recovery_progress = 0.0  # 恢复进度 (0.0 到 1.0)
        self._previous_risk_level = 'normal'  # 记录前一个风险等级
        
        # 确保VIX数据已经添加到算法中
        try:
            self.algorithm.AddIndex(self.vix_symbol, Resolution.Daily)
            self.algorithm.log_debug("VIX指数已添加到数据源", log_type="risk")
        except Exception as e:
            self.algorithm.log_debug(f"添加VIX指数时出错: {e}", log_type="risk")
    
    def update_vix_data(self, current_time):
        """更新VIX数据并分析市场风险状态"""
        try:
            self.algorithm.log_debug(f"========== VIX监控更新开始 ({current_time}) ==========", log_type="risk")
            
            # 获取VIX历史数据
            vix_data = self._get_vix_historical_data()
            
            if vix_data is None or len(vix_data) < 2:
                self.algorithm.log_debug("警告: VIX数据不足，无法进行完整风险分析", log_type="risk")
                # 仍然要设置一个基础的VIX值，避免_last_vix_value为空
                self._last_vix_value = 20.0
                self.algorithm.log_debug(f"设置默认VIX值: {self._last_vix_value:.1f}", log_type="risk")
                return self._get_default_risk_state()
            
            # 确定数据来源类型
            data_source = "未知"
            if len(vix_data) >= 10 and all(isinstance(v, (int, float)) for v in vix_data):
                # 检查是否为真实数据（通常有更多变化）
                vix_std = np.std(vix_data) if len(vix_data) > 1 else 0
                if vix_std > 2.0:
                    data_source = "真实VIX数据"
                elif 10 <= min(vix_data) and max(vix_data) <= 60:
                    data_source = "估算/模拟数据"
                else:
                    data_source = "备用数据"
            
            self.algorithm.log_debug(f"VIX数据状态: 来源={data_source}, 长度={len(vix_data)}, 当前值={vix_data[-1]:.2f}", log_type="risk")
            
            # 更新VIX历史记录
            self._update_vix_history(vix_data)
            
            # 计算VIX变化率
            vix_change_rate = self._calculate_vix_change_rate()
            
            # 获取当前VIX水平
            current_vix = vix_data[-1] if len(vix_data) > 0 else 20.0
            
            # 分析风险状态
            risk_state = self._analyze_vix_risk_state(current_vix, vix_change_rate)
            
            # 记录VIX状态
            self._log_vix_status(current_vix, vix_change_rate, risk_state)
            
            self.algorithm.log_debug(f"========== VIX监控更新完成 ==========", log_type="risk")
            return risk_state
            
        except Exception as e:
            error_msg = f"严重错误: VIX监控更新异常 - {str(e)}"
            self.algorithm.log_debug(error_msg, log_type="risk")
            
            # 设置一个基础的VIX值，避免_last_vix_value为空
            self._last_vix_value = 20.0
            self.algorithm.log_debug(f"异常恢复: 设置默认VIX值={self._last_vix_value:.1f}", log_type="risk")
            return self._get_default_risk_state()
    
    def _get_vix_historical_data(self):
        """获取VIX历史数据"""
        try:
            lookback_days = self.config.RISK_CONFIG['vix_lookback_days']
            
            # 尝试获取VIX数据
            self.algorithm.log_debug(f"尝试从QuantConnect获取VIX数据，回看天数: {lookback_days}", log_type="risk")
            history = self.algorithm.History(self.vix_symbol, lookback_days, Resolution.Daily)
            history_list = list(history)
            
            self.algorithm.log_debug(f"[VIX数据获取] 返回数据长度: {len(history_list)}", log_type="risk")
            
            if len(history_list) == 0:
                # 明确报错：无法获取VIX数据
                error_msg = f"错误: 无法从QuantConnect获取VIX数据 (符号: {self.vix_symbol}, 回看: {lookback_days}天)"
                self.algorithm.log_debug(error_msg, log_type="risk")
                
                # 尝试使用SPY估算方法
                self.algorithm.log_debug("警告: 启用备用方案 - 使用SPY波动率估算VIX", log_type="risk")
                estimated_values = self._estimate_market_volatility()
                
                if estimated_values and len(estimated_values) > 0:
                    self.algorithm.log_debug(f"备用方案成功: SPY估算VIX值 = {estimated_values[-1]:.2f}", log_type="risk")
                    return estimated_values
                else:
                    self.algorithm.log_debug("警告: SPY估算方法也失败，使用时间周期默认值", log_type="risk")
                    return self._generate_default_vix_values()
            
            # 成功获取VIX数据
            vix_values = [x.Close for x in history_list]
            self.algorithm.log_debug(f"成功: 获取到真实VIX数据，最近5天: {[f'{v:.2f}' for v in vix_values[-5:]] if len(vix_values)>=5 else [f'{v:.2f}' for v in vix_values]}", log_type="risk")
            return vix_values
            
        except Exception as e:
            # 明确报错：VIX数据获取异常
            error_msg = f"严重错误: VIX数据获取异常 - {str(e)}"
            self.algorithm.log_debug(error_msg, log_type="risk")
            
            # 尝试备用方案
            self.algorithm.log_debug("启动异常恢复: 尝试SPY估算方法", log_type="risk")
            try:
                estimated_values = self._estimate_market_volatility()
                if estimated_values and len(estimated_values) > 0:
                    self.algorithm.log_debug(f"异常恢复成功: SPY估算VIX = {estimated_values[-1]:.2f}", log_type="risk")
                    return estimated_values
            except Exception as e2:
                self.algorithm.log_debug(f"异常恢复失败: SPY估算也出错 - {str(e2)}", log_type="risk")
            
            # 最后备用方案
            self.algorithm.log_debug("使用最后备用方案: 时间周期默认VIX值", log_type="risk")
            return self._generate_default_vix_values()
    
    def _estimate_market_volatility(self):
        """当无法获取VIX数据时，使用SPY等市场指数估算波动率"""
        try:
            self.algorithm.log_debug("开始SPY波动率估算: 获取SPY历史数据", log_type="risk")
            
            # 使用SPY作为市场代理计算隐含波动率
            spy_history = self.algorithm.History("SPY", 20, Resolution.Daily)
            spy_history_list = list(spy_history)
            
            self.algorithm.log_debug(f"SPY数据获取结果: 长度={len(spy_history_list)}", log_type="risk")
            
            if len(spy_history_list) < 10:
                error_msg = f"错误: SPY历史数据不足 (需要>=10天，实际{len(spy_history_list)}天)"
                self.algorithm.log_debug(error_msg, log_type="risk")
                return None
            
            # 计算波动率
            prices = np.array([x.Close for x in spy_history_list])
            returns = np.diff(prices) / prices[:-1]
            daily_vol = np.std(returns)
            
            # 将日波动率转换为VIX类似的数值（年化百分比）
            estimated_vix = daily_vol * np.sqrt(252) * 100
            
            self.algorithm.log_debug(f"SPY波动率计算: 日波动率={daily_vol:.4f}, 估算VIX={estimated_vix:.2f}", log_type="risk")
            
            # 生成最近几天的估算VIX数值
            base_vix = max(10, min(80, estimated_vix))  # 限制在合理范围内
            vix_estimates = []
            
            # 设置随机种子以确保可重现性
            np.random.seed(int(self.algorithm.Time.timestamp()))
            
            for i in range(min(10, len(spy_history_list))):
                # 添加一些随机波动来模拟VIX变化
                variation = np.random.normal(0, base_vix * 0.05)
                estimated_value = max(5, base_vix + variation)
                vix_estimates.append(estimated_value)
            
            self.algorithm.log_debug(f"SPY估算完成: 基准VIX={base_vix:.2f}, 序列长度={len(vix_estimates)}, 范围=[{min(vix_estimates):.1f}, {max(vix_estimates):.1f}]", log_type="risk")
            return vix_estimates
            
        except Exception as e:
            self.algorithm.log_debug(f"估算市场波动率出错: {e}", log_type="risk")
            return None
    
    def _generate_default_vix_values(self):
        """生成合理的默认VIX值序列，基于市场周期"""
        try:
            import random
            import math
            
            self.algorithm.log_debug("启动默认VIX生成: 基于时间周期的动态模拟", log_type="risk")
            
            # 基于日期和市场周期生成动态默认值
            current_time = self.algorithm.Time
            days_since_start = (current_time - self.algorithm.StartDate).days
            
            self.algorithm.log_debug(f"时间周期参数: 回测开始日={self.algorithm.StartDate.date()}, 当前日={current_time.date()}, 经过天数={days_since_start}", log_type="risk")
            
            # 创建基于时间的周期性VIX模拟
            base_vix = 18.0  # 长期VIX平均值
            
            # 添加季度性周期（每90天一个周期）
            quarterly_cycle = math.sin(2 * math.pi * days_since_start / 90) * 3
            
            # 添加年度性周期（每250天一个周期，对应交易日年）
            annual_cycle = math.sin(2 * math.pi * days_since_start / 250) * 5
            
            # 添加随机波动
            random.seed(days_since_start)  # 确保可重现性
            daily_noise = random.uniform(-2, 2)
            
            # 计算当前VIX估值
            current_vix = base_vix + quarterly_cycle + annual_cycle + daily_noise
            current_vix = max(10.0, min(60.0, current_vix))  # 限制在合理范围
            
            # 生成过去10天的VIX序列
            vix_sequence = []
            for i in range(10):
                day_offset = i - 9
                day_cycle = math.sin(2 * math.pi * (days_since_start + day_offset) / 90) * 3
                year_cycle = math.sin(2 * math.pi * (days_since_start + day_offset) / 250) * 5
                random.seed(days_since_start + day_offset)
                noise = random.uniform(-1.5, 1.5)
                
                day_vix = base_vix + day_cycle + year_cycle + noise
                day_vix = max(10.0, min(60.0, day_vix))
                vix_sequence.append(day_vix)
            
            self.algorithm.log_debug(f"默认VIX生成完成: 当前值={current_vix:.2f}, 序列长度={len(vix_sequence)}, 范围=[{min(vix_sequence):.1f}, {max(vix_sequence):.1f}]", log_type="risk")
            self.algorithm.log_debug(f"注意: 这是模拟VIX值，非真实市场数据", log_type="risk")
            
            return vix_sequence
            
        except Exception as e:
            error_msg = f"严重错误: 默认VIX生成失败 - {str(e)}"
            self.algorithm.log_debug(error_msg, log_type="risk")
            
            # 最后的备用方案：一个合理的固定序列
            fallback_sequence = [16.5, 17.2, 18.1, 19.3, 20.5, 19.8, 18.9, 17.6, 18.4, 19.1]
            self.algorithm.log_debug(f"使用固定备用序列: 长度={len(fallback_sequence)}, 当前值={fallback_sequence[-1]:.1f}", log_type="risk")
            return fallback_sequence
    
    def _update_vix_history(self, vix_data):
        """更新VIX历史记录"""
        max_history = self.config.RISK_CONFIG['vix_lookback_days'] * 2
        
        # 添加新数据
        for value in vix_data:
            if value not in self.vix_history:  # 避免重复
                self.vix_history.append(value)
        
        # 保持历史记录长度
        if len(self.vix_history) > max_history:
            self.vix_history = self.vix_history[-max_history:]
        
        # 更新最后的VIX值
        if len(vix_data) > 0:
            self._last_vix_value = vix_data[-1]
            self.algorithm.log_debug(f"VIX监控器更新: _last_vix_value = {self._last_vix_value:.2f}", log_type="risk")
    
    def _calculate_vix_change_rate(self):
        """计算VIX变化率"""
        if len(self.vix_history) < 2:
            return 0.0
        
        period = self.config.RISK_CONFIG['vix_rapid_rise_period']
        
        if len(self.vix_history) >= period:
            # 计算过去N天的变化率
            recent_vix = self.vix_history[-1]
            past_vix = self.vix_history[-period]
            
            if past_vix > 0:
                change_rate = (recent_vix - past_vix) / past_vix
                return change_rate
        
        # 如果历史数据不足，计算日变化率
        recent_vix = self.vix_history[-1]
        previous_vix = self.vix_history[-2]
        
        if previous_vix > 0:
            return (recent_vix - previous_vix) / previous_vix
        
        return 0.0
    
    def _analyze_vix_risk_state(self, current_vix, vix_change_rate):
        """分析VIX风险状态（增强版，包含恢复机制）"""
        risk_state = {
            'current_vix': current_vix,
            'vix_change_rate': vix_change_rate,
            'defense_mode': False,
            'extreme_mode': False,
            'recovery_mode': False,
            'hedge_allocation': 0.0,
            'max_equity_ratio': 1.0,
            'risk_level': 'normal',
            'recovery_progress': 0.0,
            'recovery_speed': 'normal'  # 'gradual' 或 'quick'
        }
        
        # 检查是否应该进入恢复模式
        rapid_decline_threshold = self.config.RISK_CONFIG['vix_rapid_decline_threshold']
        quick_recovery_threshold = self.config.RISK_CONFIG['vix_quick_recovery_threshold']
        
        # 优先检查恢复机制
        if (self._defense_mode_active or self._extreme_mode_active) and vix_change_rate < rapid_decline_threshold:
            # VIX快速下降，进入恢复模式
            risk_state = self._initiate_recovery_mode(risk_state, current_vix, vix_change_rate)
            
        elif self._recovery_mode_active:
            # 已经在恢复模式中，评估恢复进度
            risk_state = self._evaluate_recovery_progress(risk_state, current_vix, vix_change_rate)
            
        else:
            # 标准的VIX风险评估逻辑
            
            # 检查VIX快速上升
            rapid_rise_threshold = self.config.RISK_CONFIG['vix_rapid_rise_threshold']
            if vix_change_rate > rapid_rise_threshold:
                risk_state['defense_mode'] = True
                risk_state['risk_level'] = 'elevated'
                self.algorithm.Debug(f"VIX快速上升触发防御模式: {vix_change_rate:.2%}")
            
            # 检查VIX极端水平
            extreme_level = self.config.RISK_CONFIG['vix_extreme_level']
            if current_vix > extreme_level:
                risk_state['extreme_mode'] = True
                risk_state['defense_mode'] = True
                risk_state['risk_level'] = 'extreme'
                
                # 极端模式下的配置
                risk_state['max_equity_ratio'] = self.config.RISK_CONFIG['vix_defense_min_equity']
                risk_state['hedge_allocation'] = self.config.RISK_CONFIG['vix_extreme_hedging_allocation']
                
                cash_ratio = 1.0 - risk_state['max_equity_ratio']
                self.algorithm.Debug(f"VIX极端水平触发最小仓位模式: VIX={current_vix:.2f}")
                self.algorithm.Debug(f"极端防御模式: 股票仓位{risk_state['max_equity_ratio']:.1%}, 现金比例{cash_ratio:.1%}, 对冲分配{risk_state['hedge_allocation']:.1%}")
            
            elif risk_state['defense_mode']:
                # 防御模式但非极端情况
                equity_reduction = min(0.5, vix_change_rate * 2)  # 根据VIX变化率调整
                risk_state['max_equity_ratio'] = max(0.3, 1.0 - equity_reduction)
                risk_state['hedge_allocation'] = min(0.15, vix_change_rate)
            
            # 检查是否应该退出防御模式
            normalization_threshold = self.config.RISK_CONFIG['vix_normalization_threshold']
            if current_vix < normalization_threshold and vix_change_rate < 0.05:
                if self._defense_mode_active or self._extreme_mode_active:
                    self.algorithm.Debug(f"VIX正常化，退出防御模式: VIX={current_vix:.2f}")
                    risk_state['defense_mode'] = False
                    risk_state['extreme_mode'] = False
                    risk_state['risk_level'] = 'normal'
        
        # 更新内部状态
        self._previous_risk_level = risk_state['risk_level']
        self._defense_mode_active = risk_state['defense_mode']
        self._extreme_mode_active = risk_state['extreme_mode']
        self._recovery_mode_active = risk_state['recovery_mode']
        
        return risk_state
    
    def _get_default_risk_state(self):
        """获取默认风险状态（当VIX数据不可用时）"""
        return {
            'current_vix': 0,
            'vix_change_rate': 0,
            'defense_mode': False,
            'extreme_mode': False,
            'recovery_mode': False,
            'hedge_allocation': 0.0,
            'max_equity_ratio': 1.0,
            'risk_level': 'unknown',
            'recovery_progress': 0.0,
            'recovery_speed': 'normal'
        }
    
    def _log_vix_status(self, current_vix, vix_change_rate, risk_state):
        """记录VIX状态信息（增强版，包含恢复模式）"""
        risk_level = risk_state['risk_level']
        
        if risk_level != 'normal':
            self.algorithm.Debug(f"=== VIX风险监控 ===")
            self.algorithm.Debug(f"当前VIX: {current_vix:.2f}")
            self.algorithm.Debug(f"VIX变化率: {vix_change_rate:.2%}")
            self.algorithm.Debug(f"风险等级: {risk_level}")
            
            if risk_state.get('recovery_mode', False):
                self.algorithm.Debug(f"恢复模式: True")
                self.algorithm.Debug(f"恢复速度: {risk_state.get('recovery_speed', 'gradual')}")
                self.algorithm.Debug(f"恢复进度: {risk_state.get('recovery_progress', 0):.1%}")
            else:
                self.algorithm.Debug(f"防御模式: {risk_state['defense_mode']}")
                self.algorithm.Debug(f"极端模式: {risk_state['extreme_mode']}")
                
            self.algorithm.Debug(f"最大股票仓位: {risk_state['max_equity_ratio']:.1%}")
            self.algorithm.Debug(f"对冲分配: {risk_state['hedge_allocation']:.1%}")
    
    def get_hedging_recommendations(self, risk_state):
        """获取对冲建议"""
        if not risk_state['defense_mode']:
            return {}
        
        hedge_allocation = risk_state['hedge_allocation']
        if hedge_allocation <= 0:
            return {}
        
        # 推荐的对冲产品和比例
        hedging_instruments = {
            'SQQQ': hedge_allocation * 0.7,  # 反向纳斯达克ETF
            'SPXS': hedge_allocation * 0.3,  # 反向标普500ETF (如果可用)
        }
        
        return hedging_instruments
    
    def is_defense_mode_active(self):
        """检查是否处于防御模式"""
        return self._defense_mode_active
    
    def is_extreme_mode_active(self):
        """检查是否处于极端模式"""
        return self._extreme_mode_active
    
    def is_recovery_mode_active(self):
        """检查是否处于恢复模式"""
        return self._recovery_mode_active
    
    def _initiate_recovery_mode(self, risk_state, current_vix, vix_change_rate):
        """启动VIX恢复模式"""
        try:
            self.algorithm.Debug(f"=== 启动VIX恢复模式 ===")
            self.algorithm.Debug(f"当前VIX: {current_vix:.2f}, 变化率: {vix_change_rate:.2%}")
            
            # 设置恢复模式状态
            risk_state['recovery_mode'] = True
            risk_state['risk_level'] = 'recovery'
            
            # 记录恢复开始的状态
            if self._recovery_start_time is None:
                self._recovery_start_time = self.algorithm.Time
                if self._extreme_mode_active:
                    self._recovery_start_equity_ratio = self.config.RISK_CONFIG['vix_defense_min_equity']
                else:
                    self._recovery_start_equity_ratio = 0.4  # 防御模式的典型仓位
                
                self.algorithm.Debug(f"恢复起始股票仓位: {self._recovery_start_equity_ratio:.1%}")
            
            # 确定恢复速度
            quick_recovery_threshold = self.config.RISK_CONFIG['vix_quick_recovery_threshold']
            if current_vix <= quick_recovery_threshold:
                risk_state['recovery_speed'] = 'quick'
                risk_state = self._apply_quick_recovery(risk_state, current_vix)
            else:
                risk_state['recovery_speed'] = 'gradual'
                risk_state = self._apply_gradual_recovery(risk_state, current_vix, vix_change_rate)
            
            return risk_state
            
        except Exception as e:
            self.algorithm.Debug(f"启动恢复模式出错: {e}")
            return risk_state
    
    def _evaluate_recovery_progress(self, risk_state, current_vix, vix_change_rate):
        """评估恢复进度"""
        try:
            risk_state['recovery_mode'] = True
            risk_state['risk_level'] = 'recovery'
            
            # 检查是否应该切换到快速恢复
            quick_recovery_threshold = self.config.RISK_CONFIG['vix_quick_recovery_threshold']
            if current_vix <= quick_recovery_threshold and risk_state.get('recovery_speed') != 'quick':
                self.algorithm.Debug(f"VIX降至{current_vix:.2f}，切换到快速恢复模式")
                risk_state['recovery_speed'] = 'quick'
                return self._apply_quick_recovery(risk_state, current_vix)
            
            # 检查是否应该完全退出恢复模式
            normalization_threshold = self.config.RISK_CONFIG['vix_normalization_threshold']
            if current_vix < normalization_threshold and abs(vix_change_rate) < 0.05:
                self.algorithm.Debug(f"VIX完全正常化，退出恢复模式: VIX={current_vix:.2f}")
                return self._complete_recovery(risk_state)
            
            # 根据当前恢复速度模式继续恢复
            if risk_state.get('recovery_speed') == 'quick':
                return self._apply_quick_recovery(risk_state, current_vix)
            else:
                return self._apply_gradual_recovery(risk_state, current_vix, vix_change_rate)
                
        except Exception as e:
            self.algorithm.Debug(f"评估恢复进度出错: {e}")
            return risk_state
    
    def _apply_gradual_recovery(self, risk_state, current_vix, vix_change_rate):
        """应用逐步恢复策略"""
        try:
            # 基于VIX下降幅度和速度计算恢复幅度
            decline_magnitude = abs(vix_change_rate)  # VIX下降幅度
            step_size = self.config.RISK_CONFIG['vix_recovery_step_size']
            min_increment = self.config.RISK_CONFIG['vix_recovery_min_increment']
            max_equity = self.config.RISK_CONFIG['vix_recovery_max_equity']
            
            # 计算恢复增量（基于VIX下降速度）
            recovery_increment = max(min_increment, decline_magnitude * step_size)
            
            # 计算当前应有的股票仓位
            current_equity_ratio = self._recovery_start_equity_ratio + (self._recovery_progress * recovery_increment * 3)
            current_equity_ratio = min(current_equity_ratio, max_equity)
            
            # 更新恢复进度
            self._recovery_progress = min(1.0, self._recovery_progress + recovery_increment)
            
            # 设置风险状态
            risk_state['max_equity_ratio'] = current_equity_ratio
            risk_state['recovery_progress'] = self._recovery_progress
            
            # 逐步减少对冲仓位
            hedge_reduction = self.config.RISK_CONFIG['vix_recovery_hedge_reduction']
            # 根据恢复前的状态确定基础对冲分配
            if self._extreme_mode_active:
                base_hedge = self.config.RISK_CONFIG['vix_extreme_hedging_allocation']
            else:
                base_hedge = self.config.RISK_CONFIG['vix_hedging_allocation']
            risk_state['hedge_allocation'] = max(0, base_hedge * (1 - self._recovery_progress * hedge_reduction))
            
            self.algorithm.Debug(f"逐步恢复 - 股票仓位: {current_equity_ratio:.1%}, 进度: {self._recovery_progress:.1%}, 对冲: {risk_state['hedge_allocation']:.1%}")
            
            return risk_state
            
        except Exception as e:
            self.algorithm.Debug(f"逐步恢复出错: {e}")
            return risk_state
    
    def _apply_quick_recovery(self, risk_state, current_vix):
        """应用快速恢复策略"""
        try:
            # 快速恢复到接近正常仓位
            max_equity = self.config.RISK_CONFIG['vix_recovery_max_equity']
            quick_recovery_ratio = 0.85  # 快速恢复到85%股票仓位
            
            risk_state['max_equity_ratio'] = min(max_equity, quick_recovery_ratio)
            risk_state['hedge_allocation'] = 0.05  # 保留少量对冲
            risk_state['recovery_progress'] = 0.9  # 快速恢复进度设为90%
            
            self.algorithm.Debug(f"快速恢复 - VIX降至{current_vix:.2f}，股票仓位恢复到{risk_state['max_equity_ratio']:.1%}")
            
            return risk_state
            
        except Exception as e:
            self.algorithm.Debug(f"快速恢复出错: {e}")
            return risk_state
    
    def _complete_recovery(self, risk_state):
        """完成恢复，返回正常模式"""
        try:
            self.algorithm.Debug(f"=== 完成VIX恢复，回到正常模式 ===")
            
            # 重置所有恢复状态
            risk_state['recovery_mode'] = False
            risk_state['defense_mode'] = False
            risk_state['extreme_mode'] = False
            risk_state['risk_level'] = 'normal'
            risk_state['max_equity_ratio'] = 1.0
            risk_state['hedge_allocation'] = 0.0
            risk_state['recovery_progress'] = 1.0
            risk_state['recovery_speed'] = 'normal'
            
            # 重置内部状态
            self._recovery_start_time = None
            self._recovery_progress = 0.0
            
            return risk_state
            
        except Exception as e:
            self.algorithm.Debug(f"完成恢复出错: {e}")
            return risk_state

class HedgingManager:
    """对冲产品管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.hedging_symbols = ['SQQQ']  # 主要使用SQQQ进行对冲
        
        # 初始化对冲产品数据
        for symbol in self.hedging_symbols:
            try:
                self.algorithm.AddEquity(symbol, Resolution.Daily)
                self.algorithm.Debug(f"对冲产品 {symbol} 已添加到数据源")
            except Exception as e:
                self.algorithm.Debug(f"添加对冲产品 {symbol} 时出错: {e}")
    
    def calculate_hedging_allocation(self, vix_risk_state, regular_symbols):
        """计算对冲产品的分配（支持恢复模式）"""
        if not vix_risk_state:
            return {}
        
        # 恢复模式下也需要对冲分配
        if not (vix_risk_state['defense_mode'] or vix_risk_state.get('recovery_mode', False)):
            return {}
        
        try:
            hedge_allocation = vix_risk_state.get('hedge_allocation', 0)
            if hedge_allocation <= 0:
                return {}
            
            # 检查对冲产品可用性
            available_hedging_symbols = []
            for symbol in self.hedging_symbols:
                if self._is_hedging_symbol_tradable(symbol):
                    available_hedging_symbols.append(symbol)
            
            if not available_hedging_symbols:
                self.algorithm.Debug("没有可用的对冲产品")
                return {}
            
            # 分配对冲比例
            hedging_weights = {}
            if 'SQQQ' in available_hedging_symbols:
                hedging_weights['SQQQ'] = hedge_allocation
                
                # 根据模式记录不同的日志
                if vix_risk_state.get('recovery_mode', False):
                    recovery_speed = vix_risk_state.get('recovery_speed', 'gradual')
                    self.algorithm.Debug(f"恢复模式({recovery_speed})下SQQQ对冲比例: {hedge_allocation:.1%}")
                else:
                    self.algorithm.Debug(f"防御模式下SQQQ对冲比例: {hedge_allocation:.1%}")
            
            return hedging_weights
            
        except Exception as e:
            self.algorithm.Debug(f"计算对冲分配出错: {e}")
            return {}
    
    def _is_hedging_symbol_tradable(self, symbol):
        """检查对冲产品是否可交易"""
        try:
            # 检查是否有当前价格数据
            if hasattr(self.algorithm, 'data_slice') and self.algorithm.data_slice:
                symbol_obj = Symbol.Create(symbol, SecurityType.Equity, Market.USA)
                if self.algorithm.data_slice.ContainsKey(symbol_obj):
                    symbol_data = self.algorithm.data_slice[symbol_obj]
                    if symbol_data is not None and hasattr(symbol_data, 'Price'):
                        current_price = symbol_data.Price
                        if current_price > 0:
                            return True
            
            # 备用检查：使用历史数据
            history = self.algorithm.History(symbol, 2, Resolution.Daily)
            history_list = list(history)
            if len(history_list) > 0:
                latest_price = history_list[-1].Close
                return latest_price > 0
            
            return False
            
        except Exception as e:
            self.algorithm.Debug(f"检查对冲产品 {symbol} 可交易性出错: {e}")
            return False
    
    def integrate_hedging_with_portfolio(self, regular_weights, regular_symbols, vix_risk_state):
        """将对冲产品与常规投资组合整合"""
        try:
            # 计算对冲分配
            hedging_weights = self.calculate_hedging_allocation(vix_risk_state, regular_symbols)
            
            if not hedging_weights:
                return regular_weights, regular_symbols
            
            total_hedge_weight = sum(hedging_weights.values())
            
            # 调整常规投资组合权重
            remaining_weight = 1.0 - total_hedge_weight
            if remaining_weight <= 0:
                self.algorithm.Debug("对冲分配过多，使用最小常规仓位")
                remaining_weight = 0.1
                total_hedge_weight = 0.9
                # 重新调整对冲权重
                adjustment_factor = 0.9 / sum(hedging_weights.values())
                hedging_weights = {k: v * adjustment_factor for k, v in hedging_weights.items()}
            
            adjusted_regular_weights = regular_weights * remaining_weight
            
            # 整合权重和符号
            all_weights = np.concatenate([adjusted_regular_weights, 
                                        np.array(list(hedging_weights.values()))])
            
            # 转换对冲符号为算法可用的Symbol对象
            hedging_symbols = []
            for symbol_str in hedging_weights.keys():
                try:
                    symbol_obj = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
                    hedging_symbols.append(symbol_obj)
                except Exception as e:
                    self.algorithm.Debug(f"创建对冲符号 {symbol_str} 出错: {e}")
                    # 如果创建失败，尝试直接使用字符串
                    hedging_symbols.append(symbol_str)
            
            all_symbols = list(regular_symbols) + hedging_symbols
            
            self.algorithm.Debug(f"整合对冲产品后的投资组合:")
            self.algorithm.Debug(f"  常规投资比例: {remaining_weight:.1%}")
            self.algorithm.Debug(f"  对冲产品比例: {total_hedge_weight:.1%}")
            self.algorithm.Debug(f"  对冲产品: {list(hedging_weights.keys())}")
            
            return all_weights, all_symbols
            
        except Exception as e:
            self.algorithm.Debug(f"整合对冲产品出错: {e}")
            return regular_weights, regular_symbols
    
    def get_hedging_status(self):
        """获取当前对冲状态"""
        try:
            hedging_status = {}
            for symbol in self.hedging_symbols:
                if hasattr(self.algorithm.Portfolio, symbol):
                    portfolio_item = getattr(self.algorithm.Portfolio, symbol)
                    hedging_status[symbol] = {
                        'quantity': portfolio_item.Quantity,
                        'holdings_value': portfolio_item.HoldingsValue,
                        'weight': portfolio_item.HoldingsValue / self.algorithm.Portfolio.TotalPortfolioValue if self.algorithm.Portfolio.TotalPortfolioValue > 0 else 0
                    }
            
            return hedging_status
            
        except Exception as e:
            self.algorithm.Debug(f"获取对冲状态出错: {e}")
            return {}
    
    def log_hedging_summary(self, vix_risk_state):
        """记录对冲情况摘要"""
        if not vix_risk_state or not vix_risk_state['defense_mode']:
            return
        
        try:
            hedging_status = self.get_hedging_status()
            if hedging_status:
                self.algorithm.Debug("=== 对冲产品状态 ===")
                for symbol, status in hedging_status.items():
                    self.algorithm.Debug(f"{symbol}: 数量={status['quantity']}, 价值=${status['holdings_value']:.0f}, 权重={status['weight']:.1%}")
            
        except Exception as e:
            self.algorithm.Debug(f"记录对冲摘要出错: {e}") 
