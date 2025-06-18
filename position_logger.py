# 仓位调整日志记录模块
from AlgorithmImports import *
import json
from datetime import datetime

class PositionLogger:
    """仓位调整日志记录器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.position_history = []
        self.rebalance_sequence = 0
        
    def log_rebalance_start(self, current_time):
        """记录调仓开始"""
        self.rebalance_sequence += 1
        self.current_rebalance = {
            'sequence': self.rebalance_sequence,
            'timestamp': current_time,
            'start_portfolio_value': float(self.algorithm.Portfolio.TotalPortfolioValue),
            'start_cash': float(self.algorithm.Portfolio.Cash),
            'start_invested': float(self.algorithm.Portfolio.TotalHoldingsValue),
            'start_position_count': len([h for h in self.algorithm.Portfolio.Values if h.Quantity != 0]),
            'decisions': {},
            'risk_analysis': {},
            'optimization_result': {},
            'execution_result': {}
        }
        
        self.algorithm.log_debug(f"=== 调仓开始 #{self.rebalance_sequence} ===", log_type="portfolio")
        self.algorithm.log_debug(f"时间: {current_time}", log_type="portfolio")
        self.algorithm.log_debug(f"组合总值: ${self.current_rebalance['start_portfolio_value']:,.2f}", log_type="portfolio")
        self.algorithm.log_debug(f"现金: ${self.current_rebalance['start_cash']:,.2f}", log_type="portfolio")
        self.algorithm.log_debug(f"已投资: ${self.current_rebalance['start_invested']:,.2f}", log_type="portfolio")
        self.algorithm.log_debug(f"持仓数量: {self.current_rebalance['start_position_count']}", log_type="portfolio")
        
    def log_current_positions(self):
        """记录当前持仓情况"""
        current_positions = []
        total_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        
        self.algorithm.log_debug("=== 当前持仓分析 ===", log_type="portfolio")
        
        for kvp in self.algorithm.Portfolio:
            symbol = kvp.Key
            holding = kvp.Value
            if holding.Quantity != 0:
                position_value = float(holding.HoldingsValue)
                position_weight = position_value / total_value if total_value > 0 else 0
                
                position_info = {
                    'symbol': str(symbol),
                    'quantity': float(holding.Quantity),
                    'avg_price': float(holding.AveragePrice),
                    'current_price': float(holding.Price),
                    'value': position_value,
                    'weight': position_weight,
                    'unrealized_pnl': float(holding.UnrealizedProfit),
                    'pnl_percent': float(holding.UnrealizedProfitPercent)
                }
                current_positions.append(position_info)
                
                self.algorithm.log_debug(f"{symbol}: 数量={holding.Quantity:,.0f}, "
                                   f"价值=${position_value:,.2f} ({position_weight:.2%}), "
                                   f"盈亏=${holding.UnrealizedProfit:,.2f} ({holding.UnrealizedProfitPercent:.2%})", log_type="portfolio")
        
        if not current_positions:
            self.algorithm.log_debug("当前无持仓", log_type="portfolio")
        
        self.current_rebalance['current_positions'] = current_positions
        return current_positions
    
    def log_risk_analysis(self, vix_risk_state, symbols_before_risk, symbols_after_risk, 
                         expected_returns_before, expected_returns_after):
        """记录风险分析结果"""
        self.algorithm.log_debug("=== 风险控制分析 ===", log_type="risk")
        
        risk_analysis = {
            'vix_risk_state': vix_risk_state,
            'symbols_filtered': {
                'before_count': len(symbols_before_risk),
                'after_count': len(symbols_after_risk),
                'filtered_out': len(symbols_before_risk) - len(symbols_after_risk),
                'filtered_symbols': [str(s) for s in symbols_before_risk if s not in symbols_after_risk]
            },
            'expected_returns_adjustment': {}
        }
        
        # VIX风险状态分析
        self.algorithm.log_debug(f"VIX风险状态: {vix_risk_state.get('risk_level', 'unknown')}", log_type="risk")
        self.algorithm.log_debug(f"VIX当前值: {vix_risk_state.get('current_vix', 'unknown')}", log_type="risk")
        self.algorithm.log_debug(f"VIX变化率: {vix_risk_state.get('vix_change_rate', 'unknown')}", log_type="risk")
        
        # 符号筛选分析
        self.algorithm.log_debug(f"股票筛选: {len(symbols_before_risk)} -> {len(symbols_after_risk)}", log_type="risk")
        if risk_analysis['symbols_filtered']['filtered_symbols']:
            self.algorithm.log_debug(f"被过滤的股票: {risk_analysis['symbols_filtered']['filtered_symbols']}", log_type="risk")
        
        # 预期收益调整分析
        if len(symbols_after_risk) > 0 and expected_returns_before is not None and expected_returns_after is not None:
            for i, symbol in enumerate(symbols_after_risk):
                if i < len(expected_returns_before) and i < len(expected_returns_after):
                    before_return = expected_returns_before[i] if hasattr(expected_returns_before, '__getitem__') else expected_returns_before
                    after_return = expected_returns_after[i] if hasattr(expected_returns_after, '__getitem__') else expected_returns_after
                    adjustment = after_return - before_return
                    
                    risk_analysis['expected_returns_adjustment'][str(symbol)] = {
                        'before': float(before_return),
                        'after': float(after_return),
                        'adjustment': float(adjustment)
                    }
                    
                    if abs(adjustment) > 0.001:  # 只记录有显著调整的
                        self.algorithm.log_debug(f"{symbol} 预期收益调整: {before_return:.4f} -> {after_return:.4f} (调整: {adjustment:+.4f})", log_type="risk")
        
        self.current_rebalance['risk_analysis'] = risk_analysis
    
    def log_optimization_result(self, target_weights, target_symbols, equity_ratio, 
                              optimization_method=None, panic_score=None):
        """记录优化结果"""
        self.algorithm.log_debug("=== 投资组合优化结果 ===", log_type="portfolio")
        
        optimization_result = {
            'method': optimization_method or 'unknown',
            'equity_ratio': float(equity_ratio) if equity_ratio is not None else 0.0,
            'panic_score': panic_score,
            'target_positions': [],
            'position_concentration': {},
            'target_symbols_count': len(target_symbols) if target_symbols else 0
        }
        
        self.algorithm.log_debug(f"优化方法: {optimization_result['method']}", log_type="portfolio")
        self.algorithm.log_debug(f"股票仓位比例: {optimization_result['equity_ratio']:.2%}", log_type="portfolio")
        if panic_score is not None:
            self.algorithm.log_debug(f"恐慌评分: {panic_score}", log_type="portfolio")
        
        # 目标仓位分析 - 修复：增强仓位一致性检查
        total_portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        invested_amount = total_portfolio_value * optimization_result['equity_ratio']
        
        self.algorithm.log_debug(f"目标股票投资额: ${invested_amount:,.2f}", log_type="portfolio")
        self.algorithm.log_debug(f"目标持仓数量: {optimization_result['target_symbols_count']}", log_type="portfolio")
        
        # 仓位一致性检查
        if optimization_result['equity_ratio'] == 0.0 and target_weights is not None and len(target_weights) > 0:
            weights_sum = sum(target_weights) if target_weights else 0
            self.algorithm.log_debug(f"⚠️ 仓位不一致警告: equity_ratio=0% 但存在目标权重 (总和={weights_sum:.4f})", log_type="portfolio")
        elif optimization_result['equity_ratio'] > 0.0 and (target_weights is None or len(target_weights) == 0):
            self.algorithm.log_debug(f"⚠️ 仓位不一致警告: equity_ratio={optimization_result['equity_ratio']:.2%} 但无目标权重", log_type="portfolio")
        
        if target_weights is not None and len(target_weights) > 0 and target_symbols:
            max_weight = 0
            min_weight = float('inf')
            weights_sum = 0
            
            for i, (symbol, weight) in enumerate(zip(target_symbols, target_weights)):
                if i < len(target_weights):
                    target_value = invested_amount * weight
                    position_info = {
                        'symbol': str(symbol),
                        'weight': float(weight),
                        'target_value': float(target_value)
                    }
                    optimization_result['target_positions'].append(position_info)
                    
                    max_weight = max(max_weight, weight)
                    min_weight = min(min_weight, weight)
                    weights_sum += weight
                    
                    self.algorithm.log_debug(f"{symbol}: 权重={weight:.2%}, 目标价值=${target_value:,.2f}", log_type="portfolio")
            
            # 权重分布统计
            optimization_result['position_concentration'] = {
                'max_weight': float(max_weight),
                'min_weight': float(min_weight),
                'weights_sum': float(weights_sum),
                'average_weight': float(weights_sum / len(target_weights)) if len(target_weights) > 0 else 0
            }
            
            self.algorithm.log_debug(f"权重分布: 最大={max_weight:.2%}, 最小={min_weight:.2%}, "
                               f"平均={optimization_result['position_concentration']['average_weight']:.2%}", log_type="portfolio")
            self.algorithm.log_debug(f"权重总和: {weights_sum:.4f}", log_type="portfolio")
        else:
            self.algorithm.log_debug("无目标仓位 - 全现金策略", log_type="portfolio")
        
        self.current_rebalance['optimization_result'] = optimization_result
    
    def log_position_changes(self, trades_executed):
        """记录仓位变化"""
        self.algorithm.log_debug("=== 仓位变化执行 ===", log_type="portfolio")
        
        execution_result = {
            'trades_count': len(trades_executed) if trades_executed else 0,
            'trades_executed': [],
            'total_trade_value': 0,
            'position_changes': [],
            'skipped_small_trades': 0,
            'estimated_transaction_cost': 0
        }
        
        if trades_executed:
            for trade in trades_executed:
                # 兼容新旧交易数据格式
                trade_value = trade.get('trade_value', trade.get('value', 0))
                price = trade.get('price', 0)
                
                trade_info = {
                    'symbol': str(trade.get('symbol', 'unknown')),
                    'action': trade.get('action', 'unknown'),
                    'quantity': float(trade.get('quantity', 0)),
                    'price': float(price),
                    'value': float(trade_value),
                    'reason': trade.get('reason', ''),
                    'target_weight': float(trade.get('target_weight', 0))
                }
                execution_result['trades_executed'].append(trade_info)
                execution_result['total_trade_value'] += abs(trade_info['value'])
                
                # 根据动作显示中文
                action_text_map = {
                    'BUY': '买入',
                    'SELL': '卖出', 
                    'LIQUIDATE': '清仓'
                }
                action_text = action_text_map.get(trade_info['action'], trade_info['action'])
                
                self.algorithm.log_debug(f"{action_text} {trade_info['symbol']}: "
                                   f"数量={trade_info['quantity']:,.0f}, "
                                   f"价格=${trade_info['price']:.2f}, "
                                   f"价值=${trade_info['value']:,.2f}", log_type="portfolio")
                
                if trade_info['target_weight'] > 0:
                    self.algorithm.log_debug(f"  目标权重: {trade_info['target_weight']:.2%}", log_type="portfolio")
                
                if trade_info['reason']:
                    self.algorithm.log_debug(f"  原因: {trade_info['reason']}", log_type="portfolio")
        else:
            self.algorithm.log_debug("无交易执行", log_type="portfolio")
        
        # 计算预估交易成本
        execution_result['estimated_transaction_cost'] = execution_result['total_trade_value'] * 0.001
        
        self.algorithm.log_debug(f"总交易价值: ${execution_result['total_trade_value']:,.2f}", log_type="portfolio")
        self.algorithm.log_debug(f"预估交易成本: ${execution_result['estimated_transaction_cost']:.2f} (0.1%)", log_type="portfolio")
        
        # 检查是否有小额交易被跳过的情况
        min_trade_value = getattr(self.algorithm.config, 'PORTFOLIO_CONFIG', {}).get('min_trade_value', 2000)
        if execution_result['total_trade_value'] > 0:
            cost_ratio = execution_result['estimated_transaction_cost'] / execution_result['total_trade_value']
            self.algorithm.log_debug(f"交易成本比例: {cost_ratio:.3%} (最小交易金额: ${min_trade_value:,.0f})", log_type="portfolio")
        
        self.current_rebalance['execution_result'] = execution_result
    
    def log_rebalance_summary(self):
        """记录调仓总结"""
        end_portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        end_cash = float(self.algorithm.Portfolio.Cash)
        end_invested = float(self.algorithm.Portfolio.TotalHoldingsValue)
        end_position_count = len([h for h in self.algorithm.Portfolio.Values if h.Quantity != 0])
        
        self.current_rebalance.update({
            'end_portfolio_value': end_portfolio_value,
            'end_cash': end_cash,
            'end_invested': end_invested,
            'end_position_count': end_position_count,
            'portfolio_value_change': end_portfolio_value - self.current_rebalance['start_portfolio_value'],
            'cash_change': end_cash - self.current_rebalance['start_cash'],
            'invested_change': end_invested - self.current_rebalance['start_invested']
        })
        
        self.algorithm.log_debug("=== 调仓总结 ===", log_type="portfolio")
        self.algorithm.log_debug(f"组合价值变化: ${self.current_rebalance['start_portfolio_value']:,.2f} -> "
                           f"${end_portfolio_value:,.2f} "
                           f"({self.current_rebalance['portfolio_value_change']:+,.2f})", log_type="portfolio")
        self.algorithm.log_debug(f"现金变化: ${self.current_rebalance['start_cash']:,.2f} -> "
                           f"${end_cash:,.2f} "
                           f"({self.current_rebalance['cash_change']:+,.2f})", log_type="portfolio")
        self.algorithm.log_debug(f"投资额变化: ${self.current_rebalance['start_invested']:,.2f} -> "
                           f"${end_invested:,.2f} "
                           f"({self.current_rebalance['invested_change']:+,.2f})", log_type="portfolio")
        self.algorithm.log_debug(f"持仓数量变化: {self.current_rebalance['start_position_count']} -> {end_position_count}", log_type="portfolio")
        
        # 实际股票仓位比例
        actual_equity_ratio = end_invested / end_portfolio_value if end_portfolio_value > 0 else 0
        target_equity_ratio = self.current_rebalance.get('optimization_result', {}).get('equity_ratio', 0)
        
        self.algorithm.log_debug(f"实际股票仓位: {actual_equity_ratio:.2%}", log_type="portfolio")
        if target_equity_ratio > 0:
            ratio_diff = actual_equity_ratio - target_equity_ratio
            self.algorithm.log_debug(f"目标仓位差异: {ratio_diff:+.2%}", log_type="portfolio")
        
        # 保存调仓记录
        self.position_history.append(self.current_rebalance.copy())
        
        # 限制历史记录长度
        if len(self.position_history) > 50:
            self.position_history = self.position_history[-50:]
        
        self.algorithm.log_debug(f"=== 调仓完成 #{self.rebalance_sequence} ===", log_type="portfolio")
    
    def log_position_decision_factors(self, decision_factors):
        """记录仓位决策因素"""
        self.algorithm.log_debug("=== 仓位决策因素分析 ===", log_type="portfolio")
        
        for factor_name, factor_data in decision_factors.items():
            self.algorithm.log_debug(f"{factor_name}:", log_type="portfolio")
            if isinstance(factor_data, dict):
                for key, value in factor_data.items():
                    if isinstance(value, float):
                        self.algorithm.log_debug(f"  {key}: {value:.4f}", log_type="portfolio")
                    else:
                        self.algorithm.log_debug(f"  {key}: {value}", log_type="portfolio")
            else:
                self.algorithm.log_debug(f"  {factor_data}", log_type="portfolio")
        
        self.current_rebalance['decision_factors'] = decision_factors
    
    def get_position_statistics(self):
        """获取仓位统计信息"""
        if not self.position_history:
            return {}
        
        recent_rebalances = self.position_history[-10:]  # 最近10次调仓
        
        avg_equity_ratio = sum(r.get('optimization_result', {}).get('equity_ratio', 0) 
                              for r in recent_rebalances) / len(recent_rebalances)
        
        avg_position_count = sum(r.get('end_position_count', 0) 
                               for r in recent_rebalances) / len(recent_rebalances)
        
        total_trades = sum(r.get('execution_result', {}).get('trades_count', 0) 
                          for r in recent_rebalances)
        
        stats = {
            'total_rebalances': len(self.position_history),
            'recent_avg_equity_ratio': avg_equity_ratio,
            'recent_avg_position_count': avg_position_count,
            'recent_total_trades': total_trades,
            'recent_rebalances_count': len(recent_rebalances)
        }
        
        return stats
    
    def log_daily_position_summary(self):
        """记录每日仓位摘要"""
        portfolio_value = float(self.algorithm.Portfolio.TotalPortfolioValue)
        cash = float(self.algorithm.Portfolio.Cash)
        invested = float(self.algorithm.Portfolio.TotalHoldingsValue)
        equity_ratio = invested / portfolio_value if portfolio_value > 0 else 0
        position_count = len([h for h in self.algorithm.Portfolio.Values if h.Quantity != 0])
        
        self.algorithm.log_debug(f"每日仓位摘要 - 总值: ${portfolio_value:,.2f}, "
                           f"现金: ${cash:,.2f}, 股票: ${invested:,.2f} ({equity_ratio:.1%}), "
                           f"持仓数: {position_count}", log_type="portfolio")
    
    def log_daily_report(self):
        """每日投资组合报告，严格按DAILY_REPORT_CONFIG控制内容"""
        cfg = getattr(self.algorithm.config, 'LOGGING_CONFIG', {})
        daily_cfg = cfg.get('DAILY_REPORT_CONFIG', {})
        portfolio = self.algorithm.Portfolio
        # 1. 组合总览
        if daily_cfg.get('enable_portfolio_overview', True):
            total_value = float(portfolio.TotalPortfolioValue)
            cash = float(portfolio.Cash)
            invested = float(portfolio.TotalHoldingsValue)
            equity_ratio = invested / total_value if total_value > 0 else 0
            self.algorithm.log_debug(f"[每日报告] 组合总值: ${total_value:,.2f}, 现金: ${cash:,.2f}, 股票: ${invested:,.2f} ({equity_ratio:.1%})", log_type="portfolio")
        # 2. 持仓明细
        if daily_cfg.get('enable_holdings_details', True):
            holdings = [h for h in portfolio.Values if h.Quantity != 0]
            max_n = daily_cfg.get('max_holdings_display', 10)
            holdings = sorted(holdings, key=lambda h: -abs(h.Quantity * h.Price))[:max_n]
            for h in holdings:
                value = h.Quantity * h.Price
                self.algorithm.log_debug(f"[每日报告] 持仓: {h.Symbol} 数量={h.Quantity}, 价值=${value:,.2f}", log_type="portfolio")
        # 3. 业绩分析
        if daily_cfg.get('enable_performance_analysis', True):
            # 这里只做简单收益率统计
            returns = getattr(self.algorithm, 'performance_metrics', {}).get('daily_returns', [])
            if returns:
                avg_return = sum(returns) / len(returns)
                self.algorithm.log_debug(f"[每日报告] 近{len(returns)}日平均日收益率: {avg_return:.3%}", log_type="portfolio")
        # 4. 成交量分析 - 修复：优化QuantConnect API使用和数据结构解析
        if daily_cfg.get('enable_volume_analysis', True):
            try:
                symbols = [h.Symbol for h in portfolio.Values if h.Quantity != 0]
                if symbols:
                    max_n = daily_cfg.get('max_volume_display', 5)
                    symbols_to_check = symbols[:max_n]
                    
                    # 使用安全获取方法获取成交量数据
                    for symbol in symbols_to_check:
                        volume, source = self.get_symbol_volume_safe(symbol)
                        if volume is not None:
                            self.algorithm.log_debug(f"[每日报告] {symbol} 成交量: {volume:,.0f} (来源: {source})", log_type="portfolio")
                        else:
                            self.algorithm.log_debug(f"[每日报告] {symbol} 成交量获取失败: {source}", log_type="portfolio")
                else:
                    self.algorithm.log_debug("[每日报告] 无持仓股票，跳过成交量分析", log_type="portfolio")
            except Exception as e:
                self.algorithm.log_debug(f"[每日报告] 成交量分析出错: {str(e)}", log_type="portfolio")
        # 5. 预测分析
        if daily_cfg.get('enable_prediction_analysis', True):
            # 若有预测结果缓存，可输出部分预测
            preds = getattr(self.algorithm, 'last_predictions', None)
            if preds and isinstance(preds, dict):
                for i, (symbol, pred) in enumerate(preds.items()):
                    if i >= daily_cfg.get('max_holdings_display', 10):
                        break
                    self.algorithm.log_debug(f"[每日报告] 预测: {symbol} -> {pred}", log_type="prediction")

    def get_symbol_volume_safe(self, symbol):
        """安全获取股票成交量数据 - 新增：解决成交量获取问题"""
        try:
            # 方法1：尝试获取TradeBar历史数据
            try:
                trade_bars = self.algorithm.History(symbol, 1, Resolution.Daily)
                if trade_bars is not None and not trade_bars.empty and 'volume' in trade_bars.columns:
                    volume = trade_bars['volume'].iloc[-1]
                    if volume > 0:
                        return volume, "历史数据"
            except Exception:
                pass
            
            # 方法2：尝试获取Security对象的当前数据
            try:
                if symbol in self.algorithm.Securities:
                    security = self.algorithm.Securities[symbol]
                    if hasattr(security, 'Volume'):
                        volume = float(security.Volume)
                        if volume > 0:
                            return volume, "实时数据"
                    
                    # 尝试获取最新的TradeBar
                    if hasattr(security, 'GetLastData'):
                        last_data = security.GetLastData()
                        if last_data and hasattr(last_data, 'Volume'):
                            volume = float(last_data.Volume)
                            if volume > 0:
                                return volume, "最新数据"
            except Exception:
                pass
            
            # 方法3：尝试使用明确的TradeBar类型请求
            try:
                from QuantConnect.Data.Market import TradeBar
                bars = self.algorithm.History(TradeBar, symbol, 1, Resolution.Daily)
                if bars and len(bars) > 0:
                    latest_bar = bars[-1]
                    if hasattr(latest_bar, 'Volume'):
                        volume = float(latest_bar.Volume)
                        if volume > 0:
                            return volume, "TradeBar数据"
            except Exception:
                pass
            
            return None, "所有方法均失败"
            
        except Exception as e:
            return None, f"获取失败: {str(e)}" 