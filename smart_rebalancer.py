# 智能再平衡模块
from AlgorithmImports import *
import numpy as np
from config import AlgorithmConfig

class SmartRebalancer:
    """智能再平衡器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self._last_trades_executed = []
        
    def execute_smart_rebalance(self, target_weights, 
                              symbols):
        """执行智能再平衡"""
        try:
            if not self._ensure_current_data_available():
                self.algorithm.log_debug("当前数据不可用，跳过再平衡")
                return False
                
            # 获取当前持仓
            current_holdings = self._get_current_holdings()
            
            # 计算目标持仓
            target_holdings = self._calculate_target_holdings(target_weights, 
                                                            symbols)
            
            # 生成交易指令
            trades = self._generate_trade_instructions(current_holdings, 
                                                     target_holdings)
            
            if not trades:
                self.algorithm.log_debug("无需进行交易")
                return True
                
            # 执行交易
            success = self._execute_trades(trades)
            
            if success:
                self._log_rebalance_summary(trades)
                
            return success
            
        except Exception as e:
            self.algorithm.log_debug(f"智能再平衡失败: {str(e)}")
            return False
    
    def _ensure_current_data_available(self):
        """确保当前数据可用"""
        try:
            # 检查是否有足够的现金或持仓
            total_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
            if total_portfolio_value <= 0:
                self.algorithm.log_debug("投资组合价值为零或负数")
                return False
                
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"数据可用性检查失败: {str(e)}")
            return False
    
    def _get_current_holdings(self):
        """获取当前持仓"""
        try:
            current_holdings = {}
            total_value = self.algorithm.Portfolio.TotalPortfolioValue
            
            for kvp in self.algorithm.Portfolio:
                symbol_obj = kvp.Key
                holding = kvp.Value
                if holding.Quantity != 0:
                    symbol_str = str(symbol_obj)
                    market_value = holding.HoldingsValue
                    weight = market_value / total_value if total_value > 0 else 0
                    
                    current_holdings[symbol_str] = {
                        'quantity': holding.Quantity,
                        'market_value': market_value,
                        'weight': weight,
                        'price': holding.Price
                    }
            
            return current_holdings
            
        except Exception as e:
            self.algorithm.log_debug(f"获取当前持仓失败: {str(e)}")
            return {}
    
    def _calculate_target_holdings(self, target_weights, 
                                 symbols):
        """计算目标持仓"""
        try:
            target_holdings = {}
            
            # 获取投资组合总价值
            total_value = self.algorithm.Portfolio.TotalPortfolioValue
            
            # 检查是否使用杠杆
            leverage_ratio = 1.0
            if hasattr(self.algorithm, 'leverage_manager'):
                try:
                    leverage_ratio = self.algorithm.leverage_manager.get_current_leverage_ratio()
                    self.algorithm.log_debug(f"当前杠杆比率: {leverage_ratio:.2f}")
                except Exception as e:
                    self.algorithm.log_debug(f"获取杠杆比率失败: {str(e)}")
            
            # 调整总投资金额（包含杠杆）
            total_investment = total_value * leverage_ratio
            
            for i, symbol in enumerate(symbols):
                if i < len(target_weights) and target_weights[i] > 0:
                    symbol_str = str(symbol)
                    target_value = total_investment * target_weights[i]
                    
                    # 获取当前价格
                    current_price = self.get_current_price(symbol_str)
                    
                    if current_price > 0:
                        target_quantity = int(target_value / current_price)
                        
                        target_holdings[symbol_str] = {
                            'target_quantity': target_quantity,
                            'target_value': target_value,
                            'target_weight': target_weights[i],
                            'current_price': current_price
                        }
            
            self.algorithm.log_debug(f"目标持仓计算完成，包含{len(target_holdings)}只股票")
            if leverage_ratio != 1.0:
                self.algorithm.log_debug(f"使用杠杆{leverage_ratio:.2f}倍，总投资金额: ${total_investment:,.0f}")
            
            return target_holdings
            
        except Exception as e:
            self.algorithm.log_debug(f"计算目标持仓失败: {str(e)}")
            return {}
    
    def get_current_price(self, symbol_str):
        """获取当前价格"""
        try:
            symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            security = self.algorithm.Securities.get(symbol)
            
            if security is not None:
                return float(security.Price)
            else:
                self.algorithm.log_debug(f"无法找到证券: {symbol_str}")
                return 0
                
        except Exception as e:
            self.algorithm.log_debug(f"获取{symbol_str}价格失败: {str(e)}")
            return 0
    
    def _generate_trade_instructions(self, current_holdings, 
                                   target_holdings):
        """生成交易指令"""
        try:
            trades = []
            
            # 处理需要买入/调整的股票
            for symbol_str, target_info in target_holdings.items():
                target_quantity = target_info['target_quantity']
                current_quantity = 0
                
                if symbol_str in current_holdings:
                    current_quantity = current_holdings[symbol_str]['quantity']
                
                quantity_diff = target_quantity - current_quantity
                
                if abs(quantity_diff) > 0:  # 需要交易
                    trades.append({
                        'symbol': symbol_str,
                        'action': 'BUY' if quantity_diff > 0 else 'SELL',
                        'quantity': abs(quantity_diff),
                        'target_quantity': target_quantity,
                        'current_quantity': current_quantity,
                        'target_weight': target_info['target_weight']
                    })
            
            # 处理需要清仓的股票
            for symbol_str, current_info in current_holdings.items():
                if symbol_str not in target_holdings:
                    trades.append({
                        'symbol': symbol_str,
                        'action': 'LIQUIDATE',
                        'quantity': abs(current_info['quantity']),
                        'target_quantity': 0,
                        'current_quantity': current_info['quantity'],
                        'target_weight': 0
                    })
            
            return trades
            
        except Exception as e:
            self.algorithm.log_debug(f"生成交易指令失败: {str(e)}")
            return []
    
    def _execute_trades(self, trades):
        """执行交易"""
        try:
            executed_trades = []
            
            for trade in trades:
                try:
                    symbol_str = trade['symbol']
                    action = trade['action']
                    quantity = trade['quantity']
                    
                    symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
                    
                    if action == 'BUY':
                        order_ticket = self.algorithm.MarketOrder(symbol, quantity)
                    elif action == 'SELL':
                        order_ticket = self.algorithm.MarketOrder(symbol, -quantity)
                    elif action == 'LIQUIDATE':
                        order_ticket = self.algorithm.Liquidate(symbol)
                    else:
                        continue
                    
                    if order_ticket:
                        executed_trades.append({
                            'symbol': symbol_str,
                            'action': action,
                            'quantity': quantity,
                            'order_ticket': order_ticket,
                            'target_weight': trade.get('target_weight', 0)
                        })
                        
                except Exception as trade_error:
                    self.algorithm.log_debug(f"执行交易失败 {trade['symbol']}: {str(trade_error)}")
                    continue
            
            self._last_trades_executed = executed_trades
            return len(executed_trades) > 0
            
        except Exception as e:
            self.algorithm.log_debug(f"交易执行失败: {str(e)}")
            return False
    
    def _log_rebalance_summary(self, executed_trades):
        """记录再平衡摘要"""
        try:
            if not executed_trades:
                return
                
            self.algorithm.log_debug("=== 再平衡执行摘要 ===")
            self.algorithm.log_debug(f"执行交易数量: {len(executed_trades)}")
            
            for trade in executed_trades:
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                target_weight = trade.get('target_weight', 0)
                
                self.algorithm.log_debug(f"  {symbol}: {action} {quantity}股 (目标权重: {target_weight:.1%})")
                
        except Exception as e:
            self.algorithm.log_debug(f"记录再平衡摘要失败: {str(e)}")
    
    def get_last_trades_executed(self):
        """获取最后执行的交易"""
        return self._last_trades_executed 