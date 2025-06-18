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
                self.algorithm.log_debug("当前数据不可用，跳过再平衡", log_type="trading")
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
                self.algorithm.log_debug("无需进行交易", log_type="trading")
                return True
                
            # 执行交易
            success = self._execute_trades(trades)
            
            if success:
                self._log_rebalance_summary(trades)
                
            return success
            
        except Exception as e:
            self.algorithm.log_debug(f"智能再平衡失败: {str(e)}", log_type="trading")
            return False
    
    def _ensure_current_data_available(self):
        """确保当前数据可用"""
        try:
            # 检查是否有足够的现金或持仓
            total_portfolio_value = self.algorithm.Portfolio.TotalPortfolioValue
            if total_portfolio_value <= 0:
                self.algorithm.log_debug("投资组合价值为零或负数", log_type="trading")
                return False
                
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"数据可用性检查失败: {str(e)}", log_type="trading")
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
            self.algorithm.log_debug(f"获取当前持仓失败: {str(e)}", log_type="trading")
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
                    self.algorithm.log_debug(f"当前杠杆比率: {leverage_ratio:.2f}", log_type="trading")
                except Exception as e:
                    self.algorithm.log_debug(f"获取杠杆比率失败: {str(e)}", log_type="trading")
            
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
            
            self.algorithm.log_debug(f"目标持仓计算完成，包含{len(target_holdings)}只股票", log_type="trading")
            if leverage_ratio != 1.0:
                self.algorithm.log_debug(f"使用杠杆{leverage_ratio:.2f}倍，总投资金额: ${total_investment:,.0f}", log_type="trading")
            
            return target_holdings
            
        except Exception as e:
            self.algorithm.log_debug(f"计算目标持仓失败: {str(e)}", log_type="trading")
            return {}
    
    def get_current_price(self, symbol_str):
        """获取当前价格"""
        try:
            symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
            security = self.algorithm.Securities.get(symbol)
            
            if security is not None:
                return float(security.Price)
            else:
                self.algorithm.log_debug(f"无法找到证券: {symbol_str}", log_type="trading")
                return 0
                
        except Exception as e:
            self.algorithm.log_debug(f"获取{symbol_str}价格失败: {str(e)}", log_type="trading")
            return 0
    
    def _generate_trade_instructions(self, current_holdings, 
                                   target_holdings):
        """生成交易指令"""
        try:
            trades = []
            min_trade_value = self.config.PORTFOLIO_CONFIG.get('min_trade_value', 2000)  # 最小交易金额
            
            # 处理需要买入/调整的股票
            for symbol_str, target_info in target_holdings.items():
                target_quantity = target_info['target_quantity']
                current_quantity = 0
                current_price = target_info['current_price']
                
                if symbol_str in current_holdings:
                    current_quantity = current_holdings[symbol_str]['quantity']
                
                quantity_diff = target_quantity - current_quantity
                
                if abs(quantity_diff) > 0:  # 需要交易
                    # 计算交易金额
                    trade_value = abs(quantity_diff) * current_price
                    
                    # 检查是否满足最小交易金额要求
                    if trade_value >= min_trade_value:
                        trades.append({
                            'symbol': symbol_str,
                            'action': 'BUY' if quantity_diff > 0 else 'SELL',
                            'quantity': abs(quantity_diff),
                            'target_quantity': target_quantity,
                            'current_quantity': current_quantity,
                            'target_weight': target_info['target_weight'],
                            'trade_value': trade_value,
                            'price': current_price
                        })
                        self.algorithm.log_debug(f"交易指令: {symbol_str} {quantity_diff:+.0f}股, "
                                               f"金额: ${trade_value:,.2f}")
                    else:
                        self.algorithm.log_debug(f"跳过小额交易: {symbol_str} {quantity_diff:+.0f}股, "
                                               f"金额: ${trade_value:,.2f} < 最小限额: ${min_trade_value:,.0f}")
            
            # 处理需要清仓的股票 - 清仓不受最小金额限制
            for symbol_str, current_info in current_holdings.items():
                if symbol_str not in target_holdings:
                    liquidate_value = abs(current_info['quantity']) * current_info['price']
                    trades.append({
                        'symbol': symbol_str,
                        'action': 'LIQUIDATE',
                        'quantity': abs(current_info['quantity']),
                        'target_quantity': 0,
                        'current_quantity': current_info['quantity'],
                        'target_weight': 0,
                        'trade_value': liquidate_value,
                        'price': current_info['price']
                    })
                    self.algorithm.log_debug(f"清仓指令: {symbol_str} {current_info['quantity']}股, "
                                           f"金额: ${liquidate_value:,.2f}")
            
            if trades:
                total_trade_value = sum(trade['trade_value'] for trade in trades)
                self.algorithm.log_debug(f"生成 {len(trades)} 个交易指令，总交易金额: ${total_trade_value:,.2f}", log_type="trading")
            else:
                self.algorithm.log_debug("无符合条件的交易指令生成", log_type="trading")
            
            return trades
            
        except Exception as e:
            self.algorithm.log_debug(f"生成交易指令失败: {str(e)}", log_type="trading")
            return []
    
    def _execute_trades(self, trades):
        """执行交易"""
        try:
            executed_trades = []
            total_trade_value = 0
            
            self.algorithm.log_debug(f"=== 开始执行交易 - 共{len(trades)}个指令 ===", log_type="trading")
            
            for trade in trades:
                try:
                    symbol_str = trade['symbol']
                    action = trade['action']
                    quantity = trade['quantity']
                    trade_value = trade.get('trade_value', 0)
                    price = trade.get('price', 0)
                    
                    symbol = Symbol.Create(symbol_str, SecurityType.Equity, Market.USA)
                    
                    # 执行订单
                    order_ticket = None
                    if action == 'BUY':
                        order_ticket = self.algorithm.MarketOrder(symbol, quantity)
                    elif action == 'SELL':
                        order_ticket = self.algorithm.MarketOrder(symbol, -quantity)
                    elif action == 'LIQUIDATE':
                        order_ticket = self.algorithm.Liquidate(symbol)
                    else:
                        self.algorithm.log_debug(f"未知交易动作: {action}", log_type="trading")
                        continue
                    
                    if order_ticket:
                        executed_trades.append({
                            'symbol': symbol_str,
                            'action': action,
                            'quantity': quantity,
                            'order_ticket': order_ticket,
                            'target_weight': trade.get('target_weight', 0),
                            'trade_value': trade_value,
                            'price': price
                        })
                        
                        total_trade_value += trade_value
                        
                        # 详细交易日志
                        self.algorithm.log_debug(f"✅ 执行成功: {action} {symbol_str} "
                                               f"{quantity:,}股 @ ${price:.2f} = ${trade_value:,.2f}")
                    else:
                        self.algorithm.log_debug(f"❌ 订单创建失败: {symbol_str}", log_type="trading")
                        
                except Exception as trade_error:
                    self.algorithm.log_debug(f"❌ 交易执行异常 {trade['symbol']}: {str(trade_error)}", log_type="trading")
                    continue
            
            # 交易执行总结
            if executed_trades:
                self.algorithm.log_debug(f"=== 交易执行完成 ===", log_type="trading")
                self.algorithm.log_debug(f"成功执行: {len(executed_trades)}/{len(trades)} 个交易", log_type="trading")
                self.algorithm.log_debug(f"总交易金额: ${total_trade_value:,.2f}", log_type="trading")
                
                # 估算交易成本（假设0.1%的交易成本）
                estimated_cost = total_trade_value * 0.001
                self.algorithm.log_debug(f"预估交易成本: ${estimated_cost:.2f} (0.1%)", log_type="trading")
                
                # 按动作分类统计
                buy_count = sum(1 for t in executed_trades if t['action'] == 'BUY')
                sell_count = sum(1 for t in executed_trades if t['action'] == 'SELL')
                liquidate_count = sum(1 for t in executed_trades if t['action'] == 'LIQUIDATE')
                
                self.algorithm.log_debug(f"交易分布: 买入{buy_count}个, 卖出{sell_count}个, 清仓{liquidate_count}个", log_type="trading")
            else:
                self.algorithm.log_debug("⚠️ 无交易成功执行", log_type="trading")
            
            self._last_trades_executed = executed_trades
            return len(executed_trades) > 0
            
        except Exception as e:
            self.algorithm.log_debug(f"❌ 交易执行系统错误: {str(e)}", log_type="trading")
            return False
    
    def _log_rebalance_summary(self, executed_trades):
        """记录再平衡摘要"""
        try:
            if not executed_trades:
                return
                
            self.algorithm.log_debug("=== 再平衡执行摘要 ===", log_type="trading")
            self.algorithm.log_debug(f"执行交易数量: {len(executed_trades)}", log_type="trading")
            
            for trade in executed_trades:
                symbol = trade['symbol']
                action = trade['action']
                quantity = trade['quantity']
                target_weight = trade.get('target_weight', 0)
                
                self.algorithm.log_debug(f"  {symbol}: {action} {quantity}股 (目标权重: {target_weight:.1%})", log_type="trading")
                
        except Exception as e:
            self.algorithm.log_debug(f"记录再平衡摘要失败: {str(e)}", log_type="trading")
    
    def get_last_trades_executed(self):
        """获取最后执行的交易"""
        return self._last_trades_executed 