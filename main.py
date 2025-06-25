# 主算法文件 - 多时间跨度CNN+LSTM+Attention策略

# region imports
from AlgorithmImports import *
# endregion

import numpy as np
import pandas as pd
import time
import tensorflow as tf
from datetime import datetime, timedelta

# 导入自定义模块
from config import AlgorithmConfig
from data_processing import DataProcessor
from model_training import ModelTrainer
from prediction import (PredictionEngine, ExpectedReturnCalculator, 
                       PredictionValidator)
from risk_management import (RiskManager, VIXMonitor, HedgingManager, 
                           DrawdownMonitor, VolatilityMonitor, ConcentrationLimiter)
from portfolio_optimization import PortfolioOptimizer
from covariance_calculator import CovarianceCalculator
from smart_rebalancer import SmartRebalancer
from system_monitor import SystemMonitor
from position_logger import PositionLogger
from diversification_enforcer import DiversificationEnforcer
from leverage_manager import LeverageManager


class CNNTransformTradingAlgorithm(QCAlgorithm):

    def Initialize(self):
        """
        初始化多时间跨度预测策略：
        1. 设置基础参数和配置
        2. 初始化各功能模块
        3. 设置调度和预热期
        """
        try:
            # 加载配置
            self.config = AlgorithmConfig()
            
            # 设置基础参数 - 直接使用具体值避免解包问题
            start_date = self.config.START_DATE
            end_date = self.config.END_DATE
            
            self.SetStartDate(start_date[0], start_date[1], start_date[2])
            self.SetEndDate(end_date[0], end_date[1], end_date[2])
            self.SetCash(self.config.INITIAL_CASH)
            
            # 添加股票到算法环境
            for symbol in self.config.SYMBOLS:
                self.AddEquity(symbol, Resolution.Daily)
            
            # 添加VIX指数数据源 - 参考vix.py的简洁方法
            try:
                self.vix_symbol = self.AddIndex("VIX").Symbol
                self.log_debug("VIX指数已添加到数据源", log_type="risk")
            except Exception as e:
                self.log_debug(f"添加VIX指数失败: {e}", log_type="risk")
                self.vix_symbol = None
            
            # 添加对冲产品
            try:
                self.AddEquity("SQQQ", Resolution.Daily)
                self.log_debug("对冲产品 SQQQ 已添加到数据源", log_type="risk")
            except Exception as e:
                self.log_debug(f"添加对冲产品失败: {e}", log_type="risk")
            
            # 初始化功能模块
            self._initialize_modules()
            
            # 设置预热期
            self.SetWarmUp(self.config.WARMUP_DAYS)
            self.log_debug("Multi-horizon trading algorithm initialized", log_type="algorithm")
            
        except Exception as e:
            self.log_debug(f"Initialization error: {str(e)}", log_type="algorithm")
            raise
        
        # 存储当前数据slice
        self.data_slice = None
        
        # 性能监控
        self.performance_metrics = {
            'rebalance_count': 0,
            'total_training_time': 0,
            'prediction_success_rate': 0,
            'daily_returns': [],
            'total_prediction_time': 0
        }
        
        # 警报时间限制
        self.last_drawdown_alert_time = None
        self.last_volatility_alert_time = None
        
        # 时间跟踪
        self.algorithm_start_time = time.time()
        self.daily_start_time = None
        self.daily_execution_times = []
        self.current_trading_day = None
        
        # 调度设置：每日执行调仓
        self.Schedule.On(
            self.DateRules.EveryDay("SPY"), 
            self.TimeRules.AfterMarketOpen("SPY"), 
            self.Rebalance
        )
        
        self.log_debug("Multi-horizon trading algorithm initialized", log_type="algorithm")
        
        # 波动指标分析相关变量
        self.volatility_analysis_results = None
        self.preferred_volatility_indicator = '自动选择'
        
        # 添加每日收盘后自动输出投资组合报告
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.AfterMarketClose("SPY", 1), self._log_daily_report_wrapper)
        
        # 记录初始投资组合价值用于性能计算
        self._initial_portfolio_value = self.Portfolio.TotalPortfolioValue
        
        # 初始化投资组合价值历史记录（用于夏普比率计算）
        self._portfolio_value_history = []
        self._daily_returns = []
        self._last_portfolio_value = self._initial_portfolio_value
        
    def _initialize_modules(self):
        """初始化各功能模块"""
        # 数据处理模块
        self.data_processor = DataProcessor(self)
        
        # 模型训练模块
        self.model_trainer = ModelTrainer(self)
        
        # 高级训练管理器
        from training_manager import TrainingManager
        self.training_manager = TrainingManager(self)
        
        # 不在初始化时进行预训练，避免初始化时间过长
        # 预训练将在非交易时间进行
        self.pretrain_completed = False
        self.pretrain_scheduled = False
        
        # 初始化训练系统（仅基础初始化，不包含预训练）
        if hasattr(self.training_manager, 'initialize_training_system'):
            self.training_manager.initialize_training_system()
        
        # 预测模块
        self.prediction_engine = PredictionEngine(self)
        self.expected_return_calculator = ExpectedReturnCalculator(self)
        self.prediction_validator = PredictionValidator
        
        # 风险管理模块
        self.risk_manager = RiskManager(self)
        self.drawdown_monitor = DrawdownMonitor(self)
        self.volatility_monitor = VolatilityMonitor(self)
        self.concentration_limiter = ConcentrationLimiter(self)
        
        # 将VIX监控器暴露给算法实例，供其他模块访问
        self.vix_monitor = self.risk_manager.vix_monitor
        self.log_debug("VIX监控器已暴露给算法实例", log_type="system")
        
        # VIX风险管理和对冲模块 - 通过RiskManager访问
        self.hedging_manager = HedgingManager(self)
        
        # 投资组合优化模块
        self.portfolio_optimizer = PortfolioOptimizer(self)
        
        # 系统资源监控模块
        self.system_monitor = SystemMonitor(self)
        
        # 仓位日志记录模块
        self.position_logger = PositionLogger(self)
        
        # 多元化强制执行器
        self.diversification_enforcer = DiversificationEnforcer(self)
        
        # 杠杆管理器
        self.leverage_manager = LeverageManager(self)
        
        # 添加自动优化系统（可选功能）
        try:
            from auto_optimization_manager import OptimizationScheduler
            self.optimization_scheduler = OptimizationScheduler(self)
            self.optimization_scheduler.initialize()
            self.log_debug("自动参数优化系统已启用", log_type="system")
            
            # 记录初始投资组合价值用于性能计算
            self._initial_portfolio_value = self.Portfolio.TotalPortfolioValue
            
        except ImportError:
            self.log_debug("自动优化模块未找到，跳过自动优化功能", log_type="system")
            self.optimization_scheduler = None
        except Exception as e:
            self.log_debug(f"自动优化系统初始化失败: {e}", log_type="system")
            self.optimization_scheduler = None
        
        self.log_debug("All modules initialized", log_type="algorithm")
        
        # 调度预训练在非交易时间进行
        self._schedule_pretrain_in_non_trading_hours()
    
    def _schedule_pretrain_in_non_trading_hours(self):
        """调度预训练在非交易时间进行"""
        try:
            # 在市场开盘前30分钟进行预训练
            self.Schedule.On(
                self.DateRules.EveryDay("SPY"),
                self.TimeRules.BeforeMarketOpen("SPY", 30),
                self._perform_pretrain_if_needed
            )
            
            # 在市场收盘后进行预训练（备用时间）
            self.Schedule.On(
                self.DateRules.EveryDay("SPY"),
                self.TimeRules.AfterMarketClose("SPY", 30),
                self._perform_pretrain_if_needed
            )
            
            self.log_debug("预训练已调度在非交易时间进行", log_type="algorithm")
            
        except Exception as e:
            self.log_debug(f"调度预训练失败: {e}", log_type="algorithm")
    
    def _perform_pretrain_if_needed(self):
        """在非交易时间执行预训练（如果需要）"""
        try:
            # 检查是否已经完成预训练
            if self.pretrain_completed:
                return
            
            # 检查是否正在进行预训练
            if self.pretrain_scheduled:
                self.log_debug("预训练正在进行中，跳过", log_type="algorithm")
                return
            
            # 检查是否在预热期
            if self.IsWarmingUp:
                #self.Debug("算法仍在预热期，跳过预训练")
                return
            
            self.log_debug("开始非交易时间预训练...", log_type="algorithm")
            self.pretrain_scheduled = True
            
            # 尝试加载缓存的预训练模型
            if hasattr(self.training_manager, '_load_cached_models'):
                if self.training_manager._load_cached_models():
                    self.log_debug("成功加载缓存的预训练模型", log_type="algorithm")
                    self.training_manager.pretrained_models_loaded = True
                    self.training_manager.update_algorithm_models()
                    self.pretrain_completed = True
                    self.pretrain_scheduled = False
                    return
            
            # 如果没有缓存，进行实际预训练
            if hasattr(self.training_manager, '_perform_pretrain'):
                self.log_debug("开始执行历史数据预训练...", log_type="algorithm")
                pretrain_start_time = time.time()
                
                if self.training_manager._perform_pretrain():
                    # 缓存训练好的模型
                    if hasattr(self.training_manager, '_cache_models'):
                        self.training_manager._cache_models()
                    
                    self.training_manager.pretrained_models_loaded = True
                    self.training_manager.update_algorithm_models()
                    self.pretrain_completed = True
                    
                    pretrain_time = time.time() - pretrain_start_time
                    self.log_debug(f"预训练完成，用时: {pretrain_time:.1f}秒", log_type="algorithm")
                else:
                    self.log_debug("预训练失败，将使用运行时训练", log_type="algorithm")
            
            self.pretrain_scheduled = False
            
        except Exception as e:
            self.log_debug(f"非交易时间预训练失败: {e}", log_type="algorithm")
            self.pretrain_scheduled = False
  
    def _should_retrain(self):
        """判断是否应该重新训练模型"""
        # 使用训练管理器的智能判断
        if hasattr(self, 'training_manager'):
            return self.training_manager.should_perform_training()
        
        # 备用逻辑：检查全局开关
        if not self.config.TRAINING_CONFIG['enable_retraining']:
            return False
        
        # 检查是否有现有模型
        if not hasattr(self.model_trainer, 'models') or not self.model_trainer.models:
            self.log_debug("No existing models found, training required", log_type="algorithm")
            return True
        
        # 检查训练频率
        frequency = self.config.TRAINING_CONFIG['retraining_frequency']
        
        if frequency == 'always':
            return True
        elif frequency == 'monthly':
            return True
        elif frequency == 'weekly':
            current_day = self.Time.weekday()
            return current_day == 0  # 周一
        elif frequency == 'quarterly':
            current_month = self.Time.month
            return current_month in [1, 4, 7, 10] and self.Time.day <= 7
        elif frequency == 'semi_annually':
            # 每6个月重训练一次：1月和7月的第一周
            current_month = self.Time.month
            return current_month in [1, 7] and self.Time.day <= 7
        elif frequency == 'never':
            return False
        
        return True
    
    def _perform_training(self):
        """执行模型训练"""
        training_start_time = time.time()
        
        # 设置训练状态标志
        self._is_training = True
        
        try:
            self.log_debug("Starting optimized training process", log_type="algorithm")
            
            # 优先使用训练管理器
            if hasattr(self, 'training_manager'):
                # 如果预训练已完成，优先使用预训练模型
                if self.pretrain_completed and self.training_manager.pretrained_models_loaded:
                    self.log_debug("使用已有的预训练模型", log_type="algorithm")
                    successful_symbols = self.training_manager.get_tradable_symbols()
                    if successful_symbols:
                        self.log_debug(f"预训练模型可用: {len(successful_symbols)}个股票", log_type="algorithm")
                        return True
                
                # 否则进行优化训练
                training_success = self.training_manager.perform_optimized_training()
                
                if training_success:
                    # 更新模型引用
                    self.training_manager.update_algorithm_models()
                    successful_symbols = self.training_manager.get_tradable_symbols()
                    
                    # 记录训练统计
                    total_time = time.time() - training_start_time
                    self.performance_metrics['total_training_time'] += total_time
                    
                    self.log_debug("优化训练完成: " + str(len(successful_symbols)) + "个可交易股票", log_type="algorithm")
                    self.log_debug("训练用时: " + str(round(total_time, 1)) + "秒", log_type="algorithm")
                    
                    # 执行波动指标比较分析
                    if successful_symbols:
                        self._perform_volatility_indicator_analysis(successful_symbols)
                    
                    return True
            
            # 备用：使用标准训练器
            self.log_debug("使用备用训练器", log_type="algorithm")
            
            # 检查训练器是否已初始化
            if not hasattr(self, 'model_trainer') or self.model_trainer is None:
                self.log_debug("ERROR: Model trainer not initialized", log_type="algorithm")
                return False
            
            # 使用ModelTrainer进行训练
            models, successful_symbols = self.model_trainer.train_all_models()
            
            if successful_symbols:
                # 记录训练统计
                total_time = time.time() - training_start_time
                self.performance_metrics['total_training_time'] += total_time
                
                self.log_debug("Multi-horizon training completed successfully", log_type="algorithm")
                self.log_debug("Tradable symbols: " + str(len(successful_symbols)), log_type="algorithm")
                self.log_debug("Training time: " + str(round(total_time, 1)) + "s", log_type="algorithm")
                
                # 执行波动指标比较分析
                self._perform_volatility_indicator_analysis(successful_symbols)
                
                return True
            else:
                self.log_debug("Training failed - no successful symbols returned", log_type="algorithm")
                return False
                
        except Exception as e:
            self.log_debug("CRITICAL ERROR in training: " + str(e), log_type="algorithm")
            return False
        finally:
            # 清除训练状态标志
            self._is_training = False
    
    def _perform_rebalancing(self):
        """执行投资组合调仓"""
        # 设置调仓状态标志
        self._is_rebalancing = True
        
        try:
            self.log_debug("Starting rebalancing execution", log_type="algorithm")
            
            # 检查是否有可用的训练模型
            try:
                # 优先使用训练管理器的模型
                if hasattr(self, 'training_manager') and self.training_manager.pretrained_models_loaded:
                    tradable_symbols = self.training_manager.get_tradable_symbols()
                    self.log_debug("Retrieved tradable symbols from training_manager: " + str(len(tradable_symbols)), log_type="algorithm")
                else:
                    tradable_symbols = self.model_trainer.get_tradable_symbols()
                    self.log_debug("Retrieved tradable symbols from model_trainer: " + str(len(tradable_symbols)), log_type="algorithm")
            except Exception as symbol_error:
                self.log_debug("Error getting tradable symbols: " + str(symbol_error), log_type="algorithm")
                tradable_symbols = []
            
            if tradable_symbols:
                self.log_debug("Using model-based rebalancing", log_type="algorithm")
                # 使用训练好的模型进行预测和调仓
                self._rebalance_with_models()
            else:
                self.log_debug("No tradable symbols available, using fallback strategy", log_type="algorithm")
                # 使用备用策略
                self._rebalance_with_fallback_strategy()
                
        except Exception as e:
            self.log_debug("CRITICAL ERROR in rebalancing execution: " + str(e), log_type="algorithm")
            raise  # 重新抛出异常以便上层处理
        finally:
            # 清除调仓状态标志
            self._is_rebalancing = False
    
    def _rebalance_with_models(self):
        """使用训练好的模型进行调仓"""
        try:
            self.log_debug("Starting model-based rebalancing", log_type="algorithm")
            
            # 获取可交易的股票列表
            if hasattr(self, 'training_manager') and self.training_manager.pretrained_models_loaded:
                tradable_symbols = self.training_manager.get_tradable_symbols()
                self.log_debug(f"从训练管理器获取可交易股票: {len(tradable_symbols) if tradable_symbols else 0}个", log_type="algorithm")
            else:
                tradable_symbols = self.model_trainer.get_tradable_symbols()
                self.log_debug(f"从模型训练器获取可交易股票: {len(tradable_symbols) if tradable_symbols else 0}个", log_type="algorithm")
            
            if not tradable_symbols:
                self.log_debug("No tradable symbols after training, skipping rebalance", log_type="algorithm")
                return

            # 使用预测引擎生成预测
            if hasattr(self, 'training_manager') and self.training_manager.pretrained_models_loaded:
                # 从训练管理器获取模型
                models = {}
                for symbol in tradable_symbols:
                    model = self.training_manager.get_model_for_symbol(symbol)
                    if model is not None:
                        models[symbol] = {
                            'model': model,
                            'scaler': self.training_manager.get_scaler_for_symbol(symbol),
                            'effective_lookback': self.training_manager.get_effective_lookback_for_symbol(symbol)
                        }
                self.log_debug(f"从训练管理器获取训练模型: {len(models)}个", log_type="algorithm")
            else:
                models = self.model_trainer.get_trained_models()
                self.log_debug(f"从模型训练器获取训练模型: {len(models) if models else 0}个", log_type="algorithm")
            
            if not models:
                self.log_debug("No trained models available, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                return
            
            # 显示模型详情
            for symbol, model_info in models.items():
                if model_info:
                    self.log_debug(f"  {symbol}: 模型可用", log_type="algorithm")
                else:
                    self.log_debug(f"  {symbol}: 模型不可用", log_type="algorithm")
            
            # 尝试生成预测
            try:
                predictions = self.prediction_engine.generate_multi_horizon_predictions(
                    models, tradable_symbols
                )
                self.log_debug(f"预测生成结果: {len(predictions) if predictions else 0}个预测", log_type="algorithm")
            except Exception as e:
                self.log_debug(f"预测生成异常: {str(e)}", log_type="algorithm")
                predictions = None
            
            # 更新训练管理器的预测成功率统计
            had_valid_predictions = predictions is not None and len(predictions) > 0
            if hasattr(self, 'training_manager') and hasattr(self.training_manager, 'update_prediction_success_rate'):
                self.training_manager.update_prediction_success_rate(had_valid_predictions)
            
            if not predictions:
                self.log_debug("No valid predictions generated, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                return
            
            # 显示原始预测详情
            self.log_debug(f"Generated predictions for {len(predictions)} symbols", log_type="algorithm")
            for symbol, pred_data in predictions.items():
                confidence = pred_data.get('confidence', {}).get('overall_confidence', 0)
                expected_return = pred_data.get('expected_return', 0)
                self.log_debug(f"  {symbol}: 置信度={confidence:.3f}, 预期收益={expected_return:.4f}", log_type="algorithm")
            
            # 验证预测结果 - 使用严格验证提升胜率
            validation_results = self.prediction_validator.strict_validation_for_winning_rate(predictions)
            valid_predictions = {k: v for k, v in predictions.items() 
                               if validation_results.get(k, False)}
            
            # 记录验证统计
            total_predictions = len(predictions)
            valid_count = len(valid_predictions)
            self.log_debug(f"严格验证结果: {valid_count}/{total_predictions} 通过 ({valid_count/total_predictions*100:.1f}%)", log_type="algorithm")
            
            # 显示被过滤掉的预测
            filtered_out = set(predictions.keys()) - set(valid_predictions.keys())
            if filtered_out:
                self.log_debug(f"被过滤的股票: {list(filtered_out)}", log_type="algorithm")
                self.log_debug(f"置信度阈值: {self.config.PREDICTION_CONFIG['confidence_threshold']}", log_type="algorithm")
            
            if not valid_predictions:
                self.log_debug("No valid predictions after validation, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                return
            
            # 计算预期收益
            expected_returns, valid_symbols = self.expected_return_calculator.calculate_expected_returns(
                valid_predictions
            )
            
            if not expected_returns:
                self.log_debug("No valid expected returns, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                return
            
            # 计算协方差矩阵
            covariance_matrix = self.portfolio_optimizer.covariance_calculator.calculate_covariance_matrix(
                valid_symbols
            )
            
            if covariance_matrix is None:
                self.log_debug("Failed to calculate covariance matrix, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                return
            
            # 应用风险管理限制
            expected_returns_before_risk = expected_returns.copy()
            symbols_before_risk = valid_symbols.copy()
            
            risk_adjusted_returns, risk_filtered_symbols = self.risk_manager.apply_risk_controls(
                expected_returns, valid_symbols
            )
            
            # 记录风险控制分析
            vix_risk_state = getattr(self, '_vix_risk_state', {})
            expected_returns_after_risk_array = np.array([risk_adjusted_returns[s] for s in risk_filtered_symbols]) if risk_filtered_symbols else np.array([])
            expected_returns_before_risk_array = np.array([expected_returns_before_risk[s] for s in symbols_before_risk]) if symbols_before_risk else np.array([])
            
            self.position_logger.log_risk_analysis(
                vix_risk_state, 
                symbols_before_risk, 
                risk_filtered_symbols,
                expected_returns_before_risk_array,
                expected_returns_after_risk_array
            )
            
            # 使用风险调整后的收益重新优化投资组合
            if risk_filtered_symbols != valid_symbols:
                covariance_matrix = self.portfolio_optimizer.covariance_calculator.calculate_covariance_matrix(risk_filtered_symbols)
                if covariance_matrix is None:
                    self.log_debug("Failed to recalculate covariance matrix after risk filtering", log_type="algorithm")
                    self._rebalance_with_fallback_strategy()
                    return
            
            # 重新优化投资组合以确保权重和符号匹配
            risk_adjusted_returns_array = np.array([risk_adjusted_returns[s] for s in risk_filtered_symbols])
            
            self.log_debug(f"杠杆更新开始", log_type="leverage")

            # 更新杠杆管理器状态
            if hasattr(self, 'leverage_manager'):
                self.log_debug("Before update leverage ratio", log_type="algorithm")
                self.log_debug("正式回测期间调用杠杆管理器", log_type="risk")  # 添加调试日志
                self.leverage_manager.update_leverage_ratio()
                current_leverage = self.leverage_manager.get_current_leverage_ratio()
                self.log_debug("After update leverage ratio", log_type="algorithm")
                self.log_debug(f"杠杆更新完成，当前杠杆比例: {current_leverage:.2f}", log_type="leverage")
            else:
                self.log_debug("杠杆管理器未初始化", log_type="risk")  # 添加调试日志
            
            weights, final_valid_symbols = self.portfolio_optimizer.optimize_portfolio(
                risk_adjusted_returns_array, covariance_matrix, risk_filtered_symbols
            )
            
            # 获取投资组合优化决策因素
            if hasattr(self.portfolio_optimizer, 'get_last_decision_factors'):
                decision_factors = self.portfolio_optimizer.get_last_decision_factors()
                self.position_logger.log_position_decision_factors(decision_factors)
            
            # 记录优化结果 - 修复：确保equity_ratio正确获取和传递
            equity_ratio = getattr(self.portfolio_optimizer, '_last_equity_ratio', 0.98)  # 修复：使用98%作为默认值
            panic_score = getattr(self.portfolio_optimizer, '_last_panic_score', None)
            
            # 调试信息：确认equity_ratio的值
            self.log_debug(f"从优化器获取的equity_ratio: {equity_ratio:.2%}", log_type="algorithm")
            if equity_ratio <= 0.05:  # 修改：小于5%才报警
                self.log_debug("警告: equity_ratio过低，可能存在计算问题", log_type="algorithm")
            
            self.position_logger.log_optimization_result(
                weights, 
                final_valid_symbols, 
                equity_ratio,
                optimization_method="模型驱动优化",
                panic_score=panic_score
            )
            
            # 强制多元化检查和修复
            weights, final_valid_symbols = self.diversification_enforcer.enforce_diversification(
                weights, final_valid_symbols, risk_adjusted_returns
            )
            
            # 应用杠杆管理
            self.leverage_manager.update_leverage_ratio()
            leveraged_weights, final_valid_symbols = self.leverage_manager.apply_leverage_to_weights(
                weights, final_valid_symbols
            )
            
            # 检查保证金要求
            margin_status = self.leverage_manager.check_margin_requirements()
            if margin_status in ['margin_call', 'liquidation']:
                self.log_debug(f"保证金警告: {margin_status}，调整杠杆策略", log_type="risk")
                # 在紧急情况下降低杠杆
                if margin_status == 'liquidation':
                    # 强制去杠杆，权重恢复到100%以下
                    leveraged_weights = weights * 0.8  # 80%持仓
                    self.log_debug("执行紧急去杠杆：降至80%持仓", log_type="risk")
            
            # 使用杠杆调整后的权重
            weights = leveraged_weights
            
            # VIX对冲产品整合
            vix_risk_state = getattr(self, '_vix_risk_state', None)
            if vix_risk_state and vix_risk_state.get('mode', 'NORMAL') in ['DEFENSE', 'EXTREME']:
                self.log_debug("应用VIX对冲策略", log_type="algorithm")
                final_weights, final_symbols = self.hedging_manager.integrate_hedging_with_portfolio(
                    weights, final_valid_symbols, vix_risk_state
                )
                
                # 记录对冲状态
                self.hedging_manager.log_hedging_summary(vix_risk_state)
                
                # 更新优化结果记录（对冲后）
                self.position_logger.log_optimization_result(
                    final_weights, 
                    final_symbols, 
                    equity_ratio,
                    optimization_method="VIX对冲整合",
                    panic_score=panic_score
                )
            else:
                final_weights = weights
                final_symbols = final_valid_symbols
            
            # 最终多元化验证
            final_weights, final_symbols = self.diversification_enforcer.enforce_diversification(
                final_weights, final_symbols, risk_adjusted_returns
            )
            
            # 仓位一致性验证 - 新增：确保显示与执行一致
            if hasattr(self.portfolio_optimizer, 'validate_position_consistency'):
                self.portfolio_optimizer.validate_position_consistency(final_weights, final_symbols)
            
            # 执行交易
            if final_weights is not None and len(final_weights) == len(final_symbols):
                trade_result = self.portfolio_optimizer.smart_rebalancer.execute_smart_rebalance(final_weights, final_symbols)
                
                # 记录交易执行结果
                if hasattr(self.portfolio_optimizer.smart_rebalancer, 'get_last_trades_executed'):
                    trades_executed = self.portfolio_optimizer.smart_rebalancer.get_last_trades_executed()
                    self.position_logger.log_position_changes(trades_executed)
                
                self.log_debug("Trade execution completed", log_type="algorithm")
            else:
                self.log_debug("Weights and symbols mismatch, using fallback strategy", log_type="algorithm")
                self._rebalance_with_fallback_strategy()
                
        except Exception as e:
            self.log_debug("ERROR in model-based rebalancing: " + str(e), log_type="algorithm")
            self._rebalance_with_fallback_strategy()
    
    def _rebalance_with_fallback_strategy(self):
        """使用备用策略进行调仓"""
        try:
            fallback_strategy = self.config.TRAINING_CONFIG['fallback_strategy']
            
            if fallback_strategy == 'equal_weights':
                # 等权重策略
                valid_symbols = self.config.SYMBOLS
                n_symbols = len(valid_symbols)
                weights = [1.0 / n_symbols] * n_symbols
                
                self.portfolio_optimizer.smart_rebalancer.execute_smart_rebalance(weights, valid_symbols)
                self.log_debug("Applied equal weights strategy", log_type="algorithm")
                
            elif fallback_strategy == 'momentum':
                # 动量策略
                self._apply_momentum_strategy()
                
            elif fallback_strategy == 'skip':
                # 跳过此次调仓
                self.log_debug("Skipping rebalancing due to fallback strategy", log_type="algorithm")
                
        except Exception as e:
            self.log_debug("Error in fallback strategy: " + str(e), log_type="algorithm")
    
    def _apply_momentum_strategy(self):
        """应用基于动量的权重分配策略"""
        try:
            momentum_scores = {}
            
            for symbol in self.config.SYMBOLS:
                try:
                    # 获取过去20天的数据计算动量
                    history = self.History(symbol, 20, Resolution.Daily)
                    prices = [x.Close for x in history]
                    
                    if len(prices) >= 20:
                        momentum = (prices[-1] - prices[0]) / prices[0]
                        momentum_scores[symbol] = momentum
                        self.log_debug(f"{symbol} 动量分数: {momentum:.4f}", log_type="algorithm")
                    
                except Exception:
                    continue
            
            if momentum_scores:
                # 过滤出正动量的股票
                positive_momentum = {k: v for k, v in momentum_scores.items() if v > 0}
                
                if positive_momentum:
                    # 根据动量强度分配权重
                    symbols_list = list(positive_momentum.keys())
                    momentum_values = list(positive_momentum.values())
                    
                    # 将负动量设为0，正动量进行归一化
                    momentum_values = np.array(momentum_values)
                    momentum_values = np.maximum(momentum_values, 0)  # 确保非负
                    
                    # 权重与动量成正比
                    if np.sum(momentum_values) > 0:
                        weights = momentum_values / np.sum(momentum_values)
                    else:
                        # 如果所有动量都是0，使用等权重
                        weights = np.ones(len(symbols_list)) / len(symbols_list)
                    
                    # 记录权重分配
                    self.log_debug("=== 动量策略权重分配 ===", log_type="algorithm")
                    for i, symbol in enumerate(symbols_list):
                        self.log_debug(f"  {symbol}: 动量={momentum_values[i]:.4f}, 权重={weights[i]:.4f} ({weights[i]*100:.2f}%)", log_type="algorithm")
                    
                    self.portfolio_optimizer.smart_rebalancer.execute_smart_rebalance(weights, symbols_list)
                    self.log_debug(f"Applied momentum strategy with {len(symbols_list)} positive momentum stocks", log_type="algorithm")
                    
                else:
                    # 如果没有正动量股票，选择动量最好的前50%进行等权重
                    sorted_symbols = sorted(momentum_scores.keys(), 
                                          key=lambda x: momentum_scores[x], reverse=True)
                    top_symbols = sorted_symbols[:max(1, len(sorted_symbols) // 2)]
                    n_top = len(top_symbols)
                    weights = [1.0 / n_top] * n_top
                    
                    self.log_debug("No positive momentum stocks, using top 50% with equal weights", log_type="algorithm")
                    self.portfolio_optimizer.smart_rebalancer.execute_smart_rebalance(weights, top_symbols)
                    
            else:
                self.log_debug("No momentum data available, keeping current positions", log_type="algorithm")
                
        except Exception as e:
            self.log_debug("Error in momentum strategy: " + str(e), log_type="algorithm")
        
    def OnData(self, data):
        """数据更新事件处理"""
        try:
            # 存储最新数据slice
            self.data_slice = data
            
            # 维护投资组合价值历史记录（用于夏普比率计算）
            current_portfolio_value = float(self.Portfolio.TotalPortfolioValue)
            
            # 每日更新投资组合价值历史
            current_day = self.Time.date()
            if not hasattr(self, '_last_update_day') or self._last_update_day != current_day:
                self._last_update_day = current_day
                
                # 添加到历史记录
                self._portfolio_value_history.append(current_portfolio_value)
                
                # 计算日收益率
                if len(self._portfolio_value_history) > 1:
                    daily_return = (current_portfolio_value - self._last_portfolio_value) / self._last_portfolio_value
                    self._daily_returns.append(daily_return)
                
                self._last_portfolio_value = current_portfolio_value
                
                # 限制历史记录长度（保留最近252个交易日）
                if len(self._portfolio_value_history) > 252:
                    self._portfolio_value_history = self._portfolio_value_history[-252:]
                if len(self._daily_returns) > 252:
                    self._daily_returns = self._daily_returns[-252:]
            
            # 数据诊断 - 检查data_slice状态
            if data is None:
                self.log_debug("WARNING: OnData received None data", log_type="data")
            else:
                # 检查数据包含的股票
                available_symbols = []
                expected_symbols = self.config.SYMBOLS
                
                for symbol_str in expected_symbols:
                    if data.ContainsKey(symbol_str):
                        symbol_data = data[symbol_str]
                        if hasattr(symbol_data, 'Price') and symbol_data.Price > 0:
                            available_symbols.append(symbol_str)
                        else:
                            self.log_debug(f"Symbol {symbol_str} in data but no valid price", log_type="data")
                    else:
                        self.log_debug(f"Symbol {symbol_str} not in current data slice", log_type="data")
                
                self.log_debug(f"Data slice contains {len(available_symbols)}/{len(expected_symbols)} expected symbols: {available_symbols}", log_type="data")
                
                # 记录data slice的总体状态
                total_symbols_in_slice = len([key for key in data.Keys])
                self.log_debug(f"Total symbols in data slice: {total_symbols_in_slice}", log_type="data")
            
            # 跟踪每日开始时间
            if self.current_trading_day != current_day:
                self.current_trading_day = current_day
                self.daily_start_time = time.time()
            
            # 更新风险监控指标
            self._update_risk_monitoring()
                    
        except Exception as e:
            self.log_debug("ERROR in OnData: " + str(e), log_type="algorithm")
    
    def _update_risk_monitoring(self):
        """更新风险监控指标"""
        try:
            # 更新VIX数据监控 - 每天只更新一次
            if hasattr(self, 'vix_monitor'):
                vix_risk_state = self.vix_monitor.update_vix_data(self.Time)
                data_source = vix_risk_state.get('data_source', '未知')
                
                # 只有在实际更新数据时才记录详细日志
                if "今日首次更新" in data_source:
                    self.log_debug(f"VIX风险状态更新: VIX={vix_risk_state.get('vix_value', 'N/A'):.2f}, 模式={vix_risk_state.get('mode', 'N/A')}, 来源={data_source}", log_type="risk")
            
            # 更新回撤监控
            current_value = self.Portfolio.TotalPortfolioValue
            current_drawdown = self.drawdown_monitor.update_portfolio_value(current_value)
            
            # 保存回撤值到实例变量
            self._current_drawdown = current_drawdown if current_drawdown is not None else 0.0
            
            # 计算日收益率和波动率
            if hasattr(self, 'previous_portfolio_value') and self.previous_portfolio_value > 0:
                daily_return = (current_value - self.previous_portfolio_value) / self.previous_portfolio_value
                current_volatility = self.volatility_monitor.update_return(daily_return)
                
                # 保存波动率值到实例变量
                self._current_volatility = current_volatility if current_volatility is not None else 0.0
                
                # 每天首次计算时记录日志
                current_date = self.Time.date()
                if not hasattr(self, '_last_risk_log_date') or self._last_risk_log_date != current_date:
                    self._last_risk_log_date = current_date
                    self.log_debug(f"风险监控数据更新: 波动率={self._current_volatility:.3f}, 回撤={self._current_drawdown:.3f}, 日收益率={daily_return:.4f}", log_type="risk")
            else:
                self._current_volatility = 0.0
                if not hasattr(self, 'previous_portfolio_value'):
                    self.log_debug("首次运行，设置初始投资组合价值", log_type="risk")
            
            # 检查警报条件并记录
            drawdown_threshold = self.config.RISK_CONFIG.get('max_drawdown_threshold', 0.15)
            volatility_threshold = self.config.RISK_CONFIG.get('volatility_threshold', 0.25)
            
            if self._current_drawdown > drawdown_threshold:
                # 限制警报频率：每小时最多一次
                if (self.last_drawdown_alert_time is None or 
                    (self.Time - self.last_drawdown_alert_time).total_seconds() >= 3600):
                    self.log_debug(f"Drawdown alert: {self._current_drawdown:.2%}", log_type="risk")
                    self.last_drawdown_alert_time = self.Time
            
            if self._current_volatility > volatility_threshold:
                # 限制警报频率：每小时最多一次
                if (self.last_volatility_alert_time is None or 
                    (self.Time - self.last_volatility_alert_time).total_seconds() >= 3600):
                    self.log_debug(f"Volatility alert: {self._current_volatility:.2%}", log_type="risk")
                    self.last_volatility_alert_time = self.Time
            
            # 保存当前值用于下次计算
            self.previous_portfolio_value = current_value
                
        except Exception as e:
            self.log_debug("Error in risk monitoring: " + str(e), log_type="algorithm")
    
    def _log_system_resources(self, phase=""):
        """记录系统资源使用情况"""
        try:
            self.system_monitor.log_resources(phase)
        except Exception:
            pass
    
    def Rebalance(self):
        """定期调仓入口"""
        if self.IsWarmingUp:
            self.log_debug("Skipping rebalance during warmup period", log_type="algorithm")
            return
        
        # 检查是否需要运行参数优化
        if self.optimization_scheduler:
            try:
                if self.optimization_scheduler.optimization_manager.should_run_optimization():
                    self.log_debug("触发自动参数优化", log_type="optimization")
                    self.optimization_scheduler.force_optimization()
            except Exception as e:
                self.log_debug(f"自动优化检查失败: {e}", log_type="optimization")
        
        rebalance_start_time = time.time()
        
        try:
            # 开始仓位日志记录
            self.position_logger.log_rebalance_start(self.Time)
            
            # 记录当前持仓情况
            current_positions = self.position_logger.log_current_positions()
            
            # 记录每日仓位摘要
            self.position_logger.log_daily_position_summary()
            
            # 记录调仓前状态
            current_value = self.Portfolio.TotalPortfolioValue
            
            # 检查是否需要重新训练
            should_retrain = self._should_retrain()
            self.log_debug(f"训练检查结果: should_retrain={should_retrain}", log_type="algorithm")
            
            # 预先检查当前模型状态
            current_models = self.model_trainer.get_trained_models()
            current_tradable = self.model_trainer.get_tradable_symbols()
            self.log_debug(f"训练前模型状态: {len(current_models) if current_models else 0}个模型, {len(current_tradable) if current_tradable else 0}个可交易股票", log_type="algorithm")
            
            if should_retrain:
                self.log_debug("Starting model retraining process", log_type="algorithm")
                training_success = self._perform_training()
                self.log_debug(f"模型训练结果: success={training_success}", log_type="algorithm")
                
                # 训练后再次检查模型状态
                after_models = self.model_trainer.get_trained_models()
                after_tradable = self.model_trainer.get_tradable_symbols()
                self.log_debug(f"训练后模型状态: {len(after_models) if after_models else 0}个模型, {len(after_tradable) if after_tradable else 0}个可交易股票", log_type="algorithm")
                
                if not training_success:
                    self.log_debug("Training failed, proceeding with fallback strategy", log_type="algorithm")
            else:
                self.log_debug("Using existing models", log_type="algorithm")
            
            # 执行调仓
            self.log_debug("Starting rebalancing process", log_type="algorithm")
            self._perform_rebalancing()
            self.log_debug("Rebalancing process completed", log_type="algorithm")
            
            # 更新性能指标
            self.performance_metrics['rebalance_count'] += 1
            
            # 记录调仓总结
            self.position_logger.log_rebalance_summary()
            
            # 记录调仓完成
            rebalance_time = time.time() - rebalance_start_time
            self.log_debug(f"=== REBALANCE COMPLETED in {rebalance_time:.2f}s ===", log_type="algorithm")
            
        except Exception as e:
            self.log_debug("ERROR in main rebalance method: " + str(e), log_type="algorithm")
            # 尝试使用最简单的备用策略
            try:
                self._emergency_fallback_strategy()
                # 记录紧急策略执行
                self.position_logger.log_rebalance_summary()
            except Exception:
                pass
    
    def _emergency_fallback_strategy(self):
        """紧急备用策略 - 最简单的资产配置"""
        try:
            self.log_debug("Executing emergency fallback: liquidating all positions", log_type="algorithm")
            # 简单地清空所有持仓，保持现金
            for holding in self.Portfolio.Values:
                if holding.Invested:
                    self.Liquidate(holding.Symbol)
        except Exception as e:
            self.log_debug("Emergency fallback failed: " + str(e), log_type="algorithm")
    
    def OnEndOfAlgorithm(self):
        """算法结束时的清理工作"""
        try:
            total_runtime = time.time() - self.algorithm_start_time
            self.log_debug("=== ALGORITHM EXECUTION SUMMARY ===", log_type="algorithm")
            self.log_debug("Total algorithm runtime: " + str(round(total_runtime, 1)) + " seconds", log_type="algorithm")
            self.log_debug("Total rebalances: " + str(self.performance_metrics['rebalance_count']), log_type="algorithm")
        except Exception:
            pass

    def _perform_volatility_indicator_analysis(self, successful_symbols):
        """执行波动指标比较分析 - 静默版本"""
        try:
            analysis_results = {
                'total_symbols': len(successful_symbols),
                'cci_better_count': 0,
                'bollinger_better_count': 0,
                'similar_count': 0,
                'failed_analysis_count': 0,
                'detailed_results': {}
            }
            
            for symbol in successful_symbols:
                try:
                    # 获取历史数据 - 增加缓冲以避免技术指标计算的NaN值
                    historical_data = self.data_processor.get_historical_data(symbol, 800)
                    if historical_data is None or len(historical_data) < 60:
                        analysis_results['failed_analysis_count'] += 1
                        continue
                    
                    # 执行CCI vs 布林带比较分析
                    comparison_result = self.data_processor.compare_volatility_indicators(historical_data)
                    
                    if comparison_result is None:
                        analysis_results['failed_analysis_count'] += 1
                        continue
                    
                    # 统计结果
                    preferred = comparison_result['recommendation']['preferred_indicator']
                    if preferred == 'CCI':
                        analysis_results['cci_better_count'] += 1
                    elif preferred == '布林带':
                        analysis_results['bollinger_better_count'] += 1
                    else:
                        analysis_results['similar_count'] += 1
                    
                    # 保存详细结果
                    analysis_results['detailed_results'][symbol] = {
                        'preferred_indicator': preferred,
                        'cci_score': (comparison_result['cci_effectiveness']['correlation'] + 
                                    comparison_result['cci_effectiveness']['accuracy'] + 
                                    comparison_result['cci_effectiveness']['sensitivity']) / 3,
                        'bollinger_score': (comparison_result['bollinger_effectiveness']['correlation'] + 
                                          comparison_result['bollinger_effectiveness']['accuracy'] + 
                                          comparison_result['bollinger_effectiveness']['sensitivity']) / 3,
                        'score_difference': comparison_result['recommendation']['score_difference']
                    }
                    
                except Exception as symbol_error:
                    analysis_results['failed_analysis_count'] += 1
                    continue
            
            # 保存分析结果到实例变量，供后续使用（不生成报告）
            self.volatility_analysis_results = analysis_results
            
            # 设置首选指标（无日志输出）
            successful = analysis_results['total_symbols'] - analysis_results['failed_analysis_count']
            if successful > 0 and analysis_results['cci_better_count'] > analysis_results['bollinger_better_count']:
                self.preferred_volatility_indicator = 'CCI'
            elif successful > 0 and analysis_results['bollinger_better_count'] > analysis_results['cci_better_count']:
                self.preferred_volatility_indicator = '布林带'
            else:
                self.preferred_volatility_indicator = '自动选择'
            
        except Exception as e:
            pass
    
    def _generate_volatility_analysis_summary(self, results):
        """生成波动指标分析总结报告 - 已禁用"""
        # 此函数已被禁用以减少日志输出
        pass

    def log_debug(self, message, log_type="general"):
        """增强的调试日志方法 - 添加消息限流和延时控制"""
        try:
            # WarmUp期间完全禁用日志输出
            if hasattr(self, 'IsWarmingUp') and self.IsWarmingUp:
                return
            
            # 临时调试：先尝试直接输出，如果失败再使用复杂逻辑
            if hasattr(self, '_simple_log_mode') and self._simple_log_mode:
                current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
                if log_type != "general":
                    prefix = f"[{current_date}] {log_type}: "
                else:
                    prefix = f"[{current_date}] "
                self.Debug(f"{prefix}{message}")
                return
            
            # 获取消息控制配置
            message_config = self.config.MESSAGE_CONTROL_CONFIG
            logging_config = self.config.LOGGING_CONFIG
            
            # 检查是否启用限流
            if message_config.get('enable_rate_limiting', False):
                current_time = self.time  # 使用算法时间
                
                # 初始化消息计数器
                if not hasattr(self, '_message_counter'):
                    self._message_counter = {}
                    self._last_minute = current_time.minute
                
                # 检查是否进入新的分钟
                if current_time.minute != self._last_minute:
                    self._message_counter.clear()
                    self._last_minute = current_time.minute
                
                # 检查消息频率
                max_per_minute = message_config.get('max_messages_per_minute', 10)
                current_count = self._message_counter.get(log_type, 0)
                
                if current_count >= max_per_minute:
                    return  # 超过限制，丢弃消息
                
                # 调试消息节流
                if log_type == 'data':
                    throttle_rate = message_config.get('debug_message_throttle', 0.1)
                    import random
                    if random.random() > throttle_rate:
                        return  # 随机丢弃调试消息
                
                # 抑制重复消息
                if message_config.get('suppress_repetitive_messages', True):
                    if not hasattr(self, '_recent_messages'):
                        self._recent_messages = {}
                    
                    message_hash = hash(message)
                    max_repetitions = message_config.get('max_repetitions', 3)
                    
                    if message_hash in self._recent_messages:
                        if self._recent_messages[message_hash] >= max_repetitions:
                            return  # 重复消息过多，丢弃
                        self._recent_messages[message_hash] += 1
                    else:
                        self._recent_messages[message_hash] = 1
                        
                        # 清理旧消息（保持字典大小合理）
                        if len(self._recent_messages) > 1000:
                            # 清理一半的旧消息
                            items = list(self._recent_messages.items())
                            self._recent_messages = dict(items[-500:])
                
                # 更新计数器
                self._message_counter[log_type] = current_count + 1
            
            # 输出消息
            if self.config.DEBUG_LEVEL.get(log_type, True):
                # 格式化日期时间：显示完整日期和时间
                current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
                if log_type != "general":
                    prefix = f"[{current_date}] {log_type}: "
                else:
                    prefix = f"[{current_date}] "
                self.Debug(f"{prefix}{message}")
                time.sleep(0.2)
                #self.Debug("After sleep 0.2")
                # === 实现日志延迟功能 ===
                if logging_config.get('enable_log_delay', False):
                    try:
                        # 获取延迟时间（毫秒）
                        base_delay_ms = logging_config.get('log_delay_ms', 20)
                        
                        # 获取分类别延迟配置
                        category_delays = logging_config.get('category_delays', {})
                        type_delay_ms = category_delays.get(log_type, base_delay_ms)
                        
                        # 动态延迟调整
                        dynamic_config = logging_config.get('dynamic_delay', {})
                        if dynamic_config.get('enable_dynamic_delay', False):
                            # 检测高频日志
                            if not hasattr(self, '_log_frequency_tracker'):
                                self._log_frequency_tracker = {}
                                self._log_frequency_window_start = time.time()
                            
                            current_real_time = time.time()
                            window_duration = 1.0  # 1秒窗口
                            
                            # 重置频率窗口
                            if current_real_time - self._log_frequency_window_start > window_duration:
                                self._log_frequency_tracker.clear()
                                self._log_frequency_window_start = current_real_time
                            
                            # 更新频率计数
                            self._log_frequency_tracker[log_type] = self._log_frequency_tracker.get(log_type, 0) + 1
                            
                            # 调整延迟时间
                            current_frequency = self._log_frequency_tracker[log_type]
                            high_frequency_threshold = dynamic_config.get('high_frequency_threshold', 10)
                            
                            if current_frequency > high_frequency_threshold:
                                delay_increment = dynamic_config.get('delay_increment_ms', 5)
                                max_delay = dynamic_config.get('max_delay_ms', 100)
                                type_delay_ms = min(type_delay_ms + delay_increment, max_delay)
                        
                        # 特殊情况延迟
                        special_delays = logging_config.get('special_delays', {})
                        
                        # 训练期间延迟
                        if hasattr(self, '_is_training') and self._is_training:
                            type_delay_ms = special_delays.get('training_period_delay', type_delay_ms)
                        
                        # 预热期延迟
                        if hasattr(self, 'IsWarmingUp') and self.IsWarmingUp:
                            type_delay_ms = special_delays.get('warmup_period_delay', type_delay_ms)
                        
                        # 调仓期延迟
                        if hasattr(self, '_is_rebalancing') and self._is_rebalancing:
                            type_delay_ms = special_delays.get('rebalance_period_delay', type_delay_ms)
                        
                        # 错误和紧急日志不延迟
                        if log_type in ['error', 'emergency', 'critical']:
                            type_delay_ms = special_delays.get('error_log_delay', 0)
                        
                        # 执行延迟（转换为秒）
                        if type_delay_ms > 0:
                            delay_seconds = type_delay_ms / 1000.0
                            #self.Debug(f"Delaying for {delay_seconds} seconds")
                            time.sleep(delay_seconds)
                            #self.Debug(f"After sleep{delay_seconds}")
                    
                    except Exception as delay_error:
                        # 延迟功能出错时不应影响正常日志输出
                        pass
                
        except Exception as e:
            # 避免日志方法本身出错导致算法崩溃
            # WarmUp期间也不输出错误日志
            if hasattr(self, 'IsWarmingUp') and self.IsWarmingUp:
                return
            try:
                current_date = self.Time.strftime('%Y-%m-%d %H:%M:%S')
                self.Debug(f"[{current_date}] ERROR in log_debug: {e}")
            except:
                # 如果连时间格式化都失败，使用最简单的格式
                self.Debug(f"ERROR in log_debug: {e}")

    def _log_daily_report_wrapper(self):
        """收盘后自动输出每日投资组合报告"""
        if hasattr(self, 'position_logger') and hasattr(self.position_logger, 'log_daily_report'):
            self.position_logger.log_daily_report()
