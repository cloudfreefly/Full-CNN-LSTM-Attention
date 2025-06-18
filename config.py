# region imports
from AlgorithmImports import *
# endregion
# 算法配置文件
class AlgorithmConfig:
    """算法主要配置参数"""
    # 时间设置
    START_DATE = (2010, 1, 1)
    END_DATE = (2025, 5, 31)
    INITIAL_CASH = 100000
    
    # 股票池配置
    #SYMBOLS = [
    #    "SPY", "GLD"
    #]
    SYMBOLS = [
        "SPY", "AVGO", "AAPL", "INTC", "NVDA", "AMZN", "LLY", "MSFT", "GOOG", "META", "TSLA", "NFLX", "GLD","TQQQ","SQQQ","TLT","IEF","BIL"
    ]
    # 数据和训练配置 - 重大改进
    LOOKBACK_DAYS = 252
    MIN_HISTORY_DAYS = 250
    MAX_HISTORY_DAYS = 500
    TRAINING_WINDOW = 504  # 扩大到2年的交易日数据（从126天）
    WARMUP_DAYS = 400
    
    # 新增：基本面数据配置
    FUNDAMENTAL_CONFIG = {
        'enable_fundamental_features': True,  # 启用基本面特征
        'pe_ratio_lookback': 252,            # PE比率历史数据
        'revenue_growth_periods': 4,         # 收入增长周期（季度）
        'debt_to_equity_threshold': 2.0,     # 债务股权比阈值
        'roe_threshold': 0.15,               # ROE阈值
        'sector_rotation_weight': 0.3,       # 行业轮动权重
        'macro_indicators': ['VIX', 'DXY', 'TNX'],  # 宏观指标
    }
    
    # 新增：数据质量控制
    DATA_QUALITY_CONFIG = {
        'min_data_points': 1000,             # 最少数据点（约4年）
        'max_missing_ratio': 0.08,          # 最大缺失值比例提高到8%（从5%）
        'outlier_detection_method': 'iqr',   # 异常值检测方法
        'outlier_threshold': 2.5,            # 异常值阈值（更严格）
        'data_consistency_check': True,      # 数据一致性检查
        'price_jump_threshold': 0.15,        # 价格跳跃阈值15%
        'enable_nan_interpolation': True,    # 启用NaN值插值修复
        'nan_interpolation_method': 'linear', # NaN插值方法
        'max_consecutive_nans': 10,          # 最大连续NaN值数量
        'feature_drop_threshold': 0.50,     # 特征丢弃阈值（50%缺失时丢弃该特征）
    }
    
    # 新增：消息控制配置
    MESSAGE_CONTROL_CONFIG = {
        'enable_rate_limiting': True,        # 启用消息限流
        'max_messages_per_minute': 100,      # 每分钟最大消息数
        'debug_message_throttle': 0.1,       # 调试消息节流比例（10%）
        'suppress_repetitive_messages': True, # 抑制重复消息
        'max_repetitions': 3,                # 最大重复次数
        'critical_only_mode': False,         # 仅关键消息模式
        'log_consolidation': True,           # 日志合并
    }
    
    # 新增：杠杆配置 - 支持150%最高持仓
    LEVERAGE_CONFIG = {
        'enable_leverage': True,              # 启用杠杆功能
        'max_leverage_ratio': 1.5,            # 最大杠杆比例150%
        'low_risk_leverage_ratio': 1.5,       # 低风险环境下杠杆比例150%
        'medium_risk_leverage_ratio': 1.2,    # 中等风险环境下杠杆比例120%
        'high_risk_leverage_ratio': 0.8,      # 高风险环境下杠杆比例80%
        'extreme_risk_leverage_ratio': 0.5,   # 极端风险环境下杠杆比例50%
        
        # 风险阈值配置
        'low_risk_vix_threshold': 18,         # 低风险VIX阈值
        'medium_risk_vix_threshold': 25,      # 中等风险VIX阈值
        'high_risk_vix_threshold': 35,        # 高风险VIX阈值
        'low_risk_volatility_threshold': 0.12, # 低风险波动率阈值
        'medium_risk_volatility_threshold': 0.20, # 中等风险波动率阈值
        'high_risk_volatility_threshold': 0.30,   # 高风险波动率阈值
        
        # 杠杆成本配置
        'borrowing_cost_annual': 0.03,        # 年化借款成本3%
        'margin_requirement': 0.25,           # 保证金要求25%
        'maintenance_margin': 0.20,           # 维持保证金20%
        
        # 动态调整配置
        'leverage_adjustment_frequency': 1,    # 杠杆调整频率（天）
        'leverage_smooth_factor': 0.3,        # 杠杆平滑因子（避免频繁调整）
        'min_leverage_change': 0.05,          # 最小杠杆变动幅度5%
    }
    
    # 多时间跨度预测配置
    PREDICTION_HORIZONS = [1, 5, 10]  # 预测时间跨度（天）
    HORIZON_WEIGHTS = {1: 0.5, 5: 0.3, 10: 0.2}  # 各时间跨度的权重
    
    # 模型配置 - 优化复杂度
    MODEL_CONFIG = {
        'conv_filters': [8, 16, 32],       # 减少CNN滤波器（从[16,32,64]）
        'conv_kernels': [3, 5, 7],         # 多尺度CNN核大小
        'lstm_units': [32, 16],            # 减少LSTM单元数（从[64,32]）
        'attention_heads': 2,              # 减少注意力头数（从4）
        'dropout_rate': 0.3,               # 增加Dropout率（从0.2）
        'batch_size': 32,                  # 增加批次大小（从16）
        'epochs': 50,                      # 增加训练轮数（从20）
        'patience': 10,                    # 增加早停耐心（从5）
        'l2_regularization': 0.01,         # 增加L2正则化（从0.001）
    }
    
    # 新增：时间序列验证配置
    VALIDATION_CONFIG = {
        'validation_method': 'time_series_split',  # 时间序列分割（替代随机分割）
        'n_splits': 5,                            # 时间序列交叉验证分割数
        'test_size': 0.2,                         # 测试集比例
        'validation_size': 0.2,                   # 验证集比例
        'gap_days': 5,                            # 训练和验证间的间隔天数
        'purged_validation': True,                # 清洗验证集（避免数据泄露）
        'walk_forward_validation': True,          # 滚动窗口验证
        'min_train_size': 252,                    # 最小训练集大小（1年）
    }
    
    # 投资组合优化配置 - 强制多元化设置，支持杠杆交易
    PORTFOLIO_CONFIG = {
        'min_weight': 0.05,           # 最小权重降至5%（从8%）
        'max_weight': 0.12,           # 最大权重降至12%（从15%）- 进一步限制集中度
        'weight_threshold': 0.001,    # 权重筛选阈值（0.1%）- 大幅降低
        'rebalance_tolerance': 0.003, # 调仓容忍度降至0.3%（从0.5%）- 更敏感
        'cash_buffer': -0.02,         # 现金缓冲调整为-2%（轻微杠杆）
        'sector_max_weight': 0.5,     # 单行业最大权重降至50%（从60%）
        'concentration_limit': 0.75,   # 集中度限制收紧至75%（从90%）
        'negative_return_threshold': -0.08,  # 负收益阈值收紧至-8%（从-10%）
        'min_positive_symbols': 10,    # 保留最少10只股票（从8只）
        'target_portfolio_size': 12,  # 目标12只股票（从10只）
        'max_portfolio_size': 15,     # 最大15只股票（从12只）
        'min_portfolio_size': 10,      # 最小10只股票（从8只）
        'diversification_preference': 0.9,  # 90%多元化偏好（从80%）
        'allow_cash_when_all_negative': True,  # 允许在全负收益时持有现金
        'force_equal_weights': False,  # 强制等权重模式
        'equal_weight_fallback': False, # 启用等权重备用策略
        'disable_optimization': False, # 禁用复杂优化，直接等权重
        'severe_loss_threshold': -0.12,  # 严重损失阈值收紧至-12%（从-15%）
        'moderate_loss_threshold': -0.08, # 中等损失阈值收紧至-8%（从-10%）
        'defensive_max_cash_ratio': 0.2,  # 防御模式最大现金降至20%（杠杆环境下）
        'defensive_min_equity_ratio': 0.5,  # 防御模式最小股票降至50%（杠杆环境下去杠杆）
        'defensive_cash_scaling': True,   # 启用复杂现金缩放
        
        # === 杠杆相关投资组合参数 ===
        'max_leverage_equity_ratio': 1.5,    # 最大杠杆股票仓位150%
        'target_leverage_equity_ratio': 1.5, # 目标杠杆股票仓位150%
        'leverage_rebalance_threshold': 0.02, # 杠杆环境下调仓阈值2%（更敏感）
        'leverage_cash_buffer': -0.45,       # 杠杆现金缓冲-45%（接近150%持仓）
        
        # === 交易成本控制 ===
        'min_trade_value': 2000,             # 最小交易金额（美元）- 防止小额交易产生过高手续费
        'min_trade_percentage': 0.02,        # 最小交易比例（2%）- 避免过小的仓位调整
        'transaction_cost_threshold': 0.001, # 交易成本阈值（0.1%）- 当预期收益超过成本时才交易
    }
    
    # 风险控制开关配置
    RISK_CONTROL_SWITCHES = {
        # === 核心风险控制开关 ===
        'enable_vix_defense': True,              # VIX防御机制
        'enable_liquidity_filter': False,         # 流动性筛选
        'enable_volatility_filter': False,        # 波动率筛选
        'enable_correlation_filter': False,       # 相关性筛选
        'enable_concentration_limits': False,     # 集中度限制
        'enable_diversification_enforcer': False, # 分散化强制执行
        'enable_portfolio_optimization': False,   # 组合优化
        'enable_hedging': True,                  # 对冲策略 - 启用！
        'enable_recovery_mechanism': True,       # 恢复机制
        
        # === 优化方法开关 ===
        'enable_mean_variance_optimization': True,    # 均值方差优化
        'enable_risk_parity_optimization': True,      # 风险平价优化
        'enable_max_diversification_optimization': True, # 最大分散化优化
        'enable_equal_weight_fallback': True,         # 等权重备用策略
        'disable_complex_optimization': False,        # 禁用复杂优化（直接等权重）
        
        # === VIX相关开关 ===
        'enable_vix_monitoring': True,           # VIX监控
        'enable_vix_defensive_filter': True,    # VIX防御性筛选
        'enable_vix_panic_score': True,         # VIX恐慌评分
        'enable_vix_risk_adjustment': True,     # VIX风险调整
        'enable_vix_recovery_tracking': True,   # VIX恢复跟踪
        
        # === 风险筛选开关 ===
        'enable_negative_return_filter': True,   # 负收益筛选
        'enable_volatility_indicator_filter': True, # 波动指标筛选
        'enable_volume_quality_check': True,     # 成交量质量检查
        'enable_price_stability_check': True,    # 价格稳定性检查
        
        # === 仓位管理开关 ===
        'enable_position_size_limits': True,     # 仓位大小限制
        'enable_sector_limits': True,            # 行业限制
        'enable_rebalance_threshold': True,      # 再平衡阈值
        'enable_minimum_position_filter': True,  # 最小仓位筛选
        
        # === 紧急控制开关 ===
        'enable_drawdown_protection': True,      # 回撤保护
        'enable_stop_loss': True,               # 止损（启用以改善回撤控制）
        'enable_take_profit': True,             # 止盈（启用以锁定收益）
        'enable_emergency_cash_mode': True,      # 紧急现金模式
        
        # === 调试和测试开关 ===
        'enable_risk_logging': True,             # 风险日志记录
        'enable_detailed_risk_analysis': True,   # 详细风险分析
        'enable_backtesting_mode': False,        # 回测模式（简化风险控制）
        'enable_paper_trading_mode': False,      # 模拟交易模式
    }
    
    # 仓位管理配置 - 更新支持杠杆交易
    POSITION_MANAGEMENT_CONFIG = {
        # === 仓位大小控制 ===
        'max_single_position_weight': 0.30,      # 单个仓位最大权重
        'min_single_position_weight': 0.03,      # 单个仓位最小权重
        'max_sector_weight': 0.60,               # 单个行业最大权重
        'max_concentration_ratio': 0.80,         # 前3大持仓最大集中度
        
        # === 仓位数量控制 ===
        'min_position_count': 2,                 # 最少持仓数量
        'max_position_count': 12,                # 最多持仓数量
        'target_position_count': 8,              # 目标持仓数量
        'optimal_position_range': (4, 10),       # 最优持仓数量范围
        
        # === 再平衡控制 ===
        'rebalance_threshold': 0.05,             # 再平衡阈值（5%）
        'min_trade_size': 0.01,                  # 最小交易规模（1%）
        'max_turnover_per_rebalance': 0.5,       # 单次调仓最大换手率
        'rebalance_frequency_limit': 1,          # 调仓频率限制（天）
        
        # === 多元化控制 ===
        'diversification_preference': 0.4,       # 多元化偏好强度
        'min_effective_positions': 2,            # 最少有效仓位数
        'correlation_diversification_threshold': 0.75, # 相关性多元化阈值
        
        # === 现金管理 - 支持杠杆交易 ===
        'min_cash_ratio': -0.50,                 # 最小现金比例（负值表示可借款50%）
        'max_cash_ratio': 0.05,                  # 最大现金比例降至5%（低风险时）
        'default_cash_buffer': -0.02,            # 默认现金缓冲（负值表示轻微杠杆）
        'emergency_cash_ratio': 0.20,            # 紧急现金比例20%（去杠杆）
        'leverage_cash_buffer': -0.45,           # 杠杆缓冲（150%持仓时的现金状态）
        
        # === 杠杆相关现金管理 ===
        'max_leverage_cash_ratio': -0.50,        # 最大杠杆时现金比例（-50%表示借款50%）
        'leverage_emergency_threshold': -0.30,   # 杠杆紧急阈值（借款30%时触发警告）
        'leverage_margin_call_threshold': -0.40, # 保证金追缴阈值（借款40%）
        
        # === 风险调整参数 ===
        'risk_budget_per_position': 0.015,        # 每个仓位风险预算降至1.5%（从2%）
        'total_risk_budget': 0.10,               # 总风险预算降至10%（从15%）
        'volatility_scaling_factor': 1.0,        # 波动率缩放因子
        'correlation_penalty_factor': 0.6,       # 相关性惩罚因子提高至0.6（从0.5）
    }
    
    # 风险管理配置 - 增强杠杆风险控制
    RISK_CONFIG = {
        'max_drawdown': 0.08,        # 最大回撤调整至8%（杠杆环境下适度放宽）
        'volatility_threshold': 1.30,  # 年化波动率阈值130%（放宽到合理水平）
        'correlation_threshold': 0.75,  # 相关性阈值75%
        'liquidity_min_volume': 200000,  # 最小日均成交量20万（降低流动性要求）
        'max_weight_per_stock': 0.08,  # 单股最大权重降至8%
        'max_weight_per_sector': 0.20,  # 单行业最大权重降至25%
        'signal_volatility_threshold': 2.0,  # 波动指标信号的波动率阈值
        'stop_loss_threshold': -0.05,  # 止损阈值调整至-5%（杠杆环境下适度放宽）
        'take_profit_threshold': 0.15,   # 止盈阈值调整至15%（杠杆环境下适度提高）
        
        # === 杠杆相关风险控制 ===
        'leverage_max_drawdown': 0.12,       # 杠杆环境下最大回撤12%
        'leverage_volatility_threshold': 0.25, # 使用杠杆时最大波动率阈值25%
        'leverage_stop_loss': -0.08,         # 杠杆止损阈值-8%
        'leverage_margin_call_level': 0.25,  # 保证金追缴水平25%
        'leverage_liquidation_level': 0.20,  # 强制平仓水平20%
        'leverage_risk_budget': 0.15,        # 杠杆总风险预算15%
        
        # VIX相关配置
        'vix_rapid_rise_threshold': 0.15,  # VIX快速上升阈值进一步降至15%（从20%）
        'vix_rapid_rise_period': 2,        # VIX快速上升检测周期缩短至2天（从3天）
        'vix_extreme_level': 30,           # VIX极端水平进一步降至30（从35）
        'vix_defense_min_equity': 0.30,    # VIX防御模式最小股票仓位30%（现金比例70%）
        'vix_defense_max_cash': 0.70,      # VIX防御模式最大现金比例70%
        'vix_hedging_allocation': 0.30,    # 对冲产品分配比例提高至30%（从25%）
        'vix_normalization_threshold': 18, # VIX正常化阈值进一步降至18（从20）
        'vix_lookback_days': 10,           # VIX历史数据回望天数
        
        # VIX恢复机制配置
        'vix_rapid_decline_threshold': -0.08,  # VIX快速下降阈值调至-8%（从-10%）
        'vix_recovery_step_size': 0.10,        # 逐步恢复步长降至10%（从12%）
        'vix_recovery_min_increment': 0.03,    # 最小恢复增量降至3%（从4%）
        'vix_quick_recovery_threshold': 15,    # 快速恢复VIX阈值降至15（从16）
        'vix_recovery_max_equity': 0.80,       # 恢复过程中最大股票仓位降至80%（从85%）
        'vix_recovery_hedge_reduction': 0.7,   # 恢复时对冲仓位减少比例提高至70%（从60%）
        'vix_recovery_evaluation_days': 1,     # 恢复状态评估周期缩短至1天（从2天）
    }
    
    # 训练控制配置 - 重大改进
    TRAINING_CONFIG = {
        'enable_retraining': True,     # 是否启用重新训练
        'retraining_frequency': 'monthly',  # 改为月度重训练（从半年）
        'max_training_time': 12000,      # 增加最大训练时间（从6000秒到200分钟）
        'memory_cleanup_interval': 5,  # 内存清理间隔
        'model_validation_ratio': 0.0,  # 禁用随机验证集（改用时间序列验证）
        'early_stopping_delta': 0.001,  # 早停最小改善
        'mc_dropout_samples': 10,     # Monte Carlo Dropout采样数
        'fallback_strategy': 'momentum',  # 无模型时的备用策略：'equal_weights', 'momentum', 'skip'
        
        # === 训练时间分离配置 ===
        'training_time_separation': {
            'enable_pretrain': True,           # 重新启用历史数据预训练
            'pretrain_history_months': 24,    # 预训练使用2年历史数据（从6个月）
            'pretrain_on_startup': False,     # 禁用启动时预训练，改为非交易时间进行
            'pretrain_strict_mode': True,     # 启用严格模式：提高数据要求
            'pretrain_model_cache': True,     # 启用缓存以提高效率
            'pretrain_cache_file': 'pretrained_models',  # ObjectStore键名（不需要.pkl扩展名）
            
            'enable_weekend_training': True,   # 启用周末深度训练
            'weekend_training_day': 6,         # 周末训练日（6=周日）
            'weekend_training_hour': 2,        # 周末训练时间（凌晨2点）
            'weekend_training_duration': 14400, # 周末训练最长时间（4小时，从2小时）
            'weekend_full_retrain': True,      # 周末是否完全重训练
            
            'enable_async_training': True,     # 启用异步后台训练
            'async_training_interval': 3600,   # 异步训练间隔（1小时）
            'async_max_duration': 3600,        # 异步训练最长时间（1小时，从30分钟）
            'async_model_update': True,        # 异步更新模型
            
            'trading_time_fast_mode': True,    # 交易时间快速模式
            'fast_mode_max_training': 600,     # 快速模式最大训练时间（10分钟，从5分钟）
            'fast_mode_incremental_only': True, # 快速模式仅增量训练
            
            'model_persistence': {
                'enable_model_saving': True,   # 启用模型保存
                'model_save_path': 'models/',  # 模型保存路径
                'save_frequency': 'daily',     # 保存频率
                'model_version_control': True, # 模型版本控制
                'max_model_versions': 14,      # 最大保留版本数（从7增加到14）
                'auto_load_latest': True,      # 自动加载最新模型
            }
        },
        
        # === 新增：训练时间调度配置 ===
        'training_schedule': {
            'after_market_close': True,    # 是否在收盘后训练
            'training_delay_minutes': 30,  # 收盘后延迟训练时间（分钟）
            'weekend_training': True,      # 是否启用周末深度训练
            'weekend_training_day': 6,     # 周末训练日（0=周一，6=周日）
            'weekend_training_hour': 20,   # 周末训练时间（小时）
            'model_validity_days': 30,     # 模型有效期（从7天增加到30天）
            'weekend_model_validity_days': 60,  # 周末训练模型有效期（从14天增加到60天）
            'max_weekend_training_time': 3600,  # 周末最大训练时间（从20分钟增加到60分钟）
            'weekend_epochs_multiplier': 2.0,   # 周末训练epochs倍数（从1.5增加到2.0）
            'emergency_retrain_threshold': 3,   # 连续失败后强制重训练（次数）
        },
        
        # === 训练性能优化配置 ===
        'performance_optimization': {
            'separate_training_trading': True,  # 分离训练和交易时间
            'fast_rebalance_mode': True,       # 快速调仓模式（交易时间不训练）
            'training_memory_limit': 0.8,      # 训练时内存使用限制（比例）
            'batch_processing': True,          # 批量处理模式
            'parallel_model_training': False,  # 并行模型训练（暂时关闭，避免资源冲突）
        }
    }
    
    # 自动优化配置
    OPTIMIZATION_CONFIG = {
        'enable_auto_optimization': True,  # 是否启用自动优化
        'optimization_frequency_days': 30,  # 优化频率（天）
        'performance_degradation_threshold': -5.0,  # 性能下降阈值（%）
        'max_optimization_time': 1800,  # 最大优化时间（秒）
        'optimization_method': 'random_search',  # 优化方法：random_search, grid_search, bayesian
        'max_iterations': 20,  # 最大迭代次数
        'weekend_optimization': True,  # 是否在周末进行优化
        'market_close_optimization': True,  # 是否在收盘后进行优化
        'save_optimization_history': True,  # 是否保存优化历史
        'emergency_optimization_threshold': -10.0,  # 紧急优化触发阈值（%）
        
        # 可优化参数的默认范围
        'default_parameter_ranges': {
            'max_drawdown': [0.05, 0.08, 0.10, 0.12, 0.15],
            'volatility_threshold': [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30],
            'max_leverage_ratio': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            'target_portfolio_size': [8, 10, 12, 15, 18],
            'max_weight': [0.08, 0.10, 0.12, 0.15, 0.18],
            'rebalance_tolerance': [0.003, 0.005, 0.008, 0.010, 0.015],
            'vix_extreme_level': [25, 28, 30, 32, 35],
            'vix_high_level': [20, 22, 25, 27, 30],
            'defensive_max_cash_ratio': [0.3, 0.4, 0.5, 0.6],
        },
        
        # 市场状态特定的参数范围
        'market_specific_ranges': {
            'high_volatility': {  # VIX > 30
                'max_drawdown': [0.08, 0.10, 0.12, 0.15],
                'max_leverage_ratio': [0.8, 1.0, 1.2],
                'defensive_max_cash_ratio': [0.3, 0.4, 0.5, 0.6]
            },
            'low_volatility': {  # VIX < 18
                'max_drawdown': [0.05, 0.08, 0.10],
                'max_leverage_ratio': [1.2, 1.5, 1.8],
                'target_portfolio_size': [10, 12, 15, 18]
            },
            'normal_volatility': {  # 18 <= VIX <= 30
                'max_drawdown': [0.06, 0.08, 0.10, 0.12],
                'max_leverage_ratio': [1.0, 1.2, 1.5],
                'target_portfolio_size': [8, 10, 12, 15]
            }
        },
        
        # 优化目标函数权重
        'objective_weights': {
            'sharpe_ratio': 0.4,
            'calmar_ratio': 0.3,
            'max_drawdown': 0.2,  # 负向权重，回撤越小越好
            'win_rate': 0.1
        }
    }
    
    # 预测配置
    PREDICTION_CONFIG = {
        'confidence_threshold': 0.75,  # 置信度阈值 (提升胜率：从0.6提升至0.75)
        'trend_consistency_weight': 0.4, # 趋势一致性权重 (提升胜率：从0.3提升至0.4)
        'uncertainty_penalty': 0.3,   # 不确定性惩罚 (提升胜率：从0.2提升至0.3)
        'volatility_adjustment': True, # 是否进行波动率调整
        'regime_detection': True      # 是否启用市场状态检测
    }
    
    # 日志配置（引用LoggingConfig）
    # 日志级别控制
    DEBUG_LEVEL = {
        'training': True,
        'data': True,
        'prediction': True,
        'portfolio': True,
        'risk': True,
        'diversification': True,
        'system': True,
        'optimizer': True,
        'algorithm': True
    }
    
    LOGGING_CONFIG = {
        'enable_detailed_logging': False,
        'log_to_file': True,
        'log_file_path': 'algorithm_log.txt',
        'max_log_file_size': 100,  # MB
        'log_rotation_count': 5,
        'console_log_level': 'INFO',
        'file_log_level': 'DEBUG',
        'enable_performance_logging': True,
        'enable_trade_logging': True,
        'enable_error_logging': True,
        'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        
        # === 日志延迟控制配置 ===
        'enable_log_delay': False,  # 是否启用日志延迟（默认关闭）
        'log_delay_ms': 20,         # 基础日志延迟时间（毫秒）
        
        # 分类别日志延迟配置
        'category_delays': {
            'algorithm': 10,        # 算法日志延迟（毫秒）
            'training': 50,         # 训练日志延迟（毫秒）
            'portfolio': 15,        # 投资组合日志延迟（毫秒）
            'risk': 20,            # 风险日志延迟（毫秒）
            'system': 30,          # 系统日志延迟（毫秒）
            'data': 40,            # 数据处理日志延迟（毫秒）
            'prediction': 25,      # 预测日志延迟（毫秒）
            'optimizer': 35,       # 优化器日志延迟（毫秒）
            'diversification': 20  # 多元化日志延迟（毫秒）
        },
        
        # 动态延迟调整配置
        'dynamic_delay': {
            'enable_dynamic_delay': True,    # 启用动态延迟调整
            'min_delay_ms': 5,               # 最小延迟时间
            'max_delay_ms': 100,             # 最大延迟时间
            'high_frequency_threshold': 10,  # 高频日志阈值（每秒条数）
            'delay_increment_ms': 5,         # 延迟递增步长
            'delay_reset_interval': 60       # 延迟重置间隔（秒）
        },
        
        # 特殊情况延迟配置
        'special_delays': {
            'warmup_period_delay': 5,        # 预热期日志延迟（毫秒）
            'training_period_delay': 100,    # 训练期日志延迟（毫秒）
            'rebalance_period_delay': 30,    # 调仓期日志延迟（毫秒）
            'error_log_delay': 0,            # 错误日志延迟（毫秒，通常不延迟）
            'emergency_log_delay': 0,        # 紧急日志延迟（毫秒，不延迟）
            'debug_mode_delay': 50           # 调试模式延迟（毫秒）
        },
        
        # 延迟策略配置
        'delay_strategy': {
            'strategy_type': 'adaptive',        # 延迟策略：'fixed'固定, 'adaptive'自适应, 'burst'突发控制
            'burst_detection_window': 5,     # 突发检测窗口（秒）
            'burst_threshold': 20,           # 突发阈值（每窗口日志条数）
            'burst_delay_multiplier': 2.0,   # 突发延迟倍数
            'adaptive_learning_rate': 0.1,   # 自适应学习率
            'performance_target_ms': 50      # 性能目标（毫秒）
        }
    }

class TechnicalConfig:
    """技术指标和信号配置"""
    # 技术指标参数
    TECHNICAL_INDICATORS = {
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'atr_period': 14,
        'volume_ma_period': 20,
        'cci_period': 20  # CCI指标周期
    }
    
    # 市场状态检测参数
    MARKET_REGIME = {
        'volatility_lookback': 60,
        'trend_lookback': 40,
        'momentum_lookback': 20,
        'correlation_lookback': 30
    }

class LoggingConfig:
    """日志和调试配置"""
    # DEBUG_LEVEL已移至AlgorithmConfig中，避免重复定义
    
    LOG_INTERVALS = {
        'training_progress': 5,  # 每5个epoch记录一次
        'memory_usage': 10,      # 每10次操作记录内存使用
        'portfolio_status': 1,   # 每次调仓记录组合状态
        'daily_report_frequency': 1  # 每日报告频率
    }
    
    # 每日报告配置
    DAILY_REPORT_CONFIG = {
        'enable_portfolio_overview': True,
        'enable_holdings_details': True,
        'enable_performance_analysis': True,
        'enable_volume_analysis': True,
        'enable_prediction_analysis': True,
        'max_holdings_display': 10,
        'max_volume_display': 5
    } 