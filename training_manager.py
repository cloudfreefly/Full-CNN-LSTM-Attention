# region imports
from AlgorithmImports import *
# endregion
# 高级训练管理模块 - 支持时间分离训练
import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from datetime import datetime, timedelta
import threading
import json
import re
from typing import List

# MinMaxScaler将在data_processing模块中处理

class AdvancedTrainingManager:
    """高级训练管理器 - 支持预训练和非工作日训练"""
    
    def __init__(self, algorithm):
        """初始化高级训练管理器"""
        self.algorithm = algorithm
        self.config = algorithm.config.TRAINING_CONFIG['training_time_separation']
        
        # 模型存储
        self.lstm_models = {}
        self.scalers = {}
        self.effective_lookbacks = {}
        self.tradable_symbols = []
    
        # 训练状态管理
        self.last_training_time = None
        self.model_cache_key = self.config['pretrain_cache_file']  # ObjectStore键名
        self.background_training_thread = None
        self.is_background_training = False
        
        # 预训练状态
        self.pretrained_models_loaded = False
        self.weekend_training_scheduled = False
        
        self.algorithm.log_debug('training', "高级训练管理器初始化完成", log_type="training")
    
    def initialize_training_system(self):
        """初始化训练系统（基础初始化，不进行预训练）"""
        try:
            # 0. 测试ObjectStore连接
            if not self._test_object_store_connection():
                self.algorithm.log_debug('training', "⚠️ ObjectStore连接测试失败，模型缓存功能可能不可用", log_type="training")
            
            # 1. 设置周末训练调度（如果启用）
            if self.config['enable_weekend_training']:
                self._schedule_weekend_training()
            
            # 2. 启动异步训练线程（如果启用）
            if self.config['enable_async_training']:
                self._start_async_training()
            
            # 注意：预训练将在非交易时间单独调度执行，不在此处进行
            self.algorithm.log_debug('training', "训练系统基础初始化完成（预训练将在非交易时间进行）", log_type="training")
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug('training', f"训练系统初始化失败: {e}", log_type="training")
            return False
        
    def _load_or_create_pretrained_models(self):
        """加载或创建预训练模型"""
        try:
            # 尝试加载缓存的模型
            if self.config['pretrain_model_cache'] and self._load_cached_models():
                self.algorithm.log_debug('training', f"成功加载缓存的预训练模型: {len(self.lstm_models)}个", log_type="training")
                self.pretrained_models_loaded = True
                return True
            
            # 如果没有缓存，进行预训练
            if self.config['enable_pretrain'] and self.config['pretrain_on_startup']:
                self.algorithm.log_debug('training', "开始历史数据预训练...", log_type="training")
                if self._perform_pretrain():
                    self.algorithm.log_debug('training', f"预训练完成: {len(self.lstm_models)}个模型，正在缓存模型...", log_type="training")
                    if self.config['pretrain_model_cache']:
                        self._cache_models()
                    self.pretrained_models_loaded = True
                    return True
                else:
                    self.algorithm.log_debug('training', "预训练失败，将在运行时使用紧急训练", log_type="training")
                    self.pretrained_models_loaded = False
                    return False
            else:
                self.algorithm.log_debug('training', "预训练已禁用，将在运行时使用快速训练", log_type="training")
                self.pretrained_models_loaded = False
                return False
            
        except Exception as e:
            self.algorithm.log_debug('training', f"预训练模型加载/创建失败: {e}", log_type="training")
            self.pretrained_models_loaded = False
            return False
    
    def _perform_pretrain(self):
        """执行历史数据预训练"""
        try:
            pretrain_start = time.time()
            successful_models = 0
            
            # 使用更长的历史数据进行预训练
            pretrain_history_days = self.config['pretrain_history_months'] * 30

            for symbol in self.algorithm.config.SYMBOLS:
                try:
                    # 获取长期历史数据
                    history_data = self._get_extended_history(symbol, pretrain_history_days)
                    
                    if history_data is not None:
                        data_length = len(history_data)
                        self.algorithm.log_debug('training', f"获取 {symbol} 历史数据: {data_length}天 (需要>{500 if self.config.get('pretrain_strict_mode', True) else 100}天)", log_type="training")
                        
                        # 降低数据长度要求，从500天降到100天
                        min_required = 500 if self.config.get('pretrain_strict_mode', True) else 100
                        
                        if data_length > min_required:
                            # 使用历史数据训练模型
                            self.algorithm.log_debug('training', f"开始预训练 {symbol} (数据长度: {data_length})", log_type="training")
                            if self._train_pretrain_model(symbol, history_data):
                                successful_models += 1
                                self.algorithm.log_debug('training', f"预训练完成: {symbol}", log_type="training")
                            else:
                                self.algorithm.log_debug('training', f"预训练模型构建失败: {symbol}", log_type="training")
                        else:
                            self.algorithm.log_debug('training', f"跳过 {symbol}: 数据长度不足 ({data_length} <= {min_required})", log_type="training")
                    else:
                        self.algorithm.log_debug('training', f"跳过 {symbol}: 无法获取历史数据", log_type="training")
                    
                    # 控制预训练时间
                    if time.time() - pretrain_start > 1800:  # 30分钟限制
                        self.algorithm.log_debug('training', "预训练时间限制，停止预训练", log_type="training")
                        break
                    
                except Exception as e:
                    self.algorithm.log_debug('training', f"预训练 {symbol} 失败: {e}", log_type="training")
                    continue

            total_time = time.time() - pretrain_start
            self.algorithm.log_debug('training', f"预训练完成: {successful_models}个模型, 用时{total_time:.1f}秒", log_type="training")
            self.algorithm.log_debug('training', f"可交易股票列表: {self.tradable_symbols} (共{len(self.tradable_symbols)}个)", log_type="training")
            
            # 立即同步模型到model_trainer，确保主算法能识别预训练模型
            if successful_models > 0:
                self.update_algorithm_models()
                self.algorithm.log_debug('training', "预训练模型已同步到主算法", log_type="training")
            
            return successful_models > 0
            
        except Exception as e:
            self.algorithm.log_debug('training', f"预训练执行失败: {e}", log_type="training")
            return False
    
    def _get_extended_history(self, symbol, days):
        """获取扩展历史数据用于预训练（增强版）"""
        try:
            self.algorithm.log_debug('training', f"尝试获取 {symbol} {days}天历史数据...", log_type="training")
            
            # 尝试多种获取方式
            history_data = None
            
            # 方法1：使用时间范围
            try:
                end_date = self.algorithm.Time
                start_date = end_date - timedelta(days=days)
                history = self.algorithm.History(symbol, start_date, end_date, Resolution.Daily)
                history_list = list(history)
                if len(history_list) > 0:
                    history_data = np.array([x.Close for x in history_list])
                    self.algorithm.log_debug('training', f"方法1成功: {symbol} 获得{len(history_data)}个数据点", log_type="training")
                else:
                    self.algorithm.log_debug('training', f"方法1失败: {symbol} 无历史数据", log_type="training")
            except Exception as e1:
                self.algorithm.log_debug('training', f"方法1异常: {symbol} - {e1}", log_type="training")
            
            # 方法2：直接使用天数（如果方法1失败）
            if history_data is None:
                try:
                    history = self.algorithm.History(symbol, days, Resolution.Daily)
                    history_list = list(history)
                    if len(history_list) > 0:
                        history_data = np.array([x.Close for x in history_list])
                        self.algorithm.log_debug('training', f"方法2成功: {symbol} 获得{len(history_data)}个数据点", log_type="training")
                    else:
                        self.algorithm.log_debug('training', f"方法2失败: {symbol} 无历史数据", log_type="training")
                except Exception as e2:
                    self.algorithm.log_debug('training', f"方法2异常: {symbol} - {e2}", log_type="training")
            
            # 方法3：降级获取较少数据（如果方法2也失败）
            if history_data is None:
                for fallback_days in [days//2, days//4, 100, 50]:
                    try:
                        self.algorithm.log_debug('training', f"降级尝试 {symbol}: {fallback_days}天", log_type="training")
                        history = self.algorithm.History(symbol, fallback_days, Resolution.Daily)
                        history_list = list(history)
                        if len(history_list) > 20:  # 至少需要20个数据点
                            history_data = np.array([x.Close for x in history_list])
                            self.algorithm.log_debug('training', f"降级成功: {symbol} 获得{len(history_data)}个数据点", log_type="training")
                            break
                    except Exception as e3:
                        self.algorithm.log_debug('training', f"降级失败 {symbol} ({fallback_days}天): {e3}", log_type="training")
                        continue
            
            return history_data
            
        except Exception as e:
            self.algorithm.log_debug('training', f"获取 {symbol} 扩展历史数据完全失败: {e}", log_type="training")
            return None
    
    def _train_pretrain_model(self, symbol, historical_prices):
        """使用历史数据训练预训练模型"""
        try:
            # 数据预处理
            if not self._preprocess_pretrain_data(symbol, historical_prices):
                return False
            
            # 构建和训练模型
            return self._build_and_train_pretrain_model(symbol)
            
        except Exception as e:
            self.algorithm.log_debug('training', f"预训练模型 {symbol} 失败: {e}", log_type="training")
            return False

    def _preprocess_pretrain_data(self, symbol, prices):
        """预处理预训练数据（增强调试版）"""
        try:
            self.algorithm.log_debug('training', f"开始预处理 {symbol} 的数据，原始长度: {len(prices)}", log_type="training")
            
            # 数据清洗和验证
            prices_array = np.array(prices)
            
            # 检查NaN和无穷值
            nan_count = np.sum(np.isnan(prices_array))
            inf_count = np.sum(np.isinf(prices_array))
            if nan_count > 0 or inf_count > 0:
                self.algorithm.log_debug('training', f"{symbol} 数据质量问题: {nan_count}个NaN, {inf_count}个无穷值", log_type="training")
                return False
            
            # 检查数据长度
            min_required = 100 if self.config.get('pretrain_strict_mode', True) else 50
            if len(prices_array) < min_required:
                self.algorithm.log_debug('training', f"{symbol} 数据长度不足: {len(prices_array)} < {min_required}", log_type="training")
                return False
            
            self.algorithm.log_debug('training', f"{symbol} 数据验证通过，开始特征提取", log_type="training")
            
            # 使用完整的特征矩阵而不是只用价格
            try:
                feature_matrix = self.algorithm.data_processor.create_feature_matrix(prices_array)
                if feature_matrix is None:
                    self.algorithm.log_debug('training', f"{symbol} 特征矩阵创建失败", log_type="training")
                    return False
                self.algorithm.log_debug('training', f"{symbol} 特征矩阵创建成功: {feature_matrix.shape}", log_type="training")
            except Exception as feature_error:
                self.algorithm.log_debug('training', f"{symbol} 特征矩阵创建失败: {feature_error}", log_type="training")
                return False
            
            # 数据缩放（使用24个特征）
            try:
                prices_scaled = self.algorithm.data_processor.scale_data(feature_matrix, symbol, fit=True)
                self.algorithm.log_debug('training', f"{symbol} 特征数据缩放成功: {prices_scaled.shape}", log_type="training")
            except Exception as scale_error:
                self.algorithm.log_debug('training', f"{symbol} 特征数据缩放失败: {scale_error}", log_type="training")
                return False
            
            # 创建训练序列
            effective_lookback = min(60, len(prices_array) // 10)  # 预训练使用较长lookback
            self.algorithm.log_debug('training', f"{symbol} 使用lookback: {effective_lookback}", log_type="training")
            
            try:
                X, y = self.create_sequences(prices_scaled, effective_lookback)
                self.algorithm.log_debug('training', f"{symbol} 序列创建成功: X={X.shape}, y={y.shape}", log_type="training")
            except Exception as seq_error:
                self.algorithm.log_debug('training', f"{symbol} 序列创建失败: {seq_error}", log_type="training")
                return False
            
            # 检查序列数量
            min_sequences = 50 if self.config.get('pretrain_strict_mode', True) else 20
            if len(X) < min_sequences:
                self.algorithm.log_debug('training', f"{symbol} 序列数量不足: {len(X)} < {min_sequences}", log_type="training")
                return False
        
            # 保存预训练数据
            setattr(self, f'_pretrain_{symbol}_X', X)
            setattr(self, f'_pretrain_{symbol}_y', y)
            self.effective_lookbacks[symbol] = effective_lookback
            
            self.algorithm.log_debug('training', f"{symbol} 数据预处理完成: {len(X)}个训练序列", log_type="training")
            return True
            
        except Exception as e:
            self.algorithm.log_debug('training', f"预处理预训练数据 {symbol} 失败: {e}", log_type="training")
            return False
    
    def _build_and_train_pretrain_model(self, symbol):
        """构建和训练预训练模型（增强调试版）"""
        try:
            self.algorithm.log_debug('training', f"开始构建 {symbol} 的预训练模型", log_type="training")
            
            X = getattr(self, f'_pretrain_{symbol}_X')
            y = getattr(self, f'_pretrain_{symbol}_y')
            
            self.algorithm.log_debug('training', f"{symbol} 训练数据形状: X={X.shape}, y={y.shape}", log_type="training")
            
            # 根据模式选择模型复杂度
            use_simple_model = not self.config.get('pretrain_strict_mode', True)
            
            # 获取特征维度（应该是24）
            feature_dim = X.shape[2] if len(X.shape) > 2 else 1
            self.algorithm.log_debug('training', f"{symbol} 检测到特征维度: {feature_dim}", log_type="training")
            
            if use_simple_model:
                self.algorithm.log_debug('training', f"{symbol} 使用简化模型架构", log_type="training")
                # 简化模型架构 - 适配多特征输入
                input_layer = Input(shape=(X.shape[1], feature_dim))
                lstm = LSTM(units=32, return_sequences=False, dropout=0.1)(input_layer)
                dense = Dense(16, activation='relu')(lstm)
                output = Dense(1)(dense)
                model = Model(inputs=input_layer, outputs=output)
                epochs = 10  # 简化模式使用较少epochs
            else:
                self.algorithm.log_debug('training', f"{symbol} 使用完整模型架构", log_type="training")
                # 构建更深的预训练模型 - 适配多特征输入
                input_layer = Input(shape=(X.shape[1], feature_dim))
                
                # 多尺度CNN
                conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
                conv2 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same')(input_layer)
                
                # 合并卷积特征
                merged = tf.keras.layers.Concatenate()([conv1, conv2])
                
                # 多层LSTM
                lstm1 = LSTM(units=64, return_sequences=True, dropout=0.2)(merged)
                lstm2 = LSTM(units=32, return_sequences=True, dropout=0.2)(lstm1)
                
                # 注意力机制
                attention_weights = tf.keras.layers.Dense(1, activation='tanh')(lstm2)
                attention_weights = tf.keras.layers.Softmax(axis=1)(attention_weights)
                attended = tf.keras.layers.Multiply()([lstm2, attention_weights])
                
                # 全局池化和输出
                pooled = tf.keras.layers.GlobalAveragePooling1D()(attended)
                dense = tf.keras.layers.Dense(16, activation='relu')(pooled)
                output = tf.keras.layers.Dense(1)(dense)
                
                model = Model(inputs=input_layer, outputs=output)
                epochs = 50  # 完整模式使用更多epochs
            
            self.algorithm.log_debug('training', f"{symbol} 模型架构构建完成，开始编译", log_type="training")
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            self.algorithm.log_debug('training', f"{symbol} 开始训练模型: {epochs} epochs", log_type="training")
            
            # 训练模型
            start_train_time = time.time()
            history = model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ] if not use_simple_model else []
            )
            
            train_time = time.time() - start_train_time
            self.algorithm.log_debug('training', f"{symbol} 模型训练完成，用时{train_time:.1f}秒", log_type="training")
            
                        # 保存模型
            self.lstm_models[symbol] = model
            
            # 更新可交易股票列表
            if symbol not in self.tradable_symbols:
                self.tradable_symbols.append(symbol)
                self.algorithm.log_debug('training', f"添加到可交易列表: {symbol}", log_type="training")
            
            # 清理预训练数据  
            delattr(self, f'_pretrain_{symbol}_X')
            delattr(self, f'_pretrain_{symbol}_y')
            
            return True
    
        except Exception as e:
            self.algorithm.log_debug('training', f"构建预训练模型 {symbol} 失败: {e}", log_type="training")
            return False
    
    def should_perform_training(self):
        """判断是否应该执行训练"""
        current_time = self.algorithm.Time
        
        # 如果在交易时间且启用了快速模式
        if self.config['trading_time_fast_mode']:
            if self._is_trading_hours(current_time):
                return self._should_fast_training()
        
        # 如果是周末且启用了周末训练
        if self.config['enable_weekend_training']:
            if current_time.weekday() == self.config['weekend_training_day']:
                return self._should_weekend_training(current_time)
        
        # 默认训练判断逻辑
        return self._should_regular_training()
    
    def _is_trading_hours(self, current_time):
        """判断是否在交易时间"""
        # 简化实现：工作日的9:30-16:00认为是交易时间
        if current_time.weekday() >= 5:  # 周末
            return False

        trading_start = current_time.replace(hour=9, minute=30, second=0)
        trading_end = current_time.replace(hour=16, minute=0, second=0)
        
        return trading_start <= current_time <= trading_end
    
    def _should_fast_training(self):
        """交易时间快速训练判断"""
        # 只在必要时进行快速增量训练
        if not self.pretrained_models_loaded:
            return True  # 没有预训练模型时需要快速训练
        
        # 检查模型是否需要更新
        if self.last_training_time is None:
            return True
        
        # 超过一定时间没有训练则进行快速更新
        time_since_last = (self.algorithm.Time - self.last_training_time).total_seconds()
        return time_since_last > self.config['async_training_interval']
    
    def _should_weekend_training(self, current_time):
        """周末训练判断"""
        target_hour = self.config['weekend_training_hour']
        
        # 检查是否到了周末训练时间
        if current_time.hour == target_hour and not self.weekend_training_scheduled:
            self.weekend_training_scheduled = True
            return True
        
        # 重置周末训练标记
        if current_time.hour != target_hour:
            self.weekend_training_scheduled = False
        
        return False
    
    def _should_regular_training(self):
        """常规训练判断逻辑"""
        # 使用原有的训练判断逻辑
        return True
    
    def perform_optimized_training(self):
        """执行优化的训练流程"""
        current_time = self.algorithm.Time
        
        if self._is_trading_hours(current_time) and self.config['trading_time_fast_mode']:
            # 交易时间：快速增量训练
            return self._perform_fast_training()
        
        elif current_time.weekday() == self.config['weekend_training_day']:
            # 周末：深度训练
            return self._perform_weekend_training()
        
        else:
            # 其他时间：常规训练
            return self._perform_regular_training()
    
    def _perform_fast_training(self):
        """执行快速训练（交易时间）"""
        try:
            self.algorithm.log_debug('training', "开始快速增量训练...", log_type="training")
            fast_start = time.time()
            successful_updates = 0
            max_time = self.config['fast_mode_max_training']
            if not self.lstm_models:
                self.algorithm.log_debug('training', "警告: 没有预训练模型，尝试快速基础训练", log_type="training")
                return self._perform_emergency_training()
            for symbol in self.algorithm.config.SYMBOLS:
                if time.time() - fast_start > max_time:
                    break
                try:
                    if symbol not in self.lstm_models:
                        self.algorithm.log_debug('training', f"跳过 {symbol}: 无预训练模型", log_type="training")
                        continue
                    # 快速训练时，指标计算需求最少60天
                    if self._perform_incremental_update(symbol, min_days=60):
                        successful_updates += 1
                        self.algorithm.log_debug('training', f"成功更新 {symbol}", log_type="training")
                        if symbol not in self.tradable_symbols:
                            self.tradable_symbols.append(symbol)
                    else:
                        self.algorithm.log_debug('training', f"更新失败 {symbol}: 数据不足或处理错误", log_type="training")
                except Exception as symbol_error:
                    self.algorithm.log_debug('training', f"处理 {symbol} 时发生错误: {symbol_error}", log_type="training")
                    continue
            total_time = time.time() - fast_start
            self.algorithm.log_debug('training', f"快速训练完成: {successful_updates}个更新, 用时{total_time:.1f}秒", log_type="training")
            self.algorithm.log_debug('training', f"当前可交易股票: {self.tradable_symbols} (共{len(self.tradable_symbols)}个)", log_type="training")
            if successful_updates > 0:
                self.update_algorithm_models()
                self.algorithm.log_debug('training', "增量训练模型已同步到主算法", log_type="training")
            if successful_updates == 0:
                self.algorithm.log_debug('training', "所有增量更新失败，尝试备用快速训练", log_type="training")
                return self._perform_emergency_training()
            self.last_training_time = self.algorithm.Time
            return successful_updates > 0
        except Exception as e:
            self.algorithm.log_debug('training', f"快速训练失败: {e}", log_type="training")
            return False
    
    def _perform_incremental_update(self, symbol, min_days=60):
        """执行增量模型更新（修正版 - 考虑交易日）"""
        try:
            # 获取最新的较多数据 - 提高数据量
            recent_data = None
            # 尝试不同的交易日数量：120, 90, 60, 30天，最小不少于min_days
            for trading_days in [120, 90, 60, 30]:
                if trading_days < min_days:
                    continue
                recent_data = self._get_recent_data(symbol, days=trading_days)
                if recent_data is not None and len(recent_data) >= min_days:
                    self.algorithm.log_debug('training', f"成功获取 {symbol} {len(recent_data)}个交易日数据（请求{trading_days}天）", log_type="training")
                    break
            if recent_data is None or len(recent_data) < min_days:
                self.algorithm.log_debug('training', f"获取 {symbol} 历史数据失败，尝试紧急模型构建", log_type="training")
                # 备用方案1：尝试构建紧急模型
                return self._try_emergency_model_for_symbol(symbol)
            
            # 如果有预训练模型，进行增量更新
            if symbol in self.lstm_models:
                success = self._update_model_incrementally(symbol, recent_data)
                if success:
                    return True
                else:
                    self.algorithm.log_debug('training', f"增量更新失败，尝试重建模型: {symbol}", log_type="training")
                    # 备用方案2：增量更新失败时重建模型
                    return self._try_emergency_model_for_symbol(symbol)
            
            # 备用方案3：没有预训练模型时构建新模型
            self.algorithm.log_debug('training', f"无预训练模型，构建新模型: {symbol}", log_type="training")
            return self._try_emergency_model_for_symbol(symbol)
            
        except Exception as e:
            self.algorithm.log_debug('training', f"增量更新 {symbol} 异常: {e}", log_type="training")
            # 备用方案4：异常时尝试紧急模型
            return self._try_emergency_model_for_symbol(symbol)
    
    def _perform_emergency_training(self):
        """紧急快速训练 - 当没有预训练模型时使用（增强版）"""
        try:
            self.algorithm.log_debug('training', "开始增强紧急训练...", log_type="training")
            emergency_start = time.time()
            successful_models = 0
            max_emergency_time = 300  # 增加到5分钟紧急训练限制
            
            for symbol in self.algorithm.config.SYMBOLS:
                if time.time() - emergency_start > max_emergency_time:
                    self.algorithm.log_debug('training', "紧急训练时间限制，停止训练", log_type="training")
                    break
                
                try:
                    # 增强的历史数据获取策略
                    history_data = None
                    for days in [150, 100, 75, 50]:  # 逐步降低数据需求
                        try:
                            history = self.algorithm.History(symbol, days, Resolution.Daily)
                            history_list = list(history)
                            
                            if len(history_list) >= 30:  # 降低最小要求到30天
                                history_data = np.array([x.Close for x in history_list])
                                self.algorithm.log_debug('training', f"{symbol}: 获得{len(history_data)}天历史数据", log_type="training")
                                break
                        except Exception:
                            continue
                    
                    if history_data is None:
                        self.algorithm.log_debug('training', f"跳过 {symbol}: 无法获取历史数据", log_type="training")
                        continue
                    
                    # 快速构建简单模型
                    if self._build_emergency_model(symbol, history_data):
                        successful_models += 1
                        self.algorithm.log_debug('training', f"紧急模型构建成功: {symbol}", log_type="training")
                    
                except Exception as symbol_error:
                    self.algorithm.log_debug('training', f"紧急训练 {symbol} 失败: {symbol_error}", log_type="training")
                    continue
            
            total_time = time.time() - emergency_start
            self.algorithm.log_debug('training', f"紧急训练完成: {successful_models}个模型, 用时{total_time:.1f}秒", log_type="training")
            
            return successful_models > 0
            
        except Exception as e:
            self.algorithm.log_debug('training', f"紧急训练失败: {e}", log_type="training")
            return False
    
    def _try_emergency_model_for_symbol(self, symbol):
        """为单个股票尝试构建紧急模型（修正版 - 考虑交易日）"""
        try:
            # 尝试获取不同长度的历史数据（使用交易日思维）
            # 考虑到周末和假期，实际日历天数需要更多
            for trading_days in [150, 120, 90, 60]:  # 交易日数量
                try:
                    # 转换为日历天数（交易日 * 1.4 约等于日历天数）
                    calendar_days = int(trading_days * 1.4)
                    
                    self.algorithm.log_debug('training', f"尝试获取 {symbol} {trading_days}个交易日数据（{calendar_days}个日历天数）", log_type="training")
                    
                    history = self.algorithm.History(symbol, calendar_days, Resolution.Daily)
                    history_list = list(history)
                    
                    # 检查是否有足够的交易日数据
                    if len(history_list) >= max(50, trading_days // 2):  # 至少需要目标的一半
                        prices = np.array([x.Close for x in history_list])
                        
                        # 只使用最近的交易日数据
                        if len(prices) > trading_days:
                            prices = prices[-trading_days:]
                        
                        if self._build_emergency_model(symbol, prices):
                            self.algorithm.log_debug('training', f"紧急模型构建成功: {symbol} (使用{len(prices)}个交易日数据)", log_type="training")
                            return True
                        else:
                            self.algorithm.log_debug('training', f"紧急模型构建失败: {symbol} (数据长度: {len(prices)})", log_type="training")
                            
                except Exception as history_error:
                    self.algorithm.log_debug('training', f"获取{trading_days}个交易日历史数据失败 {symbol}: {history_error}", log_type="training")
                    continue
            
            self.algorithm.log_debug('training', f"所有紧急模型构建尝试失败: {symbol}", log_type="training")
            return False
            
        except Exception as e:
            self.algorithm.log_debug('training', f"紧急模型构建异常 {symbol}: {e}", log_type="training")
            return False

    def _build_emergency_model(self, symbol, prices):
        """构建紧急简单模型（增强版）"""
        try:
            # 使用24维特征矩阵，保持与预训练的一致性
            if not hasattr(self.algorithm, 'data_processor') or self.algorithm.data_processor is None:
                self.algorithm.log_debug('training', f"数据处理器不可用 {symbol}，无法创建特征矩阵", log_type="training")
                return False
            
            try:
                # 创建24维特征矩阵
                feature_matrix = self.algorithm.data_processor.create_feature_matrix(prices)
                if feature_matrix is None:
                    self.algorithm.log_debug('training', f"特征矩阵创建失败 {symbol}", log_type="training")
                    return False
                
                # 使用数据处理器进行缩放
                prices_scaled = self.algorithm.data_processor.scale_data(feature_matrix, symbol, fit=True)
                self.algorithm.log_debug('training', f"特征矩阵缩放成功 {symbol}: {prices_scaled.shape}", log_type="training")
                
            except Exception as feature_error:
                self.algorithm.log_debug('training', f"特征矩阵处理失败 {symbol}: {feature_error}", log_type="training")
                return False
            
            # 动态调整lookback
            lookback = min(15, max(5, len(prices_scaled) // 6))  # 更灵活的lookback
            X, y = self.create_sequences(prices_scaled, lookback)
            
            if len(X) < 5:  # 降低最小序列要求
                self.algorithm.log_debug('training', f"序列数量不足 {symbol}: {len(X)} < 5", log_type="training")
                return False
            
            # 构建模型，使用24维特征
            feature_dim = X.shape[2] if len(X.shape) > 2 else 24  # 确保使用24维
            input_layer = Input(shape=(X.shape[1], feature_dim))
            
            # 使用更简单的架构
            if len(X) < 20:
                # 数据很少时使用最简单的模型
                lstm = LSTM(units=8, return_sequences=False, dropout=0.1)(input_layer)
                output = Dense(1)(lstm)
            else:
                # 数据较多时使用稍复杂的模型
                lstm = LSTM(units=16, return_sequences=False, dropout=0.2)(input_layer)
                dense = Dense(8, activation='relu')(lstm)
                output = Dense(1)(dense)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # 较高学习率快速收敛
                loss='mse', 
                metrics=['mae']
            )
            
            # 动态调整训练参数
            epochs = min(10, max(3, len(X) // 5))  # 根据数据量调整epochs
            batch_size = min(len(X), max(4, len(X) // 4))  # 动态batch size
            
            # 快速训练
            model.fit(
                X, y, 
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=0,
                validation_split=0.1 if len(X) > 10 else 0
            )
            
            # 保存模型
            self.lstm_models[symbol] = model
            self.effective_lookbacks[symbol] = lookback
            
            if symbol not in self.tradable_symbols:
                self.tradable_symbols.append(symbol)
            
            self.algorithm.log_debug('training', f"紧急模型构建成功 {symbol}: {len(X)}序列, {epochs}epochs, lookback={lookback}", log_type="training")
            return True
            
        except Exception as e:
            self.algorithm.log_debug('training', f"构建紧急模型 {symbol} 失败: {e}", log_type="training")
            return False
    
    def _clean_price_data(self, prices):
        """清洗价格数据，移除异常值"""
        try:
            prices = np.array(prices)
            
            # 移除NaN和无穷值
            prices = prices[~np.isnan(prices)]
            prices = prices[~np.isinf(prices)]
            
            if len(prices) < 10:
                return prices
            
            # 移除极端异常值（超过3个标准差）
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            
            if std_price > 0:
                z_scores = np.abs((prices - mean_price) / std_price)
                prices = prices[z_scores < 3]
            
            return prices
            
        except Exception:
            return prices

    def _get_recent_data(self, symbol, days=60):
        """获取最近的数据（修正版 - 默认60天，覆盖指标需求）"""
        try:
            # 计算实际需要的日历天数（考虑周末和假期）
            # 经验公式：交易日 * 1.4 ≈ 日历天数（考虑周末和假期）
            calendar_days_needed = max(days * 2, days + 10)  # 至少增加10天缓冲
            
            self.algorithm.log_debug(f"请求 {symbol} {days}个交易日数据，使用{calendar_days_needed}个日历天数", log_type="training")
            
            # 方案1：尝试获取足够的日历天数来确保有足够的交易日
            try:
                history = self.algorithm.History(symbol, calendar_days_needed, Resolution.Daily)
                history_list = list(history)
                
                if len(history_list) >= days:
                    prices = np.array([x.Close for x in history_list])
                    # 验证数据质量
                    if len(prices) >= days and not np.any(np.isnan(prices)) and not np.any(np.isinf(prices)):
                        # 只取最近的指定天数
                        recent_prices = prices[-days:] if len(prices) > days else prices
                        self.algorithm.log_debug(f"成功获取 {symbol} {len(recent_prices)}个交易日数据", log_type="training")
                        return recent_prices
                else:
                    self.algorithm.log_debug(f"数据不足 {symbol}: 需要{days}天，实际获得{len(history_list)}天", log_type="training")
            except Exception as daily_error:
                self.algorithm.log_debug(f"获取日线数据失败 {symbol}: {daily_error}", log_type="training")
            
            # 方案2：如果还是不够，尝试更长的时间窗口
            try:
                extended_days = calendar_days_needed * 2  # 进一步扩大时间窗口
                self.algorithm.log_debug(f"尝试扩展时间窗口到{extended_days}天", log_type="training")
                
                history = self.algorithm.History(symbol, extended_days, Resolution.Daily)
                history_list = list(history)
                
                if len(history_list) >= days:
                    prices = np.array([x.Close for x in history_list])
                    if len(prices) >= days and not np.any(np.isnan(prices)) and not np.any(np.isinf(prices)):
                        recent_prices = prices[-days:]
                        self.algorithm.log_debug(f"扩展窗口成功获取 {symbol} {len(recent_prices)}个交易日数据", log_type="training")
                        return recent_prices
            except Exception as extended_error:
                self.algorithm.log_debug(f"扩展时间窗口失败 {symbol}: {extended_error}", log_type="training")
            
            # 方案3：尝试获取小时数据并降采样
            try:
                # 使用更多小时数据来确保有足够的交易日
                hour_periods = calendar_days_needed * 8  # 每天8小时交易时间
                history = self.algorithm.History(symbol, hour_periods, Resolution.Hour)
                history_list = list(history)
                
                if len(history_list) > 0:
                    prices = np.array([x.Close for x in history_list])
                    if len(prices) > 0 and not np.any(np.isnan(prices)) and not np.any(np.isinf(prices)):
                        # 降采样到日线（取每8个小时的最后一个价格，近似每日收盘价）
                        daily_prices = prices[7::8] if len(prices) >= 8 else prices[::max(1, len(prices)//days)]
                        if len(daily_prices) >= days:
                            recent_prices = daily_prices[-days:]
                            self.algorithm.log_debug(f"小时数据降采样成功 {symbol}: {len(recent_prices)}个数据点", log_type="training")
                            return recent_prices
            except Exception as hour_error:
                self.algorithm.log_debug(f"获取小时数据失败 {symbol}: {hour_error}", log_type="training")
            
            # 方案4：尝试从当前持仓获取价格信息（作为最后备用）
            try:
                if hasattr(self.algorithm, 'Portfolio') and hasattr(self.algorithm.Portfolio, symbol):
                    portfolio_item = getattr(self.algorithm.Portfolio, symbol)
                    if portfolio_item.Price > 0:
                        current_price = portfolio_item.Price
                        # 创建基于当前价格的合理价格序列（模拟小幅波动）
                        synthetic_prices = []
                        for i in range(days):
                            # 使用随机游走模拟价格变化
                            price_change = current_price * 0.001 * (np.random.random() - 0.5) * 2  # ±0.1%随机变化
                            synthetic_prices.append(current_price + price_change * i)
                        
                        synthetic_prices = np.array(synthetic_prices)
                        self.algorithm.log_debug(f"使用合成价格数据 {symbol}: {len(synthetic_prices)}个数据点", log_type="training")
                        return synthetic_prices
            except Exception as portfolio_error:
                self.algorithm.log_debug(f"从投资组合获取价格失败 {symbol}: {portfolio_error}", log_type="training")
            
            # 方案5：尝试从数据slice获取当前价格
            try:
                if hasattr(self.algorithm, 'data_slice') and self.algorithm.data_slice is not None:
                    if self.algorithm.data_slice.ContainsKey(symbol):
                        current_data = self.algorithm.data_slice[symbol]
                        if hasattr(current_data, 'Price') and current_data.Price > 0:
                            current_price = current_data.Price
                            # 创建基于当前价格的价格序列
                            synthetic_prices = []
                            for i in range(days):
                                price_change = current_price * 0.001 * (i - days//2)  # 线性变化
                                synthetic_prices.append(current_price + price_change)
                            
                            synthetic_prices = np.array(synthetic_prices)
                            self.algorithm.log_debug(f"使用当前数据价格 {symbol}: {current_price}", log_type="training")
                            return synthetic_prices
            except Exception as slice_error:
                self.algorithm.log_debug(f"从数据slice获取价格失败 {symbol}: {slice_error}", log_type="training")
            
            self.algorithm.log_debug(f"所有数据获取方案失败 {symbol}", log_type="training")
            return None
            
        except Exception as e:
            self.algorithm.log_debug(f"获取数据异常 {symbol}: {e}", log_type="training")
            return None
    
    def _update_model_incrementally(self, symbol, recent_prices):
        """增量更新模型（增强版 - 多重容错机制）"""
        try:
            # 数据预处理 - 使用24维特征矩阵
            prices_scaled = None
            
            # 方案1：使用数据处理器创建24维特征矩阵
            if hasattr(self.algorithm, 'data_processor') and self.algorithm.data_processor is not None:
                try:
                    # 创建24维特征矩阵
                    feature_matrix = self.algorithm.data_processor.create_feature_matrix(recent_prices)
                    if feature_matrix is not None:
                        prices_scaled = self.algorithm.data_processor.scale_data(
                            feature_matrix, symbol, fit=False
                        )
                        self.algorithm.log_debug(f"特征矩阵缩放成功 {symbol}: {prices_scaled.shape}", log_type="training")
                    else:
                        self.algorithm.log_debug(f"特征矩阵创建失败 {symbol}", log_type="training")
                except Exception as scale_error:
                    self.algorithm.log_debug(f"特征矩阵处理失败 {symbol}: {scale_error}", log_type="training")
            
            # 方案2：如果特征矩阵创建失败，返回False
            if prices_scaled is None:
                self.algorithm.log_debug(f"无法创建24维特征矩阵 {symbol}，跳过增量更新", log_type="training")
                return False
            
            # 动态调整lookback
            original_lookback = self.effective_lookbacks.get(symbol, 20)
            data_length = len(prices_scaled)
            
            # 获取现有模型的期望输入形状
            model = self.lstm_models[symbol]
            expected_shape = model.input_shape
            expected_lookback = expected_shape[1]  # 期望的序列长度
            
            self.algorithm.log_debug(f"模型期望输入形状 {symbol}: {expected_shape}, 需要lookback={expected_lookback}", log_type="training")
            
            # 优先尝试匹配原始模型的lookback
            lookback_candidates = [expected_lookback]
            
            # 如果数据不足，尝试数据填充或其他策略
            if data_length <= expected_lookback:
                self.algorithm.log_debug(f"数据长度不足 {symbol}: {data_length} <= {expected_lookback}，尝试数据填充", log_type="training")
                
                # 方案1：数据填充（重复最后几个值）
                if data_length >= 5:
                    padding_needed = expected_lookback - data_length + 1
                    last_values = prices_scaled[-3:]  # 取最后3个值
                    padding = np.tile(last_values, (padding_needed // 3 + 1, 1))[:padding_needed]
                    prices_scaled_padded = np.vstack([padding, prices_scaled])
                    
                    self.algorithm.log_debug(f"数据填充 {symbol}: {data_length} -> {len(prices_scaled_padded)}", log_type="training")
                    
                    # 使用填充后的数据
                    X_new, y_new = self.create_sequences(prices_scaled_padded, expected_lookback)
                    
                    if len(X_new) >= 1:
                        try:
                            # 动态调整训练参数
                            epochs = min(3, max(1, len(X_new) // 2))  # 填充数据用更少epochs
                            batch_size = min(len(X_new), max(1, len(X_new)))
                            
                            # 进行增量训练
                            model.fit(
                                X_new, y_new,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0
                            )
                            
                            self.algorithm.log_debug(f"增量更新成功(填充数据) {symbol}: {len(X_new)}序列, epochs={epochs}", log_type="training")
                            return True
                            
                        except Exception as train_error:
                            self.algorithm.log_debug(f"填充数据训练失败 {symbol}: {train_error}", log_type="training")
                
                # 方案2：如果填充也失败，返回False让上层处理
                self.algorithm.log_debug(f"数据填充方案失败 {symbol}", log_type="training")
                return False
            
            # 如果数据足够，尝试使用期望的lookback
            for lookback in lookback_candidates:
                if data_length > lookback:
                    try:
                        # 创建训练序列
                        X_new, y_new = self.create_sequences(prices_scaled, lookback)
                        
                        if len(X_new) >= 1:
                            # 验证模型输入形状兼容性
                            if X_new.shape[1:] != expected_shape[1:]:
                                self.algorithm.log_debug(f"输入形状仍不匹配 {symbol}: {X_new.shape[1:]} vs {expected_shape[1:]}", log_type="training")
                                continue
                            
                            # 动态调整训练参数
                            epochs = min(5, max(1, len(X_new) // 2))
                            batch_size = min(len(X_new), max(1, len(X_new) // 2))
                            
                            # 进行增量训练
                            model.fit(
                                X_new, y_new,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0
                            )
                            
                            # 更新lookback（如果改变了）
                            if lookback != original_lookback:
                                self.effective_lookbacks[symbol] = lookback
                            
                            self.algorithm.log_debug(f"增量更新成功 {symbol}: {len(X_new)}序列, lookback={lookback}, epochs={epochs}", log_type="training")
                            return True
                            
                    except Exception as sequence_error:
                        self.algorithm.log_debug(f"序列创建失败 {symbol} (lookback={lookback}): {sequence_error}", log_type="training")
                        continue
            
            self.algorithm.log_debug(f"所有lookback尝试失败 {symbol}", log_type="training")
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"增量更新异常 {symbol}: {e}", log_type="training")
            return False
    
    def create_sequences(self, data, seq_length):
        """创建训练序列"""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def get_model_for_symbol(self, symbol):
        """获取指定股票的模型"""
        return self.lstm_models.get(symbol, None)
    
    def get_scaler_for_symbol(self, symbol):
        """获取指定股票的scaler"""
        return self.algorithm.data_processor.scalers.get(symbol, None)
    
    def get_effective_lookback_for_symbol(self, symbol):
        """获取指定股票的有效lookback"""
        return self.effective_lookbacks.get(symbol, 30)
    
    def get_tradable_symbols(self):
        """获取可交易的股票列表"""
        return self.tradable_symbols
    
    def update_algorithm_models(self):
        """更新算法的模型引用（增强版 - 同步到model_trainer）"""
        # 更新算法的lstm_models
        if hasattr(self.algorithm, 'lstm_models'):
            self.algorithm.lstm_models = self.lstm_models
        if hasattr(self.algorithm, 'scalers'):
            self.algorithm.scalers = self.algorithm.data_processor.scalers
        
        # 同步模型到model_trainer，确保get_tradable_symbols()能正确工作
        if hasattr(self.algorithm, 'model_trainer') and self.algorithm.model_trainer is not None:
            try:
                # 将预训练模型同步到model_trainer.models
                for symbol, model in self.lstm_models.items():
                    model_info = {
                        'model': model,
                        'effective_lookback': self.effective_lookbacks.get(symbol, 30),
                        'training_time': 0,
                        'final_loss': 'pretrained',
                        'source': 'advanced_training_manager'
                    }
                    self.algorithm.model_trainer.models[symbol] = model_info
                
                self.algorithm.log_debug(f"同步了{len(self.lstm_models)}个预训练模型到ModelTrainer", log_type="training")
                
                # 也同步到multi_horizon_model
                if hasattr(self.algorithm.model_trainer, 'multi_horizon_model'):
                    for symbol, model in self.lstm_models.items():
                        model_info = {
                            'model': model,
                            'effective_lookback': self.effective_lookbacks.get(symbol, 30),
                            'training_time': 0,
                            'final_loss': 'pretrained',
                            'source': 'advanced_training_manager'
                        }
                        self.algorithm.model_trainer.multi_horizon_model.models[symbol] = model_info
                
            except Exception as sync_error:
                self.algorithm.log_debug(f"同步模型到ModelTrainer失败: {sync_error}", log_type="training")
    
    def _cache_models(self):
        """缓存模型到ObjectStore"""
        try:
            if not self.config['pretrain_model_cache']:
                return
            
            # 将scalers转换为可序列化的格式
            serializable_scalers = {}
            for symbol, scaler in self.algorithm.data_processor.scalers.items():
                try:
                    serializable_scalers[symbol] = {
                        'data_min_': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                        'data_max_': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
                        'data_range_': scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None,
                        'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                        'min_': scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                        'feature_range': scaler.feature_range
                    }
                except Exception as scaler_error:
                    self.algorithm.log_debug(f"序列化scaler {symbol} 失败: {scaler_error}", log_type="training")
                    continue
            
            cache_data = {
                'models': {},
                'scalers': serializable_scalers,
                'effective_lookbacks': self.effective_lookbacks,
                'cache_time': self.algorithm.Time.isoformat(),
                'tradable_symbols': list(self.lstm_models.keys())
            }
            
            # 保存模型权重而不是完整模型
            for symbol, model in self.lstm_models.items():
                try:
                    cache_data['models'][symbol] = {
                        'weights': [w.tolist() for w in model.get_weights()],  # 转换为可序列化的列表
                        'config': model.get_config(),
                        'architecture': model.to_json()
                    }
                except Exception as model_error:
                    self.algorithm.log_debug(f"缓存模型 {symbol} 失败: {model_error}", log_type="training")
                    continue
            
            # 使用ObjectStore保存缓存数据
            cache_key = self.model_cache_key
            
            # 将数据序列化为JSON字符串
            cache_json = json.dumps(cache_data, ensure_ascii=False, indent=None)
            
            # 保存到ObjectStore
            success = self.algorithm.ObjectStore.Save(cache_key, cache_json)
            
            if success:
                self.algorithm.log_debug(f"模型缓存已保存到ObjectStore: {len(cache_data['models'])}个模型", log_type="training")
            else:
                self.algorithm.log_debug("模型缓存保存到ObjectStore失败", log_type="training")
            
        except Exception as e:
            self.algorithm.log_debug(f"模型缓存失败: {e}", log_type="training")
    
    def _load_cached_models(self):
        """从ObjectStore加载缓存的模型"""
        try:
            cache_key = self.model_cache_key
            
            # 检查ObjectStore中是否存在缓存
            if not self.algorithm.ObjectStore.ContainsKey(cache_key):
                self.algorithm.log_debug("ObjectStore中未找到缓存的预训练模型", log_type="training")
                return False
            
            # 从ObjectStore读取缓存数据
            cache_json = self.algorithm.ObjectStore.Read(cache_key)
            
            if not cache_json:
                self.algorithm.log_debug("从ObjectStore读取缓存数据失败", log_type="training")
                return False
            
            # 解析JSON数据
            cache_data = json.loads(cache_json)
            
            # 检查缓存有效性（例如：不超过一周）
            cache_time = datetime.fromisoformat(cache_data['cache_time'])
            if (self.algorithm.Time - cache_time).days > 7:
                self.algorithm.log_debug("缓存模型过期，将重新训练", log_type="training")
                return False
            
            # 重建模型
            models_loaded = 0
            for symbol, model_data in cache_data['models'].items():
                try:
                    # 从JSON重建模型架构
                    model = tf.keras.models.model_from_json(model_data['architecture'])
                    
                    # 恢复权重（从列表转换回numpy数组）
                    weights = [np.array(w) for w in model_data['weights']]
                    model.set_weights(weights)
                    
                    # 重新编译模型
                    model.compile(
                        optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mae']
                    )
                    
                    self.lstm_models[symbol] = model
                    models_loaded += 1
                    
                except Exception as model_error:
                    self.algorithm.log_debug(f"加载模型 {symbol} 失败: {model_error}", log_type="training")
                    continue
            
            # 恢复scalers（重建MinMaxScaler对象）
            restored_scalers = {}
            cached_scalers = cache_data.get('scalers', {})
            for symbol, scaler_data in cached_scalers.items():
                try:
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler(feature_range=scaler_data.get('feature_range', (0, 1)))
                    
                    # 恢复scaler的属性
                    if scaler_data.get('data_min_') is not None:
                        scaler.data_min_ = np.array(scaler_data['data_min_'])
                    if scaler_data.get('data_max_') is not None:
                        scaler.data_max_ = np.array(scaler_data['data_max_'])
                    if scaler_data.get('data_range_') is not None:
                        scaler.data_range_ = np.array(scaler_data['data_range_'])
                    if scaler_data.get('scale_') is not None:
                        scaler.scale_ = np.array(scaler_data['scale_'])
                    if scaler_data.get('min_') is not None:
                        scaler.min_ = np.array(scaler_data['min_'])
                    
                    restored_scalers[symbol] = scaler
                    
                except Exception as scaler_restore_error:
                    self.algorithm.log_debug(f"恢复scaler {symbol} 失败: {scaler_restore_error}", log_type="training")
                    continue
            
            self.algorithm.data_processor.scalers = restored_scalers
            self.effective_lookbacks = cache_data.get('effective_lookbacks', {})
            self.tradable_symbols = cache_data.get('tradable_symbols', [])
            
            self.algorithm.log_debug(f"成功从ObjectStore加载 {models_loaded} 个缓存模型", log_type="training")
            return models_loaded > 0
            
        except Exception as e:
            self.algorithm.log_debug(f"从ObjectStore加载缓存模型失败: {e}", log_type="training")
            return False

    def _schedule_weekend_training(self):
        """设置周末训练调度"""
        try:
            self.algorithm.log_debug("周末训练调度已设置", log_type="training")
            # 这里可以添加更复杂的调度逻辑
            return True
        except Exception as e:
            self.algorithm.log_debug(f"设置周末训练调度失败: {e}", log_type="training")
            return False
    
    def _start_async_training(self):
        """启动异步训练线程"""
        try:
            if self.background_training_thread is not None and self.background_training_thread.is_alive():
                return
            
            self.algorithm.log_debug("异步训练线程已设置（暂时禁用以避免资源冲突）", log_type="training")
            # 暂时注释掉异步训练，避免在QuantConnect环境中的资源冲突
            # self.background_training_thread = threading.Thread(target=self._async_training_loop)
            # self.background_training_thread.daemon = True
            # self.background_training_thread.start()
            
        except Exception as e:
            self.algorithm.log_debug(f"启动异步训练失败: {e}", log_type="training")
    
    def _async_training_loop(self):
        """异步训练循环（暂时禁用）"""
        # 暂时禁用异步训练以避免在QuantConnect环境中的资源冲突
        pass

    def _test_object_store_connection(self):
        """测试ObjectStore连接"""
        try:
            test_key = "test_connection"
            test_data = "ObjectStore connection test"
            
            # 测试保存
            save_success = self.algorithm.ObjectStore.Save(test_key, test_data)
            if not save_success:
                self.algorithm.log_debug("ObjectStore保存测试失败", log_type="training")
                return False
            
            # 测试读取
            if not self.algorithm.ObjectStore.ContainsKey(test_key):
                self.algorithm.log_debug("ObjectStore键检查失败", log_type="training")
                return False
            
            read_data = self.algorithm.ObjectStore.Read(test_key)
            if read_data != test_data:
                self.algorithm.log_debug("ObjectStore读取测试失败", log_type="training")
                return False
            
            # 清理测试数据
            self.algorithm.ObjectStore.Delete(test_key)
            
            self.algorithm.log_debug("ObjectStore连接测试成功", log_type="training")
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"ObjectStore连接测试失败: {e}", log_type="training")
            return False


# === 兼容性：保留原始TrainingManager类 ===
class TrainingManager(AdvancedTrainingManager):
    """兼容性训练管理器"""
    
    def __init__(self, algorithm):
        super().__init__(algorithm)
        self.algorithm.log_debug("使用高级训练管理器（兼容模式）", log_type="training")
    
    def should_retrain(self):
        """兼容性方法"""
        return self.should_perform_training()
    
    def perform_training(self):
        """兼容性方法"""
        return self.perform_optimized_training()

def fix_log_debug_calls(lines: List[str]) -> List[str]:
    pattern = re.compile(r'self\.algorithm\.log_debug\((f?"[^"]*"|f?\'[^"]*\')\)')
    fixed = []
    for line in lines:
        m = pattern.search(line)
        if m:
            msg = m.group(1)
            newline = line.replace(m.group(0), f"self.algorithm.log_debug({msg}, log_type='training')")
            fixed.append(newline)
        else:
            fixed.append(line)
    return fixed