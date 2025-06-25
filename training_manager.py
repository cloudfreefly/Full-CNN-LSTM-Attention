# region imports
from AlgorithmImports import *
# endregion

import time
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from datetime import datetime, timedelta
import json

class TrainingManager:
    """精简训练管理器 - 核心功能版本"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.config = algorithm.config.TRAINING_CONFIG['training_time_separation']
        
        # 模型存储
        self.lstm_models = {}
        self.scalers = {}
        self.effective_lookbacks = {}
        self.tradable_symbols = []
        
        # 训练状态
        self.last_training_time = None
        self.pretrained_models_loaded = False
        self.last_full_training_time = None
        self._recent_prediction_success_rate = 1.0
        self._prediction_success_history = []
        
        self.algorithm.log_debug("训练管理器初始化完成", log_type="training")
    
    def initialize_training_system(self):
        """初始化训练系统"""
        try:
            return self._load_or_create_pretrained_models()
        except Exception as e:
            self.algorithm.log_debug(f"训练系统初始化失败: {e}", log_type="training")
            return False
    
    def _load_or_create_pretrained_models(self):
        """加载或创建预训练模型"""
        try:
            # 尝试加载缓存模型
            if self.config.get('pretrain_model_cache', True) and self._load_cached_models():
                self.algorithm.log_debug(f"加载缓存模型: {len(self.lstm_models)}个", log_type="training")
                self.pretrained_models_loaded = True
                return True
            
            # 启动时预训练
            if self.config.get('enable_pretrain', True) and self.config.get('pretrain_on_startup', True):
                self.algorithm.log_debug("✅ 启动预训练...", log_type="training")
                if self._perform_pretrain():
                    if self.config.get('pretrain_model_cache', True):
                        self._cache_models()
                    self.pretrained_models_loaded = True
                    return True
                else:
                    self.algorithm.log_debug("预训练失败，将使用紧急训练", log_type="training")
                    self.pretrained_models_loaded = False
                    return False
            else:
                self.algorithm.log_debug("预训练已禁用", log_type="training")
                self.pretrained_models_loaded = False
                return False
                
        except Exception as e:
            self.algorithm.log_debug(f"预训练模型处理失败: {e}", log_type="training")
            self.pretrained_models_loaded = False
            return False
    
    def _perform_pretrain(self):
        """执行预训练"""
        try:
            start_time = time.time()
            successful_models = 0
            history_days = self.config.get('pretrain_history_months', 36) * 30
            
            for symbol in self.algorithm.config.SYMBOLS:
                try:
                    history_data = self._get_history_data(symbol, history_days)
                    if history_data is not None and len(history_data) >= 100:
                        if self._train_model(symbol, history_data):
                            successful_models += 1
                            self.algorithm.log_debug(f"预训练完成: {symbol}", log_type="training")
                    
                    # 时间控制
                    if time.time() - start_time > 1800:  # 30分钟限制
                        break
                        
                except Exception as e:
                    self.algorithm.log_debug(f"预训练{symbol}失败: {e}", log_type="training")
                    continue
            
            total_time = time.time() - start_time
            self.algorithm.log_debug(f"预训练完成: {successful_models}个模型, {total_time:.1f}秒", log_type="training")
            
            if successful_models > 0:
                self.update_algorithm_models()
                return True
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"预训练执行失败: {e}", log_type="training")
            return False
    
    def _get_history_data(self, symbol, days):
        """获取历史数据"""
        try:
            # 方法1：时间范围
            try:
                end_date = self.algorithm.Time
                start_date = end_date - timedelta(days=days)
                history = self.algorithm.History(symbol, start_date, end_date, Resolution.Daily)
                history_list = list(history)
                if len(history_list) > 0:
                    return np.array([x.Close for x in history_list])
            except:
                pass
            
            # 方法2：直接天数
            try:
                history = self.algorithm.History(symbol, days, Resolution.Daily)
                history_list = list(history)
                if len(history_list) > 0:
                    return np.array([x.Close for x in history_list])
            except:
                pass
            
            # 方法3：降级获取
            for fallback_days in [days//2, days//4, 100]:
                try:
                    history = self.algorithm.History(symbol, fallback_days, Resolution.Daily)
                    history_list = list(history)
                    if len(history_list) >= 30:
                        return np.array([x.Close for x in history_list])
                except:
                    continue
            
            return None
            
        except Exception as e:
            self.algorithm.log_debug(f"获取{symbol}历史数据失败: {e}", log_type="training")
            return None
    
    def _train_model(self, symbol, prices):
        """训练单个模型"""
        try:
            # 数据预处理
            if not self._preprocess_data(symbol, prices):
                return False
            
            # 构建和训练模型
            return self._build_and_train_model(symbol)
            
        except Exception as e:
            self.algorithm.log_debug(f"训练{symbol}模型失败: {e}", log_type="training")
            return False
    
    def _preprocess_data(self, symbol, prices):
        """预处理数据"""
        try:
            prices_array = np.array(prices)
            
            # 数据验证
            if len(prices_array) < 50 or np.sum(np.isnan(prices_array)) > 0:
                return False
            
            # 创建特征矩阵
            try:
                feature_matrix = self.algorithm.data_processor.create_feature_matrix(prices_array)
                if feature_matrix is None:
                    return False
            except:
                return False
            
            # 数据缩放
            try:
                prices_scaled = self.algorithm.data_processor.scale_data(feature_matrix, symbol, fit=True)
                if prices_scaled is None:
                    return False
            except:
                return False
            
            # 创建序列
            lookback = min(60, len(prices_array) // 10)
            X, y = self.create_sequences(prices_scaled, lookback)
            
            if len(X) < 10:
                return False
            
            # 保存预处理数据
            setattr(self, f'_{symbol}_X', X)
            setattr(self, f'_{symbol}_y', y)
            self.effective_lookbacks[symbol] = lookback
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"预处理{symbol}数据失败: {e}", log_type="training")
            return False
    
    def _build_and_train_model(self, symbol):
        """构建和训练模型"""
        try:
            X = getattr(self, f'_{symbol}_X')
            y = getattr(self, f'_{symbol}_y')
            
            # 构建模型
            feature_dim = X.shape[2] if len(X.shape) > 2 else 24
            input_layer = Input(shape=(X.shape[1], feature_dim))
            
            # 简化的模型架构
            if len(X) < 50:
                lstm = LSTM(units=16, return_sequences=False, dropout=0.2)(input_layer)
                output = Dense(1)(lstm)
            else:
                conv = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
                lstm = LSTM(units=32, return_sequences=False, dropout=0.3)(conv)
                dense = Dense(16, activation='relu')(lstm)
                output = Dense(1)(dense)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            # 训练参数
            epochs = min(20, max(5, len(X) // 10))
            batch_size = min(len(X), max(8, len(X) // 5))
            
            # 训练模型
            model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=0.1 if len(X) > 20 else 0
            )
            
            # 保存模型
            self.lstm_models[symbol] = model
            if symbol not in self.tradable_symbols:
                self.tradable_symbols.append(symbol)
            
            # 清理临时数据
            delattr(self, f'_{symbol}_X')
            delattr(self, f'_{symbol}_y')
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"构建{symbol}模型失败: {e}", log_type="training")
            return False
    
    def should_perform_training(self):
        """判断是否需要训练"""
        try:
            current_time = self.algorithm.Time
            
            # 检查是否需要全量重训练
            if self._should_full_retrain() and self._is_non_trading_hours():
                return True
            
            # 检查是否需要快速训练
            if self._is_trading_hours(current_time) and self._should_fast_training():
                return True
            
            # 检查常规训练
            if not self.pretrained_models_loaded or not self.lstm_models:
                return True
            
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"训练判断失败: {e}", log_type="training")
            return False
    
    def _is_trading_hours(self, current_time):
        """判断是否在交易时间"""
        try:
            trading_start = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            trading_end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            return trading_start <= current_time <= trading_end
        except:
            return True
    
    def _is_non_trading_hours(self):
        """判断是否在非交易时间"""
        return not self._is_trading_hours(self.algorithm.Time)
    
    def _should_full_retrain(self):
        """判断是否需要全量重训练"""
        # 没有模型
        if not self.lstm_models:
            return True
        
        # 模型数量不足
        available_models = len([m for m in self.lstm_models.values() if m is not None])
        required_models = len(self.algorithm.config.SYMBOLS)
        if available_models < required_models * 0.5:
            return True
        
        # 模型过旧
        if hasattr(self, 'last_full_training_time') and self.last_full_training_time:
            time_since_last = (self.algorithm.Time - self.last_full_training_time).total_seconds()
            if time_since_last > 7 * 24 * 3600:  # 7天
                return True
        
        # 预测成功率过低
        if hasattr(self, '_recent_prediction_success_rate'):
            if self._recent_prediction_success_rate < 0.3:
                return True
        
        return False
    
    def _should_fast_training(self):
        """判断是否需要快速训练"""
        if not self.pretrained_models_loaded or not self.lstm_models:
            return True
        
        # 检查训练频率
        if self.last_training_time:
            time_since_last = (self.algorithm.Time - self.last_training_time).total_seconds()
            if time_since_last < 3600:  # 1小时内不重复训练
                return False
        
        return True
    
    def perform_optimized_training(self):
        """执行优化训练"""
        try:
            current_time = self.algorithm.Time
            
            if self._is_trading_hours(current_time):
                # 交易时间：快速训练
                return self._perform_fast_training()
            else:
                # 非交易时间：全量重训练或常规训练
                if self._should_full_retrain():
                    return self._perform_full_retrain()
                else:
                    return self._perform_fast_training()
                    
        except Exception as e:
            self.algorithm.log_debug(f"优化训练失败: {e}", log_type="training")
            return False
    
    def _perform_fast_training(self):
        """执行快速训练"""
        try:
            self.algorithm.log_debug("开始快速训练...", log_type="training")
            successful_models = 0
            
            for symbol in self.algorithm.config.SYMBOLS:
                try:
                    # 获取最近数据
                    recent_data = self._get_recent_data(symbol, days=90)
                    if recent_data is not None and len(recent_data) >= 30:
                        if symbol in self.lstm_models:
                            # 增量更新
                            if self._update_model_incrementally(symbol, recent_data):
                                successful_models += 1
                        else:
                            # 构建新模型
                            if self._build_emergency_model(symbol, recent_data):
                                successful_models += 1
                                
                except Exception as e:
                    self.algorithm.log_debug(f"快速训练{symbol}失败: {e}", log_type="training")
                    continue
            
            self.last_training_time = self.algorithm.Time
            self.algorithm.log_debug(f"快速训练完成: {successful_models}个模型", log_type="training")
            
            return successful_models > 0
            
        except Exception as e:
            self.algorithm.log_debug(f"快速训练失败: {e}", log_type="training")
            return False
    
    def _get_recent_data(self, symbol, days=60):
        """获取最近数据"""
        try:
            history = self.algorithm.History(symbol, days, Resolution.Daily)
            history_list = list(history)
            if len(history_list) > 0:
                return np.array([x.Close for x in history_list])
            return None
        except:
            return None
    
    def _update_model_incrementally(self, symbol, recent_prices):
        """增量更新模型"""
        try:
            if symbol not in self.lstm_models:
                return False
            
            # 预处理新数据
            if not self._preprocess_data(symbol, recent_prices):
                return False
            
            X = getattr(self, f'_{symbol}_X')
            y = getattr(self, f'_{symbol}_y')
            
            # 增量训练
            model = self.lstm_models[symbol]
            model.fit(X, y, epochs=3, batch_size=min(len(X), 16), verbose=0)
            
            # 清理临时数据
            delattr(self, f'_{symbol}_X')
            delattr(self, f'_{symbol}_y')
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"增量更新{symbol}失败: {e}", log_type="training")
            return False
    
    def _build_emergency_model(self, symbol, prices):
        """构建紧急模型"""
        try:
            # 简化的紧急模型构建
            if not self._preprocess_data(symbol, prices):
                return False
            
            X = getattr(self, f'_{symbol}_X')
            y = getattr(self, f'_{symbol}_y')
            
            if len(X) < 5:
                return False
            
            # 简单模型架构
            feature_dim = X.shape[2] if len(X.shape) > 2 else 24
            input_layer = Input(shape=(X.shape[1], feature_dim))
            lstm = LSTM(units=8, return_sequences=False, dropout=0.1)(input_layer)
            output = Dense(1)(lstm)
            
            model = Model(inputs=input_layer, outputs=output)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
            
            # 快速训练
            epochs = min(5, max(2, len(X) // 5))
            model.fit(X, y, epochs=epochs, batch_size=min(len(X), 8), verbose=0)
            
            # 保存模型
            self.lstm_models[symbol] = model
            if symbol not in self.tradable_symbols:
                self.tradable_symbols.append(symbol)
            
            # 清理临时数据
            delattr(self, f'_{symbol}_X')
            delattr(self, f'_{symbol}_y')
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"构建紧急模型{symbol}失败: {e}", log_type="training")
            return False
    
    def _perform_full_retrain(self):
        """执行全量重训练"""
        try:
            self.algorithm.log_debug("开始全量重训练...", log_type="training")
            start_time = time.time()
            
            # 清理现有模型
            self._clear_existing_models()
            
            successful_models = 0
            max_time = 3600  # 1小时限制
            
            for symbol in self.algorithm.config.SYMBOLS:
                if time.time() - start_time > max_time:
                    break
                
                try:
                    history_data = self._get_history_data(symbol, 365)  # 1年数据
                    if history_data is not None and len(history_data) >= 60:
                        if self._train_model(symbol, history_data):
                            successful_models += 1
                            
                except Exception as e:
                    self.algorithm.log_debug(f"全量重训练{symbol}失败: {e}", log_type="training")
                    continue
            
            total_time = time.time() - start_time
            self.algorithm.log_debug(f"全量重训练完成: {successful_models}个模型, {total_time:.1f}秒", log_type="training")
            
            if successful_models > 0:
                self.update_algorithm_models()
                self.last_full_training_time = self.algorithm.Time
                self.pretrained_models_loaded = True
                
                if self.config.get('pretrain_model_cache', True):
                    self._cache_models()
                
                return True
            
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"全量重训练失败: {e}", log_type="training")
            return False
    
    def _clear_existing_models(self):
        """清理现有模型"""
        try:
            if hasattr(self, 'lstm_models'):
                self.lstm_models.clear()
            if hasattr(self, 'tradable_symbols'):
                self.tradable_symbols.clear()
            if hasattr(self, 'scalers'):
                self.scalers.clear()
            self.pretrained_models_loaded = False
        except Exception as e:
            self.algorithm.log_debug(f"清理模型失败: {e}", log_type="training")
    
    def update_prediction_success_rate(self, had_valid_predictions):
        """更新预测成功率"""
        try:
            self._prediction_success_history.append(1 if had_valid_predictions else 0)
            
            if len(self._prediction_success_history) > 20:
                self._prediction_success_history = self._prediction_success_history[-20:]
            
            if len(self._prediction_success_history) >= 5:
                self._recent_prediction_success_rate = sum(self._prediction_success_history) / len(self._prediction_success_history)
                
                if self._recent_prediction_success_rate < 0.3:
                    self.algorithm.log_debug(f"预测成功率偏低: {self._recent_prediction_success_rate:.1%}", log_type="training")
                    
        except Exception as e:
            self.algorithm.log_debug(f"更新预测成功率失败: {e}", log_type="training")
    
    def create_sequences(self, data, seq_length):
        """创建训练序列"""
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(data[i, 0] if len(data.shape) > 1 else data[i])
        return np.array(X), np.array(y)
    
    def get_model_for_symbol(self, symbol):
        """获取指定股票的模型"""
        return self.lstm_models.get(symbol, None)
    
    def get_scaler_for_symbol(self, symbol):
        """获取指定股票的缩放器"""
        return self.scalers.get(symbol, None)
    
    def get_effective_lookback_for_symbol(self, symbol):
        """获取指定股票的有效lookback"""
        return self.effective_lookbacks.get(symbol, 15)
    
    def get_tradable_symbols(self):
        """获取可交易股票列表"""
        return self.tradable_symbols.copy()
    
    def update_algorithm_models(self):
        """更新算法模型"""
        try:
            if hasattr(self.algorithm, 'model_trainer'):
                self.algorithm.model_trainer.lstm_models = self.lstm_models.copy()
                self.algorithm.model_trainer.scalers = self.scalers.copy()
                self.algorithm.model_trainer.effective_lookbacks = self.effective_lookbacks.copy()
                self.algorithm.model_trainer.tradable_symbols = self.tradable_symbols.copy()
                
                self.algorithm.log_debug(f"模型同步完成: {len(self.lstm_models)}个模型", log_type="training")
            else:
                self.algorithm.log_debug("model_trainer不存在，无法同步模型", log_type="training")
                
        except Exception as e:
            self.algorithm.log_debug(f"模型同步失败: {e}", log_type="training")
    
    def _cache_models(self):
        """缓存模型"""
        try:
            cache_data = {
                'models_count': len(self.lstm_models),
                'tradable_symbols': self.tradable_symbols,
                'effective_lookbacks': self.effective_lookbacks,
                'cache_time': self.algorithm.Time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            cache_key = self.config.get('pretrain_cache_file', 'pretrained_models')
            cache_json = json.dumps(cache_data)
            
            if self.algorithm.ObjectStore.Save(cache_key, cache_json):
                self.algorithm.log_debug(f"模型缓存成功: {cache_key}", log_type="training")
            else:
                self.algorithm.log_debug("模型缓存失败", log_type="training")
                
        except Exception as e:
            self.algorithm.log_debug(f"模型缓存异常: {e}", log_type="training")
    
    def _load_cached_models(self):
        """加载缓存模型"""
        try:
            cache_key = self.config.get('pretrain_cache_file', 'pretrained_models')
            
            if not self.algorithm.ObjectStore.ContainsKey(cache_key):
                self.algorithm.log_debug("缓存文件不存在，需要重新训练", log_type="training")
                return False
            
            cache_json = self.algorithm.ObjectStore.Read(cache_key)
            cache_data = json.loads(cache_json)
            
            # 验证缓存有效性
            models_count = cache_data.get('models_count', 0)
            if models_count > 0:
                self.tradable_symbols = cache_data.get('tradable_symbols', [])
                self.effective_lookbacks = cache_data.get('effective_lookbacks', {})
                
                # 重要：缓存中没有实际模型，需要重新训练
                # 但可以使用缓存的symbol列表快速重建模型
                self.algorithm.log_debug(f"缓存数据加载成功，需要重建{models_count}个模型", log_type="training")
                
                # 快速重建模型
                if self._rebuild_models_from_cache():
                    self.algorithm.log_debug(f"从缓存重建{len(self.lstm_models)}个模型", log_type="training")
                    return True
                else:
                    self.algorithm.log_debug("模型重建失败，需要完整预训练", log_type="training")
                    return False
            
            return False
            
        except Exception as e:
            self.algorithm.log_debug(f"加载缓存模型失败: {e}", log_type="training")
            return False
    
    def _rebuild_models_from_cache(self):
        """从缓存快速重建模型"""
        try:
            successful_models = 0
            max_rebuild_time = 300  # 5分钟限制
            start_time = time.time()
            
            for symbol in self.tradable_symbols[:10]:  # 限制重建数量
                if time.time() - start_time > max_rebuild_time:
                    break
                
                try:
                    # 获取较短的历史数据进行快速训练
                    history_data = self._get_history_data(symbol, 180)  # 6个月数据
                    if history_data is not None and len(history_data) >= 60:
                        if self._train_model(symbol, history_data):
                            successful_models += 1
                            
                except Exception as e:
                    self.algorithm.log_debug(f"重建模型{symbol}失败: {e}", log_type="training")
                    continue
            
            self.algorithm.log_debug(f"快速重建完成: {successful_models}个模型", log_type="training")
            return successful_models > 0
            
        except Exception as e:
            self.algorithm.log_debug(f"模型重建异常: {e}", log_type="training")
            return False
    
    # 兼容性方法
    def should_retrain(self):
        """兼容性方法"""
        return self.should_perform_training()
    
    def perform_training(self):
        """兼容性方法"""
        return self.perform_optimized_training()