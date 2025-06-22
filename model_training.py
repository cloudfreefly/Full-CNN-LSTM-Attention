# 模型训练模块
from AlgorithmImports import *
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization,
                             MultiHeadAttention, GlobalAveragePooling1D, Concatenate,
                             LayerNormalization, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import time
import gc
# 简化类型注解以兼容QuantConnect云端
from config import AlgorithmConfig
from data_processing import DataProcessor, DataValidator

import re

def fix_log_debug_calls(lines):
    pattern = re.compile(r'self\.algorithm\.log_debug\((f?"[^"]*"|f?\'[^"]*\')\)')
    fixed = []
    for line in lines:
        m = pattern.search(line)
        if m:
            msg = m.group(1)
            newline = line.replace(m.group(0), f"self.algorithm.log_debug('training', {msg})", log_type="training")
            fixed.append(newline)
        else:
            fixed.append(line)
    return fixed

class MultiHorizonModel:
    """多时间跨度预测模型"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        self.models = {}
        self.training_history = {}
        self.feature_importance = {}
        
    def build_model(self, input_shape, horizons):
        """构建多时间跨度CNN+LSTM+Attention模型"""
        
        inputs = Input(shape=input_shape, name='main_input')
        
        # 多尺度CNN层
        conv_outputs = []
        for i, (filters, kernel_size) in enumerate(zip(
            self.config.MODEL_CONFIG['conv_filters'],
            self.config.MODEL_CONFIG['conv_kernels']
        )):
            conv = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation='relu',
                padding='same',
                kernel_regularizer=l2(0.001),
                name=f'conv1d_{i+1}'
            )(inputs)
            conv = BatchNormalization(name=f'bn_conv_{i+1}')(conv)
            conv = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name=f'dropout_conv_{i+1}')(conv)
            conv_outputs.append(conv)
        
        # 合并多尺度特征
        if len(conv_outputs) > 1:
            merged_conv = Concatenate(axis=-1, name='conv_concat')(conv_outputs)
        else:
            merged_conv = conv_outputs[0]
        
        # 多层LSTM
        lstm_output = merged_conv
        for i, units in enumerate(self.config.MODEL_CONFIG['lstm_units']):
            return_sequences = (i < len(self.config.MODEL_CONFIG['lstm_units']) - 1) or True  # 最后一层也返回序列用于attention
            lstm_output = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.config.MODEL_CONFIG['dropout_rate'],
                recurrent_dropout=self.config.MODEL_CONFIG['dropout_rate'],
                kernel_regularizer=l2(0.001),
                name=f'lstm_{i+1}'
            )(lstm_output)
            lstm_output = BatchNormalization(name=f'bn_lstm_{i+1}')(lstm_output)
        
        # Multi-Head Attention机制
        attention_output = MultiHeadAttention(
            num_heads=self.config.MODEL_CONFIG['attention_heads'],
            key_dim=lstm_output.shape[-1] // self.config.MODEL_CONFIG['attention_heads'],
            dropout=self.config.MODEL_CONFIG['dropout_rate'],
            name='multi_head_attention'
        )(lstm_output, lstm_output)
        
        # 残差连接
        attention_output = Add(name='residual_connection')([lstm_output, attention_output])
        attention_output = LayerNormalization(name='layer_norm')(attention_output)
        
        # 全局平均池化
        pooled_output = GlobalAveragePooling1D(name='global_avg_pool')(attention_output)
        
        # 共享特征层
        shared_features = Dense(
            128, 
            activation='relu', 
            kernel_regularizer=l2(0.001),
            name='shared_dense'
        )(pooled_output)
        shared_features = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name='shared_dropout')(shared_features)
        
        # 为每个时间跨度创建专门的输出头
        outputs = []
        output_names = []
        for horizon in horizons:
            # 时间跨度特定的特征层
            horizon_features = Dense(
                64, 
                activation='relu',
                kernel_regularizer=l2(0.001),
                name=f'horizon_{horizon}_dense'
            )(shared_features)
            horizon_features = Dropout(self.config.MODEL_CONFIG['dropout_rate'], name=f'horizon_{horizon}_dropout')(horizon_features)
            
            # 输出层
            output = Dense(1, name=f'output_{horizon}d')(horizon_features)
            outputs.append(output)
            output_names.append(f'output_{horizon}d')
        
        # 创建模型
        model = Model(inputs=inputs, outputs=outputs, name='multi_horizon_model')
        
        # 编译模型 - 为不同时间跨度设置不同的损失权重
        losses = {name: 'mse' for name in output_names}
        loss_weights = {name: self.config.HORIZON_WEIGHTS[horizons[i]] for i, name in enumerate(output_names)}
        
        # 为每个输出指定metrics
        metrics_dict = {name: ['mae'] for name in output_names}
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics_dict
        )
        
        return model
    
    def prepare_training_data(self, data_processor, symbol, 
                            horizons):
        """准备训练数据 - 使用时间序列验证"""
        try:
            # 获取历史数据 - 使用更长的训练窗口
            required_data = self.config.TRAINING_WINDOW + self.config.LOOKBACK_DAYS + max(horizons) + 400
            if required_data < self.config.DATA_QUALITY_CONFIG['min_data_points']:
                required_data = self.config.DATA_QUALITY_CONFIG['min_data_points']
            
            prices = data_processor.get_historical_data(symbol, required_data)
            
            if prices is None or len(prices) < required_data:
                prices_len = len(prices) if prices is not None else 0
                self.algorithm.log_debug(f"Insufficient data for {symbol}: got {prices_len}, need {required_data}", log_type='training')
                return None
            
            # 创建增强特征矩阵（包含基本面和宏观特征）
            # 启用固定24维特征模式，确保训练和预测的一致性
            data_processor.enable_fixed_24_features(True)
            feature_matrix = data_processor.create_feature_matrix(prices, symbol=symbol)
            
            if feature_matrix is None:
                self.algorithm.log_debug(f"Failed to create feature matrix for {symbol}", log_type='training')
                # 恢复原始特征模式
                data_processor.enable_fixed_24_features(False)
                return None

            # 验证特征矩阵维度
            if feature_matrix.shape[1] != 24:
                self.algorithm.log_debug(f"Feature matrix dimension mismatch for {symbol}: got {feature_matrix.shape[1]}, expected 24", log_type='training')
                # 恢复原始特征模式
                data_processor.enable_fixed_24_features(False)
                return None
            
            # 数据质量检查
            is_valid, message = DataValidator.check_data_quality(feature_matrix, symbol)
            if not is_valid:
                self.algorithm.log_debug(f"Data quality check failed for {symbol}: {message}", log_type='training')
                # 恢复原始特征模式
                data_processor.enable_fixed_24_features(False)
                return None
            
            # 数据清洗
            cleaned_data = data_processor.clean_data(feature_matrix, symbol)
            
            # 数据缩放
            scaled_data = data_processor.scale_data(cleaned_data, symbol, fit=True)
            
            # 恢复原始特征模式
            data_processor.enable_fixed_24_features(False)
            
            # 只使用最近的训练窗口数据
            training_data = scaled_data[-self.config.TRAINING_WINDOW:]
            
            # 计算有效的lookback长度
            effective_lookback = min(self.config.LOOKBACK_DAYS, len(training_data) // 3)
            
            # 创建多时间跨度序列
            X, y_dict = data_processor.create_multi_horizon_sequences(
                training_data, effective_lookback, horizons
            )
            
            # 验证数据
            if not DataValidator.validate_sequences(X, y_dict):
                self.algorithm.log_debug(f"Invalid sequences for {symbol}", log_type='training')
                return None
            
            if X.shape[0] < self.config.VALIDATION_CONFIG['min_train_size']:
                self.algorithm.log_debug(f"Too few training sequences for {symbol}: {X.shape[0]}", log_type='training')
                return None
            
            return X, y_dict, effective_lookback
            
        except Exception as e:
            self.algorithm.log_debug(f"Error preparing training data for {symbol}: {e}", log_type='training')
            # 确保恢复原始特征模式
            if hasattr(data_processor, 'enable_fixed_24_features'):
                data_processor.enable_fixed_24_features(False)
            return None
    
    def create_time_series_splits(self, X, y_dict):
        """创建时间序列验证分割"""
        try:
            validation_config = self.config.VALIDATION_CONFIG
            n_samples = X.shape[0]
            
            # 计算分割参数
            test_size = int(n_samples * validation_config['test_size'])
            validation_size = int(n_samples * validation_config['validation_size'])
            gap_days = validation_config['gap_days']
            min_train_size = validation_config['min_train_size']
            
            splits = []
            
            if validation_config['validation_method'] == 'time_series_split':
                # 时间序列交叉验证
                n_splits = validation_config['n_splits']
                
                for i in range(n_splits):
                    # 计算训练集结束位置
                    train_end = min_train_size + i * (n_samples - min_train_size - test_size - gap_days) // n_splits
                    
                    # 添加间隔
                    val_start = train_end + gap_days
                    val_end = val_start + validation_size
                    
                    # 确保不超出数据范围
                    if val_end <= n_samples:
                        train_indices = list(range(0, train_end))
                        val_indices = list(range(val_start, val_end))
                        
                        splits.append((train_indices, val_indices))
            
            elif validation_config['validation_method'] == 'walk_forward':
                # 滚动窗口验证
                window_size = min_train_size
                step_size = validation_size
                
                for start in range(0, n_samples - window_size - validation_size - gap_days, step_size):
                    train_end = start + window_size
                    val_start = train_end + gap_days
                    val_end = val_start + validation_size
                    
                    if val_end <= n_samples:
                        train_indices = list(range(start, train_end))
                        val_indices = list(range(val_start, val_end))
                        
                        splits.append((train_indices, val_indices))
            
            else:
                # 简单的时间序列分割（默认）
                train_end = n_samples - test_size - validation_size - gap_days
                val_start = train_end + gap_days
                val_end = val_start + validation_size
                
                train_indices = list(range(0, train_end))
                val_indices = list(range(val_start, val_end))
                test_indices = list(range(val_end + gap_days, n_samples))
                
                splits.append((train_indices, val_indices))
            
            self.algorithm.log_debug(f"Created {len(splits)} time series validation splits", log_type='training')
            return splits
            
        except Exception as e:
            self.algorithm.log_debug(f"Error creating time series splits: {e}", log_type='training')
            return []
    
    def train_model_with_time_series_validation(self, symbol, data_processor):
        """使用时间序列验证训练模型"""
        start_time = time.time()
        
        try:
            # 准备训练数据
            training_data = self.prepare_training_data(
                data_processor, symbol, self.config.PREDICTION_HORIZONS
            )
            
            if training_data is None:
                return False
            
            X, y_dict, effective_lookback = training_data
            
            # 创建时间序列验证分割
            splits = self.create_time_series_splits(X, y_dict)
            
            if not splits:
                self.algorithm.log_debug(f"No valid splits created for {symbol}, falling back to simple split", log_type='training')
                return self.train_model_simple_split(symbol, X, y_dict, effective_lookback)
            
            # 数据统计
            self.algorithm.log_debug(f"Training {symbol} with time series validation:", log_type='training')
            self.algorithm.log_debug(f"  X shape: {X.shape}, effective_lookback: {effective_lookback}", log_type='training')
            self.algorithm.log_debug(f"  Validation splits: {len(splits)}", log_type='training')
            for horizon, y in y_dict.items():
                self.algorithm.log_debug(f"  {horizon}d targets: {len(y)} samples, range=[{y.min():.4f}, {y.max():.4f}]", log_type='training')
            
            # 交叉验证训练
            best_model = None
            best_score = float('inf')
            validation_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(splits):
                try:
                    # 分割数据
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train_dict = {f'output_{h}d': y_dict[h][train_idx] for h in self.config.PREDICTION_HORIZONS}
                    y_val_dict = {f'output_{h}d': y_dict[h][val_idx] for h in self.config.PREDICTION_HORIZONS}
                    
                    # 构建模型
                    model = self.build_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        horizons=self.config.PREDICTION_HORIZONS
                    )
                    
                    # 设置回调
                    callbacks = [
                        EarlyStopping(
                            monitor='val_loss',
                            patience=self.config.MODEL_CONFIG['patience'],
                            restore_best_weights=True,
                            verbose=0
                        ),
                        ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.7,
                            patience=5,
                            min_lr=1e-6,
                            verbose=0
                        )
                    ]
                    
                    # 训练模型
                    history = model.fit(
                        X_train, y_train_dict,
                        validation_data=(X_val, y_val_dict),
                        epochs=self.config.MODEL_CONFIG['epochs'],
                        batch_size=self.config.MODEL_CONFIG['batch_size'],
                        callbacks=callbacks,
                        verbose=0
                    )
                    
                    # 评估模型
                    val_loss = min(history.history['val_loss'])
                    validation_scores.append(val_loss)
                    
                    # 保存最佳模型
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = model
                    
                    self.algorithm.log_debug(f"  Fold {fold+1}: val_loss = {val_loss:.4f}", log_type='training')
                    
                except Exception as fold_error:
                    self.algorithm.log_debug(f"Error in fold {fold+1}: {fold_error}", log_type='training')
                    continue
            
            if best_model is None:
                self.algorithm.log_debug(f"No valid model trained for {symbol}", log_type='training')
                return False
            
            # 计算交叉验证统计
            mean_score = np.mean(validation_scores)
            std_score = np.std(validation_scores)
            
            # 验证训练结果
            if np.isnan(best_score) or np.isinf(best_score):
                self.algorithm.log_debug(f"Invalid training result for {symbol}", log_type='training')
                return False
            
            # 保存模型和相关信息
            self.models[symbol] = {
                'model': best_model,
                'effective_lookback': effective_lookback,
                'training_samples': X.shape[0],
                'feature_dim': X.shape[2]
            }
            
            self.training_history[symbol] = {
                'best_val_loss': best_score,
                'mean_val_loss': mean_score,
                'std_val_loss': std_score,
                'cv_scores': validation_scores,
                'n_folds': len(validation_scores),
                'training_time': time.time() - start_time
            }
            
            # 记录训练结果
            training_time = time.time() - start_time
            self.algorithm.log_debug(f"Time series validation completed for {symbol}:", log_type='training')
            self.algorithm.log_debug(f"  Best validation loss: {best_score:.4f}", log_type='training')
            self.algorithm.log_debug(f"  Mean CV score: {mean_score:.4f} ± {std_score:.4f}", log_type='training')
            self.algorithm.log_debug(f"  Training time: {training_time:.2f}s", log_type='training')
            
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in time series validation training for {symbol}: {e}", log_type='training')
            return False
    
    def train_model_simple_split(self, symbol, X, y_dict, effective_lookback):
        """简单时间序列分割训练（备用方法）"""
        try:
            # 简单的时间序列分割
            validation_config = self.config.VALIDATION_CONFIG
            n_samples = X.shape[0]
            
            # 计算分割点
            test_size = int(n_samples * validation_config['test_size'])
            val_size = int(n_samples * validation_config['validation_size'])
            gap = validation_config['gap_days']
            
            # 分割数据
            train_end = n_samples - test_size - val_size - gap * 2
            val_start = train_end + gap
            val_end = val_start + val_size
            
            X_train = X[:train_end]
            X_val = X[val_start:val_end]
            
            # 准备目标数据
            y_train_dict = {f'output_{h}d': y_dict[h][:train_end] for h in self.config.PREDICTION_HORIZONS}
            y_val_dict = {f'output_{h}d': y_dict[h][val_start:val_end] for h in self.config.PREDICTION_HORIZONS}
            
            # 构建模型
            model = self.build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                horizons=self.config.PREDICTION_HORIZONS
            )
            
            # 设置回调
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.MODEL_CONFIG['patience'],
                    restore_best_weights=True,
                    verbose=0
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train_dict,
                validation_data=(X_val, y_val_dict),
                epochs=self.config.MODEL_CONFIG['epochs'],
                batch_size=self.config.MODEL_CONFIG['batch_size'],
                callbacks=callbacks,
                verbose=0
            )
            
            # 验证训练结果
            final_loss = min(history.history['val_loss'])
            
            if np.isnan(final_loss) or np.isinf(final_loss):
                self.algorithm.log_debug(f"Invalid training result for {symbol}", log_type='training')
                return False
            
            # 保存模型和相关信息
            self.models[symbol] = {
                'model': model,
                'effective_lookback': effective_lookback,
                'training_samples': X.shape[0],
                'feature_dim': X.shape[2]
            }
            
            self.training_history[symbol] = {
                'history': history.history,
                'final_loss': final_loss,
                'training_time': time.time() - start_time
            }
            
            self.algorithm.log_debug(f"Simple split training completed for {symbol}: val_loss = {final_loss:.4f}", log_type='training')
            return True
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in simple split training for {symbol}: {e}", log_type='training')
            return False

    def train_model(self, symbol, data_processor):
        """训练单个股票的模型 - 使用改进的验证方法"""
        try:
            # 使用时间序列验证训练
            if self.config.VALIDATION_CONFIG['validation_method'] in ['time_series_split', 'walk_forward']:
                return self.train_model_with_time_series_validation(symbol, data_processor)
            else:
                # 备用：准备数据并使用简单分割
                training_data = self.prepare_training_data(
                    data_processor, symbol, self.config.PREDICTION_HORIZONS
                )
                
                if training_data is None:
                    return False
                
                X, y_dict, effective_lookback = training_data
                return self.train_model_simple_split(symbol, X, y_dict, effective_lookback)
                
        except Exception as e:
            self.algorithm.log_debug(f"Error training model for {symbol}: {e}", log_type='training')
            return False
    
    def get_model_summary(self, symbol):
        """获取模型摘要信息"""
        if symbol not in self.models:
            return {}
        
        model_info = self.models[symbol]
        history_info = self.training_history.get(symbol, {})
        
        return {
            'effective_lookback': model_info['effective_lookback'],
            'training_samples': model_info['training_samples'],
            'feature_dim': model_info['feature_dim'],
            'final_loss': history_info.get('final_loss', 0),
            'improvement': history_info.get('improvement', 0),
            'training_time': history_info.get('training_time', 0)
        }

class ModelTrainer:
    """模型训练管理器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        # 重要：直接使用算法实例的data_processor，不要创建新的实例
        self.data_processor = algorithm_instance.data_processor
        self.multi_horizon_model = MultiHorizonModel(algorithm_instance)
        
        # 重要：添加scaler管理
        self.scalers = {}
        
        # 初始化models字典
        self.models = {}
        
    def train_all_models(self):
        """训练所有股票的模型"""
        try:
            self.algorithm.log_debug("=== Starting model training for all symbols ===", log_type="training")
            
            # 清理现有模型
            try:
                self._cleanup_models()
                self.algorithm.log_debug('training', "Cleaned up existing models", log_type="training")
            except Exception as cleanup_error:
                self.algorithm.log_debug('training', f"Error during model cleanup: {cleanup_error}", log_type="training")
            
            training_results = {}
            successful_symbols = []
            
            total_symbols = len(self.algorithm.config.SYMBOLS)
            self.algorithm.log_debug('training', f"Training models for {total_symbols} symbols: {self.algorithm.config.SYMBOLS}", log_type="training")
            
            for i, symbol in enumerate(self.algorithm.config.SYMBOLS, 1):
                try:
                    self.algorithm.log_debug('training', f"Training progress: {i}/{total_symbols} - Processing {symbol}", log_type="training")
                    
                    # 检查内存和时间限制
                    if hasattr(self.algorithm, 'performance_metrics'):
                        total_training_time = self.algorithm.performance_metrics.get('total_training_time', 0)
                        max_time = self.config.TRAINING_CONFIG.get('max_training_time', 300)
                        if total_training_time > max_time:
                            self.algorithm.log_debug('training', f"Training time limit exceeded ({total_training_time}s > {max_time}s), stopping", log_type="training")
                            break
                    
                    model_result = self.train_single_model(symbol)
                    
                    if model_result:
                        training_results[symbol] = model_result
                        successful_symbols.append(symbol)
                        self.algorithm.log_debug('training', f"✓ Successfully trained model for {symbol}", log_type="training")
                    else:
                        self.algorithm.log_debug('training', f"✗ Failed to train model for {symbol}", log_type="training")
                        
                except Exception as symbol_error:
                    self.algorithm.log_debug('training', f"ERROR training {symbol}: {str(symbol_error)}", log_type="training")
                    self.algorithm.log_debug('training', f"Symbol error type: {type(symbol_error).__name__}", log_type="training")
                    import traceback
                    self.algorithm.log_debug('training', f"Symbol error traceback: {traceback.format_exc()}", log_type="training")
                    continue
            
            self.algorithm.log_debug('training', f"=== Training completed ===", log_type="training")
            self.algorithm.log_debug('training', f"Successfully trained: {len(successful_symbols)}/{total_symbols} models", log_type="training")
            self.algorithm.log_debug('training', f"Successful symbols: {successful_symbols}", log_type="training")
            
            return training_results, successful_symbols
            
        except Exception as e:
            self.algorithm.log_debug('training', f"CRITICAL ERROR in train_all_models: {str(e)}", log_type="training")
            self.algorithm.log_debug('training', f"Train all models error type: {type(e).__name__}", log_type="training")
            import traceback
            self.algorithm.log_debug('training', f"Train all models error traceback: {traceback.format_exc()}", log_type="training")
            return {}, []

    def train_single_model(self, symbol):
        """训练单个股票的模型"""
        start_time = time.time()
        
        try:
            self.algorithm.log_debug('training', f"--- Training model for {symbol} ---", log_type="training")
            
            # 准备训练数据
            self.algorithm.log_debug('training', f"Step 1: Preparing training data for {symbol}", log_type="training")
            try:
                training_data = self.multi_horizon_model.prepare_training_data(
                    self.data_processor, symbol, self.config.PREDICTION_HORIZONS
                )
                
                if training_data is None:
                    self.algorithm.log_debug('training', f"No training data available for {symbol}", log_type="training")
                    return None
                    
                X, y_dict, effective_lookback = training_data
                self.algorithm.log_debug('training', f"Training data prepared for {symbol}:", log_type="training")
                self.algorithm.log_debug('training', f"  X shape: {X.shape}", log_type="training")
                self.algorithm.log_debug('training', f"  y_dict keys: {list(y_dict.keys()) if y_dict else 'None'}", log_type="training")
                self.algorithm.log_debug('training', f"  Effective lookback: {effective_lookback}", log_type="training")
                
            except Exception as data_error:
                self.algorithm.log_debug('training', f"Error preparing training data for {symbol}: {data_error}", log_type="training")
                return None
            
            # 构建模型
            self.algorithm.log_debug('training', f"Step 2: Building model for {symbol}", log_type="training")
            try:
                model = self.multi_horizon_model.build_model(
                    (X.shape[1], X.shape[2]), 
                    self.config.PREDICTION_HORIZONS
                )
                self.algorithm.log_debug('training', f"Model built for {symbol}: {model.summary() if hasattr(model, 'summary') else 'Model object created'}", log_type="training")
            except Exception as build_error:
                self.algorithm.log_debug('training', f"Error building model for {symbol}: {build_error}", log_type="training")
                return None
            
            # 训练模型
            self.algorithm.log_debug('training', f"Step 3: Training model for {symbol}", log_type="training")
            try:
                # 准备训练目标
                y_list = [y_dict[horizon] for horizon in self.config.PREDICTION_HORIZONS]
                
                # 设置回调
                callbacks = []
                if hasattr(tf.keras.callbacks, 'EarlyStopping'):
                    early_stopping = tf.keras.callbacks.EarlyStopping(
                        monitor='loss', 
                        patience=5, 
                        restore_best_weights=True
                    )
                    callbacks.append(early_stopping)
                
                # 训练模型
                epochs = min(50, self.config.TRAINING_CONFIG.get('max_epochs', 50))
                self.algorithm.log_debug('training', f"Starting training for {symbol} with {epochs} epochs", log_type="training")
                
                history = model.fit(
                    X, y_list,
                    epochs=epochs,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=0
                )
                
                training_time = time.time() - start_time
                self.algorithm.log_debug('training', f"Training completed for {symbol} in {training_time:.2f}s", log_type="training")
                
                # 存储模型和相关信息
                model_info = {
                    'model': model,
                    'effective_lookback': effective_lookback,
                    'training_history': history.history if hasattr(history, 'history') else {},
                    'training_time': training_time,
                    'final_loss': history.history['loss'][-1] if hasattr(history, 'history') and 'loss' in history.history else 'N/A'
                }
                
                # 同时存储在两个地方以保持兼容性
                self.models[symbol] = model_info
                self.multi_horizon_model.models[symbol] = model_info
                self.algorithm.log_debug('training', f"Model stored for {symbol}, final loss: {model_info['final_loss']}", log_type="training")
                
                return model_info
                
            except Exception as train_error:
                self.algorithm.log_debug('training', f"Error during model training for {symbol}: {train_error}", log_type="training")
                import traceback
                self.algorithm.log_debug('training', f"Training error traceback: {traceback.format_exc()}", log_type="training")
                return None
                
        except Exception as e:
            self.algorithm.log_debug('training', f"CRITICAL ERROR in train_single_model for {symbol}: {str(e)}", log_type="training")
            self.algorithm.log_debug('training', f"Single model error type: {type(e).__name__}", log_type="training")
            import traceback
            self.algorithm.log_debug('training', f"Single model error traceback: {traceback.format_exc()}", log_type="training")
            return None
    
    def _cleanup_models(self):
        """清理旧模型释放内存"""
        # 清理ModelTrainer的models
        if hasattr(self, 'models'):
            for symbol, model_info in self.models.items():
                try:
                    if 'model' in model_info:
                        del model_info['model']
                except:
                    pass
        
        # 清理MultiHorizonModel的models
        if hasattr(self.multi_horizon_model, 'models'):
            for symbol, model_info in self.multi_horizon_model.models.items():
                try:
                    if 'model' in model_info:
                        del model_info['model']
                except:
                    pass
        
        # 重置字典
        self.models = {}
        self.multi_horizon_model.models = {}
        self.multi_horizon_model.training_history = {}
        gc.collect()
        
        self.algorithm.log_debug('training', "Model cleanup completed", log_type="training")
    
    def get_trained_models(self):
        """获取训练好的模型字典"""
        # 优先返回ModelTrainer的models，如果为空则返回MultiHorizonModel的models
        if hasattr(self, 'models') and self.models:
            return self.models
        else:
            return self.multi_horizon_model.models
    
    def get_tradable_symbols(self):
        """获取可交易的股票列表"""
        # 优先使用ModelTrainer的models，如果为空则使用MultiHorizonModel的models
        if hasattr(self, 'models') and self.models:
            return list(self.models.keys())
        else:
            return list(self.multi_horizon_model.models.keys())
    
    def get_training_statistics(self):
        """获取训练统计信息"""
        # 使用ModelTrainer的models来获取统计信息
        models_dict = self.models if hasattr(self, 'models') and self.models else self.multi_horizon_model.models
        
        stats = {
            'total_models': len(models_dict),
            'avg_training_time': 0,
            'avg_improvement': 0,
            'best_model': None,
            'worst_model': None
        }
        
        if not models_dict:
            return stats
        
        # 计算平均指标
        training_times = []
        final_losses = []
        
        for symbol, model_info in models_dict.items():
            if isinstance(model_info, dict):
                training_times.append(model_info.get('training_time', 0))
                final_loss = model_info.get('final_loss', 'N/A')
                if final_loss != 'N/A' and isinstance(final_loss, (int, float)):
                    final_losses.append(final_loss)
        
        if training_times:
            stats['avg_training_time'] = np.mean(training_times)
            stats['avg_improvement'] = np.mean(final_losses)
            
            # 找到最佳和最差模型
            best_idx = np.argmax(final_losses)
            worst_idx = np.argmin(final_losses)
            
            symbols = list(models_dict.keys())
            stats['best_model'] = symbols[best_idx]
            stats['worst_model'] = symbols[worst_idx]
        
        return stats
    
    def validate_model_quality(self, symbol):
        """验证模型质量"""
        if symbol not in self.multi_horizon_model.models:
            return {'quality_score': 0.0}
        
        history = self.multi_horizon_model.training_history.get(symbol, {})
        
        # 计算质量分数
        improvement = history.get('improvement', 0)
        final_loss = history.get('final_loss', 1.0)
        
        # 质量分数基于多个因素
        quality_score = 0.0
        
        # 1. 损失改善程度 (0-40分)
        if improvement > 0:
            quality_score += min(improvement * 40, 40)
        
        # 2. 最终损失水平 (0-30分)
        if final_loss < 0.1:
            quality_score += 30
        elif final_loss < 0.2:
            quality_score += 20
        elif final_loss < 0.5:
            quality_score += 10
        
        # 3. 训练样本数 (0-20分)
        training_samples = self.multi_horizon_model.models[symbol]['training_samples']
        if training_samples > 100:
            quality_score += 20
        elif training_samples > 50:
            quality_score += 15
        elif training_samples > 20:
            quality_score += 10
        
        # 4. 特征维度 (0-10分)
        feature_dim = self.multi_horizon_model.models[symbol]['feature_dim']
        if feature_dim > 10:
            quality_score += 10
        elif feature_dim > 5:
            quality_score += 5
        
        return {
            'quality_score': quality_score,
            'improvement': improvement,
            'final_loss': final_loss,
            'training_samples': training_samples,
            'feature_dim': feature_dim
        }
    
    def predict_and_optimize(self, symbol, data_processor):
        """预测并优化模型"""
        start_time = time.time()
        
        try:
            self.algorithm.log_debug('training', f"--- Predicting and optimizing model for {symbol} ---", log_type="training")
            
            # 准备预测数据
            self.algorithm.log_debug('training', f"Step 1: Preparing prediction data for {symbol}", log_type="training")
            try:
                prediction_data = self.multi_horizon_model.prepare_training_data(
                    data_processor, symbol, self.config.PREDICTION_HORIZONS
                )
                
                if prediction_data is None:
                    self.algorithm.log_debug('training', f"No prediction data available for {symbol}", log_type="training")
                    return None
                    
                X, y_dict, effective_lookback = prediction_data
                self.algorithm.log_debug('training', f"Prediction data prepared for {symbol}:", log_type="training")
                self.algorithm.log_debug('training', f"  X shape: {X.shape}", log_type="training")
                self.algorithm.log_debug('training', f"  y_dict keys: {list(y_dict.keys()) if y_dict else 'None'}", log_type="training")
                self.algorithm.log_debug('training', f"  Effective lookback: {effective_lookback}", log_type="training")
                
            except Exception as data_error:
                self.algorithm.log_debug('training', f"Error preparing prediction data for {symbol}: {data_error}", log_type="training")
                return None
            
            # 预测模型
            self.algorithm.log_debug('training', f"Step 2: Predicting model for {symbol}", log_type="training")
            try:
                model = self.multi_horizon_model.models[symbol]['model']
                predicted_returns = model.predict(X)
                self.algorithm.log_debug('training', f"Model prediction completed for {symbol}:", log_type="training")
                self.algorithm.log_debug('training', f"  Predicted returns shape: {predicted_returns.shape}", log_type="training")
            except Exception as predict_error:
                self.algorithm.log_debug('training', f"Error predicting model for {symbol}: {predict_error}", log_type="training")
                return None
            
            # 处理预测结果
            self.algorithm.log_debug('training', f"[模型预测] 每只股票预测收益: {predicted_returns}", log_type="training")
            if not predicted_returns or all(v is None or v == 0 for v in predicted_returns.values()):
                self.algorithm.log_debug('training', "[模型预测] 所有预测无效，优化器将不会被调用", log_type="training")
            
            # 优化模型
            self.algorithm.log_debug('training', f"Step 3: Optimizing model for {symbol}", log_type="training")
            try:
                # 实现优化逻辑
                optimized_model = self.optimize_model(symbol, predicted_returns)
                self.algorithm.log_debug('training', f"Model optimization completed for {symbol}:", log_type="training")
                self.algorithm.log_debug('training', f"  Optimized model: {optimized_model}", log_type="training")
                
                # 存储优化后的模型
                self.models[symbol]['optimized_model'] = optimized_model
                self.algorithm.log_debug('training', f"Optimized model stored for {symbol}", log_type="training")
                
                return optimized_model
                
            except Exception as optimize_error:
                self.algorithm.log_debug('training', f"Error optimizing model for {symbol}: {optimize_error}", log_type="training")
                return None
                
        except Exception as e:
            self.algorithm.log_debug('training', f"CRITICAL ERROR in predict_and_optimize for {symbol}: {str(e)}", log_type="training")
            self.algorithm.log_debug('training', f"Predict and optimize error type: {type(e).__name__}", log_type="training")
            import traceback
            self.algorithm.log_debug('training', f"Predict and optimize error traceback: {traceback.format_exc()}", log_type="training")
            return None
    
    def optimize_model(self, symbol, predicted_returns):
        """实现优化逻辑"""
        # 实现优化逻辑
        pass

 