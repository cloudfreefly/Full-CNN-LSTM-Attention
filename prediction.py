# 预测模块
from AlgorithmImports import *
import numpy as np
import tensorflow as tf
from config import AlgorithmConfig
import time

class PredictionEngine:
    """多时间跨度预测引擎"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        # 重要：直接使用算法实例的data_processor，不创建新实例
        self.data_processor = algorithm_instance.data_processor
        
    def generate_multi_horizon_predictions(self, models, symbols):
        """生成多时间跨度预测"""
        predictions = {}
        
        self.algorithm.log_debug(f"开始生成预测: {len(symbols)}个股票, {len(models)}个模型", log_type="prediction")
        
        for symbol in symbols:
            try:
                model_info = models.get(symbol)
                if model_info is None:
                    self.algorithm.log_debug(f"跳过{symbol}: 无可用模型", log_type="prediction")
                    continue
                    
                prediction_data = self._predict_single_symbol(symbol, model_info)
                if prediction_data:
                    predictions[symbol] = prediction_data
                    self.algorithm.log_debug(f"✓ {symbol}: 预测生成成功", log_type="prediction")
                else:
                    self.algorithm.log_debug(f"✗ {symbol}: 预测生成失败", log_type="prediction")
                    
            except Exception as e:
                self.algorithm.log_debug(f"✗ {symbol}: 预测异常: {e}", log_type="prediction")
                continue
        
        success_rate = len(predictions) / len(symbols) * 100 if symbols else 0
        self.algorithm.log_debug(f"预测生成完成: {len(predictions)}/{len(symbols)} ({success_rate:.1f}%)", log_type="prediction")
        return predictions
    
    def _predict_single_symbol(self, symbol, model_info):
        """为单个股票生成预测"""
        try:
            if model_info is None:
                self.algorithm.log_debug(f"No model available for {symbol}", log_type="prediction")
                return None
            
            model = model_info.get('model')
            if model is None:
                self.algorithm.log_debug(f"Model is None for {symbol}", log_type="prediction")
                return None
            
            # 获取训练时使用的effective_lookback，确保序列长度一致
            model_effective_lookback = model_info.get('effective_lookback')
            if model_effective_lookback is None:
                self.algorithm.log_debug(f"No effective_lookback found in model_info for {symbol}, using default", log_type="prediction")
                model_effective_lookback = 12  # 使用默认值12，基于错误信息中的期望形状
            
            # 获取历史数据 - 需要足够的数据来创建特征和序列
            required_days = max(model_effective_lookback + 50, self.config.LOOKBACK_DAYS // 2)
            prices = self.data_processor.get_historical_data(symbol, days=required_days)
            if prices is None or len(prices) < model_effective_lookback + 20:
                prices_len = len(prices) if prices is not None else 0
                self.algorithm.log_debug(f"Insufficient price data for {symbol}: need {model_effective_lookback + 20}, got {prices_len}", log_type="prediction")
                return None
            
            # 启用固定24维特征模式，确保与训练时一致
            self.data_processor.enable_fixed_24_features(True)
            
            # 创建特征矩阵
            if model_info.get('is_pretrained', False):
                # 预训练模型：使用24维特征矩阵（与预训练时保持一致）
                self.algorithm.log_debug(f"使用预训练模型 {symbol}，使用24维特征矩阵，effective_lookback={model_effective_lookback}", log_type="prediction")
                feature_matrix = self.data_processor.create_feature_matrix(prices)
            else:
                # 统一使用24维特征矩阵，保持与训练时的一致性
                self.algorithm.log_debug(f"使用24维特征矩阵进行预测 {symbol}，effective_lookback={model_effective_lookback}", log_type="prediction")
                feature_matrix = self.data_processor.create_feature_matrix(prices)
                
            if feature_matrix is None:
                self.algorithm.log_debug(f"Failed to create feature matrix for {symbol}", log_type="prediction")
                # 恢复原始特征模式
                self.data_processor.enable_fixed_24_features(False)
                return None

            # 验证特征矩阵维度
            if feature_matrix.shape[1] != 24:
                self.algorithm.log_debug(f"Feature matrix dimension mismatch for {symbol}: got {feature_matrix.shape[1]}, expected 24", log_type="prediction")
                # 恢复原始特征模式
                self.data_processor.enable_fixed_24_features(False)
                return None
            
            # 使用训练时的缩放器
            scaled_data = self.data_processor.scale_data(feature_matrix, symbol, fit=False)
            
            if scaled_data is None:
                self.algorithm.log_debug(f"Failed to scale data for {symbol}", log_type="prediction")
                # 恢复原始特征模式
                self.data_processor.enable_fixed_24_features(False)
                return None
            
            # 检查是否有足够的数据进行序列创建
            if len(scaled_data) < model_effective_lookback:
                self.algorithm.log_debug(f"Insufficient scaled data for {symbol}: need {model_effective_lookback}, got {len(scaled_data)}", log_type="prediction")
                # 恢复原始特征模式
                self.data_processor.enable_fixed_24_features(False)
                return None
            
            # 准备输入序列 - 使用训练时的exact effective_lookback
            X = scaled_data[-model_effective_lookback:].reshape(1, model_effective_lookback, -1)
            
            # 验证输入形状
            self.algorithm.log_debug(f"Input shape for {symbol}: {X.shape}, expected: (1, {model_effective_lookback}, 24)", log_type="prediction")
            
            # Monte Carlo Dropout预测（用于不确定性量化）
            predictions_mc = self._monte_carlo_predictions(model, X)
            
            # 计算预测统计信息
            predictions_stats = self._calculate_prediction_statistics(predictions_mc)
            
            # 反向缩放预测结果
            current_price = prices[-1]
            scaled_predictions = self._inverse_scale_predictions(
                predictions_stats, symbol, current_price
            )
            
            # 计算置信度和趋势分析
            confidence_analysis = self._analyze_prediction_confidence(
                predictions_mc, scaled_predictions
            )
            
            # 恢复原始特征模式
            self.data_processor.enable_fixed_24_features(False)
            
            return {
                'predictions': scaled_predictions,
                'confidence': confidence_analysis,
                'current_price': current_price,
                'market_regime': self.data_processor.calculate_market_regime(prices)
            }
            
        except Exception as e:
            self.algorithm.log_debug(f"Error in single symbol prediction for {symbol}: {e}", log_type="prediction")
            # 确保恢复原始特征模式
            if hasattr(self.data_processor, 'enable_fixed_24_features'):
                self.data_processor.enable_fixed_24_features(False)
            return None
    
    def _monte_carlo_predictions(self, model, X, n_samples = None):
        """Monte Carlo Dropout预测获取不确定性"""
        if n_samples is None:
            n_samples = self.config.TRAINING_CONFIG['mc_dropout_samples']
        
        # 启用训练模式以激活dropout
        predictions_samples = []
        
        for _ in range(n_samples):
            # 在推理时启用dropout
            pred = model(X, training=True)
            
            # 如果是多输出模型，处理每个输出
            if isinstance(pred, list):
                sample_dict = {}
                for i, horizon in enumerate(self.config.PREDICTION_HORIZONS):
                    sample_dict[horizon] = pred[i].numpy().flatten()
                predictions_samples.append(sample_dict)
            else:
                predictions_samples.append({1: pred.numpy().flatten()})
        
        # 重新组织数据结构
        mc_predictions = {}
        for horizon in self.config.PREDICTION_HORIZONS:
            horizon_samples = [sample[horizon] for sample in predictions_samples if horizon in sample]
            if horizon_samples:
                mc_predictions[horizon] = np.array(horizon_samples)
        
        return mc_predictions
    
    def _calculate_prediction_statistics(self, mc_predictions):
        """计算预测统计信息"""
        stats = {}
        
        for horizon, samples in mc_predictions.items():
            if len(samples) > 0:
                mean_pred = np.mean(samples, axis=0)
                std_pred = np.std(samples, axis=0)
                
                # 计算置信区间
                confidence_level = 0.95
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(samples, lower_percentile, axis=0)
                upper_bound = np.percentile(samples, upper_percentile, axis=0)
                
                stats[horizon] = {
                    'mean': mean_pred[0] if len(mean_pred) > 0 else 0,
                    'std': std_pred[0] if len(std_pred) > 0 else 0,
                    'lower_bound': lower_bound[0] if len(lower_bound) > 0 else 0,
                    'upper_bound': upper_bound[0] if len(upper_bound) > 0 else 0,
                    'samples': samples
                }
        
        return stats
    
    def _inverse_scale_predictions(self, predictions_stats, symbol, 
                                 current_price):
        """反向缩放预测结果到实际价格"""
        scaled_predictions = {}
        
        for horizon, stats in predictions_stats.items():
            try:
                # 获取scaler的特征数量（兼容不同版本的scikit-learn）
                scaler = self.data_processor.scalers[symbol]
                
                # 尝试多种方法获取特征数量
                if hasattr(scaler, 'n_features_in_'):
                    n_features = scaler.n_features_in_
                elif hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    n_features = len(scaler.scale_)
                elif hasattr(scaler, 'data_min_') and scaler.data_min_ is not None:
                    n_features = len(scaler.data_min_)
                else:
                    # 默认使用24维特征
                    n_features = 24
                
                # 构造虚拟数据用于反向缩放
                dummy_data = np.zeros((1, n_features))
                dummy_data[0, 0] = stats['mean']  # 假设价格在第一列
                
                # 反向缩放
                unscaled = scaler.inverse_transform(dummy_data)
                predicted_price = unscaled[0, 0]
                
                # 计算预期收益率
                expected_return = (predicted_price - current_price) / current_price
                
                # 处理置信区间
                dummy_lower = dummy_data.copy()
                dummy_lower[0, 0] = stats['lower_bound']
                dummy_upper = dummy_data.copy()
                dummy_upper[0, 0] = stats['upper_bound']
                
                lower_price = scaler.inverse_transform(dummy_lower)[0, 0]
                upper_price = scaler.inverse_transform(dummy_upper)[0, 0]
                
                scaled_predictions[horizon] = {
                    'predicted_price': predicted_price,
                    'expected_return': expected_return,
                    'price_std': stats['std'],
                    'confidence_interval': {
                        'lower': lower_price,
                        'upper': upper_price
                    }
                }
                
            except Exception as e:
                self.algorithm.log_debug(f"Error inverse scaling for {symbol} horizon {horizon}: {e}", log_type="prediction")
                scaled_predictions[horizon] = {
                    'predicted_price': current_price,
                    'expected_return': 0.0,
                    'price_std': 0.0,
                    'confidence_interval': {'lower': current_price, 'upper': current_price}
                }
        
        return scaled_predictions
    
    def _analyze_prediction_confidence(self, mc_predictions, scaled_predictions):
        """分析预测置信度"""
        confidence_analysis = {}
        
        # 1. 不确定性评分
        uncertainty_scores = {}
        for horizon in self.config.PREDICTION_HORIZONS:
            if horizon in scaled_predictions:
                std = scaled_predictions[horizon]['price_std']
                predicted_price = scaled_predictions[horizon]['predicted_price']
                
                # 标准化的不确定性（变异系数）
                cv = std / abs(predicted_price) if predicted_price != 0 else 1.0
                uncertainty_score = max(0, 1 - cv)  # 越低的变异系数，置信度越高
                uncertainty_scores[horizon] = uncertainty_score
        
        # 2. 趋势一致性分析
        trend_consistency = self._analyze_trend_consistency(scaled_predictions)
        
        # 3. 预测范围合理性
        range_reasonableness = self._analyze_prediction_range(scaled_predictions)
        
        # 4. 综合置信度评分
        overall_confidence = self._calculate_overall_confidence(
            uncertainty_scores, trend_consistency, range_reasonableness
        )
        
        confidence_analysis = {
            'uncertainty_scores': uncertainty_scores,
            'trend_consistency': trend_consistency,
            'range_reasonableness': range_reasonableness,
            'overall_confidence': overall_confidence
        }
        
        return confidence_analysis
    
    def _analyze_trend_consistency(self, predictions):
        """分析趋势一致性"""
        horizons = sorted(predictions.keys())
        if len(horizons) < 2:
            return {'score': 0.5, 'direction': 'neutral'}
        
        returns = []
        for horizon in horizons:
            returns.append(predictions[horizon]['expected_return'])
        
        # 检查趋势方向一致性
        positive_count = sum(1 for r in returns if r > 0)
        negative_count = sum(1 for r in returns if r < 0)
        
        if positive_count == len(returns):
            direction = 'bullish'
            consistency_score = 1.0
        elif negative_count == len(returns):
            direction = 'bearish'  
            consistency_score = 1.0
        else:
            direction = 'mixed'
            consistency_score = max(positive_count, negative_count) / len(returns)
        
        # 检查预测幅度的递进性（短期预测应该较小）
        magnitude_consistency = self._check_magnitude_progression(returns, horizons)
        
        overall_score = (consistency_score + magnitude_consistency) / 2
        
        return {
            'score': overall_score,
            'direction': direction,
            'magnitude_consistency': magnitude_consistency
        }
    
    def _check_magnitude_progression(self, returns, horizons):
        """检查预测幅度的递进性"""
        if len(returns) < 2:
            return 1.0
        
        # 计算每日平均收益率
        daily_returns = [abs(ret) / horizon for ret, horizon in zip(returns, horizons)]
        
        # 理想情况下，日均收益率应该相对稳定
        std_daily_returns = np.std(daily_returns)
        mean_daily_returns = np.mean(daily_returns)
        
        if mean_daily_returns == 0:
            return 1.0
        
        cv = std_daily_returns / mean_daily_returns
        consistency_score = max(0, 1 - cv)  # 变异系数越小，一致性越好
        
        return consistency_score
    
    def _analyze_prediction_range(self, predictions):
        """分析预测范围的合理性"""
        range_scores = {}
        
        for horizon, pred in predictions.items():
            expected_return = abs(pred['expected_return'])
            
            # 基于时间跨度的合理性检查
            max_reasonable_daily_return = 0.1  # 10%日涨跌幅上限
            max_reasonable_return = max_reasonable_daily_return * horizon * 0.7  # 打个折扣
            
            if expected_return <= max_reasonable_return:
                range_scores[horizon] = 1.0
            else:
                # 超出合理范围则降低评分
                range_scores[horizon] = max(0, max_reasonable_return / expected_return)
        
        overall_range_score = np.mean(list(range_scores.values())) if range_scores else 0.5
        
        return {
            'individual_scores': range_scores,
            'overall_score': overall_range_score
        }
    
    def _calculate_overall_confidence(self, uncertainty_scores, 
                                    trend_consistency, 
                                    range_reasonableness):
        """计算综合置信度评分"""
        # 权重设置
        weights = {
            'uncertainty': 0.4,
            'trend_consistency': 0.3,
            'range_reasonableness': 0.3
        }
        
        # 计算加权平均
        uncertainty_avg = np.mean(list(uncertainty_scores.values())) if uncertainty_scores else 0.5
        trend_score = trend_consistency.get('score', 0.5)
        range_score = range_reasonableness.get('overall_score', 0.5)
        
        overall_confidence = (
            weights['uncertainty'] * uncertainty_avg +
            weights['trend_consistency'] * trend_score +
            weights['range_reasonableness'] * range_score
        )
        
        return overall_confidence

class ExpectedReturnCalculator:
    """预期收益计算器"""
    
    def __init__(self, algorithm_instance):
        self.algorithm = algorithm_instance
        self.config = AlgorithmConfig()
        
    def calculate_expected_returns(self, predictions):
        """计算预期收益率"""
        expected_returns = {}
        valid_symbols = []
        raw_returns = []
        final_returns = []
        
        for symbol, pred_data in predictions.items():
            try:
                # 计算多时间跨度加权收益率
                weighted_return = self._calculate_weighted_return(pred_data)
                
                # 应用置信度调整 - 减少过度压缩
                confidence_adjusted_return = self._apply_confidence_adjustment(
                    weighted_return, pred_data['confidence']
                )
                
                # 应用市场状态调整 - 减少过度压缩
                regime_adjusted_return = self._apply_regime_adjustment(
                    confidence_adjusted_return, pred_data['market_regime']
                )
                
                expected_returns[symbol] = regime_adjusted_return
                valid_symbols.append(symbol)
                raw_returns.append(weighted_return)
                final_returns.append(regime_adjusted_return)
                
                self.algorithm.log_debug(f"{symbol} expected return calculation:", log_type="prediction")
                self.algorithm.log_debug(f"  Raw weighted: {weighted_return:.6f}", log_type="prediction")
                self.algorithm.log_debug(f"  Confidence adjusted: {confidence_adjusted_return:.6f}", log_type="prediction")
                self.algorithm.log_debug(f"  Final (regime adjusted): {regime_adjusted_return:.6f}", log_type="prediction")
                
            except Exception as e:
                self.algorithm.log_debug(f"Error calculating expected return for {symbol}: {e}", log_type="prediction")
                continue
        
        # 诊断信息：检查收益分布
        if raw_returns and final_returns:
            raw_std = np.std(raw_returns)
            final_std = np.std(final_returns)
            raw_range = max(raw_returns) - min(raw_returns)
            final_range = max(final_returns) - min(final_returns)
            
            self.algorithm.log_debug(f"预期收益分布诊断:", log_type="prediction")
            self.algorithm.log_debug(f"  原始收益标准差: {raw_std:.6f}, 范围: {raw_range:.6f}", log_type="prediction")
            self.algorithm.log_debug(f"  最终收益标准差: {final_std:.6f}, 范围: {final_range:.6f}", log_type="prediction")
            self.algorithm.log_debug(f"  调整导致的差异压缩: {((raw_std-final_std)/raw_std*100):.1f}%", log_type="prediction")
            
            # 如果差异被过度压缩，放大最终收益差异
            if final_std < raw_std * 0.3:  # 如果标准差被压缩超过70%
                self.algorithm.log_debug("检测到收益差异过度压缩，应用差异放大", log_type="prediction")
                mean_return = np.mean(final_returns)
                amplified_returns = {}
                for i, symbol in enumerate(valid_symbols):
                    deviation = final_returns[i] - mean_return
                    amplified_return = mean_return + deviation * 2.0  # 放大差异2倍
                    amplified_returns[symbol] = amplified_return
                    self.algorithm.log_debug(f"  {symbol}: {final_returns[i]:.6f} -> {amplified_return:.6f}", log_type="prediction")
                expected_returns = amplified_returns
        
        return expected_returns, valid_symbols
    
    def _calculate_weighted_return(self, pred_data):
        """计算多时间跨度加权收益率"""
        weighted_return = 0.0
        total_weight = 0.0
        
        predictions = pred_data['predictions']
        
        for horizon in self.config.PREDICTION_HORIZONS:
            if horizon in predictions:
                horizon_return = predictions[horizon]['expected_return']
                weight = self.config.HORIZON_WEIGHTS.get(horizon, 0)
                
                weighted_return += horizon_return * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_return /= total_weight
        
        return weighted_return
    
    def _apply_confidence_adjustment(self, base_return, confidence):
        """应用置信度调整 - 减少过度压缩"""
        overall_confidence = confidence.get('overall_confidence', 0.5)
        
        # 调整置信度因子，减少过度压缩，保持更多收益差异
        # 原来：0.5 + 0.5 * confidence (范围0.5-1.0)
        # 现在：0.7 + 0.3 * confidence (范围0.7-1.0)
        confidence_factor = 0.7 + 0.3 * overall_confidence
        adjusted_return = base_return * confidence_factor
        
        return adjusted_return
    
    def _apply_regime_adjustment(self, base_return, market_regime):
        """应用市场状态调整 - 减少过度压缩"""
        # 调整因子，减少压缩强度，保持更多差异
        regime_factors = {
            'high_volatility': 0.9,  # 高波动时保守 (从0.8调整为0.9)
            'trending': 1.1,         # 趋势明确时积极 (从1.2调整为1.1)
            'low_volatility': 1.0,   # 低波动时正常
            'neutral': 1.0,          # 中性时正常
            'unknown': 0.95          # 未知时略保守 (从0.9调整为0.95)
        }
        
        factor = regime_factors.get(market_regime, 1.0)
        adjusted_return = base_return * factor
        
        return adjusted_return

class PredictionValidator:
    """预测验证器"""
    
    @staticmethod
    def validate_predictions(predictions):
        """验证预测结果的有效性"""
        validation_results = {}
        
        for symbol, pred_data in predictions.items():
            is_valid = True
            
            # 检查必要字段
            required_fields = ['predictions', 'confidence', 'current_price']
            for field in required_fields:
                if field not in pred_data:
                    is_valid = False
                    break
            
            if is_valid and 'predictions' in pred_data:
                # 检查预测值的有效性
                for horizon, pred in pred_data['predictions'].items():
                    expected_return = pred.get('expected_return', 0)
                    
                    # 检查是否为有效数值
                    if np.isnan(expected_return) or np.isinf(expected_return):
                        is_valid = False
                        break
                    
                    # 检查是否在合理范围内
                    if abs(expected_return) > 1.0:  # 100%的涨跌幅上限
                        is_valid = False
                        break
            
            validation_results[symbol] = is_valid
        
        return validation_results
    
    @staticmethod
    def strict_validation_for_winning_rate(predictions):
        """严格验证以提升胜率"""
        if not predictions:
            return {}
        
        validation_results = {}
        config = AlgorithmConfig()
        
        for symbol, pred_data in predictions.items():
            try:
                # 基础验证
                if not pred_data or 'confidence' not in pred_data:
                    validation_results[symbol] = False
                    continue
                
                confidence = pred_data['confidence']
                overall_confidence = confidence.get('overall_confidence', 0)
                trend_consistency = confidence.get('trend_consistency', {}).get('score', 0)
                uncertainty_score = confidence.get('uncertainty_score', 1.0)
                
                # 严格置信度门槛 (75%)
                if overall_confidence < config.PREDICTION_CONFIG['confidence_threshold']:
                    validation_results[symbol] = False
                    continue
                
                # 趋势一致性检查 (80%)
                if trend_consistency < 0.8:
                    validation_results[symbol] = False
                    continue
                
                # 不确定性限制 (30%)
                if uncertainty_score > config.PREDICTION_CONFIG['uncertainty_penalty']:
                    validation_results[symbol] = False
                    continue
                
                # 预测范围检查
                predictions_dict = pred_data.get('predictions', {})
                if not predictions_dict:
                    validation_results[symbol] = False
                    continue
                
                # 预测幅度合理性检查
                total_return_magnitude = 0
                max_single_return = 0
                for horizon, pred_info in predictions_dict.items():
                    expected_return = pred_info.get('expected_return', 0)
                    
                    # 单日收益率不应超过15%（更严格）
                    if abs(expected_return) > 0.15:
                        validation_results[symbol] = False
                        break
                    
                    total_return_magnitude += abs(expected_return)
                    max_single_return = max(max_single_return, abs(expected_return))
                else:
                    # 检查预测是否过于保守（总预测收益率太小）
                    if total_return_magnitude < 0.01:  # 1%
                        validation_results[symbol] = False
                        continue
                    
                    # 检查最大单期预测是否有足够的信号强度
                    if max_single_return < 0.005:  # 0.5%
                        validation_results[symbol] = False
                        continue
                    
                    validation_results[symbol] = True
                    
            except Exception:
                validation_results[symbol] = False
        
        return validation_results
    
    @staticmethod
    def get_prediction_summary(predictions):
        """获取预测汇总信息"""
        if not predictions:
            return {}
        
        summary = {
            'total_symbols': len(predictions),
            'avg_confidence': 0,
            'regime_distribution': {},
            'return_distribution': {}
        }
        
        confidences = []
        regimes = []
        returns = []
        
        for symbol, pred_data in predictions.items():
            # 置信度
            confidence = pred_data.get('confidence', {}).get('overall_confidence', 0)
            confidences.append(confidence)
            
            # 市场状态
            regime = pred_data.get('market_regime', 'unknown')
            regimes.append(regime)
            
            # 预期收益（使用1天期）
            predictions_dict = pred_data.get('predictions', {})
            if 1 in predictions_dict:
                expected_return = predictions_dict[1].get('expected_return', 0)
                returns.append(expected_return)
        
        # 计算统计信息
        if confidences:
            summary['avg_confidence'] = np.mean(confidences)
        
        if regimes:
            from collections import Counter
            regime_counts = Counter(regimes)
            summary['regime_distribution'] = dict(regime_counts)
        
        if returns:
            summary['return_distribution'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'positive_count': sum(1 for r in returns if r > 0),
                'negative_count': sum(1 for r in returns if r < 0)
            }
        
        return summary 