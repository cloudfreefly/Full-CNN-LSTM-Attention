# 日志延迟配置指南

## 概述

新增的日志延迟参数提供了灵活的日志输出控制，可以根据不同场景和日志类型智能调整延迟时间，防止日志输出过于频繁影响系统性能。

## 配置位置

所有日志延迟配置位于 `config.py` 文件的 `LOGGING_CONFIG` 部分。

## 配置参数详解

### 1. 基础延迟控制

```python
'enable_log_delay': False,  # 是否启用日志延迟（默认关闭）
'log_delay_ms': 20,         # 基础日志延迟时间（毫秒）
```

- **enable_log_delay**: 全局日志延迟开关
- **log_delay_ms**: 默认延迟时间，当没有特定配置时使用

### 2. 分类别日志延迟配置

```python
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
}
```

**用途**: 根据日志类型设置不同的延迟时间
- 高频日志（如训练日志）设置较长延迟
- 重要日志（如算法日志）设置较短延迟

### 3. 动态延迟调整配置

```python
'dynamic_delay': {
    'enable_dynamic_delay': False,    # 启用动态延迟调整
    'min_delay_ms': 5,               # 最小延迟时间
    'max_delay_ms': 100,             # 最大延迟时间
    'high_frequency_threshold': 10,  # 高频日志阈值（每秒条数）
    'delay_increment_ms': 5,         # 延迟递增步长
    'delay_reset_interval': 60       # 延迟重置间隔（秒）
}
```

**功能**: 根据日志频率动态调整延迟时间
- 当日志频率超过阈值时，自动增加延迟
- 定期重置延迟到初始值

### 4. 特殊情况延迟配置

```python
'special_delays': {
    'warmup_period_delay': 5,        # 预热期日志延迟（毫秒）
    'training_period_delay': 100,    # 训练期日志延迟（毫秒）
    'rebalance_period_delay': 30,    # 调仓期日志延迟（毫秒）
    'error_log_delay': 0,            # 错误日志延迟（毫秒，通常不延迟）
    'emergency_log_delay': 0,        # 紧急日志延迟（毫秒，不延迟）
    'debug_mode_delay': 50           # 调试模式延迟（毫秒）
}
```

**智能识别**:
- 自动检测算法运行状态（预热期、训练期、调仓期）
- 根据日志内容识别错误和紧急情况
- 为不同状态设置合适的延迟时间

### 5. 延迟策略配置

```python
'delay_strategy': {
    'strategy_type': 'fixed',        # 延迟策略：'fixed'固定, 'adaptive'自适应, 'burst'突发控制
    'burst_detection_window': 5,     # 突发检测窗口（秒）
    'burst_threshold': 20,           # 突发阈值（每窗口日志条数）
    'burst_delay_multiplier': 2.0,   # 突发延迟倍数
    'adaptive_learning_rate': 0.1,   # 自适应学习率
    'performance_target_ms': 50      # 性能目标（毫秒）
}
```

**策略类型**:
- **fixed**: 固定延迟策略（默认）
- **adaptive**: 自适应延迟策略（根据性能调整）
- **burst**: 突发控制策略（检测日志突发并调整延迟）

## 使用建议

### 1. 开发测试阶段
```python
'enable_log_delay': False  # 关闭延迟，便于调试
```

### 2. 生产环境
```python
'enable_log_delay': True   # 启用延迟，提高性能
'log_delay_ms': 20         # 设置适中的基础延迟
```

### 3. 高频交易场景
```python
'category_delays': {
    'training': 100,       # 训练日志延迟更长
    'algorithm': 5,        # 算法日志延迟较短
}
```

### 4. 调试模式
```python
'special_delays': {
    'debug_mode_delay': 100,  # 调试时增加延迟
    'error_log_delay': 0,     # 错误日志不延迟
}
```

## 延迟计算逻辑

系统按以下优先级计算延迟时间：

1. **紧急情况检查**: 错误/紧急日志 → 0延迟
2. **运行状态检查**: 预热期/训练期/调仓期 → 特殊延迟
3. **日志类型匹配**: 分类别延迟 → 对应延迟
4. **默认延迟**: 基础延迟时间

## 性能影响

- **启用延迟**: 降低日志输出频率，减少I/O压力
- **智能延迟**: 在关键时刻（错误）保持快速响应
- **分类延迟**: 平衡信息输出和系统性能

## 监控建议

1. 观察日志输出频率的变化
2. 监控系统整体性能指标
3. 在关键交易时段验证延迟效果
4. 根据实际情况调整延迟参数

---

**注意**: 延迟设置过长可能影响问题诊断速度，建议根据实际需求调整参数。 