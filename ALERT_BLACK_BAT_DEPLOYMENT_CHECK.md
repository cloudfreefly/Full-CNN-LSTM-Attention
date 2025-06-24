# Alert Black Bat部署检查清单

## 问题诊断
当前回测结果显示：
- ❌ 杠杆比例固定在1.0，没有动态调整
- ❌ 股票仓位始终接近100%，没有现金管理
- ❌ 日志中完全没有"Alert Black Bat"相关输出
- ❌ 代码修改没有生效

## 部署检查步骤

### 1. 确认代码已更新
检查以下文件是否包含最新修改：
- [ ] `portfolio_optimization.py` - 包含"Alert Black Bat代码已执行"日志
- [ ] `leverage_manager.py` - 包含"Alert Black Bat杠杆管理器已执行"日志  
- [ ] `config.py` - 回撤阈值已更新为更敏感的值

### 2. 验证配置正确
运行配置测试：
```python
from config import AlgorithmConfig
config = AlgorithmConfig()
print("动态现金管理:", config.RISK_CONFIG['dynamic_cash_management']['enable_dynamic_cash'])
print("杠杆调整:", config.LEVERAGE_CONFIG['enable_drawdown_based_adjustment'])
print("回撤阈值:", config.LEVERAGE_CONFIG['drawdown_thresholds'])
```

### 3. 检查语法错误
运行语法检查：
```bash
python3 -m py_compile portfolio_optimization.py
python3 -m py_compile leverage_manager.py
python3 -m py_compile config.py
```

### 4. QuantConnect平台操作
1. **保存所有文件**：确保修改已保存
2. **重新编译**：点击编译按钮，检查是否有错误
3. **清除缓存**：重启算法或刷新页面
4. **重新回测**：开始新的回测

### 5. 验证部署成功
新回测日志应包含：
- ✅ "Alert Black Bat代码已执行"
- ✅ "Alert Black Bat杠杆管理器已执行"  
- ✅ "Alert Black Bat: volatile/crisis"（市场状况）
- ✅ "Alert Black Bat杠杆: 回撤X% -> Yx"（杠杆调整）

## 预期效果
- 杠杆比例应在0.6-1.5之间动态变化
- 股票仓位应根据市场状况调整（60-105%）
- 在市场回撤时应及时降低杠杆和仓位

## 故障排除
如果仍然没有效果：
1. 检查是否使用了缓存的旧代码
2. 确认QuantConnect平台版本兼容性
3. 检查是否有异常导致代码回退
4. 联系技术支持或重新创建算法项目 