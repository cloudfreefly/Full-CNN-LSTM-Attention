# Portfolio遍历问题修复

## 问题描述
在QuantConnect平台中遍历`Portfolio`对象时出现错误：
```
cannot unpack non-iterable KeyValuePair[Symbol,SecurityHolding] object
```

## 原因分析
在QuantConnect中，当遍历`algorithm.Portfolio`时，返回的是`KeyValuePair[Symbol, SecurityHolding]`对象，不能直接解包为元组。

## 修复方案

### 错误写法：
```python
for symbol_obj, holding in self.algorithm.Portfolio:
    # 这会导致KeyValuePair解包错误
```

### 正确写法：
```python
for kvp in self.algorithm.Portfolio:
    symbol_obj = kvp.Key
    holding = kvp.Value
    # 正确访问KeyValuePair的Key和Value属性
```

## 修复文件
1. **smart_rebalancer.py** - `_get_current_holdings`方法第70行
2. **position_logger.py** - `log_current_positions`方法第43行

## 影响
修复后，Portfolio遍历功能恢复正常，不再出现KeyValuePair解包错误，确保以下功能正常运行：
- 当前持仓获取
- 仓位分析记录
- 智能再平衡执行

## 注意事项
在QuantConnect平台开发时，应该使用以下模式遍历Portfolio：
- `for kvp in Portfolio:` 然后使用 `kvp.Key` 和 `kvp.Value`
- 或使用 `Portfolio.Values` 直接获取所有持仓对象
- 避免假设Portfolio可以直接解包的用法 