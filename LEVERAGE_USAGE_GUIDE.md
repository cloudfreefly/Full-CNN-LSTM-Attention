# 杠杆交易功能使用指南

## 📈 **功能概述**

本系统已成功配置支持**最高150%持仓**的杠杆交易功能，通过智能风险评估实现动态杠杆调整。

## 🔧 **主要配置参数**

### 1. **杠杆配置 (LEVERAGE_CONFIG)**
```python
LEVERAGE_CONFIG = {
    'enable_leverage': True,              # 启用杠杆功能
    'max_leverage_ratio': 1.5,            # 最大杠杆比例150%
    'low_risk_leverage_ratio': 1.5,       # 低风险环境：150%
    'medium_risk_leverage_ratio': 1.2,    # 中等风险环境：120%
    'high_risk_leverage_ratio': 0.8,      # 高风险环境：80%
    'extreme_risk_leverage_ratio': 0.5,   # 极端风险环境：50%
}
```

### 2. **现金管理配置**
```python
'min_cash_ratio': -0.50,                 # 可借款50%（150%持仓）
'max_cash_ratio': 0.05,                  # 低风险时最大现金5%
'leverage_cash_buffer': -0.45,           # 杠杆缓冲-45%
```

### 3. **风险控制参数**
```python
'leverage_max_drawdown': 0.12,           # 杠杆环境最大回撤12%
'leverage_stop_loss': -0.08,             # 杠杆止损-8%
'leverage_margin_call_level': 0.25,      # 保证金追缴25%
```

## 🎯 **杠杆动态调整机制**

### **风险等级评估**
- **低风险** (VIX < 18, 波动率 < 12%): 150%杠杆
- **中等风险** (VIX < 25, 波动率 < 20%): 120%杠杆  
- **高风险** (VIX < 35, 波动率 < 30%): 80%杠杆
- **极端风险** (VIX > 35, 波动率 > 30%): 50%杠杆

### **实时调整策略**
1. **每日评估**：根据VIX、波动率、回撤水平评估风险
2. **平滑调整**：避免频繁变动，使用30%平滑因子
3. **最小变动**：杠杆变化超过5%才执行调整

## ⚠️ **风险控制机制**

### **保证金管理**
- **警告线**: 借款30%时触发警告
- **追缴线**: 借款40%时要求减仓
- **平仓线**: 借款45%时强制平仓

### **紧急保护**
- **自动去杠杆**: 触发平仓线时自动降至80%持仓
- **回撤保护**: 回撤超过12%时降低杠杆
- **VIX防护**: VIX快速上升时立即调整杠杆

## 💼 **实际应用示例**

### **低风险环境下的150%持仓**
```
市场条件: VIX=15, 波动率=10%, 回撤=2%
杠杆设置: 150%
现金状态: -50% (借款50%)
持仓状态: 股票150%, 现金-50%
```

### **风险上升时的自动调整**
```
市场条件: VIX=30, 波动率=25%, 回撤=8%
杠杆设置: 80% (自动降杠杆)
现金状态: 20% (持有现金)
持仓状态: 股票80%, 现金20%
```

## 📊 **监控和报告**

### **杠杆状态监控**
- 当前杠杆比例
- 目标杠杆比例  
- 风险等级评估
- 保证金使用情况

### **每日报告内容**
- 杠杆使用统计
- 风险等级分布
- 保证金安全边际
- 杠杆调整历史

## 🚀 **使用建议**

### **新手用户**
1. 建议从120%持仓开始测试
2. 密切关注保证金水平
3. 理解杠杆风险和成本

### **进阶用户**
1. 可以使用完整150%持仓
2. 自定义风险阈值参数
3. 结合VIX对冲策略使用

### **风险提醒**
⚠️ **杠杆交易显著增加风险，可能导致更大损失**
⚠️ **确保理解借款成本和保证金要求**  
⚠️ **建议先在模拟环境充分测试**

## 🔄 **启用步骤**

1. **确认配置**: 检查 `config.py` 中的 `LEVERAGE_CONFIG`
2. **账户准备**: 确保QuantConnect账户支持杠杆交易
3. **风险评估**: 理解杠杆交易的风险和成本
4. **监控设置**: 启用详细的杠杆监控和报告
5. **测试运行**: 先在小资金环境下测试

## 📝 **技术实现**

### **核心组件**
- `LeverageManager`: 杠杆计算和管理
- `风险评估模块`: 动态风险等级评估  
- `保证金监控`: 实时保证金水平检查
- `紧急控制`: 自动去杠杆保护机制

### **集成流程**
1. 每日风险评估 → 2. 杠杆比例计算 → 3. 权重调整 → 4. 保证金检查 → 5. 执行交易

---

**⚡ 现在您的系统已完全支持150%最高持仓的杠杆交易功能！** 