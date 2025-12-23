# 数据目录

此目录用于存放数据文件。

## 数据文件要求

### 训练数据 (train_data.csv)

用于拟合数据标准化器。必须包含：

- `ID`: 患者ID（可选）
- `Time`: 生存时间（单位：月）
- `Event`: 事件状态（1=发生事件，0=删失）
- 特征列：其余所有列

### 测试数据 (test_data.csv)

用于模型验证。格式与训练数据相同。

### 数据格式示例

```csv
ID,Time,Event,Clinical_Age,Clinical_Gender,ChestMuscle_Feature1,WholeLung_Feature1
1,24.5,1,65,1,0.523,1.234
2,36.0,0,72,0,0.678,0.987
3,18.2,1,58,1,0.345,1.456
```

### 注意事项

1. **特征顺序**: 特征列的顺序必须与训练时保持一致
2. **缺失值**: 使用前必须处理所有缺失值（建议使用中位数填充）
3. **数据类型**: 所有特征应为数值型
4. **时间单位**: 确保时间单位为月
5. **事件编码**: Event列应为0或1（0=删失，1=发生事件）

### 数据预处理建议

```python
import pandas as pd
import numpy as np

# 读取数据
data = pd.read_csv('data.csv')

# 处理缺失值（使用中位数填充）
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# 检查数据
print(f"数据形状: {data.shape}")
print(f"缺失值: {data.isnull().sum().sum()}")
print(f"事件率: {data['Event'].mean():.2%}")
```

## 隐私和安全

⚠️ **重要**: 
- 不要将包含患者隐私信息的数据上传到公共仓库
- 建议使用脱敏数据
- 遵守相关数据保护法规

