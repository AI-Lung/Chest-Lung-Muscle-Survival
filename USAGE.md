# 使用指南

## 快速开始

### 步骤1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤2: 准备数据

确保您的数据文件符合以下格式：

- CSV格式
- 包含列：`ID`（可选）、`Time`（生存时间，单位：月）、`Event`（事件状态：1=发生事件，0=删失）
- 其余列为特征列

示例：
```csv
ID,Time,Event,Feature1,Feature2,Feature3
1,24.5,1,0.5,1.2,3.4
2,36.0,0,0.8,1.5,2.9
```

### 步骤3: 运行验证脚本

```bash
cd examples
python validate_model.py
```

**注意**: 运行前请修改`validate_model.py`中的文件路径：
- `model_path`: 模型文件路径（.pkl文件）
- `config_path`: 配置文件路径（model_config.json）
- `train_data_path`: 训练数据路径（用于拟合标准化器）
- `test_data_path`: 测试数据路径

## 详细使用说明

### 1. 模型加载

```python
from utils.model_loader import DeepHitModelLoader

loader = DeepHitModelLoader(
    model_path="path/to/model.pkl",
    config_path="path/to/model_config.json"
)

# 加载配置和模型
loader.load_config()
loader.load_model()
```

### 2. 数据标准化

**重要**: 必须使用与训练时相同的标准化方法！

```python
import pandas as pd

# 加载训练数据（用于拟合标准化器）
train_data = pd.read_csv("train_data.csv")
X_train = train_data.drop(['ID', 'Time', 'Event'], axis=1)

# 拟合标准化器
loader.fit_scaler(X_train)
```

### 3. 预测

```python
# 加载测试数据
test_data = pd.read_csv("test_data.csv")
X_test = test_data.drop(['ID', 'Time', 'Event'], axis=1)

# 预测生存概率（返回DataFrame，行为时间点，列为样本）
survival_probs = loader.predict_survival(X_test, return_df=True)

# 预测风险评分（返回numpy数组）
risk_scores = loader.predict_risk_score(X_test)
```

### 4. 评估

```python
from utils.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# C-index
c_index = evaluator.calculate_c_index(
    risk_scores, 
    test_data['Time'], 
    test_data['Event']
)

# ROC AUC (特定时间点)
roc_auc_36m = evaluator.calculate_roc_auc_at_time(
    survival_probs.loc[36].values,  # 36个月的生存概率
    test_data['Time'],
    test_data['Event'],
    time_point=36
)

# Integrated Brier Score
ibs = evaluator.calculate_ibs(
    survival_probs,
    test_data['Time'],
    test_data['Event']
)

# Kaplan-Meier P值
km_p = evaluator.calculate_km_pvalue(
    risk_scores,
    test_data['Time'],
    test_data['Event'],
    n_groups=3  # 2组或3组
)
```

## 常见问题

### Q1: 如何获取特定时间点的生存概率？

```python
survival_probs = loader.predict_survival(X_test, return_df=True)

# 获取36个月的生存概率
if 36 in survival_probs.index:
    surv_prob_36m = survival_probs.loc[36].values
else:
    # 如果精确时间点不存在，使用插值
    closest_idx = np.argmin(np.abs(survival_probs.index - 36))
    surv_prob_36m = survival_probs.iloc[closest_idx].values
```

### Q2: 如何对单个样本进行预测？

```python
# 单个样本（需要是DataFrame格式）
sample = pd.DataFrame([sample_values], columns=feature_names)
survival_prob = loader.predict_survival(sample)
risk_score = loader.predict_risk_score(sample)
```

### Q3: 特征顺序必须与训练时一致吗？

**是的！** 特征列的顺序必须与训练时完全一致。建议：
- 使用相同的特征名称
- 按照相同的顺序排列特征列

### Q4: 如何处理缺失值？

在使用模型前，必须处理所有缺失值。建议方法：
- 数值特征：使用中位数填充
- 分类特征：使用众数填充

```python
# 示例：使用中位数填充
X_test = X_test.fillna(X_test.median())
```

### Q5: 模型文件在哪里？

模型文件（.pkl格式）需要从训练结果中获取。如果您只有模型参数配置，可以使用`create_model_from_config`方法重新创建模型结构（但需要重新训练）。

### Q6: 如何可视化结果？

参见`examples/validate_model.py`中的`plot_risk_stratification`函数，它展示了如何绘制风险分层图。

## 注意事项

1. ⚠️ **数据标准化至关重要**: 必须使用与训练时相同的标准化方法
2. ⚠️ **特征顺序**: 确保特征列顺序与训练时一致
3. ⚠️ **时间单位**: 确保时间单位为月
4. ⚠️ **缺失值**: 使用前必须处理所有缺失值
5. ⚠️ **模型文件**: 需要提供完整的训练好的模型文件

## 获取帮助

如果遇到问题，请：
1. 检查数据格式是否正确
2. 确认所有依赖包已正确安装
3. 查看错误信息并检查文件路径
4. 在GitHub上提交Issue

