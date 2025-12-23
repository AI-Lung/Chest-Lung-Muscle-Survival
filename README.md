# DeepHit生存分析模型 - COPD预后预测

这是一个用于COPD（慢性阻塞性肺疾病）预后预测的DeepHit生存分析模型。该模型基于深度学习技术，能够预测患者的生存概率和风险评分。

## 📋 目录

- [模型概述](#模型概述)
- [模型参数](#模型参数)
- [安装说明](#安装说明)
- [使用方法](#使用方法)
- [数据格式](#数据格式)
- [评估指标](#评估指标)
- [文件结构](#文件结构)
- [示例代码](#示例代码)
- [引用](#引用)

## 🎯 模型概述

DeepHit是一个基于深度学习的生存分析模型，用于处理右删失的生存数据。本模型专门针对COPD患者的预后预测进行了优化。

### 模型特点

- **深度学习架构**: 使用多层感知机（MLP）网络
- **离散时间建模**: 将连续时间离散化为多个时间点
- **竞争风险处理**: 能够处理多种事件类型
- **高性能**: 在多个评估指标上表现优异

## 📊 模型参数

最佳模型参数（通过超参数搜索获得）：

```json
{
  "alpha": 0.25,
  "batch_size": 32,
  "dropout": 0.4,
  "epochs": 150,
  "hidden_layers": [256],
  "learning_rate": 0.0001,
  "num_durations": 30,
  "sigma": 0.1
}
```

### 参数说明

- **alpha**: 排序损失权重（0-1之间，平衡似然损失和排序损失）
- **batch_size**: 批次大小
- **dropout**: Dropout比率（防止过拟合）
- **epochs**: 训练轮数
- **hidden_layers**: 隐藏层结构（[256]表示单层256个神经元）
- **learning_rate**: 学习率
- **num_durations**: 离散时间点数量
- **sigma**: 排序损失平滑参数

## 🔧 安装说明

### 1. 环境要求

- Python >= 3.7
- PyTorch >= 1.9.0

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```python
import torch
import pycox
import torchtuples
print("安装成功！")
```

## 📖 使用方法

### 快速开始

1. **准备数据**

   确保您的数据格式符合要求（见[数据格式](#数据格式)部分）

2. **加载模型**

```python
from utils.model_loader import DeepHitModelLoader

# 初始化加载器
loader = DeepHitModelLoader(
    model_path="models/deephit_model.pkl",
    config_path="models/model_config.json"
)

# 加载模型和配置
loader.load_config()
loader.load_model()
```

3. **拟合标准化器**

```python
import pandas as pd

# 加载训练数据（用于拟合标准化器）
train_data = pd.read_csv("data/train_data.csv")
X_train = train_data.drop(['ID', 'Time', 'Event'], axis=1)

# 拟合标准化器
loader.fit_scaler(X_train)
```

4. **进行预测**

```python
# 加载测试数据
test_data = pd.read_csv("data/test_data.csv")
X_test = test_data.drop(['ID', 'Time', 'Event'], axis=1)

# 预测生存概率
survival_probs = loader.predict_survival(X_test, return_df=True)

# 预测风险评分
risk_scores = loader.predict_risk_score(X_test)
```

5. **评估模型**

```python
from utils.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# 计算C-index
c_index = evaluator.calculate_c_index(
    risk_scores, 
    test_data['Time'], 
    test_data['Event']
)

print(f"C-index: {c_index:.4f}")
```

### 完整示例

参见 `examples/validate_model.py` 文件，其中包含完整的验证流程。

## 📁 数据格式

### 输入数据要求

数据应为CSV格式，包含以下列：

- **ID**: 患者ID（可选）
- **Time**: 生存时间（月）
- **Event**: 事件状态（1=发生事件，0=删失）
- **特征列**: 其余列为模型输入特征

### 示例数据格式

```csv
ID,Time,Event,Feature1,Feature2,Feature3,...
1,24.5,1,0.5,1.2,3.4,...
2,36.0,0,0.8,1.5,2.9,...
3,18.2,1,0.3,0.9,4.1,...
```

### 特征要求

- 特征应为数值型
- 缺失值应在使用前处理（建议使用中位数填充）
- 特征顺序应与训练时保持一致

## 📈 评估指标

模型提供以下评估指标：

### 1. C-index（一致性指数）

衡量模型预测风险排序的准确性，范围0-1，越高越好。

```python
c_index = evaluator.calculate_c_index(risk_scores, time_data, event_data)
```

### 2. ROC AUC

特定时间点的ROC曲线下面积，用于评估二分类性能。

```python
roc_auc = evaluator.calculate_roc_auc_at_time(
    survival_prob, time_data, event_data, time_point=36
)
```

### 3. Integrated Brier Score (IBS)

综合Brier评分，衡量预测校准度，越低越好。

```python
ibs = evaluator.calculate_ibs(survival_probs_df, time_data, event_data)
```

### 4. Kaplan-Meier Log-rank P值

用于评估风险分组的显著性。

```python
p_value = evaluator.calculate_km_pvalue(
    risk_scores, time_data, event_data, n_groups=3
)
```

## 📂 文件结构

```
DeepHit_Model_GitHub/
├── README.md                 # 本文件
├── requirements.txt          # 依赖包列表
├── models/                   # 模型文件目录
│   └── model_config.json    # 模型配置文件
├── data/                     # 数据目录（用户提供）
│   ├── train_data.csv       # 训练数据
│   └── test_data.csv        # 测试数据
├── utils/                    # 工具模块
│   ├── model_loader.py      # 模型加载器
│   └── evaluator.py         # 评估工具
└── examples/                 # 示例代码
    └── validate_model.py     # 验证示例
```

## 💡 示例代码

### 基本使用

```python
from utils.model_loader import DeepHitModelLoader
from utils.evaluator import ModelEvaluator
import pandas as pd

# 1. 加载模型
loader = DeepHitModelLoader(
    model_path="models/deephit_model.pkl",
    config_path="models/model_config.json"
)
loader.load_config()
loader.load_model()

# 2. 准备数据
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

X_train = train_data.drop(['ID', 'Time', 'Event'], axis=1)
X_test = test_data.drop(['ID', 'Time', 'Event'], axis=1)

# 3. 拟合标准化器
loader.fit_scaler(X_train)

# 4. 预测
survival_probs = loader.predict_survival(X_test)
risk_scores = loader.predict_risk_score(X_test)

# 5. 评估
evaluator = ModelEvaluator()
c_index = evaluator.calculate_c_index(
    risk_scores, 
    test_data['Time'], 
    test_data['Event']
)

print(f"C-index: {c_index:.4f}")
```

### 批量预测

```python
# 对多个样本进行预测
results = []
for idx, row in test_data.iterrows():
    X_sample = row.drop(['ID', 'Time', 'Event']).values.reshape(1, -1)
    X_sample_df = pd.DataFrame(X_sample, columns=X_train.columns)
    
    surv_prob = loader.predict_survival(X_sample_df)
    risk_score = loader.predict_risk_score(X_sample_df)
    
    results.append({
        'ID': row['ID'],
        'Risk_Score': risk_score[0],
        'Survival_Prob_36m': surv_prob.loc[36, 0] if 36 in surv_prob.index else None
    })

results_df = pd.DataFrame(results)
results_df.to_csv('predictions.csv', index=False)
```

## ⚠️ 注意事项

1. **数据标准化**: 必须使用与训练时相同的标准化方法，建议使用提供的`fit_scaler`方法
2. **特征顺序**: 确保特征列的顺序与训练时一致
3. **缺失值**: 在使用前处理所有缺失值
4. **时间单位**: 确保时间单位为月，与训练数据一致
5. **模型文件**: 需要提供训练好的模型文件（.pkl格式）

## 🔬 模型性能

在原始数据集上的性能表现：

- **C-index**: 0.72-0.78
- **ROC AUC (36月)**: 0.68-0.75
- **ROC AUC (48月)**: 0.70-0.78
- **ROC AUC (60月)**: 0.65-0.72

*注：实际性能可能因数据集而异*

## 📝 引用

如果您使用本模型，请引用相关论文：

```bibtex
@article{deephit2018,
  title={DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks},
  author={Lee, Changhee and Zame, William and Yoon, Jinsung and van der Schaar, Mihaela},
  journal={AAAI},
  year={2018}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证。

## 📧 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**注意**: 本模型仅用于研究目的，不应用于临床诊断或治疗决策。

