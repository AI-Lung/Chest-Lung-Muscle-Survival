# 项目结构说明

## 📁 目录结构

```
DeepHit_Model_GitHub/
│
├── README.md                    # 项目主文档（必读）
├── USAGE.md                     # 详细使用指南
├── PROJECT_STRUCTURE.md         # 本文件：项目结构说明
├── requirements.txt             # Python依赖包列表
├── .gitignore                   # Git忽略文件配置
├── quick_start.py              # 快速开始脚本
│
├── models/                      # 模型文件目录
│   ├── model_config.json       # 模型配置文件（包含最佳参数）
│   └── deephit_model.pkl       # 训练好的模型文件（需要用户提供）
│
├── data/                        # 数据目录
│   ├── README.md               # 数据格式说明
│   ├── train_data.csv          # 训练数据（用户提供）
│   └── test_data.csv           # 测试数据（用户提供）
│
├── utils/                       # 工具模块
│   ├── __init__.py             # 包初始化文件
│   ├── model_loader.py         # 模型加载器（核心模块）
│   └── evaluator.py            # 模型评估工具
│
└── examples/                    # 示例代码
    ├── __init__.py             # 包初始化文件
    └── validate_model.py      # 完整验证示例
```

## 📄 文件说明

### 核心文件

#### 1. `README.md`
- **作用**: 项目主文档，包含项目概述、安装说明、基本使用方法
- **必读**: ✅ 是
- **内容**: 
  - 模型概述
  - 安装说明
  - 快速开始
  - 评估指标说明

#### 2. `USAGE.md`
- **作用**: 详细使用指南，包含常见问题解答
- **必读**: ✅ 是（使用前必读）
- **内容**:
  - 详细使用步骤
  - 代码示例
  - 常见问题解答
  - 注意事项

#### 3. `requirements.txt`
- **作用**: Python依赖包列表
- **使用**: `pip install -r requirements.txt`

#### 4. `quick_start.py`
- **作用**: 快速开始脚本，最简单的使用方式
- **使用**: `python quick_start.py`

### 模型文件

#### `models/model_config.json`
- **作用**: 存储模型最佳参数配置
- **内容**: 
  - 模型类型
  - 最佳超参数
  - 时间点配置
  - 输出维度

#### `models/deephit_model.pkl`
- **作用**: 训练好的模型文件（需要用户提供）
- **格式**: Pickle格式
- **大小**: 通常几MB到几十MB

### 工具模块

#### `utils/model_loader.py`
- **作用**: 模型加载和预测的核心模块
- **主要类**: `DeepHitModelLoader`
- **功能**:
  - 加载模型和配置
  - 数据标准化
  - 生存概率预测
  - 风险评分预测

#### `utils/evaluator.py`
- **作用**: 模型评估工具
- **主要类**: `ModelEvaluator`
- **功能**:
  - C-index计算
  - ROC AUC计算
  - Integrated Brier Score计算
  - Kaplan-Meier P值计算

### 示例代码

#### `examples/validate_model.py`
- **作用**: 完整的模型验证示例
- **功能**:
  - 模型加载
  - 数据准备
  - 预测
  - 评估
  - 可视化

## 🚀 使用流程

### 方式1: 快速开始（推荐新手）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据（放在data/目录下）
# - train_data.csv
# - test_data.csv

# 3. 准备模型文件（放在models/目录下）
# - deephit_model.pkl
# - model_config.json（已提供）

# 4. 运行快速开始脚本
python quick_start.py
```

### 方式2: 使用示例代码（推荐进阶用户）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 修改examples/validate_model.py中的路径

# 3. 运行验证脚本
cd examples
python validate_model.py
```

### 方式3: 自定义使用（推荐高级用户）

```python
from utils.model_loader import DeepHitModelLoader
from utils.evaluator import ModelEvaluator

# 加载模型
loader = DeepHitModelLoader(...)
# ... 自定义代码
```

## 📋 必需文件清单

### 用户必须提供的文件

- [ ] `models/deephit_model.pkl` - 训练好的模型文件
- [ ] `data/train_data.csv` - 训练数据（用于标准化）
- [ ] `data/test_data.csv` - 测试数据

### 已提供的文件

- [x] `models/model_config.json` - 模型配置
- [x] `utils/model_loader.py` - 模型加载器
- [x] `utils/evaluator.py` - 评估工具
- [x] `examples/validate_model.py` - 验证示例
- [x] `quick_start.py` - 快速开始脚本
- [x] `README.md` - 项目文档
- [x] `USAGE.md` - 使用指南

## 🔍 文件依赖关系

```
quick_start.py
    ├── utils/model_loader.py
    └── utils/evaluator.py

examples/validate_model.py
    ├── utils/model_loader.py
    └── utils/evaluator.py

utils/model_loader.py
    ├── torch
    ├── pycox
    ├── torchtuples
    └── sklearn.preprocessing

utils/evaluator.py
    ├── lifelines
    ├── sklearn.metrics
    └── scipy
```

## 📝 注意事项

1. **模型文件**: 需要从训练结果中获取`.pkl`格式的模型文件
2. **数据格式**: 必须符合要求（见`data/README.md`）
3. **特征顺序**: 必须与训练时保持一致
4. **标准化**: 必须使用与训练时相同的标准化方法

## 🆘 获取帮助

1. 查看 `README.md` - 了解项目概述
2. 查看 `USAGE.md` - 查看详细使用说明和常见问题
3. 查看 `examples/validate_model.py` - 查看完整示例代码
4. 在GitHub上提交Issue

## 📊 项目特点

- ✅ **模块化设计**: 代码结构清晰，易于维护
- ✅ **完整文档**: 提供详细的使用文档和示例
- ✅ **易于使用**: 提供快速开始脚本
- ✅ **灵活扩展**: 可以轻松添加新功能
- ✅ **标准格式**: 遵循Python最佳实践

