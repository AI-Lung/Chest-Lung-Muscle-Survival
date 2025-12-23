# DeepHit Survival Analysis Model - COPD Prognosis Prediction

This is a DeepHit survival analysis model for COPD (Chronic Obstructive Pulmonary Disease) prognosis prediction. The model is based on deep learning technology and can predict patient survival probabilities and risk scores.

## 📋 Table of Contents

- [Model Overview](#model-overview)
- [Model Parameters](#model-parameters)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [File Structure](#file-structure)
- [Example Code](#example-code)
- [Citation](#citation)

## 🎯 Model Overview

DeepHit is a deep learning-based survival analysis model for handling right-censored survival data. This model is specifically optimized for COPD patient prognosis prediction.

### Model Features

- **Deep Learning Architecture**: Uses Multi-Layer Perceptron (MLP) network
- **Discrete Time Modeling**: Discretizes continuous time into multiple time points
- **Competing Risks Handling**: Can handle multiple event types
- **High Performance**: Excellent performance on multiple evaluation metrics

## 📊 Model Parameters

Best model parameters (obtained through hyperparameter search):

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

### Parameter Description

- **alpha**: Ranking loss weight (between 0-1, balances likelihood loss and ranking loss)
- **batch_size**: Batch size
- **dropout**: Dropout rate (prevents overfitting)
- **epochs**: Number of training epochs
- **hidden_layers**: Hidden layer structure ([256] means a single layer with 256 neurons)
- **learning_rate**: Learning rate
- **num_durations**: Number of discrete time points
- **sigma**: Ranking loss smoothing parameter

## 🔧 Installation

### 1. Requirements

- Python >= 3.7
- PyTorch >= 1.9.0

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```python
import torch
import pycox
import torchtuples
print("Installation successful!")
```

## 📖 Usage

### Quick Start

1. **Prepare Data**

   Ensure your data format meets the requirements (see [Data Format](#data-format) section)

2. **Load Model**

```python
from utils.model_loader import DeepHitModelLoader

# Initialize loader
loader = DeepHitModelLoader(
    model_path="models/deephit_model.pkl",
    config_path="models/model_config.json"
)

# Load model and configuration
loader.load_config()
loader.load_model()
```

3. **Fit Scaler**

```python
import pandas as pd

# Load training data (for fitting scaler)
train_data = pd.read_csv("data/train_data.csv")
X_train = train_data.drop(['ID', 'Time', 'Event'], axis=1)

# Fit scaler
loader.fit_scaler(X_train)
```

4. **Make Predictions**

```python
# Load test data
test_data = pd.read_csv("data/test_data.csv")
X_test = test_data.drop(['ID', 'Time', 'Event'], axis=1)

# Predict survival probabilities
survival_probs = loader.predict_survival(X_test, return_df=True)

# Predict risk scores
risk_scores = loader.predict_risk_score(X_test)
```

5. **Evaluate Model**

```python
from utils.evaluator import ModelEvaluator

evaluator = ModelEvaluator()

# Calculate C-index
c_index = evaluator.calculate_c_index(
    risk_scores, 
    test_data['Time'], 
    test_data['Event']
)

print(f"C-index: {c_index:.4f}")
```

### Complete Example

See the `examples/validate_model.py` file for a complete validation workflow.

## 📁 Data Format

### Input Data Requirements

Data should be in CSV format with the following columns:

- **ID**: Patient ID (optional)
- **Time**: Survival time (months)
- **Event**: Event status (1=event occurred, 0=censored)
- **Feature columns**: All other columns are model input features

### Example Data Format

```csv
ID,Time,Event,Feature1,Feature2,Feature3,...
1,24.5,1,0.5,1.2,3.4,...
2,36.0,0,0.8,1.5,2.9,...
3,18.2,1,0.3,0.9,4.1,...
```

### Feature Requirements

- Features should be numeric
- Missing values should be handled before use (recommended: median imputation)
- Feature order should be consistent with training time

## 📈 Evaluation Metrics

The model provides the following evaluation metrics:

### 1. C-index (Concordance Index)

Measures the accuracy of risk ranking predictions, range 0-1, higher is better.

```python
c_index = evaluator.calculate_c_index(risk_scores, time_data, event_data)
```

### 2. ROC AUC

Area under the ROC curve at specific time points, used to evaluate binary classification performance.

```python
roc_auc = evaluator.calculate_roc_auc_at_time(
    survival_prob, time_data, event_data, time_point=36
)
```

### 3. Integrated Brier Score (IBS)

Integrated Brier score, measures prediction calibration, lower is better.

```python
ibs = evaluator.calculate_ibs(survival_probs_df, time_data, event_data)
```

### 4. Kaplan-Meier Log-rank P-value

Used to evaluate the significance of risk stratification.

```python
p_value = evaluator.calculate_km_pvalue(
    risk_scores, time_data, event_data, n_groups=3
)
```

## 📂 File Structure

```
DeepHit_Model_GitHub/
├── README.md                 # This file
├── requirements.txt          # Dependency list
├── models/                   # Model files directory
│   └── model_config.json    # Model configuration file
├── data/                     # Data directory (user provided)
│   ├── train_data.csv       # Training data
│   └── test_data.csv        # Test data
├── utils/                    # Utility modules
│   ├── model_loader.py      # Model loader
│   └── evaluator.py         # Evaluation tools
└── examples/                 # Example code
    └── validate_model.py     # Validation example
```

## 💡 Example Code

### Basic Usage

```python
from utils.model_loader import DeepHitModelLoader
from utils.evaluator import ModelEvaluator
import pandas as pd

# 1. Load model
loader = DeepHitModelLoader(
    model_path="models/deephit_model.pkl",
    config_path="models/model_config.json"
)
loader.load_config()
loader.load_model()

# 2. Prepare data
train_data = pd.read_csv("data/train_data.csv")
test_data = pd.read_csv("data/test_data.csv")

X_train = train_data.drop(['ID', 'Time', 'Event'], axis=1)
X_test = test_data.drop(['ID', 'Time', 'Event'], axis=1)

# 3. Fit scaler
loader.fit_scaler(X_train)

# 4. Predict
survival_probs = loader.predict_survival(X_test)
risk_scores = loader.predict_risk_score(X_test)

# 5. Evaluate
evaluator = ModelEvaluator()
c_index = evaluator.calculate_c_index(
    risk_scores, 
    test_data['Time'], 
    test_data['Event']
)

print(f"C-index: {c_index:.4f}")
```

### Batch Prediction

```python
# Predict for multiple samples
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

## ⚠️ Important Notes

1. **Data Standardization**: Must use the same standardization method as during training, recommended to use the provided `fit_scaler` method
2. **Feature Order**: Ensure feature column order is consistent with training time
3. **Missing Values**: Handle all missing values before use
4. **Time Unit**: Ensure time unit is months, consistent with training data
5. **Model File**: Need to provide trained model file (.pkl format)

## 🔬 Model Performance

Performance on the original dataset:

- **C-index**: 0.72-0.78
- **ROC AUC (36 months)**: 0.68-0.75
- **ROC AUC (48 months)**: 0.70-0.78
- **ROC AUC (60 months)**: 0.65-0.72

*Note: Actual performance may vary depending on the dataset*

## 📝 Citation

If you use this model, please cite the relevant paper:

```bibtex
@article{deephit2018,
  title={DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks},
  author={Lee, Changhee and Zame, William and Yoon, Jinsung and van der Schaar, Mihaela},
  journal={AAAI},
  year={2018}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit Issues and Pull Requests.

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

For questions or suggestions, please contact via GitHub Issues.

---

**Note**: This model is for research purposes only and should not be used for clinical diagnosis or treatment decisions.
