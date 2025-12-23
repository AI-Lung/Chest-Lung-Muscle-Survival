"""
DeepHit模型验证示例
演示如何使用训练好的模型在新数据上进行验证
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import DeepHitModelLoader
from utils.evaluator import ModelEvaluator


def load_data(data_path):
    """
    加载数据
    
    数据格式要求：
    - CSV文件
    - 前3列为：ID, Time, Event
    - 其余列为特征
    """
    data = pd.read_csv(data_path)
    
    # 提取ID、时间和事件
    if 'ID' in data.columns:
        ids = data['ID']
    else:
        ids = data.index
    
    if 'Time' in data.columns:
        time = data['Time']
    else:
        raise ValueError("数据中必须包含'Time'列")
    
    if 'Event' in data.columns:
        event = data['Event']
    elif 'Status' in data.columns:
        event = data['Status']
    else:
        raise ValueError("数据中必须包含'Event'或'Status'列")
    
    # 提取特征（排除ID、Time、Event列）
    feature_cols = [col for col in data.columns if col not in ['ID', 'Time', 'Event', 'Status']]
    X = data[feature_cols]
    
    return X, time, event, ids


def validate_model(model_path, config_path, train_data_path, test_data_path):
    """
    验证模型性能
    
    Parameters:
    -----------
    model_path : str
        模型文件路径
    config_path : str
        配置文件路径
    train_data_path : str
        训练数据路径（用于拟合标准化器）
    test_data_path : str
        测试数据路径
    """
    print("="*80)
    print("DeepHit模型验证")
    print("="*80)
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    loader = DeepHitModelLoader(model_path=model_path, config_path=config_path)
    loader.load_config()
    loader.load_model()
    print("✓ 模型加载成功")
    
    # 2. 加载训练数据并拟合标准化器
    print("\n2. 加载训练数据并拟合标准化器...")
    X_train, time_train, event_train, ids_train = load_data(train_data_path)
    loader.fit_scaler(X_train)
    print(f"✓ 训练数据: {len(X_train)} 样本, {X_train.shape[1]} 特征")
    
    # 3. 加载测试数据
    print("\n3. 加载测试数据...")
    X_test, time_test, event_test, ids_test = load_data(test_data_path)
    print(f"✓ 测试数据: {len(X_test)} 样本, {X_test.shape[1]} 特征")
    print(f"  事件数: {event_test.sum()} ({event_test.sum()/len(event_test)*100:.1f}%)")
    
    # 4. 预测
    print("\n4. 进行预测...")
    survival_probs = loader.predict_survival(X_test, return_df=True)
    risk_scores = loader.predict_risk_score(X_test)
    print("✓ 预测完成")
    
    # 5. 评估
    print("\n5. 评估模型性能...")
    evaluator = ModelEvaluator()
    
    # C-index
    c_index = evaluator.calculate_c_index(risk_scores, time_test, event_test)
    print(f"\nC-index: {c_index:.4f}")
    
    # ROC AUC (多个时间点)
    time_points = [24, 36, 48, 60]
    print(f"\nROC AUC (各时间点):")
    for t in time_points:
        # 获取该时间点的生存概率
        if t in survival_probs.index:
            surv_prob_t = survival_probs.loc[t].values
        else:
            # 插值找到最接近的时间点
            closest_idx = np.argmin(np.abs(survival_probs.index - t))
            surv_prob_t = survival_probs.iloc[closest_idx].values
        
        roc_auc = evaluator.calculate_roc_auc_at_time(
            surv_prob_t, time_test, event_test, t
        )
        print(f"  {t}月: {roc_auc:.4f}" if not np.isnan(roc_auc) else f"  {t}月: N/A")
    
    # IBS
    ibs = evaluator.calculate_ibs(survival_probs, time_test, event_test)
    print(f"\nIntegrated Brier Score (IBS): {ibs:.4f}")
    
    # KM曲线P值
    km_p_2 = evaluator.calculate_km_pvalue(risk_scores, time_test, event_test, n_groups=2)
    km_p_3 = evaluator.calculate_km_pvalue(risk_scores, time_test, event_test, n_groups=3)
    print(f"\nKaplan-Meier Log-rank P值:")
    print(f"  2组: {km_p_2:.4f}")
    print(f"  3组: {km_p_3:.4f}")
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    results = pd.DataFrame({
        'ID': ids_test,
        'Time': time_test,
        'Event': event_test,
        'Risk_Score': risk_scores,
        'Survival_Prob_24m': survival_probs.loc[24].values if 24 in survival_probs.index else np.nan,
        'Survival_Prob_36m': survival_probs.loc[36].values if 36 in survival_probs.index else np.nan,
        'Survival_Prob_48m': survival_probs.loc[48].values if 48 in survival_probs.index else np.nan,
        'Survival_Prob_60m': survival_probs.loc[60].values if 60 in survival_probs.index else np.nan,
    })
    
    results_path = 'validation_results.csv'
    results.to_csv(results_path, index=False)
    print(f"✓ 结果已保存到: {results_path}")
    
    # 7. 绘制生存曲线（示例）
    print("\n7. 绘制风险分层图...")
    plot_risk_stratification(risk_scores, time_test, event_test)
    print("✓ 图表已保存")
    
    print("\n" + "="*80)
    print("验证完成！")
    print("="*80)
    
    return {
        'c_index': c_index,
        'ibs': ibs,
        'km_p_2groups': km_p_2,
        'km_p_3groups': km_p_3,
        'risk_scores': risk_scores,
        'survival_probs': survival_probs
    }


def plot_risk_stratification(risk_scores, time_data, event_data, save_path='risk_stratification.png'):
    """绘制风险分层图"""
    from lifelines import KaplanMeierFitter
    
    # 分组（3组）
    tertile1 = np.percentile(risk_scores, 33.33)
    tertile2 = np.percentile(risk_scores, 66.67)
    groups = np.zeros(len(risk_scores))
    groups[(risk_scores > tertile1) & (risk_scores <= tertile2)] = 1
    groups[risk_scores > tertile2] = 2
    
    group_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    colors = ['#3498db', '#f39c12', '#e74c3c']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # KM曲线
    kmf = KaplanMeierFitter()
    for i in range(3):
        mask = (groups == i)
        if np.any(mask):
            kmf.fit(time_data[mask], event_data[mask], 
                   label=f'{group_labels[i]} (n={sum(mask)}, events={sum(event_data[mask])})')
            kmf.plot_survival_function(ax=axes[0], ci_show=True, color=colors[i], 
                                      linewidth=3, alpha=0.8)
    
    axes[0].set_xlabel('Time (months)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    axes[0].set_title('Kaplan-Meier Curves by Risk Group', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 风险评分分布
    for i in range(3):
        mask = (groups == i)
        axes[1].hist(risk_scores[mask], bins=20, alpha=0.6, color=colors[i], 
                    label=group_labels[i], edgecolor='black')
    
    axes[1].set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Risk Score Distribution by Group', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # 示例用法
    # 请根据实际情况修改路径
    
    model_path = "../models/deephit_model.pkl"  # 模型文件路径
    config_path = "../models/model_config.json"  # 配置文件路径
    train_data_path = "../data/train_data.csv"  # 训练数据路径
    test_data_path = "../data/test_data.csv"  # 测试数据路径
    
    # 检查文件是否存在
    import os
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("请确保已提供训练好的模型文件")
    else:
        validate_model(model_path, config_path, train_data_path, test_data_path)

