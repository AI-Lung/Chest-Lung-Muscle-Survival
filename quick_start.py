"""
DeepHit模型快速开始脚本
简化版的使用示例
"""

import sys
import os
import pandas as pd

# 添加utils到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_loader import DeepHitModelLoader
from utils.evaluator import ModelEvaluator


def quick_predict(model_path, config_path, train_data_path, test_data_path, output_path='predictions.csv'):
    """
    快速预测函数
    
    Parameters:
    -----------
    model_path : str
        模型文件路径
    config_path : str
        配置文件路径
    train_data_path : str
        训练数据路径（用于标准化）
    test_data_path : str
        测试数据路径
    output_path : str
        输出文件路径
    """
    print("="*60)
    print("DeepHit模型快速预测")
    print("="*60)
    
    # 1. 加载模型
    print("\n[1/5] 加载模型...")
    loader = DeepHitModelLoader(model_path=model_path, config_path=config_path)
    loader.load_config()
    loader.load_model()
    print("✓ 模型加载成功")
    
    # 2. 加载并准备数据
    print("\n[2/5] 加载数据...")
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    # 提取特征
    feature_cols = [col for col in train_data.columns if col not in ['ID', 'Time', 'Event', 'Status']]
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    print(f"✓ 训练数据: {len(X_train)} 样本, {len(feature_cols)} 特征")
    print(f"✓ 测试数据: {len(X_test)} 样本")
    
    # 3. 拟合标准化器
    print("\n[3/5] 拟合数据标准化器...")
    loader.fit_scaler(X_train)
    print("✓ 标准化器拟合完成")
    
    # 4. 预测
    print("\n[4/5] 进行预测...")
    survival_probs = loader.predict_survival(X_test, return_df=True)
    risk_scores = loader.predict_risk_score(X_test)
    print("✓ 预测完成")
    
    # 5. 评估（如果有真实标签）
    if 'Time' in test_data.columns and 'Event' in test_data.columns:
        print("\n[5/5] 评估模型性能...")
        evaluator = ModelEvaluator()
        c_index = evaluator.calculate_c_index(
            risk_scores, 
            test_data['Time'], 
            test_data['Event']
        )
        print(f"✓ C-index: {c_index:.4f}")
    else:
        print("\n[5/5] 跳过评估（缺少Time或Event列）")
    
    # 6. 保存结果
    print("\n保存结果...")
    results = pd.DataFrame({
        'ID': test_data['ID'] if 'ID' in test_data.columns else range(len(X_test)),
        'Risk_Score': risk_scores,
    })
    
    # 添加特定时间点的生存概率
    for time_point in [24, 36, 48, 60]:
        if time_point in survival_probs.index:
            results[f'Survival_Prob_{time_point}m'] = survival_probs.loc[time_point].values
        else:
            # 插值
            closest_idx = abs(survival_probs.index - time_point).argmin()
            results[f'Survival_Prob_{time_point}m'] = survival_probs.iloc[closest_idx].values
    
    results.to_csv(output_path, index=False)
    print(f"✓ 结果已保存到: {output_path}")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # 使用示例
    # 请根据实际情况修改路径
    
    model_path = "models/deephit_model.pkl"
    config_path = "models/model_config.json"
    train_data_path = "data/train_data.csv"
    test_data_path = "data/test_data.csv"
    
    # 检查文件是否存在
    import os
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("\n请确保:")
        print("1. 已提供训练好的模型文件 (.pkl)")
        print("2. 已提供模型配置文件 (model_config.json)")
        print("3. 已准备训练数据和测试数据")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    if not os.path.exists(train_data_path):
        print(f"警告: 训练数据文件不存在: {train_data_path}")
        print("将无法拟合标准化器，预测可能不准确")
    
    if not os.path.exists(test_data_path):
        print(f"错误: 测试数据文件不存在: {test_data_path}")
        sys.exit(1)
    
    # 运行预测
    try:
        results = quick_predict(
            model_path=model_path,
            config_path=config_path,
            train_data_path=train_data_path,
            test_data_path=test_data_path
        )
        print(f"\n成功预测 {len(results)} 个样本")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

