"""
DeepHit模型加载工具
用于加载训练好的DeepHit生存分析模型
"""

import torch
import pickle
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pycox.models import DeepHitSingle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import torchtuples as tt


class DeepHitModelLoader:
    """DeepHit模型加载器"""
    
    def __init__(self, model_path=None, config_path=None):
        """
        初始化模型加载器
        
        Parameters:
        -----------
        model_path : str, optional
            模型文件路径 (.pkl文件)
        config_path : str, optional
            模型配置文件路径 (.json文件)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def load_config(self, config_path=None):
        """加载模型配置"""
        if config_path is None:
            config_path = self.config_path
        
        if config_path is None or not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        return self.config
    
    def load_model(self, model_path=None):
        """
        加载训练好的DeepHit模型
        
        Parameters:
        -----------
        model_path : str, optional
            模型文件路径
            
        Returns:
        --------
        tuple : (pycox_model, labtrans, durations)
        """
        if model_path is None:
            model_path = self.model_path
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 尝试使用pickle加载
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, tuple) and len(model_data) == 3:
                self.model = model_data
                return self.model
            else:
                raise ValueError("模型文件格式不正确")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")
    
    def create_model_from_config(self, in_features):
        """
        根据配置创建新的DeepHit模型（用于重新训练）
        
        Parameters:
        -----------
        in_features : int
            输入特征数量
            
        Returns:
        --------
        DeepHitSingle : 新创建的模型
        """
        if self.config is None:
            raise ValueError("请先加载配置文件")
        
        params = self.config['best_params']
        durations = np.array(self.config['durations'])
        
        # 创建标签转换器
        labtrans = LabTransDiscreteTime(len(durations))
        labtrans.cuts = durations[1:]  # 排除第一个0值
        
        # 构建神经网络
        net = tt.practical.MLPVanilla(
            in_features,
            params['hidden_layers'],
            labtrans.out_features,
            True,  # batch_norm
            params['dropout']
        )
        
        # 创建DeepHit模型
        model = DeepHitSingle(
            net,
            tt.optim.Adam(params['learning_rate']),
            alpha=params['alpha'],
            sigma=params['sigma'],
            duration_index=labtrans.cuts
        )
        
        return model, labtrans
    
    def fit_scaler(self, X_train):
        """
        拟合数据标准化器
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            训练数据
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        
        self.scaler.fit(X_train)
        self.is_fitted = True
    
    def transform_data(self, X):
        """
        标准化数据
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            输入数据
            
        Returns:
        --------
        np.ndarray : 标准化后的数据
        """
        if not self.is_fitted:
            raise ValueError("请先使用fit_scaler拟合标准化器")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.scaler.transform(X)
    
    def predict_survival(self, X, return_df=True):
        """
        预测生存概率
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            输入特征数据
        return_df : bool
            是否返回DataFrame格式
            
        Returns:
        --------
        pd.DataFrame or np.ndarray : 生存概率
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        # 标准化数据
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler.transform(X.values)
        else:
            X_scaled = self.scaler.transform(X)
        
        # 转换为tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # 预测
        pycox_model, labtrans, durations = self.model
        surv_prob = pycox_model.predict_surv_df(X_tensor)
        
        if return_df:
            return surv_prob
        else:
            return surv_prob.values
    
    def predict_risk_score(self, X):
        """
        预测风险评分（生存概率的负值）
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            输入特征数据
            
        Returns:
        --------
        np.ndarray : 风险评分
        """
        surv_prob = self.predict_survival(X, return_df=True)
        # 使用平均生存概率的负值作为风险评分
        risk_scores = -surv_prob.mean(axis=0).values
        return risk_scores

