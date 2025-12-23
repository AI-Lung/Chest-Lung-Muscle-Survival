"""
模型评估工具
用于评估DeepHit模型的性能
"""

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_curve, auc
from scipy.integrate import simpson
from scipy import stats


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self):
        pass
    
    def calculate_c_index(self, risk_scores, time_data, event_data):
        """
        计算C-index（一致性指数）
        
        Parameters:
        -----------
        risk_scores : array-like
            风险评分（越高表示风险越大）
        time_data : array-like
            生存时间
        event_data : array-like
            事件状态（1=发生事件，0=删失）
            
        Returns:
        --------
        float : C-index值
        """
        c_index = concordance_index(
            event_times=time_data,
            predicted_scores=risk_scores,
            event_observed=event_data
        )
        return c_index
    
    def calculate_roc_auc_at_time(self, survival_prob, time_data, event_data, time_point):
        """
        计算特定时间点的ROC AUC
        
        Parameters:
        -----------
        survival_prob : array-like
            生存概率
        time_data : array-like
            生存时间
        event_data : array-like
            事件状态
        time_point : float
            时间点（月）
            
        Returns:
        --------
        float : ROC AUC值
        """
        # 确定在time_point之前发生事件的样本
        actual_status = (time_data <= time_point) & (event_data == 1)
        
        n_events = actual_status.sum()
        n_non_events = (~actual_status).sum()
        
        if n_events == 0 or n_non_events == 0:
            return np.nan
        
        # 使用1-生存概率作为风险评分
        risk_scores = 1 - survival_prob
        
        fpr, tpr, _ = roc_curve(actual_status, risk_scores)
        roc_auc = auc(fpr, tpr)
        
        return roc_auc
    
    def calculate_roc_auc_at_times(self, survival_prob, time_data, event_data, time_points=[24, 36, 48, 60]):
        """
        计算多个时间点的ROC AUC
        
        Parameters:
        -----------
        survival_prob : array-like
            生存概率（单个时间点）
        time_data : array-like
            生存时间
        event_data : array-like
            事件状态
        time_points : list
            时间点列表（月）
            
        Returns:
        --------
        list : 各时间点的ROC AUC值
        """
        auc_values = []
        for t in time_points:
            auc = self.calculate_roc_auc_at_time(survival_prob, time_data, event_data, t)
            auc_values.append(auc)
        return auc_values
    
    def calculate_brier_score(self, survival_prob, time_point, actual_time, actual_status):
        """
        计算Brier Score
        
        Parameters:
        -----------
        survival_prob : array-like
            生存概率
        time_point : float
            时间点
        actual_time : array-like
            实际生存时间
        actual_status : array-like
            事件状态
            
        Returns:
        --------
        float : Brier Score
        """
        actual_survival = (actual_time > time_point).astype(int)
        censored = (actual_time <= time_point) & (actual_status == 0)
        squared_diff = (survival_prob - actual_survival) ** 2
        squared_diff[censored] = 0
        return np.mean(squared_diff)
    
    def calculate_ibs(self, survival_probs_df, time_data, event_data, time_points=None):
        """
        计算Integrated Brier Score (IBS)
        
        Parameters:
        -----------
        survival_probs_df : pd.DataFrame
            生存概率DataFrame（行为时间点，列为样本）
        time_data : array-like
            实际生存时间
        event_data : array-like
            事件状态
        time_points : array-like, optional
            时间点（如果为None，使用survival_probs_df的index）
            
        Returns:
        --------
        float : IBS值
        """
        if time_points is None:
            time_points = survival_probs_df.index.values
        
        brier_scores = []
        valid_times = []
        
        for t in time_points:
            if t in survival_probs_df.index:
                survival_prob = survival_probs_df.loc[t].values
                bs = self.calculate_brier_score(survival_prob, t, time_data, event_data)
                if not np.isnan(bs):
                    brier_scores.append(bs)
                    valid_times.append(t)
        
        if not brier_scores:
            return np.nan
        
        ibs = simpson(brier_scores, valid_times) / (max(valid_times) - min(valid_times))
        return ibs
    
    def calculate_km_pvalue(self, risk_scores, time_data, event_data, n_groups=2, method='median'):
        """
        计算Kaplan-Meier曲线的Log-rank检验P值
        
        Parameters:
        -----------
        risk_scores : array-like
            风险评分
        time_data : array-like
            生存时间
        event_data : array-like
            事件状态
        n_groups : int
            分组数量（2或3）
        method : str
            分组方法（'median'或'roc'）
            
        Returns:
        --------
        float : P值
        """
        from lifelines.statistics import logrank_test, multivariate_logrank_test
        
        risk_scores = np.array(risk_scores, dtype=float)
        time_data_array = np.array(time_data, dtype=float)
        event_data_array = np.array(event_data, dtype=int)
        
        # 分组
        if method == 'median':
            if n_groups == 2:
                median_risk = np.median(risk_scores)
                groups = (risk_scores > median_risk).astype(int)
            else:  # n_groups == 3
                tertile1 = np.percentile(risk_scores, 33.33)
                tertile2 = np.percentile(risk_scores, 66.67)
                groups = np.zeros(len(risk_scores))
                groups[(risk_scores > tertile1) & (risk_scores <= tertile2)] = 1
                groups[risk_scores > tertile2] = 2
        else:
            # 使用中位数作为简单方法（实际应用中可以使用更复杂的ROC方法）
            if n_groups == 2:
                median_risk = np.median(risk_scores)
                groups = (risk_scores > median_risk).astype(int)
            else:
                tertile1 = np.percentile(risk_scores, 33.33)
                tertile2 = np.percentile(risk_scores, 66.67)
                groups = np.zeros(len(risk_scores))
                groups[(risk_scores > tertile1) & (risk_scores <= tertile2)] = 1
                groups[risk_scores > tertile2] = 2
        
        groups = np.array(groups, dtype=int)
        
        # Log-rank检验
        if n_groups == 2:
            low_mask = (groups == 0)
            high_mask = (groups == 1)
            
            if np.any(low_mask) and np.any(high_mask):
                results = logrank_test(
                    time_data_array[low_mask], time_data_array[high_mask],
                    event_data_array[low_mask], event_data_array[high_mask]
                )
                return results.p_value
            else:
                return 1.0
        else:  # n_groups == 3
            results = multivariate_logrank_test(time_data_array, groups, event_data_array)
            return results.p_value

