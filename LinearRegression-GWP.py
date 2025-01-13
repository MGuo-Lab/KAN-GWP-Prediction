#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:39:09 2024

@author: k23070952
"""


import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold
# Suppress RDKit warnings
# 데이터 불러오기
X_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_train_selected.pkl')
X_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl')
y_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_train.pkl')
y_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_test.pkl')

# NaN 값을 처리 (예: 0 또는 평균으로 채우기)
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

# 숫자가 아닌 컬럼 제거
X_train = X_train.select_dtypes(include=[np.number])
X_val = X_val.select_dtypes(include=[np.number])

print("데이터가 불러와졌습니다.")


X_combined = pd.concat([X_train, X_val], axis=0)

# Y끼리 합치기
y_combined = pd.concat([y_train, y_val], axis=0)

# 합친 데이터 확인
X_combined.reset_index(drop=True, inplace=True)
y_combined.reset_index(drop=True, inplace=True)
# 숫자형으로 변환 (변환 불가능한 값은 NaN으로 대체)[['ATS0are', 'VMcGowan', 'SMR_VSA1']]
X_combined = X_combined.apply(pd.to_numeric, errors='coerce')[['ATS0are', 'VMcGowan', 'SMR_VSA1', 'SMR', 'Sse', 
                                                                'BCUTv-1l', 'ETA_alpha', 'AATSC1c', 'SpMax_Dzpe', 'SlogP_VSA2',
                                                                'SM1_Dzpe', 'AMID_h', 'BCUTse-1l', 'SlogP_VSA10', 'BCUTi-1h',
                                                                'AXp-0dv', 'Mare', 'BCUTpe-1h', 'AETA_beta', 'SpMAD_Dzse']]
# 숫자형으로 변환 (변환 불가능한 값은 NaN으로 대체)
y_combined = y_combined.apply(pd.to_numeric, errors='coerce')



kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 5개의 폴드로 분할

# 4번째 폴드 추출
fold_index = 1  # 0부터 시작하므로 4번째 폴드는 index 3
for i, (train_idx, test_idx) in enumerate(kf.split(X_combined)):
    if i == fold_index:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

        X_fold_train, X_fold_test = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
        y_fold_train, y_fold_test = y_combined.iloc[train_idx], y_combined.iloc[test_idx]
    
        # 데이터를 텐서로 변환
        X_train_tensor = torch.tensor(X_fold_train.values, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_fold_test.values, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_fold_train.values, dtype=torch.float32).view(-1, 1).to(device)
        y_test_tensor = torch.tensor(y_fold_test.values, dtype=torch.float32).view(-1, 1).to(device)
        
        dataset = {
            'train_input': X_train_tensor,
            'test_input': X_test_tensor,
            'train_label': y_train_tensor,
            'test_label': y_test_tensor,
        }
        

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_fold_train, y_fold_train)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)



y_pred_train = model.predict(X_fold_train)
y_pred_test = model.predict(X_fold_test)

train_mse = mean_squared_error(y_fold_train, y_pred_train)
test_mse = mean_squared_error(y_fold_test, y_pred_test)

train_r2 = r2_score(y_fold_train, y_pred_train)
test_r2 = r2_score(y_fold_test, y_pred_test)

print(f"Train MSE: {train_mse}, Train R^2: {train_r2}")
print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")


plt.figure(figsize=(14, 6),dpi=800)

# 훈련 데이터 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_fold_train, y_pred_train, label='Train Data', alpha=0.7, edgecolors='k', c='#639ace')
plt.plot([y_fold_train.min(), y_fold_train.max()], [y_fold_train.min(), y_fold_train.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Train Data\n$R^2$: {train_r2:.4f}, MSE: {train_mse:.4f}')
plt.xlabel('Actual GWP value', fontsize=12 )
plt.ylabel('Predicted GWP value', fontsize=12)
# plt.grid(True)
plt.legend(fontsize=12)


# 테스트 데이터 플롯
plt.subplot(1, 2, 2)
plt.scatter(y_fold_test, y_pred_test, label='Test Data', alpha=0.7, edgecolors='k', c='#f26b6b')
plt.plot([y_fold_test.min(), y_fold_test.max()], [y_fold_test.min(), y_fold_test.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Test Data\n$R^2$: {test_r2:.4f}, MSE: {test_mse:.4f} ')
plt.xlabel('Actual GWP value', fontsize=12)
plt.ylabel('Predicted GWP value', fontsize=12)
plt.legend(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.show()








