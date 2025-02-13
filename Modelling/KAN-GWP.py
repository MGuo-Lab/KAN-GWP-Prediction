#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:23:02 2024

@author: k23070952
"""

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy
# import asts
from sklearn.metrics import r2_score,mean_squared_error
from itertools import combinations
import time
import seaborn as sns
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from rdkit import RDLogger
from itertools import product
import optuna
from kan import KAN, LBFGS
from mordred import Calculator, descriptors
from IPython.display import clear_output

from sklearn.model_selection import KFold
# Suppress RDKit warnings


# Load data
X_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_train_selected.pkl')
X_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl')
y_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_train.pkl')
y_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_test.pkl')

# Handle NaN values (e.g., fill with 0 or mean)
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

# Remove non-numeric columns
X_train = X_train.select_dtypes(include=[np.number])
X_val = X_val.select_dtypes(include=[np.number])

print("Data has been successfully loaded.")


X_combined = pd.concat([X_train, X_val], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

X_combined.reset_index(drop=True, inplace=True)
y_combined.reset_index(drop=True, inplace=True)
X_combined = X_combined.apply(pd.to_numeric, errors='coerce')
print("컬럼 이름을 알파벳 순서로 정렬:")


indices_to_remove = X_combined[X_combined['ATS0are'] > 1000].index
print(f"\nIndices to be removed: {list(indices_to_remove)}")

# 2. 해당 인덱스를 삭제
X_combined = X_combined.drop(index=indices_to_remove)
y_combined = y_combined.drop(index=indices_to_remove)
print("\nFiltered dataset:")
print(X_combined)

# Data transformation using logarithmic scaling
shift_value = abs(X_combined.min().min()) # 모든 데이터에서 최소값을 기준으로 이동
X_combined = X_combined.applymap(lambda x: np.log1p(x + shift_value))

print("Original DataFrame:")
print(X_combined)
print("\nShift value applied for transformation:", shift_value)

# Standardization using StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data = scaler.fit_transform(X_combined)
X_combined = pd.DataFrame(scaler.fit_transform(data), columns=X_combined.columns)
y_combined = y_combined.apply(pd.to_numeric, errors='coerce')



kf = KFold(n_splits=10, shuffle=True, random_state=1004) 

fold_index = 1 
for i, (train_idx, test_idx) in enumerate(kf.split(X_combined)):
    if i == fold_index:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        X_fold_train, X_fold_test = X_combined.iloc[train_idx], X_combined.iloc[test_idx]
        y_fold_train, y_fold_test = y_combined.iloc[train_idx], y_combined.iloc[test_idx]
        
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
        
        X_tensor = torch.tensor(X_combined.values, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_combined.values, dtype=torch.float32).view(-1, 1).to(device)
        


grids = np.array([5])
k = 3
train_losses = []
test_losses = []
for i in range(grids.shape[0]):
    if i == 0:
        model = KAN(width=[X_train_tensor.shape[1], 1, 1], grid=grids[i], k=k, seed=1).to(device)
    if i != 0:
        model = model.refine(grids[i])
    results = model.fit(dataset, opt='Adam', lr=1e-2, steps=1000,lamb=0.01, lamb_coef=1.0,)
    
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    
    
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.show()

model.plot(folder = '/Users/k23070952/Desktop/kan_figs/kan_fig/', in_vars = X_combined.columns.to_list(), out_vars = ['Global Warming Potential'],
           varscale=0.2, scale = 3)
plt.show()

min_loss = min(test_losses)  
min_step = test_losses.index(min_loss)+1  

print(f"최소 Test Loss: {min_loss}")
print(f"최소 Test Loss가 발생한 스텝: {min_step}")
model = model.prune(edge_th= 0.1)

train_losses = []
test_losses = []
model_2 = model.refine(10)
results = model_2.fit(dataset, opt='Adam', lr=1e-3, lamb=0.01, steps=500)
train_losses += results['train_loss']
test_losses += results['test_loss']
model_2.plot(in_vars = X_combined.columns.to_list(), varscale=0.2)
plt.show()


# # model_2 = model_2.prune()
# train_losses = []
# test_losses = []
# model_3 = model_2.refine(20)
# results = model_3.fit(dataset, opt='LBFGS', lr=1e-3, lamb=0.2, steps=15)
# # results = model_3.fit(dataset, opt="LBFGS", steps=60)
# train_losses += results['train_loss']
# test_losses += results['test_loss']
# model_3.plot(in_vars = X_combined.columns.to_list(), varscale=0.2)
# plt.show()

lib = ['x^2', 'x^3', 'x^4',        # 다항식
        'exp', 'log', 'sqrt',            # 지수, 로그, 제곱근
        'tanh', 'sin', 'cos',            # 하이퍼볼릭 및 삼각 함수
         'x^5','abs', 'x']  


best_model = model_2.copy()

with torch.no_grad():
    y_train_pred = best_model(dataset['train_input'])
    y_test_pred = best_model(dataset['test_input']) 

best_model.auto_symbolic(lib=lib, r2_threshold=0.1, weight_simple=0)
best_model.plot()
plt.show()

with torch.no_grad():
    y_train_pred = best_model(dataset['train_input'])
    y_test_pred = best_model(dataset['test_input']) 
    

# 실제 값
y_train_actual = dataset['train_label']
y_test_actual = dataset['test_label']

# r2_score 계산을 위해 CPU로 이동
y_train_actual = y_train_actual.cpu().numpy()
y_train_pred = y_train_pred.cpu().numpy()
y_test_actual = y_test_actual.cpu().numpy()
y_test_pred = y_test_pred.cpu().numpy()


y_train_pred = np.nan_to_num(y_train_pred, nan=0.0)
y_test_pred = np.nan_to_num(y_test_pred, nan=0.0)
    
train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)
train_mse = mean_squared_error(y_train_actual, y_train_pred)
test_mse = mean_squared_error(y_test_actual, y_test_pred)
print(train_r2, test_r2)


# 실제 값 vs 예측 값 플롯
plt.figure(figsize=(14, 6),dpi=800)

# 훈련 데이터 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_train_actual, y_train_pred, label='Train Data', alpha=0.7, edgecolors='k', c='#639ace')
plt.plot([y_train_actual.min(), y_train_actual.max()], [y_train_actual.min(), y_train_actual.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Train Data\n$R^2$: {train_r2:.4f}, MSE: {train_mse:.4f}')
plt.xlabel('Actual GWP value', fontsize=12 )
plt.ylabel('Predicted GWP value', fontsize=12)
# plt.grid(True)
plt.legend(fontsize=12)


# 테스트 데이터 플롯
plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_test_pred, label='Test Data', alpha=0.7, edgecolors='k', c='#f26b6b')
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Test Data\n$R^2$: {test_r2:.4f}, MSE: {test_mse:.4f} ')
plt.xlabel('Actual GWP value', fontsize=12)
plt.ylabel('Predicted GWP value', fontsize=12)
plt.legend(fontsize=12)
# plt.grid(True)
plt.tight_layout()
plt.show()


best_model.feature_score
best_model.attribute(1,0)
best_model.suggest_symbolic(0,2,0)

x, y = best_model.get_fun(0,0,0)
plt.scatter(x,y)

from kan.utils import ex_round
ex_round(best_model.symbolic_formula()[0][0], 5)




