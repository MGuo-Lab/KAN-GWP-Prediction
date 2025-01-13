#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:39:22 2024

@author: k23070952
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 14:24:46 2024

@author: k23070952
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:23:02 2024

@author: k23070952
"""

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import copy
import ast
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




device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# 2. 텐서를 디바이스로 이동
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1).to(device)


# X_train_tensor = X_train_tensor.to(torch.float32)
# X_val_tensor = X_val_tensor.to(torch.float32)
# y_train_tensor = y_train_tensor.to(torch.float32)
# y_val_tensor = y_val_tensor.to(torch.float32)

dataset = {'train_input': X_train_tensor, 'test_input':X_val_tensor, 'train_label':y_train_tensor, 'test_label':y_val_tensor}



grids = np.array([10])
steps = 20
k = 2
# for _ in range(10):
# 26
second_layer = 20
third_layer = 2
# optimizer = torch.optim.Adam(model.parameters(), lr=lr )
train_losses = []
test_losses = []
for i in range(grids.shape[0]):
    if i == 0:
        model = KAN(width=[X_train_tensor.shape[1], second_layer, third_layer,  1], grid=grids[i], k=k, seed=1).to(device)
    if i != 0:
        model = model.refine(grids[i])
    # results = model.fit(dataset, opt="LBFGS", steps=steps)
    
    results = model.fit(dataset, opt="LBFGS", steps=steps)
    
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    # break

plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train', 'test'])
plt.ylabel('RMSE')
plt.xlabel('step')
plt.show()


# test_losses에서 최소값과 그 인덱스 찾기
min_loss = min(test_losses)  # 가장 작은 loss 값 찾기
min_step = test_losses.index(min_loss)+1  # 가장 작은 loss가 발생한 스텝 (인덱스)

print(f"최소 Test Loss: {min_loss}")
print(f"최소 Test Loss가 발생한 스텝: {min_step}")


train_losses = []
test_losses = []
model_2 = model.refine(20)
results = model_2.fit(dataset, opt="LBFGS", steps=20)
train_losses += results['train_loss']
test_losses += results['test_loss']


train_losses = []
test_losses = []
model_3 = model_2.refine(40)
results = model_3.fit(dataset, opt="LBFGS", steps=20)
train_losses += results['train_loss']
test_losses += results['test_loss']



def min_step_train(step):
    train_losses = []
    test_losses = []
    model = KAN(width=[X_train_tensor.shape[1], second_layer, third_layer, 1], grid=80, k=k, seed=1).to(device)
    results = model.fit(dataset, opt="LBFGS", steps=step)
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    
    
    model = model.refine(15)
    results = model.fit(dataset, opt="LBFGS", steps=20)
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    
    
    model = model.refine(240)
    results = model.fit(dataset, opt="LBFGS", steps=15, )
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    
    # model = model.refine(50)
    # results = model.fit(dataset, opt="LBFGS", steps=10)
    # train_losses += results['train_loss']
    # test_losses += results['test_loss']
    
    # model = model.refine(40)
    # results = model.fit(dataset, opt="LBFGS", steps=30, lamb=0.002, lamb_entropy=2.)
    # train_losses += results['train_loss']
    # test_losses += results['test_loss']
    
    
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.ylabel('RMSE')
    plt.xlabel('step')
    plt.show()
    return model, train_losses, test_losses
    
best_model, best_train_losses, best_test_losses = min_step_train(min_step)
best_min_loss = min(best_test_losses)  # 가장 작은 loss 값 찾기
best_min_step = best_test_losses.index(best_min_loss)+1  # 가장 작은 loss가 발생한 스텝 (인덱스)
print(f"최소 Test Loss: {best_min_loss}")
print(f"최소 Test Loss가 발생한 스텝: {best_min_step}")

best_model.plot()
plt.show()


# 예측 수행
best_model = model_3.copy()

best_model = temp_model

with torch.no_grad():
    y_train_pred = best_model(dataset['train_input'])
    y_test_pred =best_model(dataset['test_input']) 


# 실제 값
y_train_actual = dataset['train_label']
y_test_actual = dataset['test_label']

# r2_score 계산을 위해 CPU로 이동
y_train_actual = y_train_actual.cpu().numpy()
y_train_pred = y_train_pred.cpu().numpy()
y_test_actual = y_test_actual.cpu().numpy()
y_test_pred = y_test_pred.cpu().numpy()

train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)
train_mse = mean_squared_error(y_train_actual, y_train_pred)
test_mse = mean_squared_error(y_test_actual, y_test_pred)

print(f"Train R^2: {train_r2:.4f}")
print(f"Test R^2: {test_r2:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")



# 실제 값 vs 예측 값 플롯
plt.figure(figsize=(14, 6))

# 훈련 데이터 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_train_actual, y_train_pred, alpha=0.7, edgecolors='k')
plt.plot([y_train_actual.min(), y_train_actual.max()], [y_train_actual.min(), y_train_actual.max()], 'r--')
plt.title(f'Train Data\n$R^2$: {train_r2:.4f}, MSE: {train_mse:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)


# 테스트 데이터 플롯
plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_test_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
plt.title(f', Test Data\n$R^2$: {test_r2:.4f}, MSE: {test_mse:.4f} ')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()



best_model.plot()
plt.show()

# pruned_model.plot(metric='forward_n')

# plt.scatter(np.arange(21)+1, pruned_model.feature_score.cpu().detach().numpy())
# plt.xlabel('rank of input features', fontsize=15)
# plt.ylabel('feature attribution score', fontsize=15)
# plt.show()

pruned_model = best_model.prune_input(threshold=0.2)

pruned_model = best_model.prune(edge_th=0.2)
pruned_model.plot()

pruned_model = pruned_model.prune()
plt.show()



pruned_model.auto_symbolic()

from kan import *
sf = model.symbolic_formula()[0][0]
nsimplify(ex_round(ex_round(sf, 3),3))


model(dataset)



from kan import *

pruned_model = best_model.prune(edge_th=0.005)
pruned_model = pruned_model.prune_input(threshold=0.3)
pruned_model.plot(in_vars=input_vars, varscale=0.3,tick = True)
plt.show()

tree_model = pruned_model.tree()
tree_model.plot()

best_model.symbolic_formula()





pruned_model = best_model.prune(node_th = 0.00001)




# r2_threshold 탐색 범위 정의
r2_threshold_values = [round(x, 2) for x in np.arange(0.0, 1.05, 0.05)]  # 0.0에서 1.0까지 0.05 단위로 생성
train_r2_list = [] 
test_r2_list = []

# 실제 값
y_train_actual = dataset['train_label'].cpu().numpy()
y_test_actual = dataset['test_label'].cpu().numpy()



    
    
# r2_threshold를 변경하며 모델 평가
for r2_threshold in r2_threshold_values:
    print(f"Running auto_symbolic with r2_threshold={r2_threshold}")
    
    # 모델 복사 및 심볼릭 변환 수행
    temp_model = model_3.copy()
    y_train_pred = temp_model(dataset['train_input'])
    temp_model.auto_symbolic(lib=lib, r2_threshold=r2_threshold, weight_simple=0.2)
    
    # 예측 수행
    with torch.no_grad():
        y_train_pred = temp_model(dataset['train_input'])
        y_test_pred = temp_model(dataset['test_input'])
    
    # 예측 값
    y_train_pred = y_train_pred.cpu().numpy()
    y_test_pred = y_test_pred.cpu().numpy()
    
    # r2_score 계산
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    # 결과 저장
    train_r2_list.append(train_r2)
    test_r2_list.append(test_r2)
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

# 탐색 결과 출력
print("r2_threshold 탐색 완료")
print("Train R² 리스트:", train_r2_list)
print("Test R² 리스트:", test_r2_list)



# automatic mode
lib = ['x', 'x^2', 'x^3', 'x^4',        # 다항식
        'exp', 'log', 'sqrt',            # 지수, 로그, 제곱근
        'tanh', 'sin', 'cos',            # 하이퍼볼릭 및 삼각 함수
         'abs',                    # 거듭제곱 및 절댓값
        'x^5',]         # 부호, 내림, 올림 연산자

best_model = model_3.copy()
best_model.auto_symbolic(lib=lib, r2_threshold=0.8, weight_simple=0.2)

with torch.no_grad():
    y_train_pred = best_model(dataset['train_input'])
    y_test_pred =best_model(dataset['test_input']) 


# 실제 값
y_train_actual = dataset['train_label']
y_test_actual = dataset['test_label']

# r2_score 계산을 위해 CPU로 이동
y_train_actual = y_train_actual.cpu().numpy()
y_train_pred = y_train_pred.cpu().numpy()
y_test_actual = y_test_actual.cpu().numpy()
y_test_pred = y_test_pred.cpu().numpy()

train_r2 = r2_score(y_train_actual, y_train_pred)
test_r2 = r2_score(y_test_actual, y_test_pred)



temp_model.symbolic_formula()[0][0]






