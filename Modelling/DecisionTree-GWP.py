#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:15:22 2024

@author: k23070952
"""
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold

X_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_train_selected.pkl')
X_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl')
y_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_train.pkl')
y_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_test.pkl')

X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

X_train = X_train.select_dtypes(include=[np.number])
X_val = X_val.select_dtypes(include=[np.number])


X_combined = pd.concat([X_train, X_val], axis=0)
y_combined = pd.concat([y_train, y_val], axis=0)

X_combined.reset_index(drop=True, inplace=True)
y_combined.reset_index(drop=True, inplace=True)
X_combined = X_combined.apply(pd.to_numeric, errors='coerce')
y_combined = y_combined.apply(pd.to_numeric, errors='coerce')



kf = KFold(n_splits=10, shuffle=True, random_state=42) 

'''If you want to perform a 10-fold prediction performance evaluation, add the {for loop} below along with the model training code and store the performance metrics in a list.'''
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
        
        


# Step 1: 
model = DecisionTreeRegressor(max_depth=3, random_state=42)

# Step 2:
model.fit(X_fold_train, y_fold_train)

# Step 3:
y_pred_train = model.predict(X_fold_train)
y_pred_test = model.predict(X_fold_test)

train_mse = mean_squared_error(y_fold_train, y_pred_train)
test_mse = mean_squared_error(y_fold_test, y_pred_test)

train_r2 = r2_score(y_fold_train, y_pred_train)
test_r2 = r2_score(y_fold_test, y_pred_test)

print(f"Train MSE: {train_mse}, Train R^2: {train_r2}")
print(f"Test MSE: {test_mse}, Test R^2: {test_r2}")

# Step 4:
plt.figure(figsize=(20, 10),dpi=800)
plot_tree(model)
plt.show()


plt.figure(figsize=(14, 6),dpi=800)

plt.subplot(1, 2, 1)
plt.scatter(y_fold_train, y_pred_train, label='Train Data', alpha=0.7, edgecolors='k', c='#639ace')
plt.plot([y_fold_train.min(), y_fold_train.max()], [y_fold_train.min(), y_fold_train.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Train Data\n$R^2$: {train_r2:.4f}, MSE: {train_mse:.4f}')
plt.xlabel('Actual GWP value', fontsize=12 )
plt.ylabel('Predicted GWP value', fontsize=12)
plt.legend(fontsize=12)


plt.subplot(1, 2, 2)
plt.scatter(y_fold_test, y_pred_test, label='Test Data', alpha=0.7, edgecolors='k', c='#f26b6b')
plt.plot([y_fold_test.min(), y_fold_test.max()], [y_fold_test.min(), y_fold_test.max()], '--', label='Ideal Prediction', c='#2ca02c')
plt.title(f'Test Data\n$R^2$: {test_r2:.4f}, MSE: {test_mse:.4f} ')
plt.xlabel('Actual GWP value', fontsize=12)
plt.ylabel('Predicted GWP value', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


