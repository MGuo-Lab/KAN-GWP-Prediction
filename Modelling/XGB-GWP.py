#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:51:18 2024

@author: k23070952
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 19:32:48 2024

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

colors = sns.color_palette("bright")
RDLogger.DisableLog('rdApp.*')

def embedding_visualisation(embeddings_array):
    """PCA로 축소해서 최적의 차원 찾기"""
    pca = PCA(n_components=1000)
    pca.fit(embeddings_array)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio) 

    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    print(n_components_90)
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=750)

    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio', color=colors[0])
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color=colors[0], alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=colors[0])

    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Variance Ratio', color=colors[1])
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    plt.title('Scree Plot with Explained and Cumulative Variance')
    plt.tight_layout()
    plt.show()

def drop_dup(df):
    original_size = len(df)
    grouped_df = df.groupby(['CanonicalSMILES',], as_index=False).agg({
        'Amount': 'mean',
        'Process ID': 'first',
        'Product Name': 'first',
        'Category ID': 'first',
        'Category Name': 'first',
        'Flow_name': 'first',
        'InChIKey': 'first',
        'Description': 'first',
        'Location': 'first',
        'embedding': 'first',
        'Description_embedding': 'first',
        'Process_Name_embed': 'first'
    })
    print(f"groupby() 후 남은 행의 수: {len(grouped_df)}")
    return grouped_df

'''유기 물질 제거'''
inorganic_smiles_exceptions = ['CO2', '[U]', '[Al]', 'ClCl']

# def filter_organic_smiles(df):
#     organic_rows = []
#     for index, row in df.iterrows():
#         smiles = row['CanonicalSMILES']
#         mol = Chem.MolFromSmiles(smiles)
#         if not mol:
#             continue

#         if smiles not in inorganic_smiles_exceptions:
#             for atom in mol.GetAtoms():
#                 if atom.GetSymbol() == 'C':
#                     organic_rows.append(row.to_dict())
#                     break
#     print('유기물질선별: ', len(df)-len(organic_rows), '개 삭제됨, 남은개수: ', len(organic_rows))
#     return pd.DataFrame(organic_rows).reset_index(drop=True)


def filtering(df):
    # Q1 = df['Amount'].quantile(0.25)
    # Q3 = df['Amount'].quantile(0.75)
    # IQR = Q3 - Q1
    # df = df[(df['Amount'] >= (Q1 - 1.5 * IQR)) & (df['Amount'] <= (Q3 + 1.5 * IQR))]
    # df = df[df['Amount'] > 0]
    print(len(df))
    # 데이터 크기(Amount 값의 크기)에 따라 내림차순 정렬
    df = df.sort_values(by='Amount', ascending=False)
    
    # 상위 17% 제외 (즉, 하위 83% 선택)
    threshold_index = int(len(df) * 0.03)  # 상위 17% 인덱스 계산
    df_filtered = df.iloc[threshold_index:]  # 상위 17% 제외한 나머지
    df_filtered = df_filtered[df_filtered['Amount'] > 0]
    print(f"1. 스케일링 후 {len(df)-len(df_filtered)}개 삭제됨, 남은개수: , {len(df_filtered)}")
    # df_filtered = df_filtered[df_filtered['Amount'] < 147266]
    
    # Remove rows where 'Process Name' contains 'market' (case-insensitive)
    df_del_market = df_filtered[~df_filtered['Process Name'].str.contains('market', case=False, na=False)]
    print(f"2. 시장조사 내용 삭제 후 {len(df_filtered)-len(df_del_market)}개 삭제됨, 남은개수: , {len(df_del_market)}")
    return df_del_market
    


GWP_df = pd.read_csv("/Users/k23070952/Desktop/LCA/0. Data/description_name_location_embeddings_df_laction_description_241007.csv")
GWP_df = GWP_df.drop(['Product Value'], axis=1)

GWP_df['Description_embedding'] = GWP_df['Description_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
GWP_df['embedding'] = GWP_df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
GWP_df['Process_Name_embed'] = GWP_df['Process_Name_embed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) 

filterd_df = filtering(GWP_df)


# 로그 변환
log_data = np.log(filterd_df['Amount'] + 1)
filterd_df['Amount'] = log_data


# filterd_df = filter_organic_smiles(filterd_df)
# grouped_df = drop_dup(filterd_df)

embeddings_array = np.vstack(filterd_df['Description_embedding'].values)
pca = PCA(n_components=60,random_state=1004)
pca_result = pca.fit_transform(embeddings_array)
pca_df = pd.DataFrame(pca_result, columns=[f'process_embed_{i+1}' for i in range(pca_result.shape[1])])

location_embeddings_array = np.vstack(filterd_df['embedding'].values)
pca = PCA(n_components=10,random_state=1004)
location_pca_result = pca.fit_transform(location_embeddings_array)
location_embed_df = pd.DataFrame(location_pca_result, columns=[f'location_embed_{i+1}' for i in range(location_pca_result.shape[1])])

Process_Name_embed_array = np.vstack(filterd_df['Process_Name_embed'].values)
pca = PCA(n_components=40,random_state=1004)
Process_Name_embed_result = pca.fit_transform(Process_Name_embed_array)
Process_Name_embed_df = pd.DataFrame(Process_Name_embed_result, columns=[f'Name_embed_{i+1}' for i in range(Process_Name_embed_result.shape[1])])

filterd_df = pd.concat([filterd_df.reset_index(drop=True), pca_df, location_embed_df, Process_Name_embed_df], axis=1)

filterd_df['Amount'] = pd.to_numeric(filterd_df['Amount'], errors='coerce')

train_data = filterd_df.drop(['Process ID', 'Process Name', 'Product Name', 'Description', 'Location', 'Category ID', 'Category Name', 'Flow_name', 'InChIKey', 'embedding', 'Description_embedding', 'Process_Name_embed'], axis=1)

# Initialize Mordred calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=True)

# Counter for invalid SMILES strings
none_count = 0

# Convert SMILES to Mordred descriptors
def smiles_to_mordred(smiles):
    global none_count
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mordred_desc = calc(mol)
            return mordred_desc.fill_missing(0)  # Fill missing values with 0
        else:
            none_count += 1
            return [0] * len(calc.descriptors)
    except:
        none_count += 1
        return [0] * len(calc.descriptors)

# Apply smiles_to_mordred and create a DataFrame of Mordred descriptors
mordred_df = pd.DataFrame(train_data['CanonicalSMILES'].apply(smiles_to_mordred).tolist(),
                          columns=[str(desc) for desc in calc.descriptors])

# Reset index for consistent merging
mordred_df = mordred_df.reset_index(drop=True)
train_data = train_data.reset_index(drop=True)

# Concatenate original data with Mordred descriptors
mordred_added_df = pd.concat([train_data, mordred_df], axis=1)

# Display a summary of the 'Amount' column
print(mordred_added_df['Amount'].describe())


none_count = 0
# Convert SMILES to MACCS keys
def smiles_to_maccs(smiles):
    global none_count
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            maccs = MACCSkeys.GenMACCSKeys(mol)
            return list(maccs)
        else:
            none_count+=1
            return [0] * 167  # MACCS keys length is 167
    except:
        none_count+=1
        return [0] * 167

# Apply smiles_to_maccs and create a DataFrame of MACCS keys (167 columns)
maccs_df = pd.DataFrame(train_data['CanonicalSMILES'].apply(smiles_to_maccs).tolist(),
                        columns=[f'MACCS_{i}' for i in range(167)])

# 기존 데이터프레임과 MACCS 키 데이터프레임 병합

maccs_df = maccs_df.reset_index(drop=True)
# train_data = train_data.reset_index(drop=True)
mordred_added_df = pd.concat([mordred_added_df, maccs_df], axis=1)
print(mordred_added_df['Amount'].describe())




'''특정 컬럼 제거'''
# Feature 그룹 정의
feature_groups = {
    'L': [col for col in mordred_added_df.columns if 'location_embed_' in col],
    'T': [col for col in mordred_added_df.columns if 'process_embed' in col],
    'D': [col for col in mordred_added_df.columns if 'Name_embed_' in col],
    'FP': [col for col in mordred_added_df.columns if 'MACCS_' in col]
}


feature_combinations = []
for i in range(1, len(feature_groups) + 1):
    feature_combinations.extend(combinations(feature_groups.keys(), i))

# for combination in feature_combinations: ('L', 'T', 'D', 'FP')
combination =  ('L','D')
cols_to_drop = []
for key in feature_groups.keys():
    if key not in combination:
        cols_to_drop.extend(feature_groups[key])

modified_df = mordred_added_df.drop(columns=cols_to_drop)
combination_name = '+'.join(combination)
print(f"Feature Set: {combination_name}")
print(modified_df)

print(f'feature number: {len(modified_df.columns)}')
constant_columns = [col for col in modified_df.columns if modified_df[col].nunique() == 1]
print("모든 행이 동일한 값을 가지는 컬럼:", constant_columns)
modified_df = modified_df.drop(columns=constant_columns)
print(f'Clearned feature number: {len(modified_df.columns)}')

X = modified_df.drop(['CanonicalSMILES', 'Amount'], axis=1)
y = modified_df['Amount']

# n_var = len(input_vars)

mordred_feature = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl').columns
# 2. X와 mordred_feature의 공통 컬럼 찾기
common_columns = X.columns.intersection(mordred_feature)

# 3. X에서 공통 컬럼만 추출
X_filtered = X[common_columns]

# 1. L과 T 컬럼 추출
# L_columns = [col for col in mordred_added_df.columns if 'location_embed_' in col]
# T_columns = [col for col in mordred_added_df.columns if 'process_embed' in col]
# d_columns = [col for col in mordred_added_df.columns if 'Name_embed_' in col]

# 1. L, T, d 컬럼 추출
L_columns = [col for col in mordred_added_df.columns if 'location_embed_' in col]
T_columns = [col for col in mordred_added_df.columns if 'process_embed' in col]
d_columns = [col for col in mordred_added_df.columns if 'Name_embed_' in col]
chemical_columns = [col for col in X_filtered.columns]  # Chemical descriptors


# 2. L과 T 컬럼의 데이터 추출
L_data = mordred_added_df[L_columns]
T_data = mordred_added_df[T_columns]
d_data = mordred_added_df[d_columns]
# 3. X_filtered와 L, T 데이터를 결합
# Reset index to ensure alignment

# Concatenate along the columns axis
X_final = pd.concat([X_filtered, L_data, T_data, d_data], axis=1)

print(f"최종 컬럼 개수: {X_final.shape[1]}")
print(f"최종 컬럼 리스트: {list(X_final.columns)}")


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=1004)


# Callback function to measure time for each iteration
def on_step_complete(study, trial):
    global iteration_start_time
    iteration_end_time = time.time()
    iteration_time = iteration_end_time - iteration_start_time
    current_iteration = len(study.trials)  # Number of completed iterations
    print(f"Iteration {current_iteration}: Time taken = {iteration_time:.2f} seconds")
    iteration_start_time = time.time()  # Reset start time for the next iteration




from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
import numpy as np
from tqdm import tqdm
import xgboost as xgb

# Base model
model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', random_state=1004)

# # 리스트 초기화
# scores = []

# # 피처 개수별로 교차 검증
# for n_features in tqdm(range(1, 157)):  # 1 to 157 features
#     rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
#     X_train_rfe = rfe.fit_transform(X_train, y_train)
    
#     # 각 폴드의 스코어를 저장
#     fold_scores = cross_val_score(model, X_train_rfe, y_train, cv=20, scoring='neg_mean_squared_error')
#     scores.append((n_features, fold_scores))

# # 스코어 데이터 변환
# feature_counts = [item[0] for item in scores]
# fold_scores = [item[1] for item in scores]  # 각 피처 개수에서의 폴드별 스코어

# # 시각화
# plt.figure(figsize=(10, 6))
# for i, fold_score in enumerate(fold_scores):
#     plt.scatter([feature_counts[i]] * len(fold_score), fold_score, alpha=0.5)  # 각 피처 개수에 대한 분산 표시

# plt.xlabel("Number of Features")
# plt.ylabel("Cross-Validation Score (neg MSE)")
# plt.title("Feature Count vs Cross-Validation Scores")
# plt.show()



# plt.figure(figsize=(16, 9), dpi = 1000)  # 16:9 종횡비로 그래프 크기 설정
# plt.subplots_adjust(bottom=0.2)

# plt.boxplot(fold_scores, positions=feature_counts, widths=0.6, patch_artist=True,
#             boxprops=dict(facecolor='steelblue', color='black', alpha=0.85),
#             medianprops=dict(color='black'),
#             showmeans=False,
#             flierprops=dict(marker=''))

# mean_values = [np.mean(scores) for scores in fold_scores]
# plt.plot(feature_counts, mean_values, color='midnightblue', marker='o', linestyle='-', linewidth=2.5, markersize=4,
#          markerfacecolor="navy", markeredgecolor="black")

# # x축을 5 단위로 설정
# plt.xticks(ticks=range(0, max(feature_counts)+1, 5), labels=range(0, max(feature_counts)+1, 5))


# plt.xlabel("Number of Features")
# plt.ylabel("Cross-Validation Score (neg MSE)")
# plt.title("Feature Count vs Cross-Validation Scores")
# plt.show()



# '''선택된 feature수로 feature set 확보'''
# rfe = RFE(estimator=model, n_features_to_select=54, step=1)
# X_train_rfe = rfe.fit_transform(X_train, y_train)

# # 각 폴드의 스코어를 저장
# fold_scores = cross_val_score(model, X_train_rfe, y_train, cv=20, scoring='neg_mean_squared_error')
# # scores.append((n_features, fold_scores))
# rfe.support_

# # np.save('/Users/k23070952/Desktop/LCA/1. Code/selected_features_54.npy', rfe.support_)



# # loaded_mask = np.load('selected_features_30.npy')


# X_train_selected = X_train.values[:, rfe.support_]
# X_test_selected = X_test.values[:, rfe.support_]



# Initialize the start time for tracking iterations
global iteration_start_time
iteration_start_time = time.time()


# Objective function for Optuna
def objective(trial):
    # Define the parameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 10),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
    }

    # Initialize and fit the model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=1004,
        **params
    )
    
    model.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse  # We aim to minimize the mean squared error


# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, callbacks=[on_step_complete])  # Adjust n_trials as needed

# Get the best parameters found
best_params = study.best_params
print("Best parameters found: ", best_params)

# qwer = study.best_trial

from sklearn.metrics import r2_score

# 최적의 하이퍼파라미터로 모델을 다시 학습
best_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    random_state=1004,
    **best_params
)

# 모델 학습
best_model.fit(X_train, y_train)

# 예측 및 R² 계산
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² Score: ", r2)



import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# SHAP 값 계산 및 그룹화
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)


import shap
import pandas as pd
import matplotlib.pyplot as plt





# 1. "Top 5 + Others" 계산 함수
def calculate_top5_and_others(shap_df, group_columns, group_name, top_n=3):
    """
    특정 그룹의 Top 5 feature와 나머지 feature의 합을 계산하여 반환
    """
    group_shap_values = shap_df[group_columns].abs().mean(axis=0)  # 절대값 평균
    top_features = group_shap_values.sort_values(ascending=False).head(top_n)  # Top N
    # other_value = group_shap_values.sum() - top_features.sum()  # 나머지 합
    # top_features[f"Other ({group_name})"] = other_value  # "Other" 추가
    return top_features

shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)

# 각 그룹에 해당하는 feature 컬럼
L_columns = [col for col in shap_df.columns if 'location_embed_' in col]
T_columns = [col for col in shap_df.columns if 'process_embed' in col]
d_columns = [col for col in shap_df.columns if 'Name_embed_' in col]
chemical_columns = [col for col in shap_df.columns if col in X_filtered.columns]

# 3. 각 그룹별 Top 5 + Others 계산
results = []
for group_name, group_columns in {
    "Process location": L_columns,
    "Process description": T_columns,
    "Process title": d_columns,
    "Chemical descriptors": chemical_columns
}.items():
    group_result = calculate_top5_and_others(shap_df, group_columns, group_name, top_n=5)
    results.append(group_result)

# 4. 결과를 하나의 DataFrame으로 통합
plot_data = pd.concat(results, axis=1)



# 1. SHAP 값을 퍼센트로 변환 (전체 합 기준)
total_sum = plot_data.sum().sum()  # 모든 그룹의 SHAP 값 합계
df_percent = plot_data / total_sum * 100  # 전체 합 기준으로 퍼센트 계산

# Process location 관련 인덱스 필터링
filtered_index = ~df_percent.index.str.contains("location_embed_|Other \(Process location\)")

# 필터링된 데이터프레임 생성
df_percent_filtered = df_percent.loc[filtered_index].drop(columns=[0])  # 열도 제거


# 3. 방사형 플롯 데이터 준비
categories = df_percent_filtered.index.tolist()  # Feature 이름 리스트
num_categories = len(categories)

angles = [n / float(num_categories) * 2 * np.pi for n in range(num_categories)]
angles += angles[:1]  # 닫힌 도형을 위해 첫 값을 마지막에 추가

# 4. 방사형 플롯 생성
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'polar': True})  # 그림 크기 설정

# 각 그룹별 데이터 플롯
for group, label in zip(
    df_percent_filtered.columns,
    ["Process description", "Process title", "Chemical descriptors"],
):  # Process location 제외
    values = df_percent_filtered[group].tolist()
    values += values[:1]  # 첫 값을 마지막에 추가하여 닫힌 도형 생성
    ax.plot(angles, values, linewidth=2, label=label)  # 그룹 이름을 Legend에 추가
    ax.fill(angles, values, alpha=0.25)

# 5. 방사형 플롯 설정
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12, rotation=45, ha='right')  # 라벨 회전 및 정렬
ax.set_ylim(0, 20)  # Y축 범위를 0~20%로 설정
ax.set_yticks(np.linspace(0, 20, 5))  # 퍼센트(%) 기준으로 눈금 설정
ax.set_yticklabels([f"{int(i)}%" for i in np.linspace(0, 20, 5)])  # 퍼센트 단위로 표시
ax.set_title("Top 5 SHAP Features with Others by Group (Global %, Max 20%)", size=20, pad=20)

# Legend 수정: 그룹 이름을 올바르게 표시
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Feature Groups")

# 플롯 표시
plt.tight_layout()
plt.show()











