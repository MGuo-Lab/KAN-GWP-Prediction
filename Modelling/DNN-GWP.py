#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 19:23:02 2024

@author: k23070952
"""

import optuna
import ast
from sklearn.metrics import r2_score
from itertools import combinations
import time
import seaborn as sns
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from rdkit import RDLogger
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.decomposition import PCA
import umap
from scikeras.wrappers import KerasRegressor
from skopt.space import Real, Integer, Categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# Suppress RDKit warnings

colors = sns.color_palette("bright")
RDLogger.DisableLog('rdApp.*')

def embedding_visualisation(embeddings_array):
    """PCA로 축소해서 최적의 차원 찾기"""
    # PCA를 사용하여 주성분의 고윳값과 누적 설명 분산 비율 계산
    pca = PCA(n_components=1000)  # 100차원까지 분석
    pca.fit(embeddings_array)

    # 고윳값 (eigenvalues) 및 누적 설명 분산 비율 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio) 

    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    print(n_components_90)
    fig, ax1 = plt.subplots(figsize=(8, 6),dpi =750)

    # 첫 번째 y축: 막대 그래프 (Explained Variance Ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio', color=colors[0])
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, color=colors[0], alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=colors[0])

    # 두 번째 y축: 선 그래프 (Cumulative Variance Ratio)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Cumulative Variance Ratio', color=colors[1])
    ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, color=colors[1])
    ax2.tick_params(axis='y', labelcolor=colors[1])

    # 그래프 제목 및 라벨 설정
    plt.title('Scree Plot with Explained and Cumulative Variance')

    # 그래프 표시
    plt.tight_layout()
    plt.show()

# def drop_dup(df):
#     # 원본 데이터의 크기 저장 (삭제 전)
#     original_size = len(df)
    
#     # Process Name과 CanonicalSMILES가 중복되는 행 삭제
#     # GWP_df_deduplicated = df.drop_duplicates(subset=['Process Name', 'CanonicalSMILES'])
#     # drop_duplicates()를 사용한 중복 삭제
#     # deduplicated_df = df.drop_duplicates(subset=['Process Name', 'CanonicalSMILES'])
#     # print(f"drop_duplicates() 후 남은 행의 수: {len(deduplicated_df)}")
    
#     # groupby()를 사용한 평균 계산 후 남은 행의 수
#     grouped_df = df.groupby(['Process Name', 'CanonicalSMILES'], as_index=False).agg({
#         'Amount': 'mean',
#         'Process ID': 'first',
#         'Product Name': 'first',
#         'Category ID': 'first',
#         'Category Name': 'first',
#         'Flow_name': 'first',
#         'InChIKey': 'first',
#         'Description': 'first',
#         'Location': 'first',
#         'embedding': 'first',
#         'Description_embedding': 'first',
#         'Process_Name_embed': 'first'
#     })
#     print(f"groupby() 후 남은 행의 수: {len(grouped_df)}")

#     return grouped_df


'''유기 물질 제거'''
inorganic_smiles_exceptions = ['CO2', '[U]', '[Al]', 'ClCl']

# 유기 화합물 필터링 함수 (탄소 원자 존재 및 예외 처리)
def filter_organic_smiles(df):
    organic_rows = []
    for index, row in df.iterrows():
        smiles = row['CanonicalSMILES']
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            continue

        if smiles not in inorganic_smiles_exceptions:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    organic_rows.append(row.to_dict())  # Series를 딕셔너리로 변환하여 추가
                    break
    print('3. 유기물질선별: ',len(df)-len(organic_rows), '개 삭제됨, 남은개수: ', len(organic_rows))
    return pd.DataFrame(organic_rows).reset_index(drop=True)


# # IQR을 사용한 이상치 제거
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
# ori_df = pd.read_csv("./Process_location_embeddings_df_laction_description_241007.csv")

# 해당 데이터의 개수
GWP_df = GWP_df.drop(['Product Value',],axis=1)

print(len(GWP_df))

# 1. 'embedding' 컬럼이 문자열로 저장된 경우 이를 리스트로 변환
GWP_df['Description_embedding'] = GWP_df['Description_embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
GWP_df['embedding'] = GWP_df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
GWP_df['Process_Name_embed'] = GWP_df['Process_Name_embed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) 




"""카테고리 처리"""
filterd_df = filtering(GWP_df)

# filterd_df = drop_dup(filterd_df)

# filterd_df = filter_organic_smiles(filterd_df)
'''데이터 전처리 및 분석'''
# top_10_duplicates = df['CanonicalSMILES'].value_counts().head(10).reset_index()
# top_10_duplicates.columns = ['CanonicalSMILES', 'Count']




embeddings_array = np.vstack(filterd_df['Description_embedding'].values)
pca = PCA(n_components=60,random_state=1004)
pca_result = pca.fit_transform(embeddings_array)
pca_df = pd.DataFrame(pca_result, columns=[f'process_embed_{i+1}' for i in range(pca_result.shape[1])])

''''''
# 4. 'embedding' 컬럼 (location_embed)도 리스트를 펼쳐서 각 차원별로 컬럼 생성
location_embeddings_array = np.vstack(filterd_df['embedding'].values)  # 'embedding' 컬럼을 2D 배열로 변환
pca = PCA(n_components=10,random_state=1004)
location_pca_result = pca.fit_transform(location_embeddings_array)
location_embed_df = pd.DataFrame(location_pca_result, columns=[f'location_embed_{i+1}' for i in range(location_pca_result.shape[1])])

''''''
# 4. 'embedding' 컬럼 (location_embed)도 리스트를 펼쳐서 각 차원별로 컬럼 생성
Process_Name_embed_array = np.vstack(filterd_df['Process_Name_embed'].values)  # 'embedding' 컬럼을 2D 배열로 변환
pca = PCA(n_components=40,random_state=1004)
Process_Name_embed_result = pca.fit_transform(Process_Name_embed_array)
Process_Name_embed_df = pd.DataFrame(Process_Name_embed_result, columns=[f'Name_embed_{i+1}' for i in range(Process_Name_embed_result.shape[1])])



# umap_reducer = umap.UMAP(n_components=40, random_state=42)  # 2D로 축소
# Process_Name_embed_result = umap_reducer.fit_transform(Process_Name_embed_array)
# Process_Name_embed_df = pd.DataFrame(Process_Name_embed_result, columns=[f'Name_embed_{i+1}' for i in range(Process_Name_embed_result.shape[1])])



# 5. 원래 DataFrame에 PCA 결과 및 location_embedding 결과를 모두 붙이기
filterd_df = pd.concat([filterd_df.reset_index(drop=True), pca_df, location_embed_df, Process_Name_embed_df], axis=1)


# 시각화
# embedding_visualisation(Process_Name_embed_array)



# 'Amount' 컬럼의 데이터 타입을 수치형으로 변환
filterd_df['Amount'] = pd.to_numeric(filterd_df['Amount'], errors='coerce')

# 'Amount' 컬럼의 통계 정보 확인
print(filterd_df['Amount'].describe())







train_data = filterd_df.drop(['Process ID', 'Process Name', 'Product Name', 'Description', 'Location', 'Category ID', 'Category Name','Flow_name', 'InChIKey', 'embedding', 'Description_embedding', 'Process_Name_embed'],axis=1)

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
train_data = train_data.reset_index(drop=True)
maccs_added_df = pd.concat([train_data, maccs_df], axis=1)
print(maccs_added_df['Amount'].describe())


'''특정 컬럼 제거'''
# # 'process_embed'가 포함된 모든 컬럼명 가져오기
# cols_to_drop = [col for col in maccs_added_df.columns if 'process_embed' in col]
# maccs_added_df = maccs_added_df.drop(columns=cols_to_drop)

# 'process_embed'가 포함된 모든 컬럼명 가져오기
# cols_to_drop = [col for col in maccs_added_df.columns if 'location_embed_' in col]
# maccs_added_df = maccs_added_df.drop(columns=cols_to_drop)

# # 'process_embed'가 포함된 모든 컬럼명 가져오기
# cols_to_drop = [col for col in maccs_added_df.columns if 'Name_embed_' in col]
# maccs_added_df = maccs_added_df.drop(columns=cols_to_drop)

# cols_to_drop = [col for col in maccs_added_df.columns if 'MACCS_' in col]
# maccs_added_df = maccs_added_df.drop(columns=cols_to_drop)

# Feature 그룹 정의
feature_groups = {
    'L': [col for col in maccs_added_df.columns if 'location_embed_' in col],
    'T': [col for col in maccs_added_df.columns if 'process_embed' in col],
    'D': [col for col in maccs_added_df.columns if 'Name_embed_' in col],
    'FP': [col for col in maccs_added_df.columns if 'MACCS_' in col]
}


# 모든 조합 생성
feature_combinations = []
for i in range(1, len(feature_groups) + 1):
    feature_combinations.extend(combinations(feature_groups.keys(), i))

# 각 조합에 따라 컬럼 제거 및 데이터프레임 생성
# for combination in feature_combinations:
('D', 'L')
('D', 'FP')
combination = ('FP')
cols_to_drop = []
for key in feature_groups.keys():
    if key not in combination:
        cols_to_drop.extend(feature_groups[key])

# 학습용 데이터프레임 생성
modified_df = maccs_added_df.drop(columns=cols_to_drop)

# 조합 이름 생성
combination_name = '+'.join(combination)
print(f"Feature Set: {combination_name}")
print(modified_df)
# 모델 학습 코드 연결 가능 (e.g., train_model(modified_df))

    
'''동일한 컬럼제거'''
print(f'feature number: {len(modified_df.columns)}', )

# 모든 행이 같은 값을 가지는 컬럼을 찾기
constant_columns = [col for col in modified_df.columns if modified_df[col].nunique() == 1]

# 동일한 값인 컬럼들 출력
print("모든 행이 동일한 값을 가지는 컬럼:", constant_columns)

# 동일한 값인 컬럼 제거
modified_df = modified_df.drop(columns=constant_columns)

'''간단한 데이터 분석'''
print(f'Clearned feature number: {len(modified_df.columns)}', )



X = modified_df.drop(['CanonicalSMILES','Amount'],axis=1)

# umap_reducer = umap.UMAP(n_components=3, random_state=1004)  # 2D로 축소
# X_reduced = umap_reducer.fit_transform(X)
# Load the saved boolean mask
# loaded_mask = np.load('selected_features_mask.npy')

# Assuming 'df' is your data with all features
# Apply the loaded mask to select features
# X = X.values[:, loaded_mask]

y = modified_df['Amount']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1004)

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1004)




# print("Num GPUs Available: "1, len(tf.config.list_physical_devices('GPU')))

# 모델 생성 함수 정의
def objective(trial):
    # Suggest hyperparameters for optimization
    units_layer1 = trial.suggest_int('units_layer1', 32, 256)
    units_layer2 = trial.suggest_int('units_layer2', 32, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.3)
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    batch_size = trial.suggest_int('batch_size', 16, 128)
    epochs = trial.suggest_int('epochs', 500, 1000)

    # Build the model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1],)))
    
    # First hidden layer
    model.add(Dense(units=units_layer1, activation='relu'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    
    # Second hidden layer
    model.add(Dense(units=units_layer2, activation='relu'))
    if batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(rate=dropout_rate))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Predict and calculate validation loss
    val_loss = history.history['val_loss'][-1]
    
    return val_loss  # Minimize validation loss

# Initialize the start time for tracking iterations
global iteration_start_time
iteration_start_time = time.time()

# Run Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

# Get the best hyperparameters found
best_params = study.best_params
print("Best Hyperparameters found: ", best_params)

# Train the final model using the best parameters
final_model = Sequential()
final_model.add(Input(shape=(X_train.shape[1],)))
final_model.add(Dense(units=best_params['units_layer1'], activation='relu'))
if best_params['batch_norm']:
    final_model.add(BatchNormalization())
final_model.add(Dropout(rate=best_params['dropout_rate']))

final_model.add(Dense(units=best_params['units_layer2'], activation='relu'))
if best_params['batch_norm']: 
    final_model.add(BatchNormalization())
final_model.add(Dropout(rate=best_params['dropout_rate']))
 
final_model.add(Dense(1))
optimizer = Adam(learning_rate=best_params['learning_rate'])
final_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the final model
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = final_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=best_params['batch_size'],
    epochs=best_params['epochs'],
    callbacks=[early_stopping],
    verbose=0
)

# Evaluate the final model
y_pred = final_model.predict(X_test)
y_train_predict = final_model.predict(X_train)

# R-squared and MSE calculations
r2_test = r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
r2_train = r2_score(y_train, y_train_predict)
mse_train = mean_squared_error(y_train, y_train_predict)

print(f"R-squared (test): {r2_test}")
print(f"Mean Squared Error (test): {mse_test}")
print(f"R-squared (train): {r2_train}")
print(f"Mean Squared Error (train): {mse_train}")

# Visualization: Actual vs Predicted
plt.figure(figsize=(12, 6))

# Train set visualization
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_predict, color='blue', alpha=0.6)
plt.plot([min(y_train), max(y_train)], 
         [min(y_train), max(y_train)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Values (Train)')
plt.ylabel('Predicted Values (Train)')
plt.title(f'Train Set: Actual vs Predicted (R²: {r2_train:.2f})')
plt.grid(True)

# Test set visualization
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, color='orange', alpha=0.6)
plt.plot([min(y_test), max(y_test)], 
         [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel('Actual Values (Test)')
plt.ylabel('Predicted Values (Test)')
plt.title(f'Test Set: Actual vs Predicted (R²: {r2_test:.2f})')
plt.grid(True)

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))

# 훈련 데이터 플롯
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_predict, alpha=0.7, edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.title(f'Train Data\n$R^2$: {r2_train:.4f}, MSE: {mse_train:.4f}')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

# 테스트 데이터 플롯
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'{combination_name}, Test Data\n$R^2$: {r2_test:.4f}, MSE: {mse_test:.4f} ')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)

plt.tight_layout()
plt.show()


