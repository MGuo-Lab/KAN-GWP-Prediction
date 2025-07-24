"""
KAN 모델 구조 최적화 (간단한 버전 - Early stopping 없이)
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from kan import MultKAN as KAN
import matplotlib.pyplot as plt

# Apply numerical stability patch for spline calculations
exec(open('spline_fix.py').read())

print("KAN 모델 최적화 (간단한 버전)")
print("=" * 50)

# 데이터 로드 및 전처리
print("데이터 로드 중...")
X_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_train_selected.pkl')
X_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl')
y_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_train.pkl')
y_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_test.pkl')

X_train = X_train.fillna(0).infer_objects(copy=False).select_dtypes(include=[np.number])
X_val = X_val.fillna(0).infer_objects(copy=False).select_dtypes(include=[np.number])

X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

# 최종 테스트 셋 분리 (전체 데이터의 10%)
X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(
    X_combined, y_combined, test_size=0.1, random_state=42, stratify=None
)

print(f"CV 데이터: {len(X_train_cv)} 샘플")
print(f"최종 테스트 데이터: {len(X_test_final)} 샘플")

# 데이터 전처리
shift_value = abs(X_train_cv.min().min())
X_train_cv = X_train_cv.map(lambda x: np.log1p(x + shift_value))
scaler = StandardScaler()
X_train_cv = pd.DataFrame(scaler.fit_transform(X_train_cv), columns=X_train_cv.columns)
y_train_cv = y_train_cv.apply(pd.to_numeric, errors='coerce')

# 최종 테스트 데이터도 같은 방식으로 전처리
X_test_final = X_test_final.map(lambda x: np.log1p(x + shift_value))
X_test_final = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns)
y_test_final = y_test_final.apply(pd.to_numeric, errors='coerce')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    # 하이퍼파라미터 탐색 공간
    n_layers = trial.suggest_int('n_layers', 1, 5)
    n_units = trial.suggest_categorical('n_units', [64, 128, 256, 512])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    optimiser = trial.suggest_categorical('optimiser', ['Adam', 'LBFGS'])
    grid_size = trial.suggest_int('grid_size', 5, 20)  # 메모리 절약
    epochs = trial.suggest_int('epochs', 20, 500)  # 20부터 500까지
    k = 3
    seed = 1004

    device = torch.device('cpu')  # Force CPU to avoid MPS issues
    r2_scores = []
    
    print(f"\nTrial {trial.number + 1}: epochs={epochs}, layers={n_layers}, units={n_units}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_train_cv)):
        print(f"  Fold {fold_idx + 1}/5", end=" ")
        
        X_fold_train, X_fold_test = X_train_cv.iloc[train_idx], X_train_cv.iloc[test_idx]
        y_fold_train, y_fold_test = y_train_cv.iloc[train_idx], y_train_cv.iloc[test_idx]
        
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
        
        # 모델 생성 및 학습
        width = [X_train_tensor.shape[1]] + [n_units]*n_layers + [1]
        model = KAN(width=width, grid=grid_size, k=k, seed=seed).to(device)
        
        # 간단한 한 번의 학습!
        results = model.fit(dataset, opt=optimiser, lr=learning_rate, steps=epochs, log=max(1, epochs//5))
        
        # 평가
        with torch.no_grad():
            y_pred = model(dataset['test_input']).cpu().numpy()
            y_true = dataset['test_label'].cpu().numpy()
            y_pred = np.nan_to_num(y_pred, nan=0.0)
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)
            print(f"R2={r2:.3f}")
    
    mean_r2 = np.mean(r2_scores)
    print(f"  Trial {trial.number + 1} complete - Mean R2: {mean_r2:.4f}")
    return mean_r2

# Optuna 최적화
print("\nOptuna 최적화 시작...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)  # 3번 테스트

print('\n' + '='*50)
print('최적화 결과:')
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'Best mean R2: {best_trial.value:.4f}')

# Optuna 시각화
print("\nOptuna 시각화 생성 중...")
import optuna.visualization as vis

try:
    # 1. Optimization history
    fig1 = vis.plot_optimization_history(study)
    fig1.write_html("optuna_optimization_history.html")
    print("- optuna_optimization_history.html 저장됨")

    # 2. Parallel coordinate plot
    fig2 = vis.plot_parallel_coordinate(study)
    fig2.write_html("optuna_parallel_coordinate.html")
    print("- optuna_parallel_coordinate.html 저장됨")

    # 3. Slice plot
    fig3 = vis.plot_slice(study)
    fig3.write_html("optuna_slice.html")
    print("- optuna_slice.html 저장됨")

except Exception as e:
    print(f"시각화 중 오류: {e}")

# 최적 모델로 최종 학습 및 평가
print("\n최적 모델로 최종 학습...")
best_params = best_trial.params

device = torch.device('cpu')
X_train_tensor = torch.tensor(X_train_cv.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_final.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_cv.values, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test_final.values, dtype=torch.float32).view(-1, 1).to(device)

dataset = {
    'train_input': X_train_tensor,
    'test_input': X_test_tensor,
    'train_label': y_train_tensor,
    'test_label': y_test_tensor,
}

# 최적 파라미터로 모델 생성
width = [X_train_tensor.shape[1]] + [best_params['n_units']]*best_params['n_layers'] + [1]
model = KAN(width=width, grid=best_params['grid_size'], k=3, seed=1004).to(device)

# 최종 학습 (최적 에포크 수 사용)
print(f"최종 학습: {best_params['epochs']} epochs")
results = model.fit(
    dataset, 
    opt=best_params['optimiser'], 
    lr=best_params['learning_rate'], 
    steps=best_params['epochs'], 
    log=max(1, best_params['epochs']//5)
)

# 최종 평가
with torch.no_grad():
    y_pred = model(dataset['test_input']).cpu().numpy()
    y_true = dataset['test_label'].cpu().numpy()
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

print(f'\n최종 테스트 성능:')
print(f'R2: {r2:.4f}')
print(f'MSE: {mse:.4f}')
print(f'테스트 셋 크기: {len(y_true)} 샘플')
print(f'CV 셋 크기: {len(X_train_cv)} 샘플')

# 결과 시각화
plt.figure(figsize=(12,5))

# Plot 1: Prediction vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', c='r')
plt.xlabel('Actual GWP')
plt.ylabel('Predicted GWP')
plt.title(f'최종 테스트 결과\nR2: {r2:.4f}, MSE: {mse:.4f}')
plt.grid(True, alpha=0.3)

# Plot 2: Data split visualization
plt.subplot(1, 2, 2)
sizes = [len(X_train_cv), len(X_test_final)]
labels = ['CV Data (90%)', 'Test Data (10%)']
colors = ['lightgreen', 'lightblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('데이터 분할')

plt.tight_layout()
plt.savefig('final_results_simple.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n결과가 'final_results_simple.png'에 저장되었습니다!")
print("완료!") 