"""
KAN 모델 구조 최적화 (베이지안 최적화 기반)
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import optuna
from kan import MultKAN as KAN
import matplotlib.pyplot as plt

# Apply numerical stability patch for spline calculations
exec(open('spline_fix.py').read())

# Keras-style EarlyStopping for KAN models
class EarlyStopping:
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, 
                 mode='min', baseline=None, restore_best_weights=True, start_from_epoch=0):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        self.start_from_epoch = start_from_epoch
        
        # Internal state
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None, model=None):
        current = logs.get(self.monitor)
        if current is None:
            return False
            
        if epoch < self.start_from_epoch:
            return False
            
        if self.baseline is not None:
            if self.mode == 'min' and current >= self.baseline:
                return False
            elif self.mode == 'max' and current <= self.baseline:
                return False
        
        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
                self.best_epoch = epoch
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: early stopping")
            return True
        return False
        
    def _is_improvement(self, current, best):
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def restore_best_weights_to_model(self, model):
        if self.best_weights is not None and self.restore_best_weights:
            model.load_state_dict(self.best_weights)
            if self.verbose > 0:
                print(f"Restoring model weights from epoch {self.best_epoch + 1}")

# 데이터 로드 및 전처리
X_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_train_selected.pkl')
X_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_X_test_selected.pkl')
y_train = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_train.pkl')
y_val = pd.read_pickle('/Users/k23070952/Desktop/LCA/1. Code/241118_y_test.pkl')

X_train = X_train.fillna(0).infer_objects(copy=False).select_dtypes(include=[np.number])
X_val = X_val.fillna(0).infer_objects(copy=False).select_dtypes(include=[np.number])

X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

# 최종 테스트 셋 분리 (전체 데이터의 10%)
from sklearn.model_selection import train_test_split
X_train_cv, X_test_final, y_train_cv, y_test_final = train_test_split(
    X_combined, y_combined, test_size=0.1, random_state=42, stratify=None
)

# 데이터 전처리 (CV용 데이터만)
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
    optimiser = trial.suggest_categorical('optimiser', ['Adam', 'LBFGS', 'SGD'])
    grid_size = trial.suggest_int('grid_size', 5, 300)
    # Early stopping parameters
    max_epochs = 500  # Reduced for testing
    patience = 20  # Reduced for testing
    min_delta = 1e-6
    k = 3
    # lamb = trial.suggest_float('lamb', 0.001, 0.1, log=True)
    seed = 1004

    device = torch.device('cpu')  # Force CPU to avoid MPS issues
    r2_scores = []
    for train_idx, test_idx in kf.split(X_train_cv):
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
        # width: [input_dim, n_units, ..., n_units, 1] (n_layers hidden)
        width = [X_train_tensor.shape[1]] + [n_units]*n_layers + [1]
        model = KAN(width=width, grid=grid_size, k=k, seed=seed).to(device)
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=min_delta,
            patience=patience,
            verbose=1,
            mode='min',
            restore_best_weights=True
        )
        
        # 효율적인 배치 학습 with early stopping
        check_every = 10  # 10 스텝마다 체크 (1 스텝보다 10배 빠름)
        
        for epoch in range(0, max_epochs, check_every):
            # 한 번에 여러 스텝 학습
            steps_to_run = min(check_every, max_epochs - epoch)
            results = model.fit(dataset, opt=optimiser, lr=learning_rate, steps=steps_to_run, log=steps_to_run)
            
            # Get validation loss (마지막 스텝의 loss)
            val_loss = results['test_loss'][-1]
            
            # Check early stopping
            logs = {'val_loss': val_loss}
            if early_stopping.on_epoch_end(epoch + steps_to_run - 1, logs, model):
                break
        
        # Restore best weights
        early_stopping.restore_best_weights_to_model(model)
        with torch.no_grad():
            y_pred = model(dataset['test_input']).cpu().numpy()
            y_true = dataset['test_label'].cpu().numpy()
            y_pred = np.nan_to_num(y_pred, nan=0.0)
            r2 = r2_score(y_true, y_pred)
            r2_scores.append(r2)
    mean_r2 = np.mean(r2_scores)
    return mean_r2

# Optuna 스터디 생성 및 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3)  # Test with 3 trials

print('Best trial:')
best_trial = study.best_trial
for key, value in best_trial.params.items():
    print(f'{key}: {value}')
print(f'Best mean R2: {best_trial.value}')

# Optuna 시각화
import optuna.visualization as vis
print("\nGenerating Optuna visualizations...")

# 1. Optimization history
fig1 = vis.plot_optimization_history(study)
fig1.write_html("optuna_optimization_history.html")
fig1.show()

# 2. Parameter importance
try:
    fig2 = vis.plot_param_importances(study)
    fig2.write_html("optuna_param_importances.html")
    fig2.show()
except:
    print("Parameter importance plot requires more trials")

# 3. Parallel coordinate plot
fig3 = vis.plot_parallel_coordinate(study)
fig3.write_html("optuna_parallel_coordinate.html")
fig3.show()

# 4. Slice plot
fig4 = vis.plot_slice(study)
fig4.write_html("optuna_slice.html")
fig4.show()

print("Optuna visualization files saved:")
print("- optuna_optimization_history.html")
print("- optuna_param_importances.html") 
print("- optuna_parallel_coordinate.html")
print("- optuna_slice.html")

# 최적 모델로 전체 학습 및 결과 시각화
best_params = best_trial.params

n_layers = best_params['n_layers']
n_units = best_params['n_units']
learning_rate = best_params['learning_rate']
optimiser = best_params['optimiser']
grid_size = best_params['grid_size']
# Early stopping parameters for final training
max_epochs = 500  # Reduced for testing
patience = 30  # Reduced for testing
min_delta = 1e-6
k = 3
seed = 1004

device = torch.device('cpu')  # Force CPU to avoid MPS issues
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

width = [X_train_tensor.shape[1]] + [n_units]*n_layers + [1]
model = KAN(width=width, grid=grid_size, k=k, seed=seed).to(device)

# Setup early stopping for final training
early_stopping_final = EarlyStopping(
    monitor='train_loss',
    min_delta=min_delta,
    patience=patience,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

# 효율적인 배치 학습 for final training
check_every = 20  # 최종 학습에서는 더 큰 배치 사용

for epoch in range(0, max_epochs, check_every):
    # 한 번에 여러 스텝 학습
    steps_to_run = min(check_every, max_epochs - epoch)
    results = model.fit(dataset, opt=optimiser, lr=learning_rate, steps=steps_to_run, log=steps_to_run)
    
    # Get training loss
    train_loss = results['train_loss'][-1]
    
    # Progress logging
    print(f"Epoch {epoch + steps_to_run}, Train Loss: {train_loss:.6f}")
    
    # Check early stopping
    logs = {'train_loss': train_loss}
    if early_stopping_final.on_epoch_end(epoch + steps_to_run - 1, logs, model):
        break

# Restore best weights
early_stopping_final.restore_best_weights_to_model(model)
with torch.no_grad():
    y_pred = model(dataset['test_input']).cpu().numpy()
    y_true = dataset['test_label'].cpu().numpy()
    y_pred = np.nan_to_num(y_pred, nan=0.0)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f'\nFinal Test Set Performance:')
    print(f'R2: {r2:.4f}, MSE: {mse:.4f}')
    print(f'Test set size: {len(y_true)} samples')
    print(f'CV set size: {len(X_train_cv)} samples')

plt.figure(figsize=(12,5))

# Plot 1: Prediction vs Actual
plt.subplot(1, 2, 1)
plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', c='r')
plt.xlabel('Actual GWP')
plt.ylabel('Predicted GWP')
plt.title(f'Final Test Set Results\nR2: {r2:.4f}, MSE: {mse:.4f}')
plt.grid(True, alpha=0.3)

# Plot 2: Data split visualization
plt.subplot(1, 2, 2)
sizes = [len(X_train_cv), len(X_test_final)]
labels = ['CV Data (90%)', 'Test Data (10%)']
colors = ['lightgreen', 'lightblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Data Split')

plt.tight_layout()
plt.savefig('final_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nResults saved as 'final_results.png'")
