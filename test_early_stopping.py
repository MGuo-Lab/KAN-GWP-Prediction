"""
Test script for EarlyStopping with KAN models
"""
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
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

# Create simple synthetic data for testing
np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
# Simple linear relationship with some noise
y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
device = torch.device('cpu')
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

dataset = {
    'train_input': X_train_tensor,
    'test_input': X_test_tensor,
    'train_label': y_train_tensor,
    'test_label': y_test_tensor,
}

# Create simple KAN model
width = [n_features, 16, 1]  # Simple architecture
model = KAN(width=width, grid=5, k=3, seed=42).to(device)

# Test early stopping
print("Testing EarlyStopping with KAN model...")
print(f"Data shape: X_train={X_train.shape}, X_test={X_test.shape}")

# Setup early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-6,
    patience=10,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

max_epochs = 100
train_losses = []
val_losses = []

for epoch in range(max_epochs):
    # Train for one step
    results = model.fit(dataset, opt='Adam', lr=0.01, steps=1, log=10)
    
    # Get validation loss
    val_loss = results['test_loss'][-1]
    train_loss = results['train_loss'][-1]
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Check early stopping
    logs = {'val_loss': val_loss}
    if early_stopping.on_epoch_end(epoch, logs, model):
        break

# Restore best weights
early_stopping.restore_best_weights_to_model(model)

# Final evaluation
with torch.no_grad():
    y_pred = model(dataset['test_input']).cpu().numpy()
    y_true = dataset['test_label'].cpu().numpy()
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

print(f"\nFinal Results:")
print(f"R2 Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Total epochs trained: {epoch+1}")
print(f"Best epoch: {early_stopping.best_epoch+1}")
print(f"Best validation loss: {early_stopping.best:.6f}")

# Plot training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.axvline(x=early_stopping.best_epoch, color='r', linestyle='--', label=f'Best Epoch ({early_stopping.best_epoch+1})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Curves')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_true, y_pred, alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', c='r')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'Predictions (R² = {r2:.4f})')
plt.grid(True)

plt.tight_layout()
plt.savefig('early_stopping_test.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nEarly stopping test completed successfully!")
print("Training curves saved as 'early_stopping_test.png'") 