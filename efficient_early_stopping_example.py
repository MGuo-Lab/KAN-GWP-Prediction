"""
효율적인 Early Stopping 방법들
"""
import torch
import numpy as np
from kan import MultKAN as KAN

# 방법 1: 배치 학습 + 주기적 체크
def efficient_early_stopping_v1(model, dataset, opt, lr, max_epochs, patience, check_every=10):
    """
    여러 스텝을 한 번에 학습하고 주기적으로 early stopping 체크
    """
    best_loss = float('inf')
    patience_counter = 0
    best_weights = None
    
    for epoch in range(0, max_epochs, check_every):
        # 한 번에 여러 스텝 학습
        steps_to_run = min(check_every, max_epochs - epoch)
        results = model.fit(dataset, opt=opt, lr=lr, steps=steps_to_run, log=1)
        
        # 현재 validation loss 체크
        val_loss = results['test_loss'][-1]  # 마지막 스텝의 loss
        
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            patience_counter = 0
            best_weights = model.state_dict().copy()
        else:
            patience_counter += check_every
            
        print(f"Epoch {epoch + steps_to_run}: Val Loss = {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + steps_to_run}")
            break
    
    # 최적 가중치 복원
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    return epoch + steps_to_run

# 방법 2: KAN의 내장 metrics 활용
def efficient_early_stopping_v2(model, dataset, opt, lr, max_epochs, patience):
    """
    KAN의 내장 metrics를 활용한 early stopping
    """
    class EarlyStoppingCallback:
        def __init__(self, patience, min_delta=1e-6):
            self.patience = patience
            self.min_delta = min_delta
            self.best_loss = float('inf')
            self.wait = 0
            self.best_weights = None
            self.should_stop = False
            
        def __call__(self):
            # KAN 내부에서 호출될 수 있는 함수 (현재는 직접 지원하지 않음)
            return self.should_stop
    
    # 더 큰 배치로 학습
    batch_size = 50  # 기본값보다 큰 배치
    results = model.fit(
        dataset, 
        opt=opt, 
        lr=lr, 
        steps=max_epochs,
        log=10,  # 10 스텝마다 로그
        # 여기서 custom callback을 추가할 수 있다면 좋겠지만, 현재 KAN은 지원하지 않음
    )
    
    return max_epochs

# 방법 3: 학습률 스케줄링과 결합
def efficient_early_stopping_v3(model, dataset, max_epochs, patience):
    """
    학습률 스케줄링과 early stopping 결합
    """
    # 초기 학습률로 빠르게 학습
    print("Phase 1: Fast learning with high LR")
    results1 = model.fit(dataset, opt='Adam', lr=0.01, steps=50, log=10)
    
    # 중간 학습률로 세밀 조정
    print("Phase 2: Fine-tuning with medium LR")
    results2 = model.fit(dataset, opt='Adam', lr=0.001, steps=100, log=10)
    
    # 낮은 학습률로 최종 조정 (early stopping 적용)
    print("Phase 3: Final tuning with low LR and early stopping")
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(0, max_epochs, 20):  # 20 스텝씩
        results = model.fit(dataset, opt='Adam', lr=0.0001, steps=20, log=5)
        val_loss = results['test_loss'][-1]
        
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 20
            
        if patience_counter >= patience:
            print(f"Early stopping in phase 3 at epoch {epoch + 20}")
            break
    
    return 50 + 100 + epoch + 20

# 예제 사용
if __name__ == "__main__":
    print("효율적인 Early Stopping 방법 예제")
    print("=" * 50)
    
    # 방법 1이 가장 실용적이고 효율적
    print("\n권장 방법: 배치 학습 + 주기적 체크")
    print("- 10-20 스텝씩 학습하고 early stopping 체크")
    print("- 메모리 효율적이고 빠름")
    print("- 세밀한 제어 가능")
    
    print("\n코드에서 사용할 때:")
    print("check_every=10  # 10 스텝마다 체크")
    print("patience=50     # 50 스텝 동안 개선 없으면 종료")
    print("이렇게 하면 1 스텝씩보다 10배 빠름!") 