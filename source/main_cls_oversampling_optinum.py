import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import random_split, DataLoader
import re
from prepare import SlidingWindowECGDataset
from ecgcnncl import ECGCNNCL, train_model, evaluate, load_and_evaluate, test_and_evaluate
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from logger_config import setup_logger


data_directory = "./Shao_Results"
file_pattern = re.compile(r"JS\d{5}_labeled\.txt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "checkpoints"

logger = setup_logger()

# Validation 데이터에서 F1 Score 계산 함수
def evaluate_f1(model, data_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (outputs >= threshold).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='weighted')  # F1 Score 계산

# Bayesian Optimization을 위한 Objective Function 정의
def objective(trial):
    # 데이터 준비


    # Bayesian Optimization을 통해 최적화할 하이퍼파라미터
    window_size = trial.suggest_int("window_size", 500, 1000, step=100)  # 예: 50 ~ 500
    minorclassoverlap = trial.suggest_float("minorclassoverlap", 0.5, 0.8, step=0.1)
    
    # 데이터셋 초기화
    dataset = SlidingWindowECGDataset(data_directory, file_pattern, window_size, overlap=0.1, minorclassoverlap=minorclassoverlap)
    dataset.labels = (dataset.labels < 0.5).astype(int)  # 0.5 미만을 비정상(1), 정상(0)으로 설정
    
    labels = [dataset[i][1] for i in range(len(dataset))]  # 레이블 추출

    # Train-Test split with stratification
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        range(len(dataset)), labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Validation-Test split with stratification
    val_indices, test_indices, val_labels, test_labels = train_test_split(
        temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )

    # Subset 생성
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGCNNCL(window_size=window_size).to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    
    os.makedirs(save_path, exist_ok=True)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, save_path=save_path)

    # Validation F1 Score 계산
    val_f1 = evaluate_f1(model, val_loader, device)
    # 테스트 데이터 평가
    cm, accuracy, f1, recall, precision = test_and_evaluate(model, test_loader, device, threshold=0.5)
    print(f"val_f1 = {val_f1}, confusion matrix: \n{cm}, Test Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")
    logger.info(f"val_f1 = {val_f1}, confusion matrix: {cm}, Test Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")

    return val_f1  # 최적화할 메트릭 (F1 Score)

# Optuna Study 실행
study = optuna.create_study(direction="maximize")  # F1 Score를 최대화
study.optimize(objective, n_trials=10)  # 10번의 시도

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:")
print(study.best_params)
logger.info(f"Best Hyperparameters:{study.best_params}")
             
# 최적화된 하이퍼파라미터로 최종 테스트
best_params = study.best_params
best_window_size = best_params["window_size"]
best_minorclassoverlap = best_params["minorclassoverlap"]

# 데이터셋 재생성 및 학습
dataset = SlidingWindowECGDataset(data_directory, file_pattern, best_window_size, overlap=0.1, minorclassoverlap=best_minorclassoverlap)
dataset.labels = (dataset.labels < 0.5).astype(int)

labels = [dataset[i][1] for i in range(len(dataset))]
train_indices, temp_indices, train_labels, temp_labels = train_test_split(
    range(len(dataset)), labels, test_size=0.2, stratify=labels, random_state=42
)
val_indices, test_indices, val_labels, test_labels = train_test_split(
    temp_indices, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ECGCNNCL(window_size=best_window_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device, save_path=save_path)

# 테스트 데이터 평가
cm, accuracy, f1, recall, precision = test_and_evaluate(model, test_loader, device, threshold=0.5)
print(f"Test Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")
logger.info(f"Test Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")