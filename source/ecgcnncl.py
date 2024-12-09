import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import logging
from datetime import datetime

from logger_config import setup_logger

logger = setup_logger()

class ECGCNNCL(nn.Module):
    def __init__(self, window_size):
        super(ECGCNNCL, self).__init__()
     
        # Convolutional Layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * (window_size // (2**4)), 256)  # Adjust based on pooling
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)

        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Convolutional Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Convolutional Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Convolutional Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Convolutional Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Block 1
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Fully Connected Block 2
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Output Layer
        x = self.sigmoid(self.fc3(x))

        return x

# Train Loop 정의
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for inputs, labels in train_loader:

            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.size(0)

        # Validation Loss 계산
        train_loss = total_train_loss / len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, criterion, device)

        # 결과 출력
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # 모델 중간 저장
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

# Train 및 Validation Loss 계산 함수
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

# 모델 로드 및 테스트 데이터셋 평가
def load_and_evaluate(model, checkpoint_path, epoch_number, test_loader, criterion, device):
    # 에포크 모델 로드
    checkpoint_file = os.path.join(checkpoint_path, f"model_epoch_{epoch_number}.pth")
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()  # 모델을 평가 모드로 설정

    # 테스트 데이터 평가
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss using model from {epoch_number}: {test_loss:.4f}")


def test_and_evaluate(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Ground truth and predictions
            y_true.extend((labels.cpu().numpy() >= threshold).astype(int))
            y_pred.extend((outputs.cpu().numpy() >= threshold).astype(int))  # Apply threshold for binary classification

    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    # Print results
    print("Confusion Matrix:\n")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    return cm, accuracy, f1, recall, precision