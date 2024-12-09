# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
# import numpy as np
# import logging
# from datetime import datetime

# from logger_config import setup_logger

# logger = setup_logger()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import time

# class InceptionModule(nn.Module):
#     def __init__(self, in_channels, nb_filters, bottleneck_size=32, kernel_sizes=(41, 21, 11), use_bottleneck=True):
#         super(InceptionModule, self).__init__()
#         self.use_bottleneck = use_bottleneck

#         # Bottleneck layer
#         print(f"27:in_channels = {in_channels},{bottleneck_size}")
#         if use_bottleneck and in_channels > 1:
#             print(f"29:in_channels = {in_channels},{bottleneck_size}")
#             self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, padding=0, bias=False) 
#         else :
#             self.bottleneck = None
#         print(f"33:self.buttleneck={self.bottleneck}")
#         # Parallel Convolutional Layers
#         print(f"module: in_channels={in_channels}")
#         self.conv1 = nn.Conv1d(bottleneck_size if self.bottleneck else in_channels, nb_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, bias=False)
#         self.conv2 = nn.Conv1d(bottleneck_size if self.bottleneck else in_channels, nb_filters, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2, bias=False)
#         self.conv3 = nn.Conv1d(bottleneck_size if self.bottleneck else in_channels, nb_filters, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2, bias=False)

#         # MaxPooling + Conv1D Path
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
#         print(f"42:inchannels = {in_channels}")
#         self.conv4 = nn.Conv1d(in_channels, nb_filters, kernel_size=1, padding=0, bias=False)

#         # Batch Normalization
#         self.bn = nn.BatchNorm1d(nb_filters * 4)

#         # Activation
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         if self.bottleneck:
#             x = self.bottleneck(x)

#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(x)
#         conv3_out = self.conv3(x)
#         conv4_out = self.conv4(self.maxpool(x))
#         print(f"53:{conv1_out.shape}, {conv2_out.shape}, {conv3_out.shape}, {conv4_out.shape}")
#         timestep = conv1_out.shape[2]
#         out = torch.cat([conv1_out, conv2_out[:,:,:timestep], conv3_out[:,:,:timestep], conv4_out[:,:,:timestep]], dim=1)
#         print(f"out.shape = {out.shape}")
#         out = self.bn(out)
#         out = self.relu(out)
#         return out


# class InceptionResNet(nn.Module):
#     def __init__(self, input_shape, nb_classes, depth=6, nb_filters=32, kernel_size=41, use_residual=True, use_bottleneck=True):
#         super(InceptionResNet, self).__init__()
#         self.depth = depth
#         self.use_residual = use_residual
#         self.kernel_size = kernel_size
#         self.nb_filters = nb_filters
#         self.use_bottleneck = use_bottleneck

#         self.inception_modules = nn.ModuleList()
#         self.shortcut_layers = nn.ModuleList()

#         in_channels = input_shape #[0]

#         for d in range(depth):
#             module = InceptionModule(in_channels, nb_filters, bottleneck_size=32, kernel_sizes=(kernel_size, kernel_size // 2, kernel_size // 4), use_bottleneck=use_bottleneck)
#             self.inception_modules.append(module)
#             if use_residual and d % 3 == 2:
#                 shortcut = nn.Sequential(
#                     nn.Conv1d(in_channels, nb_filters * 4, kernel_size=1, padding=0, bias=False),
#                     nn.BatchNorm1d(nb_filters * 4)
#                 )
#                 self.shortcut_layers.append(shortcut)
#             in_channels = nb_filters * 4  # Update for next layer

#         self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Linear(nb_filters * 4, nb_classes)

#     def forward(self, x):
#         shortcut = x

#         for d, module in enumerate(self.inception_modules):
#             print(f"{d}:forward:x shape = {x.shape}" )
#             x = module(x)
#             print(f"{d}: forward: x shape after module {x.shape} ")
#             if self.use_residual and d % 3 == 2:
#                 x = x + self.shortcut_layers[d // 3](shortcut)
#                 shortcut = x

#         x = self.global_avg_pool(x).squeeze(-1)  # [batch, channels, 1] -> [batch, channels]
#         x = self.fc(x)
#         return x

# # Train Loop 정의
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
#     for epoch in range(num_epochs):
#         model.train()
#         total_train_loss = 0.0

#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
#             print(f"Input shape: {inputs.shape}")  # torch.Size([batch_size, input_channels, window_size])
#             # Forward pass
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_train_loss += loss.item() * inputs.size(0)

#         # Validation Loss 계산
#         train_loss = total_train_loss / len(train_loader.dataset)
#         val_loss = evaluate(model, val_loader, criterion, device)

#         # 결과 출력
#         logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

#         # 모델 중간 저장
#         torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))


# # Train 및 Validation Loss 계산 함수
# def evaluate(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in dataloader:
#             inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item() * inputs.size(0)
#     return total_loss / len(dataloader.dataset)


# # 모델 로드 및 테스트 데이터셋 평가
# def test_and_evaluate(model, test_loader, device, threshold=0.5):
#     model.eval()
#     y_true = []
#     y_pred = []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)

#             # Ground truth and predictions
#             y_true.extend((labels.cpu().numpy() >= threshold).astype(int))
#             y_pred.extend((outputs.cpu().numpy() >= threshold).astype(int))

#     # Calculate metrics
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # Confusion matrix and metrics
#     cm = confusion_matrix(y_true, y_pred)
#     accuracy = accuracy_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)

#     logger.info(f"Confusion Matrix:\n{cm}")
#     logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
#     return cm, accuracy, f1, recall, precision
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
import numpy as np
from logger_config import setup_logger

logger = setup_logger()

class InceptionModule(nn.Module):
    def __init__(self, in_channels, nb_filters, bottleneck_size=32, kernel_sizes=(41, 21, 11), use_bottleneck=True):
        super(InceptionModule, self).__init__()
        self.use_bottleneck = use_bottleneck
        
        # Determine input size for convolution layers
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, padding=0, bias=False)
            self.conv_input_size = bottleneck_size
        else:
            self.bottleneck = None
            self.conv_input_size = in_channels
            
        # Parallel Convolutional Layers
        self.conv1 = nn.Conv1d(self.conv_input_size, nb_filters, kernel_size=kernel_sizes[0], 
                              padding=kernel_sizes[0] // 2, bias=False)
        self.conv2 = nn.Conv1d(self.conv_input_size, nb_filters, kernel_size=kernel_sizes[1], 
                              padding=kernel_sizes[1] // 2, bias=False)
        self.conv3 = nn.Conv1d(self.conv_input_size, nb_filters, kernel_size=kernel_sizes[2], 
                              padding=kernel_sizes[2] // 2, bias=False)

        # MaxPooling + Conv1D Path
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(self.conv_input_size, nb_filters, kernel_size=1, padding=0, bias=False)

        # Batch Normalization and ReLU
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.bottleneck:
            x = self.bottleneck(x)
        
        # Process through parallel paths
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        conv4_out = self.conv4(self.maxpool(x))

        # Ensure all outputs have the same timesteps
        timestep = conv1_out.shape[2]
        out = torch.cat([
            conv1_out, 
            conv2_out[:,:,:timestep], 
            conv3_out[:,:,:timestep], 
            conv4_out[:,:,:timestep]
        ], dim=1)
        
        out = self.bn(out)
        out = self.relu(out)
        return out

class InceptionResNet(nn.Module):
    def __init__(self, input_shape, nb_classes, depth=6, nb_filters=32, kernel_size=41, use_residual=True, use_bottleneck=True):
        super(InceptionResNet, self).__init__()
        self.depth = depth
        self.use_residual = use_residual
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.use_bottleneck = use_bottleneck

        # Initialize module lists
        self.inception_modules = nn.ModuleList()
        self.shortcut_layers = nn.ModuleList()

        # Initial convolution to match channel dimensions
        self.initial_conv = nn.Conv1d(input_shape, nb_filters * 4, kernel_size=1, padding=0, bias=False)
        current_channels = nb_filters * 4

        # Create inception modules and shortcut layers
        for d in range(depth):
            module = InceptionModule(
                in_channels=current_channels,
                nb_filters=nb_filters,
                kernel_sizes=(kernel_size, kernel_size // 2, kernel_size // 4),
                use_bottleneck=use_bottleneck
            )
            self.inception_modules.append(module)
            
            if use_residual and d % 3 == 2:
                shortcut = nn.Sequential(
                    nn.Conv1d(current_channels, nb_filters * 4, kernel_size=1, padding=0, bias=False),
                    nn.BatchNorm1d(nb_filters * 4)
                )
                self.shortcut_layers.append(shortcut)

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(nb_filters * 4, nb_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial convolution to match channels
        x = self.initial_conv(x)
        shortcut = x
        shortcut_count = 0

        for d, module in enumerate(self.inception_modules):
            x = module(x)
            
            if self.use_residual and d % 3 == 2:
                x = x + self.shortcut_layers[shortcut_count](shortcut)
                shortcut = x
                shortcut_count += 1

        x = self.global_avg_pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        batch_count = 0
        
        for inputs, labels in train_loader:
            if batch_count % 10 == 0:  # Log every 10 batches
                logger.info(f"Epoch {epoch+1}, Batch {batch_count}")
            batch_count += 1
            
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, os.path.join(save_path, f'checkpoint_epoch_{epoch+1}.pth'))

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(dataloader.dataset)

def test_and_evaluate(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.squeeze() >= threshold).float().cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    return cm, accuracy, f1, recall, precision