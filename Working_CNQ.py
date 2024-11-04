import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from scipy import signal
import re
import traceback
from torch.utils.data import Dataset, DataLoader
import datetime
import json

class StatisticalFeatures(nn.Module):
    def __init__(self):
        super(StatisticalFeatures, self).__init__()
    
    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        # Calculate statistical features along the sequence dimension (dim=2)
        mean = torch.mean(x, dim=2)  # [batch_size, channels]
        std = torch.std(x, dim=2)    # [batch_size, channels]
        
        # Calculate skewness
        diff = x - mean.unsqueeze(2)  # Expand mean for broadcasting
        skew = torch.mean(torch.pow(diff, 3), dim=2) / (torch.pow(std, 3) + 1e-8)
        
        # Calculate kurtosis
        kurt = torch.mean(torch.pow(diff, 4), dim=2) / (torch.pow(std, 4) + 1e-8)
        
        # Stack features
        stats = torch.cat([mean, std, skew, kurt], dim=1)  # [batch_size, channels * 4]
        return stats

class ConvNetQuake(nn.Module):
    def __init__(self, input_length=None):
        super(ConvNetQuake, self).__init__()
        
        # Initial feature extraction with larger kernels to capture ECG patterns
        self.feature_extractor = nn.Sequential(
            # Initial layer with large kernel to capture ECG waveform patterns
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(2, stride=2),
            
            # Deeper feature extraction
            self._make_conv_block(64, 128, kernel_size=9),
            self._make_conv_block(128, 256, kernel_size=9),
            self._make_conv_block(256, 512, kernel_size=9),
            
            # Global context
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
        # Statistical features processing
        self.auxiliary_features = nn.Sequential(
            StatisticalFeatures(),
            nn.BatchNorm1d(4),  # 4 features: mean, std, skew, kurt
            nn.Linear(4, 16),
            nn.LeakyReLU(0.2)
        )
        
        # Classifier combining both feature types
        self.classifier = nn.Sequential(
            nn.Linear(512 + 16, 256),  # 512 from conv features + 16 from auxiliary
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _make_conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            # First conv with residual connection
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            
            # Second conv
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            
            # Max pooling for downsampling
            nn.MaxPool1d(2, stride=2)
        )
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Main feature extraction path
        conv_features = self.feature_extractor(x)  # [batch_size, 512]
        
        # Statistical features path
        stats = self.auxiliary_features(x)  # [batch_size, 16]
        
        # Combine features
        combined = torch.cat([conv_features, stats], dim=1)  # [batch_size, 512 + 16]
        
        # Classification
        output = self.classifier(combined)
        return output

class ECGDataManager:
    def __init__(self, data_dir, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError("Split proportions must sum to 1")

        self.data_dir = Path(data_dir)
        self.normal_dir = self.data_dir / "normal"
        self.abnormal_dir = self.data_dir / "abnormal"
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        print(f"\nInitializing with directories:")
        print(f"Normal dir: {self.normal_dir}")
        print(f"Abnormal dir: {self.abnormal_dir}")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if not self.normal_dir.exists() or not self.abnormal_dir.exists():
            raise ValueError(f"One or both directories not found: {self.normal_dir}, {self.abnormal_dir}")
        
        # Design bandpass filter
        self.nyquist = 500 / 2  # Assuming 500Hz sampling rate
        self.b, self.a = signal.butter(3, [0.5/self.nyquist, 40/self.nyquist], btype='band')
        
        # Find maximum signal length
        self.max_length = self._find_max_length()
        print(f"Maximum signal length: {self.max_length}")

    def _find_max_length(self):
        """Find the maximum signal length across all files"""
        max_len = 0
        all_files = list(self.normal_dir.glob("S*_labeled.txt")) + list(self.abnormal_dir.glob("S*_labeled.txt"))
        for file in all_files:
            try:
                data = np.loadtxt(file)
                max_len = max(max_len, len(data) - 1)  # -1 for label
            except Exception as e:
                print(f"Error reading {file}: {str(e)}")
        return max_len

    def load_and_preprocess_file(self, filename):
        try:
            filepath = Path(filename)
            data = np.loadtxt(filepath)
            features = data[:-1]  # All but last value
            
            # Apply bandpass filter
            features = signal.filtfilt(self.b, self.a, features)
            
            # Segment normalization
            segment_length = 1000  # ~2 seconds at 500Hz
            num_segments = len(features) // segment_length
            normalized_features = []
            
            for i in range(num_segments):
                segment = features[i*segment_length:(i+1)*segment_length]
                mean = np.mean(segment)
                std = np.std(segment)
                normalized_segment = (segment - mean) / (std + 1e-8)
                normalized_features.extend(normalized_segment)
            
            # Handle remaining samples
            if len(features) % segment_length != 0:
                segment = features[num_segments*segment_length:]
                mean = np.mean(segment)
                std = np.std(segment)
                normalized_segment = (segment - mean) / (std + 1e-8)
                normalized_features.extend(normalized_segment)
            
            features = np.array(normalized_features)
            
            # Pad or truncate to max_length
            if len(features) > self.max_length:
                features = features[:self.max_length]
            elif len(features) < self.max_length:
                pad_length = self.max_length - len(features)
                features = np.pad(features, (0, pad_length), mode='constant', constant_values=0)
            
            # Set label based on directory
            is_normal = filepath.parent.name == "normal"
            label = 1.0 if is_normal else 0.0
            
            # Reshape for model input (1 channel)
            features_reshaped = features.reshape(1, -1)
            
            return torch.tensor(features_reshaped, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            traceback.print_exc()
            raise

    def prepare_data(self):
        """Prepare train, validation, and test datasets"""
        # Get all files
        normal_files = list(self.normal_dir.glob("S*_labeled.txt"))
        abnormal_files = list(self.abnormal_dir.glob("S*_labeled.txt"))
        
        # Print file counts
        print(f"\nFound {len(normal_files)} normal files and {len(abnormal_files)} abnormal files")
        
        # Shuffle files
        np.random.shuffle(normal_files)
        np.random.shuffle(abnormal_files)
        
        # Split normal files
        n_normal = len(normal_files)
        normal_train_idx = int(n_normal * self.train_split)
        normal_val_idx = int(n_normal * (self.train_split + self.val_split))
        
        normal_train = normal_files[:normal_train_idx]
        normal_val = normal_files[normal_train_idx:normal_val_idx]
        normal_test = normal_files[normal_val_idx:]
        
        # Split abnormal files
        n_abnormal = len(abnormal_files)
        abnormal_train_idx = int(n_abnormal * self.train_split)
        abnormal_val_idx = int(n_abnormal * (self.train_split + self.val_split))
        
        abnormal_train = abnormal_files[:abnormal_train_idx]
        abnormal_val = abnormal_files[abnormal_train_idx:abnormal_val_idx]
        abnormal_test = abnormal_files[abnormal_val_idx:]
        
        # Combine and shuffle
        train_files = normal_train + abnormal_train
        val_files = normal_val + abnormal_val
        test_files = normal_test + abnormal_test
        
        np.random.shuffle(train_files)
        np.random.shuffle(val_files)
        np.random.shuffle(test_files)
        
        # Process files
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        print("\nProcessing training files...")
        for file in train_files:
            try:
                features, label = self.load_and_preprocess_file(file)
                X_train.append(features)
                y_train.append(label)
            except Exception as e:
                print(f"Skipping training file {file}: {str(e)}")
                continue
            
        print("Processing validation files...")
        for file in val_files:
            try:
                features, label = self.load_and_preprocess_file(file)
                X_val.append(features)
                y_val.append(label)
            except Exception as e:
                print(f"Skipping validation file {file}: {str(e)}")
                continue
            
        print("Processing test files...")
        for file in test_files:
            try:
                features, label = self.load_and_preprocess_file(file)
                X_test.append(features)
                y_test.append(label)
            except Exception as e:
                print(f"Skipping test file {file}: {str(e)}")
                continue
        
        # Convert to tensors
        X_train = torch.stack(X_train)
        y_train = torch.stack(y_train)
        X_val = torch.stack(X_val)
        y_val = torch.stack(y_val)
        X_test = torch.stack(X_test)
        y_test = torch.stack(y_test)
        
        print("\nDataset shapes:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test

def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    true_positives = ((y_pred == 1) & (y_true == 1)).sum().item()
    false_positives = ((y_pred == 1) & (y_true == 0)).sum().item()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum().item()
    true_negatives = ((y_pred == 0) & (y_true == 0)).sum().item()
    
    epsilon = 1e-7
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }

def train_model(model, train_data, val_data, epochs=100, batch_size=32, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use focal loss and AdamW optimizer
    criterion = FocalLoss(alpha=0.75, gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    best_val_loss = float('inf')
    best_model_state = None
    best_val_metrics = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output.squeeze() > 0.5).float()
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = calculate_metrics(
            torch.tensor(train_targets),
            torch.tensor(train_predictions)
        )

        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output.squeeze(), y_val)
                val_loss += loss.item()
                
                predicted = (output.squeeze() > 0.5).float()
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = calculate_metrics(
            torch.tensor(val_targets),
            torch.tensor(val_predictions)
        )

        # Update learning rate
        scheduler.step()

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Train Metrics: Acc={train_metrics['accuracy']:.1f}%, P={train_metrics['precision']:.1f}%, "
              f"R={train_metrics['recall']:.1f}%, F1={train_metrics['f1']:.1f}%")
        print(f"Val Metrics: Acc={val_metrics['accuracy']:.1f}%, P={val_metrics['precision']:.1f}%, "
              f"R={val_metrics['recall']:.1f}%, F1={val_metrics['f1']:.1f}%")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_loss, best_val_metrics

def evaluate_model(model, test_data, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loader = DataLoader(test_data, batch_size=batch_size)
    criterion = FocalLoss(alpha=0.75, gamma=2)
    
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test, y_test = X_test.to(device), y_test.to(device)
            outputs = model(X_test)
            
            loss = criterion(outputs.squeeze(), y_test)
            test_loss += loss.item()
            
            predicted = (outputs.squeeze() > 0.5).float()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_test.cpu().numpy())
    
    predictions_tensor = torch.tensor(all_predictions)
    targets_tensor = torch.tensor(all_targets)
    
    metrics = calculate_metrics(targets_tensor, predictions_tensor)
    metrics['test_loss'] = test_loss / len(test_loader)
    
    return metrics

def main():
    try:
        # Initialize data manager
        print("Initializing ECG Data Manager...")
        data_manager = ECGDataManager(data_dir="/content")
        
        # Prepare datasets
        print("\nPreparing datasets...")
        X_train, y_train, X_val, y_val, X_test, y_test = data_manager.prepare_data()
        
        # Create datasets
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        # Print dataset information
        print("\nDataset Shapes:")
        print(f"Training: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Validation: {X_val.shape}, Labels: {y_val.shape}")
        print(f"Test: {X_test.shape}, Labels: {y_test.shape}")
        
        print("\nClass Distribution:")
        print(f"Training Set - Normal: {(y_train == 1).sum().item()}, "
              f"Abnormal: {(y_train == 0).sum().item()}")
        print(f"Validation Set - Normal: {(y_val == 1).sum().item()}, "
              f"Abnormal: {(y_val == 0).sum().item()}")
        print(f"Test Set - Normal: {(y_test == 1).sum().item()}, "
              f"Abnormal: {(y_test == 0).sum().item()}")

        # Initialize model
        print("\nInitializing model...")
        input_length = X_train.shape[2]
        model = ConvNetQuake(input_length=input_length)
        
        # Print model summary
        print("\nModel Architecture:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        print("\nStarting model training...")
        model, best_val_loss, best_val_metrics = train_model(
            model, 
            train_data, 
            val_data, 
            epochs=100, 
            batch_size=32,
            lr=3e-4
        )
        
        # Print validation results
        print("\nBest Validation Results:")
        print("=" * 50)
        print(f"Loss: {best_val_loss:.4f}")
        print(f"Accuracy: {best_val_metrics['accuracy']:.2f}%")
        print(f"Precision: {best_val_metrics['precision']:.2f}%")
        print(f"Recall: {best_val_metrics['recall']:.2f}%")
        print(f"F1 Score: {best_val_metrics['f1']:.2f}%")
        print("=" * 50)
        
        # Save model
        save_path = 'best_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_metrics': best_val_metrics,
            'input_length': input_length,
            'model_architecture': str(model),
            'training_params': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 3e-4
            }
        }, save_path)
        print(f"\nBest model saved to: {save_path}")
        
        # Evaluate on test set
        print("\nEvaluating model on test set...")
        test_metrics = evaluate_model(model, test_data)
        
        # Print test results
        print("\nFinal Test Results:")
        print("=" * 50)
        print(f"Loss: {test_metrics['test_loss']:.4f}")
        print(f"Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Precision: {test_metrics['precision']:.2f}%")
        print(f"Recall: {test_metrics['recall']:.2f}%")
        print(f"F1 Score: {test_metrics['f1']:.2f}%")
        print("\nDetailed Metrics:")
        print(f"True Positives: {test_metrics['true_positives']}")
        print(f"False Positives: {test_metrics['false_positives']}")
        print(f"True Negatives: {test_metrics['true_negatives']}")
        print(f"False Negatives: {test_metrics['false_negatives']}")
        print("=" * 50)
        
        # Save results
        test_results = {
            'test_metrics': {k: float(v) if isinstance(v, (float, np.float32)) else v 
                           for k, v in test_metrics.items()},
            'timestamp': str(datetime.datetime.now()),
            'data_distribution': {
                'train': {'normal': int((y_train == 1).sum().item()), 
                         'abnormal': int((y_train == 0).sum().item())},
                'val': {'normal': int((y_val == 1).sum().item()), 
                       'abnormal': int((y_val == 0).sum().item())},
                'test': {'normal': int((y_test == 1).sum().item()), 
                        'abnormal': int((y_test == 0).sum().item())}
            }
        }
        
        with open('test_results.json', 'w') as f:
            json.dump(test_results, f, indent=4)
        print("\nTest results saved to: test_results.json")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("\nTraceback:")
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Starting ECG classification at {datetime.datetime.now()}")
    main()