import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from sklearn.model_selection import train_test_split
import re
import traceback

class ECGDataManager:
    def __init__(self, data_dir, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
        if not np.isclose(train_split + val_split + test_split, 1.0):
            raise ValueError("Split proportions must sum to 1")

        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

    def get_subject_number(self, filename):
        """Extract subject number from filename"""
        match = re.match(r'S(\d{4})_labeled\.txt', filename.name)
        if match:
            return int(match.group(1))
        return None

    def load_and_preprocess_file(self, filename):
        """Load and preprocess a single file"""
        try:
            data = np.loadtxt(filename)

            # Separate features and label
            features = data[:-1]
            label = float(data[-1] >= 0.5)  # Binary label

            # Normalize features
            mean = np.mean(features)
            std = np.std(features)
            features_normalized = (features - mean) / (std + 1e-10)

            # Reshape for model input (1 channel)
            features_reshaped = features_normalized.reshape(1, -1)

            return torch.tensor(features_reshaped, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            raise

    def split_files(self):
        """Split files into training, validation, and test sets"""
        labeled_files = []
        pattern = re.compile(r'S\d{4}_labeled\.txt')

        for file in self.data_dir.glob("S*_labeled.txt"):
            if pattern.match(file.name):
                labeled_files.append((self.get_subject_number(file), file))

        if not labeled_files:
            raise ValueError(f"No labeled files found in {self.data_dir}")

        labeled_files.sort(key=lambda x: x[0])
        files = [f[1] for f in labeled_files]

        # Calculate split indices
        total_files = len(files)
        train_size = int(total_files * self.train_split)
        val_size = int(total_files * self.val_split)

        train_files = files[:train_size]
        val_files = files[train_size:train_size + val_size]
        test_files = files[train_size + val_size:]

        return train_files, val_files, test_files

    def create_dataset(self, files):
        """Create dataset from files"""
        X_data = []
        y_data = []

        max_length = 0
        for file in files:
            try:
                features, label = self.load_and_preprocess_file(file)
                max_length = max(max_length, features.shape[1])
                X_data.append(features)
                y_data.append(label)
            except Exception as e:
                print(f"Skipping {file}: {str(e)}")
                continue

        if not X_data:
            raise ValueError("No valid data could be processed from files")

        # Pad sequences
        X_padded = [F.pad(x, (0, max_length - x.shape[1])) for x in X_data]
        X_tensor = torch.stack(X_padded)
        y_tensor = torch.stack(y_data)

        return X_tensor, y_tensor

    def prepare_data(self):
        train_files, val_files, test_files = self.split_files()

        print("\nProcessing training data...")
        X_train, y_train = self.create_dataset(train_files)

        print("\nProcessing validation data...")
        X_val, y_val = self.create_dataset(val_files)

        print("\nProcessing test data...")
        X_test, y_test = self.create_dataset(test_files)

        return X_train, y_train, X_val, y_val, X_test, y_test

class ConvNetQuake(nn.Module):
    def __init__(self, input_length=None):
        super(ConvNetQuake, self).__init__()
        
        # Calculate the output size after convolutions
        self.calculate_output_size = input_length is not None
        test_input = torch.zeros(1, 1, input_length) if input_length else None
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)  # Force output to have length 8
        )
        
        # Calculate the size of the flattened features
        if self.calculate_output_size:
            with torch.no_grad():
                conv_out = self.conv_layers(test_input)
                self.flat_features = conv_out.view(1, -1).size(1)
        else:
            self.flat_features = 32 * 8  # 32 channels * 8 (adaptive pooling output)
            
        # Define the fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

def train_model(model, train_data, val_data, epochs=50, batch_size=32, lr=1e-4):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output.squeeze(), y_val)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

def main():
    try:
        data_manager = ECGDataManager(data_dir="./")
        X_train, y_train, X_val, y_val, X_test, y_test = data_manager.prepare_data()
        
        # Get input length from training data
        input_length = X_train.shape[2]
        
        # Create datasets
        train_data = torch.utils.data.TensorDataset(X_train, y_train)
        val_data = torch.utils.data.TensorDataset(X_val, y_val)
        test_data = torch.utils.data.TensorDataset(X_test, y_test)

        # Initialize model with input length
        model = ConvNetQuake(input_length=input_length)
        
        # Train model
        train_model(model, train_data, val_data, epochs=50, batch_size=32)
        
        # Test evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
        criterion = nn.BCELoss()
        
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_test, y_test in test_loader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                outputs = model(X_test)
                
                loss = criterion(outputs.squeeze(), y_test)
                test_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                correct += (predicted == y_test).sum().item()
                total += y_test.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        print(f"\nTest Results:")
        print(f"Average Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
