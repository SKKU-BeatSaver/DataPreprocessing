import numpy as np
import pandas as pd
import os
import re
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
# Add these imports at the top
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Data loading functions
def load_single_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    time_series = np.array([float(line.strip()) for line in lines[:-1]])
    label = float(lines[-1].strip())
    
    return time_series, label

def create_dataset(data_directory, file_pattern):
    all_files = [f for f in os.listdir(data_directory) 
                 if file_pattern.match(os.path.basename(f))]
    
    all_series = []
    all_labels = []
    
    for filename in sorted(all_files):
        file_path = os.path.join(data_directory, filename)
        time_series, label = load_single_file(file_path)
        all_series.append(time_series)
        all_labels.append(label)
    
    return np.array(all_series), np.array(all_labels)

# Custom Dataset class
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Input shape: [batch_size, sequence_length]
        x = x.unsqueeze(-1)  # Add feature dimension: [batch_size, sequence_length, 1]
        
        # Position encoding
        x = self.pos_encoder(x)  # [batch_size, sequence_length, hidden_dim]
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, sequence_length, hidden_dim]
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Output layer
        x = self.fc(x)  # [batch_size, 1]
        return x.squeeze(-1)

# Lightning Module
class ECGClassifier(pl.LightningModule):
    def __init__(self, input_dim=1, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    



# Add these functions after the ECGClassifier class
def load_trained_model(checkpoint_path):
    """Load a trained model from a checkpoint"""
    model = ECGClassifier.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def make_predictions(model, dataloader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Make predictions using the trained model"""
    model = model.to(device)
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            predictions.extend((outputs > 0.5).float().cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def evaluate_model(predictions, true_labels):
    """Calculate and print various performance metrics"""
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



# Main execution
if __name__ == "__main__":
    data_directory = "../Shao_Results_100"
    file_pattern = re.compile(r"JS\d{5}_labeled\.txt")
    
    # Load and process data
    features, labels = create_dataset(data_directory, file_pattern)
    
    # Normalize features
    features = (features - features.mean()) / features.std()
    
    # Create datasets
    train_size = int(0.8 * len(features))
    train_features, val_features = features[:train_size], features[train_size:]
    train_labels, val_labels = labels[:train_size], labels[train_size:]
    
    train_dataset = ECGDataset(train_features, train_labels)
    val_dataset = ECGDataset(val_features, val_labels)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = ECGClassifier(
        input_dim=1,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        learning_rate=1e-3
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor("epoch"),
        ],
        logger=TensorBoardLogger("lightning_logs"),
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

    # [Previous training code remains the same]
    
    # After training, test the model
    # Find the best checkpoint
    checkpoint_path = "lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt"  # Replace X, Y, Z with actual values
    
    # Load the trained model
    trained_model = load_trained_model(checkpoint_path)
    
    # Create test dataloader (using validation set for this example)
    test_dataset = ECGDataset(val_features, val_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Make predictions
    predictions, true_labels = make_predictions(trained_model, test_loader)
    
    # Evaluate the model
    metrics = evaluate_model(predictions, true_labels)
    
    # Optional: Plot some predictions
    def plot_example_predictions(features, true_labels, predictions, num_examples=3):
        fig, axes = plt.subplots(num_examples, 1, figsize=(15, 5*num_examples))
        for i in range(num_examples):
            axes[i].plot(features[i])
            axes[i].set_title(f'True Label: {true_labels[i]:.2f}, Predicted: {predictions[i]:.2f}')
        plt.tight_layout()
        plt.show()

    # Plot a few examples
    plot_example_predictions(val_features, true_labels, predictions)

def test_directory(model, test_directory, file_pattern):
    # Load test data
    test_features, test_labels = create_dataset(test_directory, file_pattern)
    
    # Normalize using training parameters
    test_features = (test_features - features.mean()) / features.std()
    
    # Create dataset and dataloader
    test_dataset = ECGDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Make predictions
    predictions, true_labels = make_predictions(model, test_loader)
    
    # Evaluate
    metrics = evaluate_model(predictions, true_labels)
    return predictions, true_labels, metrics

# Example usage
test_dir = "../Shao_Results_test"
predictions, true_labels, metrics = test_directory(trained_model, test_dir, file_pattern)