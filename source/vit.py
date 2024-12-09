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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from vit_pytorch import ViT

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

class ECGClassifier(pl.LightningModule):
    def __init__(self, sequence_length=1000, patch_size=50, dim=64, depth=6, heads=4, mlp_dim=128, dropout=0.1, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Reshape sequence to image-like format (1 channel)
        self.sequence_length = sequence_length
        self.reshape_size = int(np.sqrt(sequence_length))
        
        # Initialize ViT model
        self.model = ViT(
            image_size=self.reshape_size,
            patch_size=patch_size,
            num_classes=1,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            channels=1
        )
        
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate
        
    def forward(self, x):
        # Reshape 1D sequence to 2D image-like format
        x = x.view(-1, 1, self.reshape_size, self.reshape_size)
        x = self.model(x)
        return self.sigmoid(x).squeeze()
    
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Example usage
if __name__ == "__main__":
    # First, install required package
    # !pip install vit-pytorch
    
    # Your data loading code remains the same
    data_directory = "../Shao_Results_100"
    file_pattern = re.compile(r"JS\d{5}_labeled\.txt")
    
    # Load and process data
    features, labels = create_dataset(data_directory, file_pattern)
    
    # Ensure sequence length is perfect square for 2D reshape
    sequence_length = features.shape[1]
    reshape_size = int(np.ceil(np.sqrt(sequence_length)))
    pad_size = reshape_size * reshape_size - sequence_length
    
    if pad_size > 0:
        features = np.pad(features, ((0, 0), (0, pad_size)), mode='constant')
    
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
        sequence_length=reshape_size * reshape_size,
        patch_size=16,  # Adjust based on your reshaped image size
        dim=64,
        depth=6,
        heads=4,
        mlp_dim=128,
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