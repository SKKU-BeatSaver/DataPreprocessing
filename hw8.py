import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn 
import torch.optim as optim 
 
def generate_time_series(num_samples=1500, sequence_length=50): 
    """ 
    label 0 : increase -> increase 
    label 1 : increase -> decrease 
    label 2 : decrease -> increase 
    label 3 : decrease -> decrease 
    label 4 : others (containing periodic pattern) 
    """ 
    X = [] 
    y = [] 
    for _ in range(num_samples): 
        # random pattern 
        pattern1_type = np.random.choice([0, 1, 2])  # Pattern 1  
        pattern2_type = np.random.choice([0, 1, 2])  # Pattern 2 
        noise = np.random.normal(0, 0.2, sequence_length)  # noise 
 
        # First pattern 
        if pattern1_type == 0:  # increase 
            pattern1 = np.linspace(0, 1, sequence_length // 2) 
        elif pattern1_type == 1:  # decrease 
            pattern1 = np.linspace(1, 0, sequence_length // 2) 
        else:  # periodic 
            pattern1 = np.sin(np.linspace(0, np.pi, sequence_length // 2)) 
 
        # Second pattern 
        if pattern2_type == 0:  # increase 
            pattern2 = np.linspace(0, 1, sequence_length // 2) 
        elif pattern2_type == 1:  # decrease 
            pattern2 = np.linspace(1, 0, sequence_length // 2) 
        else:  # periodic 
            pattern2 = np.sin(np.linspace(0, np.pi, sequence_length // 2)) 
 
        # concat data 
        series = np.concatenate([pattern1, pattern2]) + noise 
 
        # label 0 : increase -> increase 
        # label 1 : increase -> decrease 
        # label 2 : decrease -> increase 
        # label 3 : decrease -> decrease 
        # label 4 : others (containing periodic pattern) 
        if pattern1_type == 0 and pattern2_type == 0: label = 0 
        elif pattern1_type == 0 and pattern2_type == 1: label = 1 
        elif pattern1_type == 1 and pattern2_type == 0: label = 2 
        elif pattern1_type == 1 and pattern2_type == 1: label = 3 
        else: label = 4 
 
        X.append(series) 
        y.append(label) 
 
    return np.array(X), np.array(y) 
 
sequence_length = 50 
X, y = generate_time_series(num_samples=2000, sequence_length=sequence_length) 
 
# Dataset 
class TimeSeriesDataset(Dataset): 
    def __init__(self, data, labels): 
        self.data = torch.tensor(data, dtype=torch.float32) 
        self.labels = torch.tensor(labels, dtype=torch.long) 

    def __len__(self): 
        return len(self.labels) 
    def __getitem__(self, idx): 
        return self.data[idx].unsqueeze(-1), self.labels[idx] 
# Dataloader 
train_size = int(0.8 * len(X)) 
test_size = len(X) - train_size 
train_dataset = TimeSeriesDataset(X[:train_size], y[:train_size]) 
test_dataset = TimeSeriesDataset(X[train_size:], y[train_size:]) 
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=32) 


 
# class RNNClassifier(nn.Module): 
#     def __init__(self, input_size, hidden_size, num_layers, output_size): 
#         super(RNNClassifier, self).__init__() 
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) 
#         self.fc = nn.Linear(hidden_size, output_size) 
 
#     def forward(self, x): 
#         out, _ = self.rnn(x) 
#         out = self.fc(out[:, -1, :])  
#         return out 
 
# # hyper-parameters 
# input_size = 1 
# hidden_size = 64 
# num_layers = 1 
# output_size = 5 
# model = RNNClassifier(input_size, hidden_size, num_layers, output_size) 
 
# criterion = nn.CrossEntropyLoss() 
# optimizer = optim.Adam(model.parameters(), lr=0.01) 
 
# # training 
# num_epochs = 20 
# for epoch in range(num_epochs): 
#     model.train() 
#     total_loss = 0 
#     for batch_data, batch_labels in train_loader: 
#         outputs = model(batch_data) 
#         loss = criterion(outputs, batch_labels) 
 
#         optimizer.zero_grad() 
#         loss.backward() 
#         optimizer.step() 
 
#         total_loss += loss.item() 
 
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}") 
 
# # evaluation 
# model.eval() 
# correct = 0 
# total = 0 
# with torch.no_grad(): 
#     for batch_data, batch_labels in test_loader: 
#         outputs = model(batch_data) 
#         _, predicted = torch.max(outputs, 1) 
#         total += batch_labels.size(0) 
#         correct += (predicted == batch_labels).sum().item() 
 
# print(f"Test Accuracy: {correct / total * 100:.2f}%") 

 
class LSTMClassifier(nn.Module): 
    def __init__(self, input_size, hidden_size, num_layers, output_size): 
        super(LSTMClassifier, self).__init__() 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size) 
 
    def forward(self, x): 
        out, _ = self.lstm(x) 
        out = self.fc(out[:, -1, :])  
        return out 
 
# hyper-parameters 
input_size = 1 
hidden_size = 64 
num_layers = 1 
output_size = 5 
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size) 
 
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.01) 
 
# training 
num_epochs = 20 
for epoch in range(num_epochs): 
    model.train() 
    total_loss = 0 
    for batch_data, batch_labels in train_loader: 
        outputs = model(batch_data) 
        loss = criterion(outputs, batch_labels) 
 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
 
        total_loss += loss.item() 
 
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}") 
 
# evaluation 
model.eval() 
correct = 0 
total = 0 
with torch.no_grad(): 
    for batch_data, batch_labels in test_loader: 
        outputs = model(batch_data) 
        _, predicted = torch.max(outputs, 1) 
        total += batch_labels.size(0) 
        correct += (predicted == batch_labels).sum().item() 
 
print(f"Test Accuracy: {correct / total * 100:.2f}%")