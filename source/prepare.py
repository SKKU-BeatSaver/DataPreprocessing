import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


# 1. Custom Dataset Class 정의 (슬라이딩 윈도우 방식)
class SlidingWindowECGDataset(Dataset):
    def __init__(self, directory, file_pattern, window_size, overlap=0.1, minorclassoverlap = 0.5):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory) if file_pattern.match(f)
        ]
        self.data = []
        self.labels = []
        self.window_size = window_size  # 슬라이딩 윈도우 크기
        #self.overlap = overlap
        #self.stride = int(window_size * (1 - overlap))  # 겹치기 계산

        # 데이터 로드 및 슬라이딩 윈도우 적용
        for file in self.files:
            raw_data = pd.read_csv(file, header=None).values.flatten()
            ecg_data = raw_data[:-1]  # 마지막 값 제외 (ECG 데이터)
            label = raw_data[-1]  # 마지막 값 (레이블)

            if (label < 0.5) :  # minor class
                self.stride = int(window_size * (1 - minorclassoverlap))  # 겹치기 계산
            else :
                self.stride = int(window_size * (1 - overlap))  # 겹치기 계산
            # 슬라이딩 윈도우 생성
            for start_idx in range(0, len(ecg_data) - window_size + 1, self.stride):
                window = ecg_data[start_idx : start_idx + window_size]
                self.data.append(window)
                self.labels.append(label)    

        # NumPy 배열로 변환
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).unsqueeze(0)  # 채널 차원 추가 (1D 데이터 → (1, window_size))
        y = torch.tensor(self.labels[idx],dtype = torch.float32)
        return x, y

