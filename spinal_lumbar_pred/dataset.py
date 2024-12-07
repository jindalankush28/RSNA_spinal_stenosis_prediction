import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
#import albumentations as A
import numpy as np
import pydicom
import os
import cv2
from utils import load_dicom

class SpinalStenosisDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
        # level for mapping embeddings
        self.level_to_idx = {
            'L1/L2': 0,
            'L2/L3': 1,
            'L3/L4': 2,
            'L4/L5': 3,
            'L5/S1': 4
        }
        
        # Create severity to index mapping
        self.severity_to_idx = {
            'normal_mild': 0,
            'moderate': 1,
            'severe': 2
        }
        
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        image_path = row['image_path']
        data_row = row["row_id"]
        image = load_dicom(image_path)
        image = self.transform(image)
        
        level_idx = torch.tensor(self.level_to_idx[row['level']], dtype=torch.long)
        severity_label = torch.tensor(self.severity_to_idx[row['severity']], dtype=torch.long)
        
        return image, level_idx, severity_label,data_row
    
    
    
def create_dataloaders(train_df, val_df, test_df, batch_size=32):
    # Create datasets
    train_dataset = SpinalStenosisDataset(train_df)
    val_dataset = SpinalStenosisDataset(val_df)
    test_dataset = SpinalStenosisDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader