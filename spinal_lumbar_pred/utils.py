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
from torch import nn
from matplotlib import pyplot as plt

def load_dicom(path):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def evaluate(model, data_loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for images, level_idx, labels, row_ids in data_loader:
            images = images.to(device)
            level_idx = level_idx.to(device)
            labels = labels.to(device)
            
            outputs = model(images, level_idx)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(row_ids)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    if return_predictions:
        results_df = pd.DataFrame({
            'row_id': all_ids,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'prob_normal_mild': [p[0] for p in all_probs],
            'prob_moderate': [p[1] for p in all_probs],
            'prob_severe': [p[2] for p in all_probs]
        })
        return avg_loss, accuracy, results_df
    
    return avg_loss, accuracy


def train_epoch(model, model_name, train_loader, valid_loader, criterion, optimizer, device, 
                num_epochs=10, save_dir='./plots', model_dir='./trained_models'):
    os.makedirs(model_dir, exist_ok=True)
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Wrap model with DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, level_idx, labels, _ in train_loader:
            images = images.to(device)
            level_idx = level_idx.to(device) 
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, level_idx)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(model_dir, f'{model_name}_best_model.pth')
            
            # Save the model state properly handling DataParallel
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,  # Save the underlying model's state dict
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'train_acc': train_acc
            }, model_path)
            print(f'Saved best model to {model_path}')

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_dir, model_name)
    
    return best_val_acc,model

# Also update the evaluate function to handle DataParallel
def evaluate(model, data_loader, criterion, device, return_predictions=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_probs = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for images, level_idx, labels, row_ids in data_loader:
            images = images.to(device)
            level_idx = level_idx.to(device)
            labels = labels.to(device)
            
            outputs = model(images, level_idx)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total_loss += loss.item()
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(row_ids)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    if return_predictions:
        results_df = pd.DataFrame({
            'row_id': all_ids,
            'true_label': all_labels,
            'predicted_label': all_preds,
            'prob_normal_mild': [p[0] for p in all_probs],
            'prob_moderate': [p[1] for p in all_probs],
            'prob_severe': [p[2] for p in all_probs]
        })
        return avg_loss, accuracy, results_df
    
    return avg_loss, accuracy

# For loading the model later:
def load_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle loading with or without DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Load the state dict
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model = model.to(device)
    return model, checkpoint


    
    
def get_predictions(model, model_name, data_loader, criterion, device, save_dir='./results'):
    import os
    import pandas as pd
    
    # Get basic metrics and predictions
    avg_loss, accuracy, results_df = evaluate(model, data_loader, criterion, device, return_predictions=True)
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save detailed predictions
    results_df.to_csv(os.path.join(save_dir, f'{model_name}_predictions.csv'), index=False)
    
    # Create/update model performance summary CSV
    summary_file = os.path.join(save_dir, 'model_performance_summary.csv')
    
    # Create new row for current model
    new_row = pd.DataFrame({
        'model_name': [model_name],
        'avg_loss': [avg_loss],
        'accuracy': [accuracy]
    })
    
    # If summary file exists, append; otherwise create new
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        # Update or append row
        if model_name in summary_df['model_name'].values:
            summary_df.loc[summary_df['model_name'] == model_name] = new_row.iloc[0]
        else:
            summary_df = pd.concat([summary_df, new_row], ignore_index=True)
    else:
        summary_df = new_row
    
    # Save updated summary
    summary_df.to_csv(summary_file, index=False)
    
    return results_df 
    
    
    
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_dir,model_name):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'))
    plt.close()