import torch
import torchvision.models as models
from torch import nn
from transformers import BeitForImageClassification, SwinForImageClassification
from dataset import create_dataloaders
from model import SpinalStenosisModel
from utils import get_predictions, train_epoch
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['vit', 'swin', 'beit', 'efficientnet', 'resnet152', 'convnext'],
                        help='List of models to train')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--level_embeddings',type = int, default = 128)
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    parser.add_argument('--unfreeze_last_n', type=int, default=20)
    return parser.parse_args()

def get_base_model(model_name):
    if model_name == 'vit':
        return models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1), False
    elif model_name == 'swin':
        return SwinForImageClassification.from_pretrained('microsoft/swin-base-patch4-window7-224'), True
    elif model_name == 'beit':
        return BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224'), True
    elif model_name == 'efficientnet':
        return models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1), False
    elif model_name == 'resnet152':
        return models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2), False
    elif model_name == 'convnext':
        return models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1), False

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_df = pd.read_csv("train_df_spinal_canal_stenosis.csv")
    valid_df = pd.read_csv("val_df_spinal_canal_stenosis.csv")
    test_df = pd.read_csv("test_df_spinal_canal_stenosis.csv")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, valid_df, test_df, batch_size=args.batch_size
    )
    
    # Class weights
    class_weights = torch.tensor([1.0, 5531/485, 5531/326], dtype=torch.float32).to(device)
    
    # Train each model
    for model_name in args.models:
        print(f"\nTraining {model_name}")
        
        # Get base model
        base_model, is_huggingface = get_base_model(model_name)
        
        # Initialize model
        model = SpinalStenosisModel(
            base_model=base_model,
            num_severity_classes=3,
            num_levels=5,
            embedding_dim=args.level_embeddings,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
            is_huggingface=is_huggingface,
            model_name=model_name
        ).to(device)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Train
        best_val_acc = train_epoch(
            model=model,
            model_name=f"{model_name}_model",
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=args.num_epochs,
            save_dir='./plots',
            model_dir='./trained_models'
        )
        
        # Get predictions
        get_predictions(
            model=model,
            model_name=model_name,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            save_dir='./results'
        )

if __name__ == "__main__":
    main()