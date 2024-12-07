import torch
import torch.nn as nn
import torchvision.models as models

class SpinalStenosisModel(nn.Module):
    def __init__(self, 
                 base_model,
                 num_severity_classes=3,
                 num_levels=5,
                 embedding_dim=64,
                 freeze_backbone=True,
                 unfreeze_last_n=20,
                 is_huggingface=False,
                 model_name=None):
        super(SpinalStenosisModel, self).__init__()
        
        self.base_model = base_model
        self.is_huggingface = is_huggingface
        
        # Get feature dimension

            # Get feature dimension dynamically by checking different possible names
        if 'convnext' in model_name.lower():
            if isinstance(self.base_model.classifier[-1], nn.Linear):
                self.num_features = self.base_model.classifier[-1].in_features
                self.base_model.classifier[-1] = nn.Identity()
            
        elif hasattr(self.base_model, 'heads'):
            if hasattr(self.base_model.heads, 'head'):
                self.num_features = self.base_model.heads.head.in_features
            else:
                self.num_features = list(self.base_model.heads.children())[-1].in_features
            self.base_model.heads = nn.Identity()
                
        elif hasattr(self.base_model, 'head'):
            self.num_features = self.base_model.head.in_features
            self.base_model.head = nn.Identity()
                
        elif hasattr(self.base_model, 'classifier'):
            if isinstance(self.base_model.classifier, nn.Linear):
                self.num_features = self.base_model.classifier.in_features
            else:
                self.num_features = self.base_model.classifier[-1].in_features
            self.base_model.classifier = nn.Identity()
                
        elif hasattr(self.base_model, 'fc'):
            self.num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
                
        else:
            raise ValueError("Cannot determine feature dimension from base model")
        
        # Level embedding
        self.level_embedding = nn.Embedding(num_levels, embedding_dim)
        
        # Classifier with transformer-style architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features + embedding_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_severity_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()
            if unfreeze_last_n > 0:
                self.unfreeze_last_n_layers(unfreeze_last_n)

    def forward(self, x, level_idx):
        # Handle different model outputs
        if self.is_huggingface:
            outputs = self.base_model(x)
            features = outputs.logits
        else:
            features = self.base_model(x)
        
        level_emb = self.level_embedding(level_idx)
        combined = torch.cat([features, level_emb], dim=1)
        return self.classifier(combined)
    
    def freeze_backbone(self):
        """Freeze all parameters in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_last_n_layers(self, n):
        """Unfreeze the last n layers of the base model"""
        params = list(self.base_model.parameters())
        for param in params[-n:]:
            param.requires_grad = True