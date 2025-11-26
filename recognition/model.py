"""
ResNet50 Logo Recognition Model with ArcFace Loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class LogoRecognitionModel(nn.Module):
    """
    ResNet50-based logo classification model.
    Uses ArcFace loss during training for better feature learning.
    """
    
    def __init__(self, num_classes, pretrained=True, embedding_size=512):
        super().__init__()
        
        # Load ResNet50 backbone
        if pretrained:
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            backbone = models.resnet50(weights=None)
        
        # Remove original classification head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        
        self.backbone = backbone
        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, x, labels=None):
        """
        Forward pass
        
        Args:
            x: Input images [B, 3, H, W]
            labels: Ground truth labels (for training with ArcFace)
            
        Returns:
            If labels provided: (loss, logits, embeddings)
            Else: (logits, embeddings)
        """
        # Extract features
        feat = self.backbone(x)
        
        # Get embeddings
        emb = self.embedding(feat)
        emb = self.bn(emb)
        emb = F.relu(emb)
        emb = self.dropout(emb)
        
        # Classification logits
        logits = self.classifier(emb)
        
        # Return loss if training
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits, emb
        
        return logits, emb
    
    def get_embeddings(self, x):
        """Extract embeddings without classification"""
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = self.bn(emb)
        emb = F.relu(emb)
        return emb


# Placeholder for training script
# Recognition weights will be provided later
