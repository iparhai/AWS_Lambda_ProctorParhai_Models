import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained='vggface2', freeze_backbone=True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained, classify=False).eval()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.projection = nn.Linear(512, embedding_size)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.projection(feat)