import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, backbone, emb_dim) -> None:
        super(Classifier, self).__init__()
        self.backbone = backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.gender_encoder = nn.Linear(1, 32)
        self.gender_bn = nn.BatchNorm1d(32)

        self.fc = nn.Sequential(
            nn.Linear(emb_dim + 32, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(512, 230)

    def no_weight_decay(self):
        return {'backbone.pos_embed', 'backbone.cls_token'}

    def forward(self, image, gender):
        self.backbone.eval()
        x = self.backbone(image)

        gender_encode = self.gender_bn(self.gender_encoder(gender))
        gender_encode = F.relu(gender_encode)

        x = torch.cat([x, gender_encode], dim=1)

        x = self.fc(x)

        return self.classifier(x)