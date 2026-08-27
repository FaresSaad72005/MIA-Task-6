import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image




# ResNet-50 Encoder
class ResNetEncoder(nn.Module):

    def __init__(self, embed_size, use_attention=False):
        super(ResNetEncoder, self).__init__()
        self.use_attention = use_attention

        # Load ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze parameter gradients
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove Conv5 avgpool & fc layers 
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        
    def forward(self, images):
        # Extract features -> Output shape: (Batch, 2048, 7, 7)
        features = self.resnet(images)
        
        # Reshape to spatial sequence format: (Batch, 49, 2048)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(3))
        return features








