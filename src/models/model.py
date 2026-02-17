import torch
import torch.nn as nn
import torchvision.models as models

class ObjectDetectionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ObjectDetectionModel, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleSegmentationModel, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        
        self.dec5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        d5 = self.dec5(e5)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)
        
        return d1

def create_model(num_classes=10, device='cuda', model_type='classification'):
    if model_type == 'segmentation':
        model = SimpleSegmentationModel(num_classes=num_classes)
    else:
        model = ObjectDetectionModel(num_classes=num_classes)
    return model.to(device)