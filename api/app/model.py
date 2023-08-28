from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

def get_model():
    model = resnet50()
    model.fc = nn.Sequential(
                        nn.Linear(model.fc.in_features, 256),
                        nn.ReLU(),
                        nn.Dropout(0.4),
                        nn.Linear(256, 120),                   
                        nn.LogSoftmax(dim=1))
    return model

def get_transform():
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform