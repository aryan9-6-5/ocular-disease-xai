from src.models.cbam import CBAM

class AttentionModel(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        self.base = torchvision.models.resnet50(pretrained=True)
        # Insert CBAM after a conv layer, e.g., after layer4
        self.cbam = CBAM(in_planes=2048)  # ResNet50 layer4 has 2048 channels
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        # Example: apply CBAM to features before global pool
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.cbam(x)               # ← CBAM here
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base.fc(x)
        return x