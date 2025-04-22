import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 添加批归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
    
    def forward(self, x):
        # 展平图像 (保留batch维度)
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)