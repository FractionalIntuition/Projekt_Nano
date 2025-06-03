import torch.nn as nn
import torch

class IPAClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2))
        )
        self.lstm = nn.LSTM(32 * 5, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)  # [B, C, F, T]
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, x.size(3), -1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
