# mlp_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 8)
        self.fc4 = nn.Linear(8, 6)
        self.fc5 = nn.Linear(6, 5)
        self.fc6 = nn.Linear(5, 3)  # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # no activation here, handled by CrossEntropyLoss
        return x


class MLPClassifier_dynamic(nn.Module):
    def __init__(self, hidden_sizes):
        super().__init__()
        layers = []
        input_size = 3
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            input_size = h
        layers.append(nn.Linear(input_size, 3))  # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
