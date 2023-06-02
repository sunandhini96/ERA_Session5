from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# defining the model

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # RF_out=3
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Rf_out=5
        x = F.relu(self.conv3(x), 2)  # RF_out=9
        x = F.relu(F.max_pool2d(self.conv4(x), 2)) # RF_out=11
        x = x.view(-1, 4096) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
