import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784 ,10)
        pass

    def forward(self, x):
        x = x.view(-1, 28* 28)
        out = self.fc1.forward(x)
        out = F.softmax(out)

        return out