# Imports

import torch
from torch import nn

#%% Define model

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()   
        #self.fc = []
        self.fc = nn.ModuleList()
        for idx in range(0,3):
            self.fc.append(nn.Linear(10,1))
    
    def forward(self, x):
        out = []
        for idx in range(0,3):
            aux = self.fc[idx](x)
            out.append(aux)
        return out

#%% Instantiate model
model = TestModel()

#%% Printmodel structure
print(model)

#%% Show model parameters
for x in model.parameters():
    print(x)

#%% Forward pass
input_x = torch.rand(10)
output = model(input_x)