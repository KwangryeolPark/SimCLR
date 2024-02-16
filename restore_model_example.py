# %%
import torch

model = torch.load("/root/python/SimCLR/save/checkpoint_none_static_100.tar")
# %%
from torchvision.models import resnet18
from simclr import SimCLR
simclr_model = SimCLR(resnet18(), 64, resnet18().fc.in_features)

simclr_model.load_state_dict(model)

# %%
import torch.nn as nn
num_class = 10
resnet_model = simclr_model.encoder
resnet_model.fc = nn.Linear(simclr_model.n_features, num_class)
print(resnet_model)