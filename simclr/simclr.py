import torch.nn as nn
import torchvision

from simclr.modules.resnet_hacks import modify_resnet_model
from simclr.modules.identity import Identity


class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, projection_dim, n_features, static=False):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )
        
        if static == True:
            print("#"*10, "This is static mode", "#"*10)
            for param in self.projector.parameters():
                param.requires_grad = False
        else:
            print("#"*10, "This is original mode", "#"*10)
        

    def forward(self, x_i, x_j=None):
        if x_j != None:
            h_i = self.encoder(x_i)
            h_j = self.encoder(x_j)

            z_i = self.projector(h_i)
            z_j = self.projector(h_j)
            return h_i, h_j, z_i, z_j
        else:
            h_i = self.encoder(x_i)

            z_i = self.projector(h_i)
            return z_i
            