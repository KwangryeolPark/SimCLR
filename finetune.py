import torch
import torch.nn as nn
import wandb
import argparse
from simclr import SimCLR
from simclr.modules import get_resnet
from torchvision.datasets import STL10, CIFAR10
from simclr.modules.transformations import TransformsSimCLR

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--static', type=bool)
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--max_epoch', type=int)

args = parser.parse_args()

wandb.init(
    project='SCL',
    tags=['original_code'],
    name="static" if args.static else "none_static",
    config=args
)

if args.dataset == 'STL10':
    train_ds = STL10(
        root='./datasets',
        split='train',
        download=True,
        transform=TransformsSimCLR(size=224).test_transform
    )
    test_ds = STL10(
        root='./datasets',
        split='test',
        download=True,
        transform=TransformsSimCLR(size=224).test_transform
    )
elif args.dataset == 'CIFAR10':
    train_ds = CIFAR10(
        root='./datasets',
        train=True,
        download=True,
        transform=TransformsSimCLR(size=224).test_transform
    )
    test_ds = CIFAR10(
        root='./datasets',
        train=False,
        download=True,
        transform=TransformsSimCLR(size=224).test_transform
    )
    
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

encoder = get_resnet(args.model, pretrained=False)
in_features = encoder.fc.in_features
simclr_model = SimCLR(encoder, 64, in_features)
simclr_model.load_state_dict(torch.load(args.ckpt))

model = simclr_model.encoder
model.fc = nn.Linear(in_features, 10)
