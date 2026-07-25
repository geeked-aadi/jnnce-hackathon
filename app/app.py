import torch
from monai.networks.nets import SegResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SegResNet(
    spatial_dims=3,
    init_filters=16,
    in_channels=4,
    out_channels=3,
    blocks_down=(1, 2, 2, 4),
    blocks_up=(1, 1, 1),
    dropout_prob=0.2,
).to(device)
weights = torch.load("model.pt", map_location=device)

model.load_state_dict(weights)

model.eval()
weights = torch.load("model.pt", map_location=device)

model.load_state_dict(weights)

model.eval()