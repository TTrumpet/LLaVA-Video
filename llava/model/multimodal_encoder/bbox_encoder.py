import torch
import torch.nn as nn
import torch.optim as optim


class bboxEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, output_dim=768, target_dtype=torch.bfloat16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.target_dtype = target_dtype

    def forward(self, x):
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            x = x.to(torch.bfloat16)
        return self.encoder(x)


# This is really simple, we can make more expressive.
class bboxDecoder(nn.Module):
    def __init__(self, input_dim=768, output_dim=4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        return self.decoder(x)