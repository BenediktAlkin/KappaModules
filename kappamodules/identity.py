import torch.nn as nn

class Identity(nn.Identity):
    def __init__(self, *_, **__):
        super().__init__()