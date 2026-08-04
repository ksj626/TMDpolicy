from torch import nn


def freeze_module(module: nn.Module) -> nn.Module:
    module.requires_grad_(False)
    module.eval()
    return module
