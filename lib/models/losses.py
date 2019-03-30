import torch
import torch.nn.functional as F
import torch.distributions as distributions

L2_LOSS = 1
BERNOULLI = 2

def l2_loss(reconstructions, images, activation, reduction="sum"):
    """L2 loss, i.e. sum of squared errors"""
    if activation == "logits":
        reconstructions = torch.sigmoid(reconstructions)
    elif activation == "tanh":
        reconstructions = torch.tanh(reconstructions).div(2).add(0.5)
    else:
        raise ValueError("unknown activation function")

    if reduction == "sum":
        loss = F.mse_loss(reconstructions, images, reduction="sum")
    elif reduction == "mean":
        # Mean per image
        loss = F.mse_loss(reconstructions, images, reduction="sum").div(images.size(0))
    else:
        raise ValueError("parameter reduction must be either \"sum\" or \"mean\"")

    return loss

def bernoulli_loss(reconstructions, images, activation, subtract_entropy=False):
    """Bernoulli loss"""
    flattened_dim = images.size(1)*images.size(2)*images.size(3)
    reconstructions = reconstructions.view(-1, flattened_dim)
    images = images.view(-1, flattened_dim)

    if subtract_entropy:
        dist = distributions.Bernoulli(probs=torch.clamp(images, 1e-6, (1-1e-6)))
        lower_bound = dist.entropy().sum(dim=1)
    else:
        lower_bound = 0

    if activation == "logits":
        loss = F.binary_cross_entropy_with_logits(reconstructions, images, reduction="none").sum(dim=1)
    elif activation == "tanh":
        reconstructions = torch.clamp(torch.tanh(reconstructions).div(2).add(0.5), 1e-6, (1-1e-6))
        loss = -torch.sum(images.mul(torch.log(reconstructions)) + (1-images).mul(torch.log(1 - reconstructions)), dim=1)
    else:
        raise ValueError("unknown activation function")
    
    return loss - lower_bound

def reconstruction_loss(reconstructions, images, config):
    if config.rec_loss == L2_LOSS:
        return l2_loss(reconstructions, images, config.activation, reduction=config.reduction)
    elif config.rec_loss == BERNOULLI:
        return bernoulli_loss(reconstructions, images, config.activation, config.subtract_entropy).mean()

def kl_divergence(mu, logvar):
    """KL Divergence"""
    loss = (mu.pow(2) + torch.exp(logvar) - logvar - 1).sum(1)
    return loss.mul(0.5).mean()

