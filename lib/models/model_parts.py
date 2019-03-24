import torch
import torch.nn as nn

FC_ENCODER = 1
CONV_ENCODER = 2
CONV_BN_ENCODER = 3
FC_DECODER = 1
DECONV_DECODER = 2
DECONV_BN_DECODER = 3

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_out')
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class Reshape(nn.Module):
    def __init__(self, size):
        super(Reshape, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, tensor):
        return tensor.view(tensor.size(0), -1)

def fc_encoder(z_dim, inplanes=1, input_size=64):
    """Fully-connected encoder from \"β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK\"
    Args:
        z_dim: latent factor dimension
        inplanes: number of input channels
        input_size: (window) size of the input tensor of shape (batch_size, inplanes, input_size, input_size)
    Returns:
        output tensor of size 2*z_dim, where [:z_dim] are the latent variable means and [:z_dim] are the latent variable log variances
    """
    return nn.Sequential(
        Flatten(),
        nn.Linear(inplanes*input_size**2, 1200),
        nn.ReLU(inplace=True),
        nn.Linear(1200, 1200),
        nn.ReLU(inplace=True),
        nn.Linear(1200, 2*z_dim),
    )

def fc_decoder(z_dim, outplanes=1, output_size=64):
    """Fully-connected decoder from \"β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK\"
    Args:
        z_dim: latent factor dimension
        outplanes: number of output channels
        output_size: (window) size of the output tensor
    Returns:
        output tensor of size (batch_size, outplanes, output_size, output_size)
    """
    return nn.Sequential(
        nn.Linear(z_dim, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, 1200),
        nn.Tanh(),
        nn.Linear(1200, outplanes*output_size**2),
        Reshape((-1, outplanes, output_size, output_size)),
    )

def conv_encoder(z_dim, inplanes=3, input_size=64):
    """Convolutional encoder from \"β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK\"
    Args:
        z_dim: latent factor dimension
        inplanes: number of input channels
        input_size: (window) size of the input tensor of shape (batch_size, inplanes, input_size, input_size)
    Returns:
        output tensor of size 2*z_dim, where [:z_dim] are the latent variable means and [:z_dim] are the latent variable log variances
    """
    return nn.Sequential(
        nn.Conv2d(inplanes, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(64*(input_size//16)**2, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 2*z_dim),
    )

def deconv_decoder(z_dim, outplanes=3, output_size=64):
    """Convolutional decoder from \"β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK\"
    Args:
        z_dim: latent factor dimension
        outplanes: number of output channels
        output_size: (window) size of the output tensor
    Returns:
        output tensor of size (batch_size, outplanes, output_size, output_size)
    """
    return nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 64*(output_size//16)**2),
        nn.ReLU(inplace=True),
        Reshape((-1, 64, (output_size//16), (output_size//16))),
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, outplanes, kernel_size=4, stride=2, padding=1),
        Reshape((-1, outplanes, output_size, output_size)),
    )

def conv_bn_encoder(z_dim, inplanes=3, input_size=64):
    """Convolutional encoder with Batch Normalization from \"Isolating Sources of Disentanglement in Variational Autoencoders\"
    Args:
        z_dim: latent factor dimension
        inplanes: number of input channels
        input_size: (window) size of the input tensor of shape (batch_size, inplanes, input_size, input_size)
    Returns:
        output tensor of size 2*z_dim, where [:z_dim] are the latent variable means and [:z_dim] are the latent variable log variances
    """
    return nn.Sequential(
        nn.Conv2d(inplanes, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 512, kernel_size=4),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 2*z_dim, kernel_size=1),
        Reshape((-1, 2*z_dim))
    )

def deconv_bn_decoder(z_dim, outplanes=3, output_size=64):
    """Convolutional decoder with Batch Normalization from \"Isolating Sources of Disentanglement in Variational Autoencoders\"
    Args:
        z_dim: latent factor dimension
        outplanes: number of output channels
        output_size: (window) size of the output tensor
    Returns:
        output tensor of size (batch_size, outplanes, output_size, output_size)
    """
    assert (output_size%16) == 0, "output size must be divisible by 16"
    return nn.Sequential(
        Reshape((-1, z_dim, 1, 1)),
        nn.ConvTranspose2d(z_dim, 512, kernel_size=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(512, 64, kernel_size=4),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, outplanes, kernel_size=4, stride=2, padding=1),
        Reshape((-1, outplanes, output_size, output_size)),
    )

def fc_discriminator(z_dim):
    """Fully connected discriminator
    Args:
        z_dim: latent factor dimension
    Returns:
        output tensor of shape (batch_size, 2) with logits
    """
    return nn.Sequential(
        nn.Linear(z_dim, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 1000),
        nn.LeakyReLU(inplace=True),
        nn.Linear(1000, 2),
    )
