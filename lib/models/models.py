import torch
import torch.nn as nn
import math

import lib.models.losses as losses
import lib.models.model_parts as model_parts

class VAE(nn.Module):
    """Base model architecture"""
    def __init__(self, z_dim, inplanes=3, input_size=64,
    encoder_fn=model_parts.FC_ENCODER, decoder_fn=model_parts.FC_DECODER):
        super(VAE, self).__init__()
        self.normalization = torch.tensor(math.log(2*math.pi))
        self.z_dim = z_dim
        
        if encoder_fn == model_parts.FC_ENCODER:
            self.encoder = model_parts.fc_encoder(z_dim, inplanes, input_size)
        elif encoder_fn == model_parts.CONV_ENCODER:
            self.encoder = model_parts.conv_encoder(z_dim, inplanes, input_size)
        elif encoder_fn == model_parts.CONV_BN_ENCODER:
            self.encoder = model_parts.conv_bn_encoder(z_dim, inplanes, input_size)
        else:
            raise ValueError("unknown encoder function")
        
        if decoder_fn == model_parts.FC_DECODER:
            self.decoder = model_parts.fc_decoder(z_dim, inplanes, input_size)
        elif decoder_fn == model_parts.DECONV_DECODER:
            self.decoder = model_parts.deconv_decoder(z_dim, inplanes, input_size)
        elif decoder_fn == model_parts.DECONV_BN_DECODER:
            self.decoder = model_parts.deconv_bn_decoder(z_dim, inplanes, input_size)
        else:
            raise ValueError("unknown decoder function")

        for block in self._modules:
            for m in self._modules[block]:
                model_parts.kaiming_init(m)
    
    def sample_from_latent(self, mu, logvar):
        """Reparametrization"""
        std = logvar.div(2).exp()
        eps = torch.empty_like(std).normal_()
        return mu.add(std.mul(eps))
    
    def regularizer(self, kl_loss, *args):
        return kl_loss
    
    def forward(self, x):
        mu, logvar = self.encoder(x).split(self.z_dim, dim=1)
        z = self.sample_from_latent(mu, logvar)
        reconstructions = self.decoder(z)
        return reconstructions, z, mu, logvar
    
    def forward_with_elbo(self, x, config):
        mu, logvar = self.encoder(x).split(self.z_dim, dim=1)
        z = self.sample_from_latent(mu, logvar)
        reconstructions = self.decoder(z)
        kl_loss = losses.kl_divergence(mu, logvar)
        regularizer = self.regularizer(kl_loss, mu, logvar, z)
        reconstruction_loss = losses.reconstruction_loss(reconstructions, x, config).mean()
        elbo = reconstruction_loss.add(kl_loss)
        loss = reconstruction_loss.add(regularizer)
        return reconstructions, z, mu, logvar, reconstruction_loss, loss, -elbo, kl_loss

class BetaVAE(VAE):
    """Model architecture as proposed in \"β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK\""""
    def __init__(self, z_dim, beta, inplanes=3, input_size=64,
    encoder_fn=model_parts.FC_ENCODER, decoder_fn=model_parts.FC_DECODER):
        super(BetaVAE, self).__init__(z_dim, inplanes=inplanes, input_size=input_size,
        encoder_fn=encoder_fn, decoder_fn=decoder_fn)
        self.beta = beta
    
    def regularizer(self, kl_loss, *args):
        return self.beta * kl_loss

class TCVAE(BetaVAE):
    """Model architecture as proposed in \"Isolating Sources of Disentanglement in Variational Autoencoders\""""
    def log_density(mu, logvar, z):
        inv_sigma = torch.exp(-logvar)
        return ((z.mul(mu)).pow(2).mul(inv_sigma).add(logvar).add_(self.normalization)).mul_(-0.5)

    def total_correlation(mu, logvar, z):
        log_qz_prob = self.log_density(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))
        log_qz_prod = torch.logsumexp(log_qz_prob, dim=1).sum(dim=1)
        log_qz = torch.sum(log_qz_prob, dim=2).logsumexp(dim=1)
        return torch.mean(log_qz - log_qz_prod)

    def regularizer(self, kl_loss, mu, logvar, z):
        return (self.beta - 1) * self.total_correlation(mu, logvar, z) + kl_loss

