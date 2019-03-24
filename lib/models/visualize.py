try:
    from visdom import Visdom
    from torchvision.utils import make_grid
    VISUALIZE = True
except:
    VISUALIZE = False

import torch

class Visualizer(object):
    def __init__(self, name):
        self.visdom = Visdom()
        self.name = name
        self.plots = dict()
        self.last_iter = 0
    
    def show_reconstructions(self, images, reconstructions, iter_n=""):
        """ Plots the ELBO loss, reconstruction loss and KL divergence
        Args:
            images: (torch.tensor)  of shape batch_size x 3 x size x size
            reconstructions: (torch.Tensor) of shape batch_size x 3 x size x size
            iter_n: (str) iteration at which plotting (OPTIONAL)
        """
        original = make_grid(images, normalize=True)
        reconstructed = make_grid(torch.sigmoid(reconstructions), normalize=True)
        self.visdom.images(torch.stack([original, reconstructed], dim=0).cpu(), env=self.name+"-reconstructed",
        opts={"title": iter_n}, nrow=8)
    
    def __init_plots(self, iter_n, elbo, reconstruction_loss, kl_loss):
        self.plots["elbo"] = self.visdom.line(torch.tensor([elbo]), X=torch.tensor([iter_n]),
        env=self.name+"-stats", opts={"title": "ELBO", "width": 600, "height": 500})
        self.plots["reconstruction_loss"] = self.visdom.line(torch.tensor([reconstruction_loss]), X=torch.tensor([iter_n]),
        env=self.name+"-stats", opts={"title": "Reconstruction loss", "width": 600, "height": 500})
        self.plots["kl_loss"] = self.visdom.line(torch.tensor([kl_loss]), X=torch.tensor([iter_n]),
        env=self.name+"-stats", opts={"title": "KL divergence", "width": 600, "height": 500})
    
    def plot_stats(self, iter_n, elbo, reconstruction_loss, kl_loss):
        """ Plots the ELBO loss, reconstruction loss and KL divergence
        Args:
            iter_n: (int) iteration at which plotting
            elbo: (int)
            reconstruction_loss: (int)
            kl_loss: (int)
        """
        # Initialize the plots
        if not self.plots:
            self.__init_plots(iter_n, elbo, reconstruction_loss, kl_loss)
            return
        self.plots["elbo"] = self.visdom.line(torch.tensor([elbo]), X=torch.tensor([iter_n]),
        win=self.plots["elbo"], update="append", env=self.name+"-stats",
        opts={"title": "ELBO", "width": 400, "height": 400})
        
        self.plots["reconstruction_loss"] = self.visdom.line(torch.tensor([reconstruction_loss]), X=torch.tensor([iter_n]),
        win=self.plots["reconstruction_loss"], update="append", env=self.name+"-stats",
        opts={"title": "Reconstruction Loss", "width": 400, "height": 400})
        
        self.plots["kl_loss"] = self.visdom.line(torch.tensor([kl_loss]), X=torch.tensor([iter_n]),
        win=self.plots["kl_loss"], update="append", env=self.name+"-stats",
        opts={"title": "KL Divergence", "width": 400, "height": 400})

