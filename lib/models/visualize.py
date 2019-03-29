try:
    from visdom import Visdom
    import torchvision.utils
    VISUALIZE = True
except:
    VISUALIZE = False

import os
import torch

class Visualizer(object):
    def __init__(self, name, save=True, output_dir="."):
        self.visdom = Visdom()
        self.name = name
        self.plots = dict()
        self.save = save
        if not os.path.exists(output_dir):
            raise ValueError("output_dir does not exists")
        
        # output directory for reconstructions
        self.recon_dir = os.path.join(output_dir, "reconstructions")
        if not os.path.exists(self.recon_dir):
            os.mkdir(self.recon_dir)
        
        # output directory for traversals
        self.trav_dir = os.path.join(output_dir, "traversals")
        if not os.path.exists(self.trav_dir):
            os.mkdir(self.trav_dir)
        

    def traverse(self, decoder, latent_vector, dims=None, num_traversals=None, iter_n=""):
        """ Traverses a latent vector along a given dimension(s).
        Args:
            decoder: (torch.nn.Module) decoder model that generates
                the reconstructions from a latent vector 
            latent_vector: (torch.tensor) latent vector representation to be traversed
                of shape (z_dim)
            dims: (list, range or torch.tensor) list of dimensions to traverse in the latent vector
                (optional)
            num_traversals: (int) how many reconstructions to generate for each dimension.
                The image grid will be of shape: len(dims) x num_traversals
            iter_n: (str) iteration at which plotting and/or image index (OPTIONAL)
        """

        if dims is None:
            dims = torch.arange(latent_vector.size(0))
        elif not (isinstance(dims, list) or isinstance(dims, range) or isinstance(dims, torch.tensor)):
            raise ValueError(f"dims must either be a list or a torch.tensor, received {type(dims)}")
        
        if num_traversals is None:
            num_traversals = latent_vector.size(0)
        elif not isinstance(num_traversals, int):
            raise ValueError(f"num_traversals must either be an int, received {type(num_traversals)}")
        
        traversals = torch.linspace(-1., 1., steps=num_traversals).to(latent_vector.device)

        reconstructions = []
        for dim in dims:
            tiles = latent_vector.repeat(num_traversals, 1)
            tiles[:, dim] = traversals
            dim_recon = decoder(tiles)
            reconstructions.append(dim_recon)
        reconstructions = torch.sigmoid(torch.cat(reconstructions, dim=0))
        reconstructed = torchvision.utils.make_grid(reconstructions, normalize=True, nrow=len(dims))
        self.visdom.images(reconstructed.cpu(), env=self.name+"-traversals",
        opts={"title": iter_n}, nrow=len(dims))

        if self.save:
            torchvision.utils.save_image(reconstructions, os.path.join(self.trav_dir, f"traversals-{iter_n}.png"), normalize=True, nrow=len(dims))

    
    def show_reconstructions(self, images, reconstructions, iter_n=""):
        """ Plots the ELBO loss, reconstruction loss and KL divergence
        Args:
            images: (torch.tensor)  of shape batch_size x 3 x size x size
            reconstructions: (torch.Tensor) of shape batch_size x 3 x size x size
            iter_n: (str) iteration at which plotting (OPTIONAL)
        """
        original = torchvision.utils.make_grid(images, normalize=True)
        reconstructed = torchvision.utils.make_grid(torch.sigmoid(reconstructions), normalize=True)
        self.visdom.images(torch.stack([original, reconstructed], dim=0).cpu(), env=self.name+"-reconstructed",
        opts={"title": iter_n}, nrow=8)

        if self.save:
            torchvision.utils.save_image(original, os.path.join(self.recon_dir, f"original-{iter_n}.png"), normalize=True)
            torchvision.utils.save_image(reconstructions, os.path.join(self.recon_dir, f"reconstructed-{iter_n}.png"), normalize=True)
    
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
        opts={"title": "ELBO", "width": 600, "height": 500})
        
        self.plots["reconstruction_loss"] = self.visdom.line(torch.tensor([reconstruction_loss]), X=torch.tensor([iter_n]),
        win=self.plots["reconstruction_loss"], update="append", env=self.name+"-stats",
        opts={"title": "Reconstruction Loss", "width": 600, "height": 500})
        
        self.plots["kl_loss"] = self.visdom.line(torch.tensor([kl_loss]), X=torch.tensor([iter_n]),
        win=self.plots["kl_loss"], update="append", env=self.name+"-stats",
        opts={"title": "KL Divergence", "width": 600, "height": 500})

