import os
import logging
import torch
import torch.optim as optim

import lib.data.datasets as datasets
import lib.models.models as models
import lib.models.losses as losses
import lib.models.visualize as visualize
from lib.data.config import save_config

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        
    def avg(self):
        return self.sum / self.count

class Experiment(object):
    def __init__(self, config, logger, seed=42, multi_gpu=False, eval_interval=10000,
    log_interval=2500, visdom=True, no_snaps=False):
        """
        Args:
            config: SimpleNamespace with loaded configurations
            logger: logging.Logger instance
            seed: random seed
            multi_gpu: flag whether to train on all available GPUs
            eval_interval: interval for storring intermediate models and optionally visualizing reconstructions
            log_interval: interval for writing objective function results
            visdom: flag whether to visualize reconstructions
        """
        global visualize
        self.config = config
        self.logger = logger
        self.multi_gpu = multi_gpu
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.no_snaps = no_snaps
        self.set_seed(seed)
        if config.dataset == datasets.DSPRITES:
            self.dataset = datasets.DSprites(self.config.data_path)
        elif config.in_memory:
            # Load data into memory
            self.dataset = datasets.ImageDataset(self.config.data_path)
        else:
            self.dataset = datasets.ImageFileDataset(self.config.data_path)
        self.logger.info(f"Dataset {config.dataset} loaded")
        
        if self.config.tcvae:
            self.model = models.TCVAE(config.z_dim, config.beta, inplanes=self.dataset.inplanes,
            input_size=config.image_size, encoder_fn=config.encoder, decoder_fn=config.decoder)
        elif self.config.beta == 1:
            self.model = models.VAE(config.z_dim, inplanes=self.dataset.inplanes,
            input_size=config.image_size, encoder_fn=config.encoder, decoder_fn=config.decoder)
        else:
            self.model = models.BetaVAE(config.z_dim, config.beta, inplanes=self.dataset.inplanes,
            input_size=config.image_size, encoder_fn=config.encoder, decoder_fn=config.decoder)
        self.logger.info("Model built")
        print(self.model)

        if visualize.VISUALIZE and visdom:
            self.visualizer = visualize.Visualizer(self.config.name, save=False) if no_snaps else visualize.Visualizer(self.config.name, save=True, output_dir=config.RUN_DIR)
        else:
            self.visualizer = None
        
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        save_config(self.config)

    def set_seed(self, seed):
        self.logger.info(f"Setting seed {seed}")
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()

        torch.manual_seed(seed)
    
    def save_model(self, snap_fname):
        if not self.no_snaps:
            if self.multi_gpu:
                torch.save(self.model.module.state_dict(), os.path.join(self.config.RUN_DIR, snap_fname))
            else:
                torch.save(self.model.state_dict(), os.path.join(self.config.RUN_DIR, snap_fname))
    
    def train(self):
        global pbar
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elbo_loss = AverageMeter()
        rec_loss = AverageMeter()
        total_loss = AverageMeter()
        divergence_loss = AverageMeter()

        num_iter = 0
        for e in range(int(self.config.max_iter/len(loader))+1):
            for data in loader:
                num_iter += 1
                optimizer.zero_grad()
                data = data.to(device)
                reconstructions, z, mu, logvar, reconstruction_loss, loss, elbo, kl_loss = self.model.forward_with_elbo(data, self.config)

                loss.backward()
                optimizer.step()

                elbo_loss.update(elbo.item(), n=data.size(0))
                rec_loss.update(reconstruction_loss.item(), n=data.size(0))
                total_loss.update(loss.item(), n=data.size(0))
                divergence_loss.update(kl_loss.item(), n=data.size(0))

                if num_iter % self.log_interval == 0:
                    self.logger.info("Iter %d/%d ELBO %.2f Reconstruction loss %.2f Loss %.2f" % (num_iter, self.config.max_iter, elbo_loss.avg(), rec_loss.avg(), total_loss.avg()))

                    if self.visualizer is not None:
                        self.visualizer.plot_stats(num_iter, elbo_loss.avg(), rec_loss.avg(), divergence_loss.avg())

                if num_iter % self.eval_interval == 0:
                    self.save_model(f"iter_{num_iter}.model")

                    if self.visualizer is not None:
                        self.visualizer.show_reconstructions(data, reconstructions, iter_n=str(num_iter))
                        for index in datasets.get_traversal_indices(self.config.dataset):
                            latent_vector = self.model.encoder(self.dataset[index].to(device).unsqueeze(0))[0, :self.model.z_dim]
                            self.visualizer.traverse(self.model.decoder, latent_vector, iter_n=f"{num_iter}-{index}")
                        
                        self.visualizer.plot_means(z.cpu())
                
                if num_iter >= self.config.max_iter:
                    return

