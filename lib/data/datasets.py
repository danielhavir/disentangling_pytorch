import os
import multiprocessing as mp
from functools import partial

import torch
from torchvision.transforms import Normalize
import torchvision.transforms.functional as F
import numpy as np
import cv2
from PIL import Image

DSPRITES = 1
CELEBA = 2
CHAIRS = 3

def cv_loader(path, size):
    """ Loads 1000 images in ~7.41s """
    # OpenCV imread is faster than PIL
    img = cv2.imread(path)[:,:,::-1]
    img = np.array(F.resize(Image.fromarray(img), size))
    return img

class _Dset(torch.utils.data.Dataset):
    """Reference: https://github.com/google-research/disentanglement_lib"""
    def __init__(self, factor_sizes, latent_factor_indices):
        self.factor_sizes = factor_sizes
        self.num_factors = len(self.factor_sizes)
        self.latent_factor_indices = latent_factor_indices
        self.observation_factor_indices = [i for i in range(self.num_factors) if i not in self.latent_factor_indices]

    def sample_latent_factors(self, num):
        factors = np.zeros(shape=(num, len(self.latent_factor_indices)), dtype=np.int64)
        for pos, i in enumerate(self.latent_factor_indices):
            factors[:, pos] = self._sample_factor(i, num)
        return factors
    
    def sample_all_factors(self, latent_factors):
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self.latent_factor_indices] = latent_factors
        for i in self.observation_factor_indices:
            all_factors[:, i] = self._sample_factor(i, num_samples)
        return all_factors

    def _sample_factor(self, i, num):
        return np.random.randint(self.factor_sizes[i], size=num)

class DSprites(_Dset):
    """
    The ground-truth factors of variation are (in the default setting):
    0 - shape (3 different values)
    1 - scale (6 different values)
    2 - orientation (40 different values)
    3 - position x (32 different values)
    4 - position y (32 different values)
    """
    def __init__(self, data_dir, latent_factor_indices=[0, 1, 2, 3, 4, 5]):
        dataset_zip = np.load(os.path.join(data_dir, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"), encoding="bytes")
        self.data = dataset_zip["imgs"]
        self.latents_values = dataset_zip["latents_values"]
        self.latents_classes = dataset_zip["latents_classes"]
        self.metadata = dataset_zip["metadata"][()]
        super(DSprites, self).__init__(self.metadata[b"latents_sizes"], latent_factor_indices)
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.inplanes = 1
        self.window_size = self.data.shape[1]
        self.normalize = Normalize([self.data.mean()], [self.data.std()])
    
    def __len__(self):
        return len(self.data)

    @property
    def observation_shape(self):
        return self.data.shape
    
    def sample_observations(self, factors):
        all_factors = self.sample_all_factors(factors)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return torch.from_numpy(self.data[indices]).unsqueeze(1).float()
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx]).unsqueeze(0).float()
        #return self.normalize(img)
        return img

class ImageDataset(_Dset):
    """ Loads images into memory in advance
    
    The ground-truth factors of variation for chairs dataset are (in the default setting):
    0 - width/size 
    1 - azimuth (24 different values)
    2 - leg style
    3 - back height

    The ground-truth factors of variation for celebA dataset are (in the default setting):
    0 - background
    1 - skin color
    2 - age/gender
    3 - azimuth (24 different values)
    4 - hair parting
    5 - fringe
    6 - sunglasses / smile
    7 - saturation
    """
    def __init__(self, data_dir, num_workers=10, latent_factor_indices=[0, 1, 2, 3, 4, 5, 6, 7]):
        data_dir = os.path.join(data_dir, "images")
        self.paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
        pool = mp.Pool(processes=num_workers)
        loader_fn = partial(cv_loader, size=(64, 64))
        self.data = pool.map(loader_fn, self.paths)
        if "celeb" in data_dir:
            self.normalize = Normalize([0.50611186, 0.42542528, 0.3828167], [0.30415685, 0.28379978, 0.28333236])
        #    super(ImageDataset, self).__init__([10, 20, 100, 24, 10, 10, 20, 32], latent_factor_indices)
        elif "chairs" in data_dir:
            self.normalize = Normalize([0.9545367, 0.95292854, 0.9517893], [0.17828687, 0.18361006, 0.18755156])
        #    super(ImageDataset, self).__init__([32, 24, 10, 10], latent_factor_indices)
        #self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.inplanes = 3
        self.window_size = 64
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].transpose(2, 0, 1)
        img = torch.from_numpy(img).float().div_(255)
        return self.normalize(img)

class ImageFileDataset(_Dset):
    """ Reads images from file during training """
    def __init__(self, data_dir):
        data_dir = os.path.join(data_dir, "images")
        self.paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
        self.loader_fn = partial(cv_loader, size=(64, 64))
        if "celeb" in data_dir:
            self.normalize = Normalize([0.50611186, 0.42542528, 0.3828167], [0.30415685, 0.28379978, 0.28333236])
        #    super(ImageFileDataset, self).__init__([10, 20, 100, 24, 10, 10, 20, 32], latent_factor_indices)
        elif "chairs" in data_dir:
            self.normalize = Normalize([0.9545367, 0.95292854, 0.9517893], [0.0276624, 0.02844673, 0.02887262])
        #    super(ImageFileDataset, self).__init__([32, 24, 10, 10], latent_factor_indices)
        #self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes)
        self.inplanes = 3
        self.window_size = 64
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.loader_fn(self.paths[idx]).transpose(2, 0, 1)
        img = torch.from_numpy(img).float().div_(255)
        return self.normalize(img)

def get_traversal_indices(dataset_enum):
    """ Based on dataset int identifier (enum), returns a list of arbitrarily hand-picked
    indices for example traversing.
    """
    if dataset_enum == DSPRITES:
        return [575850, 80, 350000]
    elif dataset_enum == CELEBA:
        return [154273, 196477, 147234, 125868]
    elif dataset_enum == CHAIRS:
        return [81436, 28547, 23965]
    else:
        raise ValueError(f"unknown dataset_enum, received {dataset_enum}")
