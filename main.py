import os
import argparse
import torch

from lib.models.train import Experiment
from lib.data.config import get_config_logger

def main(args):
    config, logger = get_config_logger(args.config, no_snaps=args.no_snaps)
    exp = Experiment(config, logger, ckpt_path=args.checkpoint_path, multi_gpu=args.multi_gpu, eval_interval=args.eval_interval,
    no_snaps=args.no_snaps, visdom=(not args.no_visdom))
    exp.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disentangled lib PyTorch")

    parser.add_argument("config", type=str, help="Config file.")
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, help="Path to a checkpoint file (if resuming training).")
    parser.add_argument("--log_interval", type=int, default=2500, help="How often to log results.")
    parser.add_argument("--eval_interval", type=int, default=10000, help="How often to store checkpoints and plot reconstructions.")
    parser.add_argument("--multi_gpu", action="store_true", help="Flag whether to use all available GPUs.")
    parser.add_argument("--no_snaps", action="store_true", help="Flag whether to prevent from snapshots.")
    parser.add_argument("--no_visdom", action="store_true", help="Flag whether to prevent from plotting to visdom.")
    parser.add_argument("--gpu_device", type=int, default=None, help="ID of a GPU to use when multiple GPUs are available.")
    args = parser.parse_args()

    if args.gpu_device is not None:
        torch.cuda.set_device(args.gpu_device)
    
    main(args)

