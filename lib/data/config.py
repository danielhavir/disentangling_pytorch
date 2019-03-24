import os
import sys
import yaml
import io
from types import SimpleNamespace
from collections import namedtuple
import time
import logging

import lib.models.model_parts as model_parts
import lib.models.losses as losses

def load_config(path):
    with open(path, "r") as f:
        c = yaml.load(f)
    return c

def save_config(config):
    try:
        with open(os.path.join(config.RUN_DIR, "config.yaml"), "w") as f:
            yaml.dump(config.__dict__, f, default_flow_style=False)
    # Except applies when --no_snaps flag is passed
    except:
        pass

def setup_logger(config, no_snaps):
    logger = logging.getLogger()
    RESULTS_DIR = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s %(message)s")
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(logging.INFO)
    logger.addHandler(std_handler)

    if not no_snaps:
        RUN_DIR = os.path.join(RESULTS_DIR, time.strftime("%Y%m%d-%X"))
        if not os.path.exists(RUN_DIR):
            os.mkdir(RUN_DIR)
        file_handler = logging.FileHandler(os.path.join(RUN_DIR, "experiment.log"), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        config.RUN_DIR = RUN_DIR
    return config, logger

def get_config_logger(path, no_snaps=False):
    config = load_config(path)
    # Basename without file extension
    config["name"] = os.path.splitext(os.path.basename(path))[0]
    
    # Fill in the defaults where missing
    if not config.get("encoder", False):
        config["encoder"] = model_parts.FC_ENCODER
    if not config.get("decoder", False):
        config["decoder"] = model_parts.FC_DECODER
    if not config.get("rec_loss", False):
        config["rec_loss"] = losses.L2_LOSS
    if not config.get("activation", False):
        config["activation"] = "logits"
    if not config.get("beta", False):
        config["beta"] = 2.
    if not config.get("z_dim", False):
        config["z_dim"] = 10
    if not config.get("lr", False):
        config["lr"] = 1e-4
    if not config.get("batch_size", False):
        config["batch_size"] = 64
    if not config.get("image_size", False):
        config["image_size"] = 64
    if not config.get("dataset", False):
        config["dataset"] = "dsprites"
    if not config.get("max_iter", False):
        config["max_iter"] = 1e6
    if not config.get("data_path", False):
        config["data_path"] = os.environ["data"]
    config["tcvae"] = config.get("tcvae", False)
    if config["rec_loss"] == losses.BERNOULLI:
        config["subtract_entropy"] = config.get("subtract_entropy", False)
    if config["rec_loss"] == losses.L2_LOSS:
        config["reduction"] = config.get("reduction", "sum")
    
    config = SimpleNamespace(**config)
    config, logger = setup_logger(config, no_snaps)
    return config, logger

