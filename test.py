import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from util import distributed as du

import time
from collections import OrderedDict
from data import create_dataset
from data import shuffle_dataset
from models import create_model
from util.visualizer import Visualizer
from util import html,util
from util.visualizer import save_images

def test(cfg):
    dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    model = create_model(cfg)      # create a model given cfg.model and other options
    model.setup(cfg)               # regular setup: load and print networks; create schedulers

    # create a website
    web_dir = os.path.join(cfg.results_dir, cfg.name, '%s_%s' % (cfg.phase, cfg.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (cfg.name, cfg.phase, cfg.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if cfg.eval:
        model.eval()
    ismaster = du.is_master_proc(cfg.NUM_GPUS)

    fmse_score_list = []
    mse_scores = 0
    fmse_scores = 0
    num_image = 0
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        
        img_path = model.get_image_paths()     # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        visuals_ones = OrderedDict()
        harmonized = None
        real = None
        for j in range(len(img_path)):
            img_path_one = []
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data[j:j+1, :, :, :]
            img_path_one.append(img_path[j])
            save_images(webpage, visuals_ones, img_path_one, aspect_ratio=cfg.aspect_ratio, width=cfg.display_winsize)
            num_image += 1
            visuals_ones.clear()

    webpage.save()  # save the HTML


