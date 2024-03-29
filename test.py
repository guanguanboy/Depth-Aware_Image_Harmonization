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
from util.visualizer import Visualizer
from util import html,util
from util.visualizer import save_images
from torch.utils.data import DataLoader
from data.ihd_dataset import IhdDataset
from retinexltifpm_model import retinexltifpmModel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import ntpath

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def test():

    RNG_SEED = 1
    NUM_SHARDS = 1
    SHARD_ID = 0
    NUM_GPUS = 1
    batch_size = 1
    epoch_count = 1
    niter = 50
    niter_decay = 50
    print_freq = 100
    display_freq = 100
    display_id = 0
    update_html_freq = 1000
    save_epoch_freq = 5
    save_iter_model = False
    save_latest_freq = 5000
    save_by_iter = False
    lr_decay_iters = 50
    continue_train = False
    results_dir = './results/'
    name = 'retinexltifpm_allihd'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    results_save_dir = os.path.join(results_dir, name)
    if not os.path.exists(results_save_dir):
        os.mkdir(results_save_dir)

    test_set = IhdDataset(dataset_root='../IntrinsicHarmony/datasets/HVIDIT/', dataset_name='HVIDIT', isTrain=False, crop_size=256)
    sampler = DistributedSampler(test_set) if NUM_GPUS > 1 else None
    dataset = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, sampler=sampler)

    cur_device = torch.cuda.current_device()
    device = DEVICE
    print('cur_device=', cur_device)    
    model = retinexltifpmModel(device, False)

    model.setup('linear', epoch_count, niter, niter_decay, lr_decay_iters, continue_train)               # regular setup: load and print networks; create schedulers

    # create a website
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    ismaster = du.is_master_proc(NUM_GPUS)

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        #visuals = model.get_current_visuals()  # get image results
        
        img_path = model.get_image_paths()     # get image paths # Added by Mia
        if i % 5 == 0 and ismaster:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))

        #print(len(img_path))

        short_path = ntpath.basename(img_path[0]) #v0000_1_0.jpg
        name = os.path.splitext(short_path)[0] #v0000_1_0

        visuals = OrderedDict()
        visuals['harmonized'] = model.harmonized
        visuals['real'] = model.real
        visuals['mask'] = model.mask
        visuals['illumination'] = model.illumination
        visuals['reflectance'] = model.reflectance

        for label, im_data in visuals.items():
            im = util.tensor2im(im_data) #将tensor转换为图像
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(results_save_dir, image_name)
            util.save_image(im, save_path, aspect_ratio=1.0)


if __name__ == '__main__':
    test()