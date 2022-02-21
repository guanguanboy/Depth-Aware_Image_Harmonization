import os
import numpy as np
from sympy import im
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

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    # sampler = (
    #     loader.batch_sampler.sampler
    #     if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
    #     else loader.sampler
    # )
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

def print_current_losses(epoch, iters, losses, t_comp, t_data):
    """print current losses on console; also save the losses to the disk

    Parameters:
        epoch (int) -- current epoch
        iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
        losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
        t_comp (float) -- computational time per data point (normalized by batch_size)
        t_data (float) -- data loading time per data point (normalized by batch_size)
    """
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)

    print(message)  # print the message
    #with open(self.log_name, "a") as log_file:
        #log_file.write('%s\n' % message)  # save the message

def train():
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

    #init
    du.init_distributed_training(NUM_GPUS, SHARD_ID)
    # Set random seed from configs.
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    #init dataset
    #dataset = create_dataset(cfg)  # create a dataset given cfg.dataset_mode and other options
    test_set = IhdDataset(dataset_root='../IntrinsicHarmony/datasets/HVIDIT/', dataset_name='HVIDIT', isTrain=True, crop_size=256)
    sampler = DistributedSampler(test_set) if NUM_GPUS > 1 else None
    dataset = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    #model = create_model(cfg)      # create a model given cfg.model and other options
    cur_device = torch.cuda.current_device()
    device = DEVICE
    print('cur_device=', cur_device)    
    save_dir = os.path.join('./checkpoints', 'retinexltifpm_allihd')  # save all the checkpoints to save_dir
    model = retinexltifpmModel(cur_device, True)

    model.setup('linear', epoch_count, niter, niter_decay, lr_decay_iters, continue_train)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(cfg)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    # cur_device = torch.cuda.current_device()
    is_master = du.is_master_proc(NUM_GPUS)
    for epoch in range(epoch_count, niter + niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        if is_master:
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #shuffle_dataset(dataset, epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            if is_master:
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                    iter_data_time = time.time()
            #visualizer.reset()
            total_iters += batch_size
            epoch_iter += batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights 网络的前向传播,计算loss函数

            if total_iters % display_freq == 0 and is_master:   # display images on visdom and save images to a HTML file
                save_result = total_iters % update_html_freq == 0
                #model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            losses = model.get_current_losses()
            if NUM_GPUS > 1:
                losses = du.all_reduce(losses)
            if total_iters % print_freq == 0 and is_master:    # print training losses and save logging information to the disk
                t_comp = (time.time() - iter_start_time) / batch_size
                print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #if display_id > 0:
                #    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
            if total_iters % save_latest_freq == 0 and is_master:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if save_by_iter else 'latest'
                model.save_networks(save_suffix, save_dir)
                
        if epoch % save_epoch_freq == 0 and is_master:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest', save_dir)
            if save_iter_model and epoch>=80:
                model.save_networks(epoch, save_dir)
        if is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, niter + niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate('linear')                     # update learning rates at the end of every epoch.

if __name__ == '__main__':
    train()