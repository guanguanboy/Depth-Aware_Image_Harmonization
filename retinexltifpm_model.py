from sympy import im
import torch
import torch.nn as nn

import os
import itertools
import torch.nn.functional as F
from util import distributed as du
from util import util
import harmony_networks as networks
import util.ssim as ssim
from collections import OrderedDict
import networks as networks_init

class retinexltifpmModel(nn.Module):
    
    def __init__(self, device, isTrain):
        super(retinexltifpmModel, self).__init__()

        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.isTrain = isTrain
        self.loss_names = ['G','G_L1','G_R_grident','G_I_L2','G_I_smooth',"IF"]
            
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['mask', 'harmonized','comp','real','reflectance','illumination','ifm_mean']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        self.device = device
        self.NUM_GPUS = 1
        self.netG = networks.define_G(device, self.NUM_GPUS, 'retinex_ltm_ifm', 'normal', 0.02)
        self.cur_device = torch.cuda.current_device()
        self.ismaster = du.is_master_proc(self.NUM_GPUS)
        self.lr = 0.0001 
        self.beta1 = 0.5
        self.lambda_ifm = 100
        self.lambda_I_L2 = 1.0
        self.lambda_I_smooth = 10
        self.lambda_L1 = 100.0
        self.lambda_R_gradient = 10.0
        self.metric = 0  
        self.optimizers = []

        if self.ismaster:
            print(self.netG)  
        
        if self.isTrain:
            #if self.ismaster == 0:
                #util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss().cuda(self.cur_device)
            self.criterionL2 = torch.nn.MSELoss().cuda(self.cur_device)
            self.criterionDSSIM_CS = ssim.DSSIM(mode='c_s').to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.inputs = input['inputs'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.image_paths = input['img_path']

        self.mask_r = F.interpolate(self.mask, size=[64,64])
        self.mask_r_32 = F.interpolate(self.mask, size=[32,32])
        self.real_r = F.interpolate(self.real, size=[32,32])
        self.real_gray = util.rgbtogray(self.real_r)
        
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.harmonized, self.reflectance, self.illumination, self.ifm_mean = self.netG(self.inputs, self.comp, self.mask_r, self.mask_r_32)
        if not self.isTrain:
            self.harmonized = self.comp*(1-self.mask) + self.harmonized*self.mask
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        self.loss_IF = self.criterionDSSIM_CS(self.ifm_mean, self.real_gray)*self.lambda_ifm

        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.lambda_L1
        self.loss_G_R_grident = self.gradient_loss(self.reflectance, self.real)*self.lambda_R_gradient
        self.loss_G_I_L2 = self.criterionL2(self.illumination, self.real)*self.lambda_I_L2
        self.loss_G_I_smooth = util.compute_smooth_loss(self.illumination)*self.lambda_I_smooth
        # assert 0
        self.loss_G = self.loss_G_L1 + self.loss_G_R_grident + self.loss_G_I_L2 + self.loss_G_I_smooth + self.loss_IF
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def gradient_loss(self, input_1, input_2):
        g_x = self.criterionL1(util.gradient(input_1, 'x'), util.gradient(input_2, 'x'))
        g_y = self.criterionL1(util.gradient(input_1, 'y'), util.gradient(input_2, 'y'))
        return g_x+g_y

    def setup(self, lr_policy, epoch_count, niter, niter_decay, lr_decay_iters, continue_train):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        verbose = False
        load_iter = 0
        epoch = 'latest'

        if self.isTrain:
            self.schedulers = [networks_init.get_scheduler(optimizer, lr_policy, epoch_count, niter, niter_decay, lr_decay_iters) for optimizer in self.optimizers]
        if not self.isTrain or continue_train:
            load_suffix = 'iter_%d' % load_iter if load_iter > 0 else epoch
            #self.load_networks(load_suffix)
        #self.print_networks(verbose)    

    def update_learning_rate(self, lr_policy):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        # losses = []
        for name in self.loss_names:
            if isinstance(name, str):
                # errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
                errors_ret[name] = getattr(self, 'loss_' + name)
        #         loss = getattr(self, 'loss_' + name)
        #         losses.append(loss)
        # loss_reduce = du.all_reduce(losses)
        # print(loss_reduce)
        # i = 0
        # for name in self.loss_names:
        #     if isinstance(name, str):
        #         errors_ret[name] = float(loss_reduce[i])
        #         i += 1
        return errors_ret

    def save_networks(self, epoch, save_dir):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        # if not du.is_master_proc(self.opt.NUM_GPUS * self.opt.NUM_SHARDS):
        #     return
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if torch.cuda.is_available():
                    if self.NUM_GPUS > 1:
                        torch.save(net.module.state_dict(), save_path)
                    else:
                        torch.save(net.state_dict(), save_path)
                else:
                    torch.save(net.cpu().state_dict(), save_path)


def test_model():
    model = retinexltifpmModel("cuda:0", False)
    #print(model)

if __name__ == '__main__':
    test_model()