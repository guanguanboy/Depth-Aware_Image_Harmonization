import MegaDepth

import torch
import sys
from torch.autograd import Variable
import numpy as np
from MegaDepth.models.models import create_model
from skimage import io
from skimage.transform import resize

def test_MegeDepth():
    print(MegaDepth.__dict__)

    mega_depth_model = MegaDepth.__dict__['HourGlass']("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
    print(mega_depth_model)
    print(MegaDepth.__dict__)

def test_simple():
    img_path = 'demo.jpg'

    model = MegaDepth.__dict__['HourGlass']("MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")

    input_height = 384
    input_width  = 512

    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.eval()

    img = np.float32(io.imread(img_path))/255.0
    img = resize(img, (input_height, input_width), order = 1)
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    input_img = input_img.unsqueeze(0)
    print('input_img.shape =', input_img.shape) #input_img.shape = torch.Size([1, 3, 384, 512])

    input_images = Variable(input_img.cuda() )
    pred_log_depth = model.forward(input_images) 
    print('pred_log_depth.shape =', pred_log_depth.shape) #torch.Size([1, 1, 384, 512])

    pred_log_depth = torch.squeeze(pred_log_depth)
    print('pred_log_depth_suqeeze.shape =', pred_log_depth.shape) #torch.Size([384, 512])

    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    io.imsave('demo.png', pred_inv_depth)
    # print(pred_inv_depth.shape)
    sys.exit()


if __name__ == '__main__':
    test_MegeDepth()
    test_simple()
