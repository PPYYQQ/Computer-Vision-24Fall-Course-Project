from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skip import skip
from tqdm import tqdm

from utils import get_image, get_noise, np_to_torch, get_noisy_image, plot_image, crop_image, pil_to_np

dtype = torch.cuda.FloatTensor

def tv_loss(image):
    """Compute the Total Variation loss for an image."""
    # Shift one pixel and get the difference from the original image
    diff_i = torch.sum(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    diff_j = torch.sum(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return diff_i + diff_j

# Noise para
sigma = 25
sigma_ = sigma/255.

# deJPEG 
fname = 'data/denoising/snail.jpg'

## denoising
# fname = 'data/denoising/F16_GT.png'
img_list = [
    './data/denoising/ISO/DSC_8880.JPG',
    # './data/denoising/ISO/DSC_8881.JPG',
    # './data/denoising/ISO/DSC_8882.JPG',
    # './data/denoising/ISO/DSC_8883.JPG',
    # './data/denoising/ISO/DSC_8884.JPG',
    # './data/denoising/ISO/DSC_8885.JPG',
    # './data/denoising/ISO/DSC_8886.JPG',
    # './data/denoising/ISO/DSC_8887.JPG',
    # './data/denoising/ISO/DSC_8888.JPG',
    # './data/denoising/ISO/DSC_8889.JPG',
    # './data/denoising/herbitcrap.png',
    # "/home/yongqian/CV/CV-24Fall-Course-Project/my_dip/data/denoising/F16_GT.png"
    # 'data/denoising/snail.jpg'



]

for idx in range(len(img_list)):

    fname = img_list[idx]
    image_name = fname.split('/')[-1].split('.')[0]
    task_name = 'denoise'
    device = 'cuda:2'

    img_pil = crop_image(get_image(fname)[0], d=32)
    img_np = pil_to_np(img_pil)


    if fname == 'data/denoising/snail.jpg':
        img_noisy_pil, img_noisy_np  = img_pil, img_np    
        num_iter = 3000
        input_depth = 3
        net = skip(
                    input_depth, num_output_channels = 3, 
                    num_channels_down = [8, 16, 32, 64, 128], 
                    num_channels_up   = [8, 16, 32, 64, 128],
                    num_channels_skip = [0, 0, 0, 4, 4], 
                    upsample_mode='bilinear', downsample_mode='stride',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        
    elif 'data/denoising/F16_GT.png' in fname:
        # Add synthetic noise
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        num_iter = 3000
        input_depth = 32
        net = skip(
                    input_depth, num_output_channels = 3, 
                    num_channels_down = [128]*5,
                    num_channels_up =   [128]*5,
                    num_channels_skip = [128]*5, 
                    upsample_mode='bilinear', downsample_mode='stride',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        
    elif './data/denoising/ISO/' in fname:
        # print('i m here')
        # input()
        img_noisy_pil, img_noisy_np  = img_pil, img_np    
        num_iter = 3000
        input_depth = 32
        net = skip(
                    input_depth, num_output_channels = 3, 
                    num_channels_down = [128]*5,
                    num_channels_up =   [128]*5,
                    num_channels_skip = [128]*5, 
                    upsample_mode='bilinear', downsample_mode='stride',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        
    elif fname == './data/denoising/herbitcrap.png':
        # print('here')
        # input()
        # Add synthetic noise
        img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
        num_iter = 3000
        input_depth = 32
        net = skip(
                    input_depth, num_output_channels = 3, 
                    num_channels_down = [128]*5,
                    num_channels_up =   [128]*5,
                    num_channels_skip = [128]*5, 
                    upsample_mode='bilinear', downsample_mode='stride',
                    need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        

    net_input = get_noise(input_depth, 'noise', (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype).to(device)

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Training

    # Parameters
    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    show_every = 100
    exp_weight=0.99

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    out_avg = None
    last_net = None
    psrn_noisy_last = 0

    # log file
    save_iter = 100
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f'{task_name}_{image_name}_tv_log.txt')
    log_file = open(log_file_path, 'a')

    # Load everything to GPU
    net = net.type(dtype).to(device)
    net_input_saved = net_input_saved.to(device)
    noise = noise.to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    tv_weight = 1e-6
    for i in range(num_iter):
        optimizer.zero_grad()
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)
        
        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
                
        # Compute DIP loss
        loss_DIP = criterion(out, img_noisy_torch)
        loss_tv = tv_loss(out)
        # print(loss_DIP, tv_weight * loss_tv)
        if i > (num_iter / 3):
            tv_weight = 0
        # if i > 2000:
        #     tv_weight *= 0.1
        # print(loss_DIP, tv_weight, loss_tv)
        total_loss = loss_DIP + tv_weight * loss_tv
        
        total_loss.backward()
            
        optimizer.step()
        
        psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
        psrn_gt    = compare_psnr(img_np, out.detach().cpu().numpy()[0]) 
        psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0]) 
        
        log_message = ('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f\n' % 
                (i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm))
        print(log_message, '\r', end='')
        log_file.write(log_message)

        if i % save_iter==0:
            try: 
                os.mkdir(f'./outputs/{task_name}/{image_name}_tv')
            except:
                pass
            plot_image(out.detach().cpu(), f'./outputs/{task_name}/{image_name}_tv/{i:04d}.jpg')

        # Backtracking
        if i % show_every:
            if psrn_noisy - psrn_noisy_last < -5: 
                print('Falling back to previous checkpoint.')

                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.cuda())
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psrn_noisy
