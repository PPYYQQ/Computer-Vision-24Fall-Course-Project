import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skip import skip
from tqdm import tqdm

from utils import get_image, get_noise, np_to_torch, get_noisy_image, plot_image

dtype = torch.cuda.FloatTensor

sigma = 25
sigma_ = sigma/255.

# deJPEG 
# fname = 'data/denoising/snail.jpg'

## denoising
fname = 'data/denoising/F16_GT.png'

img_pil, img_np = get_image(fname)

if fname == 'data/denoising/snail.jpg':
    img_noisy_pil, img_noisy_np  = img_pil, img_np 
        
elif fname == 'data/denoising/F16_GT.png':
    # Add synthetic noise
    img_noisy_pil, img_noisy_np = get_noisy_image(img_np, sigma_)
    
else:
    assert False

# Get the model
if fname == 'data/denoising/snail.jpg':
    num_iter = 2400
    input_depth = 3

    net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear', downsample_mode='stride',
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')


elif fname == 'data/denoising/F16_GT.png':
    num_iter = 3000
    input_depth = 32 

    net = skip(
                input_depth, num_output_channels = 3, 
                num_channels_down = [128]*5,
                num_channels_up =   [128]*5,
                num_channels_skip = [128]*5, 
                upsample_mode='bilinear', downsample_mode='stride',
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
else:
    assert False
    
net_input = get_noise(input_depth, 'noise', (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

# Training

# Parameter
device = 'cuda:0'
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
log_file_path = os.path.join(log_dir, 'denoise_training_log.txt')
log_file = open(log_file_path, 'a')


net = net.type(dtype).to(device)
net_input_saved = net_input_saved.to(device)
noise = noise.to(device)

optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.MSELoss()

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
            
    total_loss = criterion(out, img_noisy_torch)
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
        plot_image(out.detach().cpu(), f'./outputs/denoise/{i:04d}.jpg')

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
