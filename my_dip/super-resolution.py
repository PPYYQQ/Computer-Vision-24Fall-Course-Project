import matplotlib.pyplot as plt
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skip import skip
from tqdm import tqdm

from utils import get_image, get_noise, np_to_torch, torch_to_np, plot_image, load_LR_HR_imgs_sr, get_baselines, tv_loss, put_in_center
from downsampler import Downsampler

dtype = torch.cuda.FloatTensor

factor = 4 # 8

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
path_to_image = 'data/sr/zebra_GT.png'
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)

# Starts here
imgs = load_LR_HR_imgs_sr(path_to_image , factor, enforse_div32)

imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])

print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                    compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                    compare_psnr(imgs['HR_np'], imgs['nearest_np'])))


input_depth = 32
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'


OPTIMIZER = 'adam'

if factor == 4: 
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'


net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

net = skip(
            input_depth,
            num_channels_down = [128]*5, 
            num_channels_up   = [128]*5, 
            num_channels_skip = [4]*5, 
            upsample_mode='bilinear', downsample_mode='stride',
            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')


img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)


log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file_path = os.path.join(log_dir, 'superresolution_training_log.txt')
log_file = open(log_file_path, 'a')


psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

# Parameters
device = 'cuda:0'
LR = 0.01
tv_weight = 0.0

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

# Load everything to GPU
net = net.type(dtype).to(device)
net_input_saved = net_input_saved.to(device)
noise = noise.to(device)

optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.MSELoss()

for i in range(num_iter):
    optimizer.zero_grad()

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out_HR = net(net_input)
    out_LR = downsampler(out_HR)

    total_loss = criterion(out_LR, img_LR_var) 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    optimizer.step()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
    # print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
    log_message = ('Iteration %05d    Loss %f   PSNR_LR %.3f   PSNR_HR %.3f\n' % 
               (i, total_loss.item(), psnr_LR, psnr_HR))
    print(log_message, '\r', end='')
    log_file.write(log_message)
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if i % 100 == 0:
        out_HR_np = torch_to_np(out_HR)


out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image(torch.tensor([imgs['HR_np']]), filename=f"./outputs/super/super_HR_np.jpg")
plot_image(torch.tensor([imgs['bicubic_np']]), filename=f"./outputs/super/super_bicubic_np.jpg")
plot_image(torch.tensor([out_HR_np]), filename=f"./outputs/super/super_output.jpg")
