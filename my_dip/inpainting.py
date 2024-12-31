from __future__ import print_function
import matplotlib.pyplot as plt

import os

import numpy as np
from skip import skip
import torch
import torch.optim as optim
import torch.nn as nn
from utils import get_image, get_noise, np_to_torch, get_noisy_image, plot_image, pil_to_np, crop_image


dtype = torch.cuda.FloatTensor

# PLOT = True
# imsize = -1
# dim_div_by = 64


## Fig 6
# img_path  = 'data/inpainting/vase.png'
# mask_path = 'data/inpainting/vase_mask.png'

## Fig 8
# img_path  = 'data/inpainting/library.png'
# mask_path = 'data/inpainting/library_mask.png'

## Fig 7 (top)
# img_path  = 'data/inpainting/kate.png'
# mask_path = 'data/inpainting/kate_mask.png'

# Another text inpainting example
# img_path  = 'data/inpainting/peppers.png'
# mask_path = 'data/inpainting/peppers_mask.png'

NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet

img_list = [
    'data/inpainting/vase.png',
    'data/inpainting/library.png',
    'data/inpainting/kate.png',
]
mask_list = [
    'data/inpainting/vase_mask.png',
    'data/inpainting/library_mask.png',
    'data/inpainting/kate_mask.png',
]
device = 'cuda:1'

for img_idx in range(len(img_list)):
    img_path = img_list[img_idx]
    mask_path = mask_list[img_idx]

    image_name = img_path.split('/')[-1].split('.')[0]
    task_name = 'inpaint'

    img_pil = crop_image(get_image(img_path)[0], d=64)
    img_mask_pil = crop_image(get_image(mask_path)[0], d=64)

    img_np      = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)


    if 'vase.png' in img_path:
        INPUT = 'meshgrid'
        input_depth = 2
        LR = 0.01 
        num_iter = 5001
        param_noise = False
        save_iter = 50
        reg_noise_std = 0.03
        
        net = skip(input_depth, img_np.shape[0], 
                num_channels_down = [128] * 5,
                num_channels_up   = [128] * 5,
                num_channels_skip = [0] * 5,  
                upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
        
    elif ('kate.png' in img_path) or ('peppers.png' in img_path):
        # Same params and net as in super-resolution and denoising
        INPUT = 'noise'
        input_depth = 32
        LR = 0.01 
        num_iter = 6001
        param_noise = False
        save_iter = 50
        reg_noise_std = 0.03
        
        net = skip(input_depth, img_np.shape[0], 
                num_channels_down = [128] * 5,
                num_channels_up =   [128] * 5,
                num_channels_skip = [128] * 5,  
                filter_size_up = 3, filter_size_down = 3, 
                upsample_mode='nearest', filter_skip_size=1,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
        
    elif 'library.png' in img_path:
        
        INPUT = 'noise'
        input_depth = 1
        
        num_iter = 3001
        save_iter = 50
        reg_noise_std = 0.00
        param_noise = True
        
        depth = int(NET_TYPE[-1])
        net = skip(input_depth, img_np.shape[0], 
                num_channels_down = [16, 32, 64, 128, 128, 128][:depth],
                num_channels_up =   [16, 32, 64, 128, 128, 128][:depth],
                num_channels_skip = [0,  0,  0,  0,   0,   0][:depth],  
                filter_size_up = 3,filter_size_down = 5,  filter_skip_size=1,
                upsample_mode='nearest',
                need1x1_up=False,
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
        
        LR = 0.01 
            
    else:
        assert False

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)


    # Compute number of parameters
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)


    img_var = np_to_torch(img_np).type(dtype).to(device)
    mask_var = np_to_torch(img_mask_np).type(dtype).to(device)


    # Parameters
    LR = 0.01

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # log file
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f'{task_name}_{image_name}_log.txt')
    log_file = open(log_file_path, 'a')

    # Load everything to GPU
    net = net.type(dtype).to(device)
    net_input_saved = net_input_saved.to(device)
    noise = noise.to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for i in range(num_iter):
        optimizer.zero_grad()

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        
        out = net(net_input)    
    
        total_loss = criterion(out * mask_var, img_var * mask_var)
        total_loss.backward()
        log_message = ('Iteration %05d    Loss %f \n' % 
                (i, total_loss.item(),))
        print(log_message, '\r', end='')
        log_file.write(log_message)

        optimizer.step()

        if i % save_iter==0:
            try: 
                os.mkdir(f'./outputs/{task_name}/{image_name}')
            except:
                pass
            plot_image(out.detach().cpu(), f'./outputs/{task_name}/{image_name}/{i:04d}.jpg')
