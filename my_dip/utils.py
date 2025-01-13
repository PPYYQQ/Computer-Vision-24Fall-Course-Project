import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL, dtype=np.float32) / 255.0

    if ar.ndim == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[np.newaxis, ...]

    return ar

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def get_image(img_path):
    """Load Image
    Args: 
        img_path: path to image
    """
    img = Image.open(img_path)
    img_np = pil_to_np(img)

    return img, img_np

def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: Image in np.array format with values from 0 to 1.
        sigma: Standard deviation of the Gaussian noise.
    
    Returns:
        img_noisy_pil: Noisy image in PIL format.
        img_noisy_np: Noisy image in np.array format.
    """
    noise = np.random.normal(scale=sigma, size=img_np.shape)
    img_noisy_np = np.clip(img_np + noise, 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return torch.from_numpy(img_np).unsqueeze(0)

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.squeeze(0).detach().cpu().numpy()

def fill_noise(tensor, noise_type='u'):
    """Fills tensor `tensor` with noise of type `noise_type`.

    Args:
        tensor: the tensor to fill with noise
        noise_type: 'u' for uniform noise, 'n' for normal noise
    """
    if noise_type == 'u':
        tensor.uniform_()
    elif noise_type == 'n':
        tensor.normal_()
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for filling tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplied by. Basically it is a standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var
            
    elif method == 'meshgrid': 
        assert input_depth == 2, "Input depth must be 2 for meshgrid method"
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), 
                           np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
        
    else:
        raise ValueError(f"Unsupported method: {method}")
        
    return net_input

def plot_image(tensor, filename):
    """
    Plots a tensor as an image and saves it to a file.

    Args:
        tensor (torch.Tensor): 4D tensor with shape (1, 3, H, W).
        filename (str): Path to save the image.
    """
    if tensor.ndim != 4 or tensor.size(1) != 3:
        raise ValueError("Expected a 4D tensor with three channels (C=3)")

    image = tensor[0].permute(1, 2, 0).cpu().numpy()
    image = np.clip(image, 0, 1)
    plt.imsave(filename, image)


def load_LR_HR_imgs_sr(fname, factor, enforce_div32=None):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        factor: downscaling factor
        enforce_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname)
        
    # For comparison with GT
    if enforce_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

        bbox = [
                (img_orig_pil.size[0] - new_size[0]) // 2, 
                (img_orig_pil.size[1] - new_size[1]) // 2,
                (img_orig_pil.size[0] + new_size[0]) // 2,
                (img_orig_pil.size[1] + new_size[1]) // 2,
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np

    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    img_LR_pil = img_HR_pil.resize(LR_size, Image.LANCZOS)
    img_LR_np = pil_to_np(img_LR_pil)

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str(img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }


def get_baselines(img_LR_pil, img_HR_pil):
    """
    Gets `bicubic`, sharpened bicubic, and `nearest` baselines.

    Args:
        img_LR_pil (PIL.Image): Low-resolution image.
        img_HR_pil (PIL.Image): High-resolution image.

    Returns:
        tuple: A tuple containing numpy arrays of bicubic, sharpened bicubic, and nearest baselines.
    """
    img_bicubic_pil = img_LR_pil.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_bic_sharp_np, img_nearest_np


def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))

def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out

def crop_image(img, d=32):
    """
    Crop the image so that its dimensions are divisible by `d`.

    Args:
        img (PIL.Image): Input image.
        d (int): Divisor to make dimensions divisible by.

    Returns:
        PIL.Image: Cropped image.
    """
    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
        (img.size[0] - new_size[0]) // 2, 
        (img.size[1] - new_size[1]) // 2,
        (img.size[0] + new_size[0]) // 2,
        (img.size[1] + new_size[1]) // 2,
    ]

    img_cropped = img.crop(bbox)
    return img_cropped