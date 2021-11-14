import sys
sys.path.append('/workspace/stylegan3encoder/')
import copy
import os
import os.path as osp
import re
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import legacy
from face_alignment import image_align
from landmarks_detector import LandmarksDetector


def factorize_weight(generator):
    layers = [
#        'generator.synthesis.input.affine',
        'generator.synthesis.L0_36_512.affine',
        'generator.synthesis.L1_36_512.affine',
#        'generator.synthesis.L2_52_512.affine',
#        'generator.synthesis.L3_52_512.affine',
#        'generator.synthesis.L4_84_512.affine',
#        'generator.synthesis.L5_148_512.affine',
#        'generator.synthesis.L6_148_512.affine',
#        'generator.synthesis.L7_276_323.affine',
#        'generator.synthesis.L8_276_203.affine', #        'generator.synthesis.L9_532_128.affine',
#        'generator.synthesis.L10_1044_81.affine',
#        'generator.synthesis.L11_1044_51.affine',
#        'generator.synthesis.L12_1044_32.affine',
#        'generator.synthesis.L13_1024_32.affine',
#        'generator.synthesis.L14_1024_3.affine',
    ]
    weights = list()
#    for layer in layers:
#        w = eval(layer).weight.cpu().detach().numpy()
#        weights.append(w)
#        print(w.shape)
    scope = locals()
    weights = [eval(layer, scope).weight.cpu().detach().numpy() for layer in layers]

    weight = np.concatenate(weights, axis=1).astype(np.float32)
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return layers, eigen_vectors.T, eigen_values


if __name__ == '__main__':
    # set device
    device = torch.device('cuda:3')
    # set model paths
    ckptbasedir = '/workspace/stylegan3encoder/ckpts/'
    network_pkl = osp.join(ckptbasedir, 'stylegan3-t-ffhq-1024x1024.pkl')
    landmark_ckpt_path = osp.join(ckptbasedir, 'dlib', 'shape_predictor_68_face_landmarks.dat')
    # load models
    landmark_detector = LandmarksDetector(landmark_ckpt_path)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    distances = np.linspace(-30,30,11)
    layers, boundaries, values = factorize_weight(G)

    # generate random W
    seed = 1
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    
    label = torch.zeros([1,G.c_dim],device=device)
    w = G.mapping(z,label, truncation_psi=0.7)
    codes = w.detach().cpu().numpy()

    imgs = []
    semantic_ind = 0
    boundary = boundaries[semantic_ind:semantic_ind+1]
    for dist in distances:
        temp_code = copy.deepcopy(codes)
        temp_code += boundary * dist
        temp_code[:,[1,2],:] += boundary * dist
        img = G.synthesis(torch.from_numpy(temp_code).to(device), noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        imgs.append(PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')) #.save(f'{outdir}/{fn}.png')

    
    gw, gh = (11, 1)
    H, W = (1024,1024)
    result = np.stack(imgs)
    result = result.reshape(gh, gw, H, W, 3)
    result = result.transpose(0,2,1,3,4)
    result = result.reshape(gh * H, gw * W, 3)
    PIL.Image.fromarray(result, 'RGB').save(f'sefa/seed{str(seed).zfill(5)}-semantic{str(semantic_ind).zfill(5)}-result.png')



