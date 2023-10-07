import random

import numpy as np
import skimage.color as sc

import torch

def get_patch_npy(*args, patch_size=96, scale=2, nChans=1, norm=32767.0, input_large=False):
    '''
    :return: 从96个通道中随机取一个
    '''
    lr_im = args[0]
    hr_im = args[1]

    [lH, lW, lC] = lr_im.shape
    lh = np.random.randint(low=0, high=lH - patch_size + 1)
    lw = np.random.randint(low=0, high=lW - patch_size + 1)
    hh, hw = scale * lh, scale * lw
    hr_patch_size = patch_size * scale

    if nChans == 1:
        lc = np.random.randint(low=0, high=lC)
        lr_patch = lr_im[lh:lh + patch_size, lw:lw + patch_size, lc]
        hr_patch = hr_im[hh:hh + hr_patch_size, hw:hw + hr_patch_size, lc]
        lr_patch = lr_patch[:, :, np.newaxis]
        hr_patch = hr_patch[:, :, np.newaxis]
    elif nChans == 96:
        lr_patch = lr_im[lh:lh + patch_size, lw:lw + patch_size, :]
        hr_patch = hr_im[hh:hh + hr_patch_size, hw:hw + hr_patch_size, :]
    else:
        raise ValueError('unsupported mri channels.')

    # lr_patch = (lr_patch / norm).astype(np.float32)
    # hr_patch = (hr_patch / norm).astype(np.float32)

    return lr_patch, hr_patch



def augment_npy(*args, hflip=True, rot=True):
    lr_patch, hr_patch = args
    ##############################################################
    #        we can do some augmentation here Flip + Rotate      #
    ##############################################################
    rot = np.random.randint(low=2, high=9)
    if rot == 2:  # up-down flip
        lr_patch = lr_patch[::-1, :, :]
        hr_patch = hr_patch[::-1, :, :]
    elif rot == 3:  # left-right flip
        lr_patch = lr_patch[:, ::-1, :]
        hr_patch = hr_patch[:, ::-1, :]
    elif rot == 4:  # up-down + left-right flip
        lr_patch = lr_patch[::-1, :, :]
        hr_patch = hr_patch[::-1, :, :]
    elif rot == 5:  # transpose
        lr_patch = np.transpose(lr_patch, [1, 0, 2])
        hr_patch = np.transpose(hr_patch, [1, 0, 2])
    elif rot == 6:  # transpose + up-down flip
        lr_patch = np.transpose(lr_patch, [1, 0, 2])[::-1, :, :]
        hr_patch = np.transpose(hr_patch, [1, 0, 2])[::-1, :, :]
    elif rot == 7:  # transpose + left-right flip
        lr_patch = np.transpose(lr_patch, [1, 0, 2])[:, ::-1, :]
        hr_patch = np.transpose(hr_patch, [1, 0, 2])[:, ::-1, :]
    else:  # transpose + up-down + left-right flip
        lr_patch = np.transpose(lr_patch, [1, 0, 2])[::-1, ::-1, :]
        hr_patch = np.transpose(hr_patch, [1, 0, 2])[::-1, ::-1, :]

    return lr_patch, hr_patch




def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]
    ic = args[0].ndim

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy
    if ic ==2:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip],
            *[a[ty:ty + tp, tx:tx + tp] for a in args[1:]]
        ]
    else:
        ret = [
            args[0][iy:iy + ip, ix:ix + ip, :],
            *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
        ]

    return ret


def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=32767.0, norm=32767.0):
    '''tensor * (rgb_range / 32767.0) '''
    def _np2Tensor(img, norm):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) #int16
        tensor = torch.from_numpy(np_transpose).float() # tensor float32
        tensor.mul_(rgb_range / norm) # 归一化，tensor float32

        return tensor

    return [_np2Tensor(a, norm) for a in args]



def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if img.ndim==2:
            if hflip: img = img[:, ::-1]
            if vflip: img = img[::-1, :]
            if rot90: img = img.transpose(1, 0)
        else:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

