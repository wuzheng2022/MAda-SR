import os
import math
from re import A
import time
import datetime
from multiprocessing import Process
from multiprocessing import Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.util.dtype import dtype_range
from skimage.util.arraycrop import crop
from warnings import warn


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)
            os.makedirs(self.get_path('results-{}'.format(d), 'SR'), exist_ok=True)
            os.makedirs(self.get_path('results-{}'.format(d), 'HR'), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        #trainer.loss.plot_loss(self.dir, epoch)

        #self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self, epoch):
        axis = np.linspace(1, epoch, epoch)
        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()

        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, tensor = queue.get()
                    if filename is None: break
                    slice  = tensor
                    if slice.shape[2] > 1:
                        # save the middle slice for MRI
                        i = (slice.shape[2]+1)//2
                        img = slice[:,:,i]
                    else:
                        img = slice[:,:,0]
                    img = ((img / np.max(img)) * 255.0).astype('uint8')
                    ff = filename[:-4]+'.png'
                    imageio.imwrite(ff, img)

        
        self.process = [
            Process(target=bg_target, args=(self.queue,)) \
            for _ in range(self.n_processes)
        ]
        
        for p in self.process: p.start()

    def end_background(self):
        for _ in range(self.n_processes): self.queue.put((None, None))
        while not self.queue.empty(): time.sleep(1)
        for p in self.process: p.join()

    def save_results(self, foldername, filename, save_list, scale):
        if self.args.save_results:

            postfix = ('SR', 'HR')
            for v, p in zip(save_list, postfix):
                # print('hr max{}, min{}, type{}'.format(np.max(v[0]), np.min(v[0]), v[0].dtype))
                # normalized = v[0].mul(32767 / self.args.rgb_range)
                # tensor_cpu = v[0].permute(1, 2, 0).cpu()

                tensor_cpu = v[0].transpose(1, 2, 0)

                tmp = self.get_path(
                    'results-{}/{}'.format(foldername, p),
                    '{}_x{}_'.format(filename, scale)
                )
                self.queue.put(('{}{}.png'.format(tmp, p), tensor_cpu))
    
    def write_performance(self, mess, file='/performance.txt', refresh=False):
        open_type = 'a' if os.path.exists(self.dir + file) else 'w'
        self.performance_file = open(self.dir + file, open_type)
        print(mess)
        self.performance_file.write(mess + '\n')
        if refresh:
            self.performance_file.close()
            self.performance_file = open(self.dir + file, 'a')


def quantize(img, rgb_range, norm=32767.0):
    '''将数据范围规整约束在[0-1]'''
    pixel_range = norm / rgb_range
    # return img.mul(pixel_range).clamp(0, norm).round().div(pixel_range)
    return img.mul(pixel_range).clamp(0, norm).div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range=255, dataset=None):
    if hr.nelement() == 1: return 0

    # sr = sr.mul(255 / rgb_range)
    # hr = hr.mul(255 / rgb_range)

    if hr.ndim == 3:
        sr = sr.byte().permute(1, 2, 0)
        hr = hr.byte().permute(1, 2, 0)

    diff = (sr - hr).data.div(rgb_range)
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def quantize_npy(img, norm, dtype=np.int32):
    img = img.mul(norm).clamp(0, norm)
    img = img.byte().permute(0, 2, 3, 1).cpu().numpy().astype(dtype)
    return img


def calc_batch_psnr_ssim(real_batch, pred_batch, multichannel=True):
    """Calculate psnr and ssim for a batch of examples.

    Args:
    - real_batch:
        A batch of ground truth examples.
    - pred_batch:
        A batch of predicated examples corresponding to real_batch.

    Returns:
       Average psnr and ssim values for this batch.
    """
    
    if not real_batch.shape == pred_batch.shape:
        raise ValueError('Input images must have the same dimensions.')

    if real_batch.dtype != pred_batch.dtype:
        raise ValueError("Inputs have mismatched dtype.")

    data_type = real_batch.dtype
    bSize = real_batch.shape[0]
    mean_psnr = 0.0
    mean_ssim = 0.0
    a, b = 0.0, 0.0
    for i in range(bSize):
        real, pred = real_batch[i, :, :, :], pred_batch[i, :, :, :]
        if data_type == np.uint8:
            a = calc_psnr_npy(real, pred)
            b = calc_ssim_npy(real, pred, multichannel=multichannel)
        else:
            data_range = torch.max(real) - torch.min(real)
            a = calc_psnr_npy(real, pred, data_range)
            b = calc_ssim_npy(real.numpy(), pred.numpy(),
                                   gaussian_weights=True,
                                   data_range=data_range,
                                   multichannel=multichannel,
                                   use_sample_covariance=False,
                                   pad=0)

        print(a,"and")
        mean_psnr += a
        mean_ssim += 0


    mean_psnr = mean_psnr / bSize
    mean_ssim = mean_ssim / bSize
    print("mean psnr:{}, mean ssim:{}".format(mean_psnr, mean_ssim))
    return mean_psnr, mean_ssim


def calc_batch_ssim(real_batch, pred_batch, multichannel=False):
    """Calculate psnr and ssim for a batch of examples.

    Args:
    - real_batch:
        A batch of ground truth examples.
    - pred_batch:
        A batch of predicated examples corresponding to real_batch.

    Returns:
       Average psnr and ssim values for this batch.
    """
    
    if not real_batch.shape == pred_batch.shape:
        raise ValueError('Input images must have the same dimensions.')

    if real_batch.dtype != pred_batch.dtype:
        raise ValueError("Inputs have mismatched dtype.")

    data_type = real_batch.dtype
    bSize, nc = real_batch.shape[:2]
    mean_ssim = 0.0
    for i in range(bSize):
        if nc == 1:
            real, pred = real_batch[i, 0, :, :], pred_batch[i, 0, :, :]
        else:
            real, pred = real_batch[i, :, :, :], pred_batch[i, :, :, :]
        if data_type == np.uint8:
            b = calc_ssim_npy(real, pred, multichannel=multichannel)
        else:
            data_range = np.max(real)-np.min(real)
            b = calc_ssim_npy(real, pred,
                                   gaussian_weights=True,
                                   data_range=data_range,
                                   multichannel=multichannel,
                                   use_sample_covariance=False,
                                   pad=0)
            mean_ssim = max(mean_ssim, b)

    return mean_ssim





def calc_psnr_npy(X, Y, data_range=None):
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    ndim = X.ndim
    if ndim <= 0:
        raise ValueError("Array dimension must larger than 0!")

    if X.dtype != Y.dtype:
        raise ValueError("Incompatible data type!")

    if data_range is None:
        if X.dtype != Y.dtype:
            warn("Inputs have mismatched dtype. Setting data_range based on X.dtype.")
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    # convert to float64
    # X = X.cpu().numpy().astype(np.float64)
    # Y = Y.cpu().numpy().astype(np.float64)

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    
    diff = np.abs(X - Y)
    rmse = np.sqrt((diff ** 2).mean())

    return 20*np.log10(data_range/rmse)



def calc_ssim_npy(X, Y, win_size=None, gradient=False, data_range=None,
              multichannel=False, gaussian_weights=False, full=False,
              pad=None, **kwargs):
    """Compute the mean structural similarity index between two images.

    Parameters
    ----------
    X, Y : ndarray
        Image.  Any dimensionality.
    win_size : int or None
        The side-length of the sliding window used in comparison.  Must be an
        odd value.  If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to Y.
    data_range : float, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.
    multichannel : bool, optional
        If True, treat the last dimension of the array as channels. Similarity
        calculations are done independently for each channel then averaged.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, return the full structural similarity image instead of the
        mean value.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        if True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        algorithm parameter, K1 (small constant, see [1]_)
    K2 : float
        algorithm parameter, K2 (small constant, see [1]_)
    sigma : float
        sigma for the Gaussian when `gaussian_weights` is True.
    pad: (added by XLZhao), whether to pad to avoid edges effect.

    Returns
    -------
    mssim : float
        The mean structural similarity over the image.
    grad : ndarray
        The gradient of the structural similarity index between X and Y [2]_.
        This is only returned if `gradient` is set to True.
    S : ndarray
        The full SSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    To match the implementation of Wang et. al. [1]_, set `gaussian_weights`
    to True, `sigma` to 1.5, and `use_sample_covariance` to False.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1109/TIP.2003.819861

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       http://arxiv.org/abs/0901.0065,
       DOI:10.1007/s10043-009-0119-z

    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size, gradient=gradient, data_range=data_range,
                    multichannel=False, gaussian_weights=gaussian_weights, full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = calc_ssim_npy(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError("win_size exceeds image extent. If the input is a multichannel "
                         "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if X.dtype != Y.dtype:
            warn("Inputs have mismatched dtype. Setting data_range based on X.dtype.")
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        # sigma = 1.5 to approximately match filter in Wang et. al. 2004
        # this ends up giving a 13-tap rather than 11-tap Gaussian
        # w = 2*int(truncate*sigma + 0.5) + 1, truncate = 3.0 matches matlab
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'mode': 'nearest', 'truncate': 3.0}

    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    if X.dtype != Y.dtype:
        raise ValueError('data type must be the same.')

        # ndimage filters need floating point data
    if X.dtype == 'int16':
        X = X.astype(np.float64) + 32768.0
        Y = Y.astype(np.float64) + 32768.0
    else:
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    if pad == None or pad < 0:
        pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D, **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


def make_base_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    import optimizer_adam
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    
    optimizer_class = optimizer_adam.local_adam
    kwargs_optimizer['betas'] = args.betas
    kwargs_optimizer['eps'] = args.epsilon
    

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, reg_lambda, *args, **kwargs):
            super(CustomOptimizer, self).__init__(reg_lambda, *args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, args.reg_lambda, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer