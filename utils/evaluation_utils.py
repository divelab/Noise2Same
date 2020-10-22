import numpy as np
from scipy.misc import ascent
from skimage.measure import compare_psnr, compare_mse, compare_ssim
from .predict_utils import normalize_mi_ma

def normalize(x, pmin=2, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def norm_minmse(gt, x, normalize_gt=True):
    """
    normalizes and affinely scales an image pair such that the MSE is minimized  
     
    Parameters
    ----------
    gt: ndarray
        the ground truth image      
    x: ndarray
        the image that will be affinely scaled 
    normalize_gt: bool
        set to True of gt image should be normalized (default)
    Returns
    -------
    gt_scaled, x_scaled 
    """
    if normalize_gt:
        gt = normalize(gt, 0.1, 99.9, clip=False).astype(np.float32, copy = False)
    x = x.astype(np.float32, copy=False) - np.mean(x)
    gt = gt.astype(np.float32, copy=False) - np.mean(gt)
    scale = np.cov(x.flatten(), gt.flatten())[0, 1] / np.var(x.flatten())
    return gt, scale * x


def get_scores(gt, x, multichan=False):
    
    gt_, x_ = norm_minmse(gt, x)
    
    mse = compare_mse(gt_, x_)
    psnr = compare_psnr(gt_, x_, data_range = 1.)
    ssim = compare_ssim(gt_, x_, data_range = 1., multichannel=multichan)
    
    return np.sqrt(mse), psnr, ssim

if __name__ == '__main__':

    # ground truth image
    y = ascent().astype(np.float32)
    # input image to compare to 
    x1 = y + 30*np.random.normal(0,1,y.shape)
    # a scaled and shifted version of x1
    x2 = 2*x1+100
    
    # calulate mse, psnr, and ssim of the normalized/scaled images
    mse1  = compare_mse(*norm_minmse(y, x1))
    mse2  = compare_mse(*norm_minmse(y, x2))
    # should be the same
    print("MSE1  = %.6f\nMSE2  = %.6f"%(mse1, mse2))

    psnr1 = compare_psnr(*norm_minmse(y, x1), data_range = 1.)
    psnr2 = compare_psnr(*norm_minmse(y, x2), data_range = 1.)
    # should be the same    
    print("PSNR1 = %.6f\nPSNR2 = %.6f"%(psnr1,psnr2))
    
    ssim1 = compare_ssim(*norm_minmse(y, x1), data_range = 1.)
    ssim2 = compare_ssim(*norm_minmse(y, x2), data_range = 1.)
    # should be the same    
    print("SSIM1 = %.6f\nSSIM2 = %.6f"%(ssim1,ssim2))