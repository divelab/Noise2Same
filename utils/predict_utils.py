from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import collections
import warnings
import numpy as np


def get_coord(shape, size, margin):
    n_tiles_i = int(np.ceil((shape[2]-size)/float(size-2*margin)))
    n_tiles_j = int(np.ceil((shape[1]-size)/float(size-2*margin)))
    for i in range(n_tiles_i+1):
        src_start_i = i*(size-2*margin) if i<n_tiles_i else (shape[2]-size)
        src_end_i = src_start_i+size
        left_i = margin if i>0 else 0
        right_i = margin if i<n_tiles_i else 0
        for j in range(n_tiles_j+1):
            src_start_j = j*(size-2*margin) if j<n_tiles_j else (shape[1]-size)
            src_end_j = src_start_j+size
            left_j = margin if j>0 else 0
            right_j = margin if j<n_tiles_j else 0
            src_s = (slice(None, None), 
                     slice(src_start_j, src_end_j), 
                     slice(src_start_i, src_end_i))
            
            trg_s = (slice(None, None), 
                     slice(src_start_j+left_j, src_end_j-right_j), 
                     slice(src_start_i+left_i, src_end_i-right_i))
            
            mrg_s = (slice(None, None), 
                     slice(left_j, -right_j if right_j else None), 
                     slice(left_i, -right_i if right_i else None))
            
            yield src_s, trg_s, mrg_s


# Below implementation of prediction utils inherited from CARE: https://github.com/CSBDeep/CSBDeep
# Content-Aware Image Restoration: Pushing the Limits of Fluorescence Microscopy. Martin Weigert, Uwe Schmidt, Tobias Boothe, Andreas Müller, Alexandr Dibrov, Akanksha Jain, Benjamin Wilhelm, Deborah Schmidt, Coleman Broaddus, Siân Culley, Mauricio Rocha-Martins, Fabián Segovia-Miranda, Caren Norden, Ricardo Henriques, Marino Zerial, Michele Solimena, Jochen Rink, Pavel Tomancak, Loic Royer, Florian Jug, and Eugene W. Myers. Nature Methods 15.12 (2018): 1090–1097.

def _raise(e):
    raise e
    
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes is not None or _raise(ValueError('axis cannot be None.'))
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes


def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)
    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )
    if clip:
        x = np.clip(x,0,1)
    return x

class PercentileNormalizer(object):

    def __init__(self, pmin=2, pmax=99.8, do_after=True, dtype=np.float32, **kwargs):

        (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100) or _raise(ValueError())
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, img, axes):

        len(axes) == img.ndim or _raise(ValueError())
        channel = axes_dict(axes)['C']
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def after(self, img):

        self.do_after or _raise(ValueError())
        alpha = self.ma - self.mi
        beta  = self.mi
        return ( alpha*img+beta ).astype(self.dtype,copy=False)

    def do_after(self):
        
        return self._do_after
    
    
class PadAndCropResizer(object):

    def __init__(self, mode='reflect', **kwargs):

        self.mode = mode
        self.kwargs = kwargs
        
    def _normalize_exclude(self, exclude, n_dim):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d%n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all(( isinstance(d,int) and 0<=d<n_dim for d in exclude_list )) or _raise(ValueError())
        return exclude_list

    def before(self, x, div_n, exclude):

        def _split(v):
            a = v // 2
            return a, v-a
        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [_split((div_n-s%div_n)%div_n) if (i not in exclude) else (0,0) for i,s in enumerate(x.shape)]
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x, exclude):

        pads = self.pad[:len(x.shape)]
        crop = [slice(p[0], -p[1] if p[1]>0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i,slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[tuple(crop)]

