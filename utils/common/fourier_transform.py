import numpy as np
import torch

def fftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.fftn(np.fft.ifftshift(data, axes=axes), 
                    axes=axes, 
                    norm=norm), 
        axes=axes
    )


def ifftc(data, axes=(-2, -1), norm="ortho"):
    """
    Centered inverse fast fourier transform
    """
    return np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(data, axes=axes), 
                     axes=axes, 
                     norm=norm), 
        axes=axes
    )


def fftc_torch(data, dim=(-2, -1), norm="ortho"):
    """
    Centered fast fourier transform for torch.Tensor
    """
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(data, dim=dim),
            dim=dim, norm=norm),
        dim=dim
    )


def ifftc_torch(data, dim=(-2, -1), norm="ortho"):
    """
    Centered inverse fast fourier transform for torch.Tensor
    """
    return torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(data, dim=dim),
            dim=dim, norm=norm),
        dim=dim
    )