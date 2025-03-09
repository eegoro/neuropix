from scipy import signal
import torch
import torch.fft
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def resample_scipy(fs, new_fs, axis = 1):
    def resample_func(data):
        return signal.resample(data, int((data.shape[-1]/fs)*new_fs), axis = axis)
    return resample_func

def resample_torch(fs, new_fs, axis = 0):
    def resample_func(data):
        num = round((data.shape[-1]/fs)*new_fs)
        dataset = TensorDataset(torch.from_numpy(data))
        batch_size = 5
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        data_result = []
        for batch in data_loader:
            batch_on_gpu = batch[0].to('cuda')
            processed_batch = _resample_torch(batch_on_gpu, num, axis = axis)
            data_result.append(processed_batch.cpu().numpy())
            batch_on_gpu = None
            del batch_on_gpu
            torch.cuda.empty_cache()

        # data_torch = torch.from_numpy(data).to('cuda')
        # data_result = _resample_torch(data_torch, num, axis = axis)
        # data_torch = None
        # data_result = np.array(data_result.to('cpu'))
        return np.concatenate(data_result, axis=0) #data_result
    return resample_func


def _resample_torch(x, num, axis=0, window=None):
    if hasattr(axis, "__len__") and not hasattr(num, "__len__"):
        num = [num] * len(axis)
    
    if hasattr(num, "__len__"):
        if hasattr(axis, "__len__") and len(num) == len(axis):
            _temp = x
            for i in range(len(num)):
                _num = num[i]
                _axis = axis[i]
                _temp = _resample_torch(_temp, _num, _axis, window)
            return _temp
        else:
            raise ValueError("if num is array like, then axis also has to be array like and of the same length")

    Nx = x.shape[axis]
    real_input = _isrealobj(x)

    if real_input:
        X = torch.fft.rfft(x, dim=axis)
    else:  
        X = torch.fft.fft(x, dim=axis)

    if window is not None:
        if callable(window):
            W = window(torch.fft.fftfreq(Nx))
        elif isinstance(window, torch.Tensor):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            raise ValueError("Window can only be either a function or Tensor.")

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            W_real = W.clone()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = torch.zeros(newshape, dtype=X.dtype, device=x.device)

    N = min(num, Nx)
    nyq = N // 2 + 1  
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        if N > 2: 
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    if N % 2 == 0:
        if num < Nx: 
            if real_input:
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num: 
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp
    if real_input:
        y = torch.fft.irfft(Y, num, dim=axis)
    else:
        y = torch.fft.ifft(Y, dim=axis, overwrite_x=True)
    y *= (float(num) / float(Nx))

    return y

def _isrealobj(x):
    d = x.dtype
    if d in(torch.complex32, torch.complex64, torch.complex128):
        return False
    else:
        return True