import torch
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
from scipy.signal import get_window

def get_features(chunk, electrode_coords, fs=500):
    chunk_torch = torch.from_numpy(chunk).to('cuda')

    mean_torch = chunk_torch.mean(dim=1)
    diff_torch = torch.stack([channel - mean_torch[idx] for idx,channel in enumerate(chunk_torch)])
    var_torch = torch.mean(torch.pow(diff_torch, 2.0), dim=1)
    std_torch = torch.pow(var_torch, 0.5)
    zscores_torch = torch.stack([diff_torch_ch / std_torch[idx] for idx,diff_torch_ch in enumerate(diff_torch)])

    skews_torch = torch.mean(torch.pow(zscores_torch, 3.0), dim=1)
    kurtoses_torch = torch.mean(torch.pow(zscores_torch, 4.0), dim=1) - 3.0 

    p25_torch = torch.quantile(chunk_torch, 0.25, interpolation='linear', dim=1)
    p75_torch = torch.quantile(chunk_torch, 0.75, interpolation='linear', dim=1)
    iqr_torch = p75_torch - p25_torch

    zero_crossings_torch = torch.sum(torch.diff((chunk_torch < 0).int()!= 0), dim=1)
    peak_to_peak_torch = torch.max(chunk_torch, dim=1).values - torch.min(chunk_torch, dim=1).values

    # from torch.fft import rfft, rfftfreq

    # power_spectrum = (torch.abs(rfft(chunk_torch, dim=1))**2)
    # freqs = rfftfreq(chunk_torch.shape[1], 1/fs)

    psd_torch = torch.stack([_torch_welch(ch.float(), fs=fs, nperseg= min(len(chunk_torch[0]), fs))[1] for ch in chunk_torch])
    freqs_torch = _torch_welch(chunk_torch[0], fs=fs, nperseg= min(len(chunk_torch[0]), fs))[0]

    bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'low': (0, 10),
            'mid': (10, 50),
            'high': (50, 200)
        }

    bands_res_torch = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs_torch >= low) & (freqs_torch <= high)
        bands_res_torch[band_name] = torch.stack([torch.sum(psd_torch_ch[mask]) for psd_torch_ch in psd_torch])

    power_low_high_ratio_torch = bands_res_torch['low']/bands_res_torch['high']

    features_dict = {
        "mean": mean_torch.cpu().numpy(),
        "std": std_torch.cpu().numpy(),
        "skew": skews_torch.cpu().numpy(),
        'kurtosis': kurtoses_torch.cpu().numpy(),
        'p25': p25_torch.cpu().numpy(),
        'p75': p75_torch.cpu().numpy(),
        'iqr': iqr_torch.cpu().numpy(),
        'delta': bands_res_torch['delta'].cpu().numpy(),
        'theta': bands_res_torch['theta'].cpu().numpy(),
        'alpha': bands_res_torch['alpha'].cpu().numpy(),
        'beta': bands_res_torch['beta'].cpu().numpy(),
        'gamma': bands_res_torch['gamma'].cpu().numpy(),
        'low': bands_res_torch['low'].cpu().numpy(),
        'mid': bands_res_torch['mid'].cpu().numpy(),
        'high': bands_res_torch['high'].cpu().numpy(),
        'power_low_high_ratio': power_low_high_ratio_torch.cpu().numpy(),
        'zero_crossings': zero_crossings_torch.cpu().numpy(),
        'peak_to_peak': peak_to_peak_torch.cpu().numpy(),

    }

    # Convert the dictionary to a DataFrame
    features_df = pd.DataFrame(features_dict)

    # Set the index to range from 0 to 383 (matching the length of tensors)
    features_df.index = range(384)

    for i, coord in enumerate(['x']):
        features_df[f'coord_{coord}'] = electrode_coords[i, :]

    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df.drop(['coord_x'], axis=1))
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_coords = tsne.fit_transform(features_scaled)
    
    # Add t-SNE coordinates to DataFrame
    features_df['tsne_1'] = tsne_coords[:, 0]
    features_df['tsne_2'] = tsne_coords[:, 1]

    return features_df


def _torch_welch(signal, fs, nperseg=None, noverlap=None, window='hann', scaling='density', debug=False):
    """
    Compute Welch's periodogram using PyTorch tensors, matching scipy.signal.welch behavior.
    """
    if not isinstance(signal, torch.Tensor):
        raise TypeError("Signal must be a torch.Tensor")
    
    # Convert signal to float if needed
    if not signal.is_floating_point():
        signal = signal.float()
    
    # Set default parameters
    if nperseg is None:
        nperseg = min(256, len(signal))
    if noverlap is None:
        noverlap = nperseg // 2
        
    # Create window and convert to tensor
    win = torch.from_numpy(get_window(window, nperseg)).to(signal.device, signal.dtype)
    
    # Split into segments
    step = nperseg - noverlap
    num_segments = (len(signal) - noverlap) // step
    
    shape = (num_segments, nperseg)
    strides = (step * signal.stride(-1), signal.stride(-1))
    segments = torch.as_strided(signal, shape, strides, storage_offset=0)
    
    if debug:
        print(f"Number of segments: {num_segments}")
        print(f"Segment shape: {segments.shape}")
        print(f"Window sum: {win.sum()}")
        print(f"First segment mean: {segments[0].mean()}")
    
    # Detrend segments (remove mean)
    segments = segments - segments.mean(dim=1, keepdim=True)
    
    # Apply window
    segments = segments * win.unsqueeze(0)
    
    # Compute FFT
    fft_segments = torch.fft.rfft(segments, dim=1)
    power_segments = torch.abs(fft_segments).pow(2)
    
    # Average periodograms
    power_spectrum = torch.mean(power_segments, dim=0)
    
    # Generate frequencies
    freqs = torch.fft.rfftfreq(nperseg, 1/fs, device=signal.device)
    
    # Compute scaling factor
    scale = 1.0 / (fs * (win**2).sum())
    
    # Apply scaling
    power_spectrum *= scale
    
    # Handle frequency scaling
    if nperseg % 2 == 0:
        # Even case: double all except DC and Nyquist
        power_spectrum[1:-1] *= 2
    else:
        # Odd case: double all except DC
        power_spectrum[1:] *= 2
    
    if debug:
        print(f"Scale factor: {scale}")
        print(f"Final spectrum shape: {power_spectrum.shape}")
    
    return freqs, power_spectrum