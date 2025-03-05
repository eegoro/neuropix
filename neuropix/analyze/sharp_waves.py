from .correlation import cross_corr
from ..io import get_sharp_wave_templates
from scipy import signal
import numpy as np

def get_coord_sharp_waves(data, fs_data, 
                          template=None, fs_template=None,
                          threshold_corr=0.85,
                          width = None,
                          distance = None,
                          prominence=0.5):

    if template==None:
        template, fs_template = get_sharp_wave_templates()
    if fs_data!=fs_template:
        template = signal.resample_poly(template, up = fs_data, down = fs_template, padtype='line')
    
    if width == None:
        width=0.1*fs_data
    if distance==None:
        distance=0.2*fs_data
    corr = cross_corr(data, template)
    peaks, _ = signal.find_peaks(corr, width=width, distance=distance, prominence=prominence)
    peaks_tr = peaks[np.where(corr[peaks]>threshold_corr)]

    return peaks_tr