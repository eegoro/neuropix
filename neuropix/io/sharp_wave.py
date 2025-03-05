import numpy as np
import os

def get_sharp_wave_templates(path=None):
    if path==None:
        path = os.path.join(os.path.dirname(__file__), "../templates/analysis/sharp_wave.np")
    with open(path, 'rb') as f:
        template = np.load(f)
    fs_template = 1000

    return template, fs_template