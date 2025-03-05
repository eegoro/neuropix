from .artifacts import artefacts_by_max_in_window
from .bad_channels import detect_bad_channels_by_mean, detect_bad_channels_by_std

__all__ = [
    'artefacts_by_max_in_window',
    'detect_bad_channels_by_mean',
    'detect_bad_channels_by_std'
]