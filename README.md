# Neuropix

A comprehensive Python package for analyzing neural data recorded with Neuropixels probes.

## Overview

Neuropix provides tools for processing, analyzing, and visualizing Neuropixels data with a focus on:

- Data importing and preprocessing
- Signal processing and filtering 
- Correlation and spectral analysis
- Sleep cycle detection and analysis
- Sharp wave detection
- Current source density (CSD) analysis
- Clustering and visualization

## Package Structure

- **neuropix/io**: Data import and metadata parsing
  - Binary data readers
  - Metadata parsers
  - Spike data handling
  - Trigger event processing

- **neuropix/signal**: Signal processing utilities
  - Filters (bandpass, notch, etc.)
  - Resampling functions
  - Bad channel detection
  - Artifact removal
  - Interpolation methods

- **neuropix/analyze**: Analysis algorithms
  - Correlation calculations
  - Spectral analysis
  - Sleep cycle detection
  - Sharp wave detection
  - Current source density (CSD)
  - Feature extraction and clustering

- **neuropix/pipeline**: Analysis workflow management
  - Preprocessing pipelines
  - Sleep analysis pipeline
  - Batch processing

- **neuropix/viz**: Visualization tools
  - Channel data plotting
  - Correlation matrices
  - Spectral plots
  - Sleep analysis visualizations
  - Raster plots
  - CSD visualization

- **neuropix/utils**: Utility functions
  - Logging
  - Signal extraction
  - Information tracking

## Basic Usage

### Reading Data

```python
from neuropix.io import parse_metadata, get_data

# Parse metadata
metadata, metadata_raw = parse_metadata('path/to/metadata.meta')

# Read raw data
data, time_array, bits = get_data(
    'path/to/recording.bin', 
    metadata, 
    start_times=[0], 
    window=10, 
    channels=[0, 1, 2],
    convert_to_uv=True
)
```

### Signal Processing

```python
from neuropix.signal import FuncWrap
from neuropix.signal.processing import bandpass_filter, notch_filter
from neuropix.signal.detection import detect_bad_channels_by_std
from neuropix.pipeline import PipelineWrap

# Create processing functions
bandpass = FuncWrap(bandpass_filter, 'bandpass_filter', lowcut=0.5, highcut=200, fs=30000)
notch = FuncWrap(notch_filter, 'notch_filter', freq=50, fs=30000)
detect_bad = FuncWrap(detect_bad_channels_by_std, 'detect_bad_channels_by_std', std_threshold=3.0)

pipeline = PipelineWrap(pipeline, "apply_preprocess", functions=[bandpass, notch, detect_bad], batch_mode=True, batch_size=1024*1024, overlap=1024)
# Apply processing
filtered_data, new_time_array, new_fs = pipeline_1(data, time_array, fs)
```

### Analysis

```python
from neuropix.analyze import compute_power_spectrum, corr_matrix, get_coord_sharp_waves

# Compute power spectrum
frequencies, power = compute_power_spectrum(data, fs=30000)

# Correlation analysis
correlation = corr_matrix(power)

# Detect sharp waves
peaks = get_coord_sharp_waves(data, fs=30000, threshold_corr=0.85)
```

### Visualization

```python
from neuropix.viz import plot_channels, plot_sorted_correlation_matrix, plot_csd_result

# Plot channel data
fig, ax = plot_channels(data, time_array, window=10)

# Plot correlation matrix
fig, _, _ = plot_sorted_correlation_matrix(correlation, linkage_matrix)

# Plot CSD results
fig = plot_csd_result(filtered_data, smoothed_data, csd_data, time_array, y_coords)
```

### Using Pipelines

```python
from neuropix.pipeline import apply_sleep_analysis

# Apply sleep analysis pipeline
results = apply_sleep_analysis(
    data, 
    fs=30000,
    bin_size=10,
    bin_overlap=9,
    low_freq_bound=30,
    num_clusters=20
)

# Access results
delta_beta_ratio = results['delta_beta_ratio']
fig_corr_matrix = results['fig_corr_matrix']
```

## Examples

See the included Jupyter notebooks for detailed examples.

## Requirements

- numpy
- scipy
- matplotlib
- pandas
- seaborn
- torch (for GPU-accelerated processing)
- scikit-learn

## License