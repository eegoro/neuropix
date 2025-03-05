import numpy as np
from typing import List, Dict, Tuple, Union
from pathlib import Path
import gc

def get_data(file_path, metadata, start_times: List[float],
                window: float, channels: List[int] = [0], 
                convert_to_uv: bool = True,
                get_all_channels: bool = False,
                time_unit = 's'):
    
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Recording file not found: {file_path}")
        
        if time_unit == 'ms':
            start_times = np.array(start_times)/1000
            window = window/1000
        start_times = np.array(start_times)
        
        result_time = _find_time_windows(metadata['time_points'], metadata['sample_positions'], start_times, window)
        start_pos = result_time['start_positions']
        end_pos = result_time['end_positions']

        if get_all_channels:
            channels = list(metadata['channel_map'].keys())
        chan_idxs = [metadata['channel_map'][chan][0] for chan in channels]

        try:
            results = []
            data_bits_list = []

            for start, end in zip(start_pos, end_pos):
                end += metadata['n_total_channels']
                chunk = []
                _memmap = _get_memmap(file_path, metadata['bytes_per_number'], start, end)

                for i, channel_idx in enumerate(chan_idxs):
                    chunk.append(np.array(_memmap[channel_idx::metadata['n_total_channels']]))
                chunk = np.array(chunk)
                
                del _memmap
                _memmap = None

                data_bits = []
                if metadata['is_nidq'] and (metadata['n_total_channels'] - 1) in chan_idxs:
                    chan_idx = chan_idxs.index(metadata['n_total_channels'] - 1)
                    data_bits = _int16_to_bits(chunk[chan_idx])
                
                if convert_to_uv:
                    scales = metadata['_scales'][chan_idxs]
                    chunk = chunk * scales[:, np.newaxis]
                
                results.append(chunk)
                data_bits_list.append(data_bits)

            if time_unit == 'ms':
                result_time['time_arrays'] = result_time['time_arrays']*1000
            return (np.array(results), result_time['time_arrays'], 
                    np.array(data_bits_list) if data_bits_list else None)

        finally:
            if _memmap is not None:
                del _memmap
                _memmap = None
            gc.collect()

def _get_memmap(file_path, bytes_per_number, start_pos, end_pos):
        """Initialize memory-mapped file for efficient reading."""
        _memmap = np.memmap(
                file_path, 
                dtype=np.int16, 
                mode='r',
                offset=start_pos * bytes_per_number,
                shape=(end_pos - start_pos))
                
        return _memmap

def _int16_to_bits(data):
        data = data.astype(np.int16)
        bits = np.zeros((len(data), 16), dtype=np.bool)
        for i in range(16):
            bits[:, i] = (data >> i) & 1
        return bits
    
def _find_time_windows(time_points, sample_positions, start_times, window):
    
    start_indices = np.searchsorted(time_points, start_times)
    end_indices = np.searchsorted(time_points, start_times + window)
    
    result_time = {
        'time_arrays': [time_points[start:end] for start, end in zip(start_indices, end_indices)],
        'start_times': time_points[start_indices],
        'end_times': time_points[end_indices - 1],
        'start_positions': sample_positions[start_indices],
        'end_positions': sample_positions[end_indices - 1],
        'start_indices': start_indices,
        'end_indices': end_indices
    }
    return result_time