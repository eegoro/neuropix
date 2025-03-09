from ..io import get_data
from ..utils import extract_time_segment
from typing import List
import numpy as np
import logging 

logger = logging.getLogger(__name__)

def apply_preprocess_to_direct_reading(functions, file_path, metadata, start_time, end_time,
                window, overlap, channels, get_all_channels, 
                convert_to_uv, time_unit = 's'):

    result_array, result_time = _process_by_batch(functions, file_path, metadata, start_time, end_time,
                window, overlap, channels, get_all_channels, 
                convert_to_uv, time_unit = 's')
    new_fs =  _calculate_sampling_rate(result_time)
    
    return result_array, result_time, new_fs

def _create_batch(file_path, metadata, start_time, end_time,
                window, overlap, channels, get_all_channels, 
                convert_to_uv, time_unit = 's'):
    
    duration = float(metadata.get('duration', 0))
    end_time = min(end_time, duration)

    starts = np.arange(start_time,end_time,window)
    logger.info(f"Numbers of cunks: {len(starts)}")
    number_chunk = 1
    
    for start in starts:
        end = min(start + window, duration)
        start_overlap = max(0, start - overlap)
        end_overlap = min(duration, end + overlap)
        logger.info(f"Getting {number_chunk} chunk...")
        logger.info(f"Start_overlap: {start_overlap}, start: {start}, end: {end}, end_overlap: {end_overlap} ")
        number_chunk +=1

        data_array, time_array, _ = get_data(file_path, metadata, start_times = [start_overlap],
            window = end_overlap-start_overlap, channels = channels, convert_to_uv=convert_to_uv,
            get_all_channels=get_all_channels, time_unit=time_unit)

        yield (start, end), data_array, time_array


def _process_by_batch(functions, file_path, metadata, start_time, end_time,
                window, overlap, channels, get_all_channels, 
                convert_to_uv, time_unit = 's'):
    result_array, result_time = [], []

    for (start,end), batch, time_batch in _create_batch(file_path, metadata, start_time, end_time,
                window, overlap, channels, get_all_channels, 
                convert_to_uv, time_unit = 's'):
        time_batch = time_batch[0]
        processed_batch = batch[0].copy()
        original_length = processed_batch.shape[-1]
        logger.info(f"Processing chunk...")
        for func in functions:
            processed_batch = func(processed_batch).copy()
        processed_length = processed_batch.shape[-1]
        if processed_length!=original_length:
            time_batch = np.linspace(time_batch[0], time_batch[-1], processed_length)
        batch_without_overlap, time_without_overlap = extract_time_segment(processed_batch, time_batch, start, end)
        result_array.append(batch_without_overlap)
        result_time.append(time_without_overlap)
    
    logger.info(f"All chunks is processed...")
    return np.concatenate(result_array, axis=-1), np.concatenate(result_time, axis=-1)

def _calculate_sampling_rate(time_array):
    logger.info(f"Calculate sampling rate...")
    time_diffs = np.diff(time_array)
    mean_time_diff = np.mean(time_diffs)
    sampling_rate = 1 / mean_time_diff
    
    return sampling_rate