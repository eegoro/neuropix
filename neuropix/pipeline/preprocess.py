import numpy as np

def apply_preprocess(data, time_array, fs, functions=None, batch_mode=False, batch_size=1024**2, overlap=1024):
    shape_orig = data.shape[-1]

    if functions is None or len(functions) == 0:
        return data, time_array, fs
    
    if not batch_mode:
        result = data.copy()
        for func in functions:
            result = func(result)
        shape_prep = result.shape[-1]
        new_time_array = np.linspace(time_array[0], time_array[-1], shape_prep)
        new_fs = shape_prep / (shape_orig / fs)
        return result, new_time_array, new_fs
    
    result = _process_by_batch(data, fs, functions, batch_size, overlap)
    shape_prep = result.shape[-1]
    new_time_array = np.linspace(time_array[0], time_array[-1], shape_prep)
    new_fs = shape_prep / (shape_orig / fs)
    return result, new_time_array, new_fs

def _create_batch(data: np.ndarray, batch_size=1024**2, overlap=1024):
    num_samples = data.shape[-1]
    batch_size = min(batch_size, num_samples)
    
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        start_overlap = max(0, start - overlap)
        end_overlap = min(num_samples, end + overlap)
        yield ((start - start_overlap, end - start_overlap), data[:, start_overlap:end_overlap])

def _process_by_batch(data, fs, functions, batch_size=1024**2, overlap=1024):
    if data.shape[-1] <= batch_size:
        return apply_preprocess(data, fs, functions)[0]
    
    result = []
    for (start,end), batch in _create_batch(data, batch_size=batch_size, overlap=overlap):
        processed_batch = batch.copy()
        original_length = processed_batch.shape[-1]
        for func in functions:
            processed_batch = func(processed_batch).copy()
        processed_length = processed_batch.shape[-1]
        if processed_length!=original_length:
            start = int(start * processed_length / original_length)
            end = int(end * processed_length / original_length)
        result.append(processed_batch[:, start:end])
    
    return np.concatenate(result, axis=-1)