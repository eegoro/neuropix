import numpy as np

def average_data_by_y(y, shank):
    coord = [tuple([shank_1,y_1]) for shank_1,y_1 in zip(shank,y)]
    unique_coord = np.array(list(set(coord)))
    coord = np.array(coord)
    
    def interp_func(data):
        averaged_data = np.zeros((len(unique_coord), data.shape[1]))
        for idx, y_val in enumerate(unique_coord):
            mask = np.all(coord == y_val, axis=1)
            averaged_data[idx] = np.mean(data[mask], axis=0)
        return averaged_data
    
    return interp_func, unique_coord

def interp_bad_channels_by_weighted_average():
    def interp_func(data):
        repaired_data = data.copy()
        nan_channels = np.all(np.isnan(data), axis=1)
        nan_indices = np.where(nan_channels)[0]

        if nan_indices.shape[0]==0:
            return repaired_data

        for nan_idx in nan_indices:
            distances = np.abs(np.arange(data.shape[0]) - nan_idx)
            sorted_indices = np.argsort(distances)
            nearest_good = sorted_indices[~nan_channels[sorted_indices]][:4]
            if len(nearest_good) > 0:
                weights = 1 / distances[nearest_good]
                weights = weights / np.sum(weights)
                repaired_data[nan_idx] = np.average(data[nearest_good], weights=weights, axis=0)
        
        return repaired_data
    return interp_func

def interp_bad_channels_by_mean_window():

    def interp_func(data):
        repaired_data = data.copy()
        nan_channels = np.all(np.isnan(data), axis=1)
        nan_indices = np.where(nan_channels)[0]

        if nan_indices.shape[0]==0:
            return repaired_data

        for nan_idx in nan_indices:
            window = slice(max(0, nan_idx - 3), min(len(nan_channels), nan_idx + 4))
            good_in_window = ~nan_channels[window]
            window_data = data[window]
            
            if np.any(good_in_window):
                good_data = window_data[good_in_window]
                repaired_data[nan_idx] = np.mean(good_data, axis=0)
        return repaired_data
    return interp_func

def interp_bad_channels_by_median_window():

    def interp_func(data):
        repaired_data = data.copy()
        nan_channels = np.all(np.isnan(data), axis=1)
        nan_indices = np.where(nan_channels)[0]

        if nan_indices.shape[0]==0:
            return repaired_data

        for nan_idx in nan_indices:
            window = slice(max(0, nan_idx - 3), min(len(nan_channels), nan_idx + 4))
            good_in_window = ~nan_channels[window]
            window_data = data[window]
            
            if np.any(good_in_window):
                good_data = window_data[good_in_window]
                repaired_data[nan_idx] = np.median(good_data, axis=0)
        return repaired_data
    return interp_func

def interp_nan_in_channel_by_weighted_average():
    def interp_func(data):
        return np.apply_along_axis(_interp_nan_in_channel_by_weighted_average, 1, data)
    return interp_func



def _interp_nan_in_channel_by_weighted_average(data):
    predicted_values = data.copy()
    nan_indices = np.where(np.isnan(predicted_values))[0]
    
    if len(nan_indices) == 0:
        return predicted_values
    
    parts_nan = _split_consecutive_groups(list(nan_indices))

    for part_idx in parts_nan:
        first_nan, last_nan = part_idx[0], part_idx[-1]
        gap_size = last_nan - first_nan + 1

        left_idx = np.max((0,first_nan-gap_size))
        left_none_positions = np.where(np.isnan(predicted_values[left_idx:first_nan]))[0]
        if len(left_none_positions) > 0:
            left_idx = left_idx + left_none_positions[-1] + 1
        left_end = predicted_values[left_idx:first_nan]
        if left_end.shape[-1] == 0:
            value = 0
        else:
            value = left_end[0]
        left_start = np.repeat(value, gap_size - left_end.shape[-1])
        left_data = np.append(left_start, left_end, axis=0)

        right_idx = np.min((predicted_values.shape[-1],last_nan+gap_size))
        right_none_positions = np.where(np.isnan(predicted_values[last_nan+1:right_idx]))[0]
        if len(right_none_positions) > 0:
            right_idx = last_nan + 1 + right_none_positions[0]
        right_start = predicted_values[last_nan+1:right_idx]
        if right_start.shape[-1] == 0:
            value = 0
        else:
            value = right_start[-1]
        right_end = np.repeat(value, gap_size - right_start.shape[-1])
        right_data = np.append(right_start, right_end, axis=0)
                
        weights = np.linspace(1, 0, gap_size)
        
        predict_gap = left_data[::-1] * weights + right_data[::-1] * weights[::-1]
        
        predicted_values[first_nan:last_nan+1] = predict_gap.copy()
    
    # for part_idx in parts_nan:
    #     first_nan, last_nan = part_idx[0], part_idx[-1]
    #     gap_size = last_nan - first_nan + 1
        
    #     if first_nan < gap_size:
    #         left_data = predicted_values[0:first_nan]
    #         if len(left_data) > 0:
    #             left_values = np.concatenate([left_data] * (gap_size // len(left_data) + 1))[:gap_size]
    #         else:
    #             right_data = predicted_values[last_nan+1:last_nan+1+gap_size]
    #             if len(right_data) < gap_size:
    #                 right_data = np.concatenate([right_data] * (gap_size // len(right_data) + 1))[:gap_size]
    #             predicted_values[first_nan:last_nan+1] = right_data[:gap_size]
    #             continue
    #     else:
    #         left_values = predicted_values[first_nan-gap_size:first_nan]
            
    #     if last_nan + gap_size >= len(predicted_values):
    #         right_data = predicted_values[last_nan+1:]
    #         if len(right_data) > 0:
    #             right_values = np.concatenate([right_data] * (gap_size // len(right_data) + 1))[:gap_size]
    #         else:
    #             predicted_values[first_nan:last_nan+1] = left_values[:gap_size]
    #             continue
    #     else:
    #         right_values = predicted_values[last_nan+1:last_nan+1+gap_size]
        
    #     if np.any(np.isnan(left_values)):
    #         valid_left = left_values[~np.isnan(left_values)]
    #         if len(valid_left) > 0:
    #             left_values = np.concatenate([valid_left] * (gap_size // len(valid_left) + 1))[:gap_size]
    #         else:
    #             left_values = right_values
        
    #     if np.any(np.isnan(right_values)):
    #         valid_right = right_values[~np.isnan(right_values)]
    #         if len(valid_right) > 0:
    #             right_values = np.concatenate([valid_right] * (gap_size // len(valid_right) + 1))[:gap_size]
    #         else:
    #             right_values = left_values
                
    #     weights = ((np.arange(gap_size)+1)/gap_size)[::-1]
        
    #     predict_gap = left_values[::-1][-gap_size:] * weights + \
    #                 right_values[::-1][-gap_size:] * weights[::-1]
        
    #     predicted_values[first_nan:last_nan+1] = predict_gap.copy()
    
    return predicted_values

def _split_consecutive_groups(arr):
    if not arr:
        return []
    
    result = []
    temp = [arr[0]]
    
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            temp.append(arr[i])
        else:
            result.append(temp)
            temp = [arr[i]]
    
    result.append(temp)
    return result