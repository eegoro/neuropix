import pandas as pd
from pathlib import Path

from .info import get_table_info

RECORDING_FILE_AP = "W:\Large_scale_mapping_NP\Awake_exp\SA6\SA6_experiment_4_9_23\Insertion4\SA6_experiment_4_9_23_4_g0\SA6_experiment_4_9_23_4_g0_imec0\SA6_experiment_4_9_23_4_g0_t0.imec0.ap.bin"

def _get_bin_path(Base_path, Insertion, Exp_name):
    Base_path = Base_path.replace('\\', '/').replace('/data/', '/Data/')
    file_path = (
        f"/media{Base_path[1:]}/{Exp_name}/Insertion{Insertion}/"
        f"catgt_{Exp_name}_{Insertion}_g0/{Exp_name}_{Insertion}_g0_tcat.imec0.ap.bin"
    )
    return(file_path)

def get_bin_path(Chosen_Animal = 'SA8', Chosen_Insertion = 3):
    data_info = get_table_info(Chosen_Animal, Chosen_Insertion)
    return _get_bin_path(data_info['Base_path'], int(data_info['Insertion']), data_info['Exp_name'])

def get_metadata_path(bin_path):
    bin_path = Path(bin_path)
    return bin_path.with_suffix('.meta')

