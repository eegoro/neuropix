import pandas as pd

TABLE_OF_EXPERIMENT = '/media/sil3/Data/Large_scale_mapping_NP/Experiment_Excel.xlsx'

def get_table_info(Chosen_Animal, Chosen_Insertion):
    exps_info = pd.read_excel(TABLE_OF_EXPERIMENT)
    data_info = exps_info[(exps_info.Animal_ID == Chosen_Animal) & (exps_info.Insertion == Chosen_Insertion)].iloc[0].to_dict()

    return data_info
