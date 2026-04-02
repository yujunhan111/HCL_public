#创建Demographics information的映射
import pandas as pd
def create_clean_mapping(unique_values):
    clean_map = {}
    current_idx = 0
    sorted_values = sorted([val for val in unique_values if val != 'nan'])
    for val in sorted_values:
        if pd.notna(val):
            clean_map[val] = current_idx
            current_idx += 1

    clean_map['nan']= current_idx

    return clean_map


def create_all_mappings(patient_df):
    mappings = {}

    # 对每个分类变量创建映射
    for feature in ['gender', 'race', 'marital_status', 'language']:
        patient_df[feature] = patient_df[feature].fillna('nan')
        unique_vals = patient_df[feature].unique()
        mappings[feature] = create_clean_mapping(unique_vals)

    return mappings