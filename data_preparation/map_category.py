import pandas as pd
from utils.str_utils import extract_number

def map_fov_categories(plate_layout_df:pd.DataFrame,
                       well:int,
                       experiment:str,
                       well_column:str='well',
                       experiment_column:str='experiment',
                       treatment_column:str='treatment'):
    
    # copy input dataframe
    original_plate_layout_df = plate_layout_df.copy()

    return original_plate_layout_df[(original_plate_layout_df[experiment_column]==experiment) &
                                     (original_plate_layout_df[well_column]==well)][treatment_column].values[0]


def map_fov_categories_df(metadata_df:pd.DataFrame,
                          plate__layout_df:pd.DataFrame,
                          well__column:str='well',
                          experiment__column:str='experiment',
                          treatment__column:str='treatment',
                          wellasint_column:str="int_well",
                          drop_wellasint_column:bool=True)->pd.DataFrame:
    
    # copy input dataframes
    original_metadata_df = metadata_df.copy()
    original_plate_layout_df = plate__layout_df.copy()

    # add a column transforming wells in numbers
    original_metadata_df[wellasint_column] = original_metadata_df.apply(lambda row:
                                                                        extract_number(s=row[well__column]), axis=1)
    
    # add a column with the treatment
    original_metadata_df[treatment__column] = original_metadata_df.apply(lambda row:
                                                                        map_fov_categories(plate_layout_df=original_plate_layout_df,
                                                                                           well=row[wellasint_column],
                                                                                           experiment=row[experiment__column],
                                                                                           well_column=well__column,
                                                                                           experiment_column=experiment__column,
                                                                                           treatment_column=treatment__column), axis=1)

    return original_metadata_df
