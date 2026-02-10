import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def background_x_2condition_plot(unique_condition_1:list,
                                 unique_condition_2:list,
                                 condition_dict:dict,
                                 ch_to_plot:int|None=None,
                                 channel_axis:int|None=None,
                                 ax=None):
    """
    This function is contingent to the ACID project.
    """
    if ch_to_plot is not None:
        assert channel_axis is not None, "you must pass a channel_axis if a channel is passed to ch_to_plot"

    # iterate over the firt conditions
    for cond1_pos, cond1 in enumerate(unique_condition_1):

        # iterate over the second conditions
        for cond2_pos, cond2 in enumerate(unique_condition_2):
            
            # get the background function
            condition_background_function = condition_dict[str(cond1)+str(cond2)]
            
            # get the specific channel if needed
            if ch_to_plot is not None:
                condition_background_function = np.unstack(condition_background_function, axis=channel_axis)[ch_to_plot]
                

            # map the background in the appropriate sub-plot position to display the background function
            if len(unique_condition_1)==1 and len(unique_condition_2)==1:
                ax.imshow(condition_background_function)
            elif len(unique_condition_1)==1 and len(unique_condition_2)>1:
                ax[cond2_pos].imshow(condition_background_function)
            elif len(unique_condition_1)>1 and len(unique_condition_2)==1:
                ax[cond1_pos].imshow(condition_background_function)
            else:
                ax[cond1_pos][cond2_pos].imshow(condition_background_function)
