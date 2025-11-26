import os
import datetime

def default_file_name(file_list:list,
                      separator:str="_",
                      date_position:int=0,
                      date_format:str='%Y%m%d',
                      reverse:bool=True)->str:
    
    # initialize a dictionary to link file names to their dates
    file_date_link = {}

    # initialize a list to collect dates
    dates = []

    # iterate through the target files
    for f in file_list:

        # split file name and collect the date
        f_date = datetime.datetime.strptime(f.split(separator)[date_position], date_format)

        # collect date into dates
        dates.append(f_date)

        # link date to original file name 
        file_date_link[f_date]=f
    
    # sort dates
    sorted_dates = sorted(dates, reverse=reverse)
    
    # get the clostest/furthest date
    return_date = sorted_dates[0]

    # get the file linked to the closest/furthest date
    return_file = file_date_link[return_date]

    return return_file