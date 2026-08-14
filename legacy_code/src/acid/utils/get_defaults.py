import os
import datetime

def default_file_name(file_list:list,
                      from_file_name:bool=False,
                      directory_path:str=os.PathLike,
                      separator:str|None=None,
                      date_position:int=0,
                      date_format:str|None=None,
                      reverse:bool=True)->str:
    """
    Given a list of file names (file_list), gets the file name with the closest/furthest date,
    either from the file name (from_file_name set to True) or from the last modification date
    of the file (from_file_name set to False - Default).

    If from_file_name is True, the function assumes that the date is included in the file name,
    separated by a specific separator (separator, default is "_") and located at a specific position
    (date_position, default is 0). The date is then parsed using the specified date_format
    (default is '%Y%m%d'). For example, if the file name is "20230615_experiment_data.csv",
    the date would be "20230615". separator would be "_" and date_position would be 0. date_format
    would be '%Y%m%d'.

    If reverse is True (default), the function returns the file with the most recent date.
    If reverse is False, it returns the file with the oldest date.

    Args:
    - file_list (list): List of file names to evaluate.
    - from_file_name (bool): Whether to extract the date from the file name or from the last
    modification date.
    - directory_path (str): Directory path where the files are located (used if from_file_name
    is False).
    - separator (str): Separator used in the file name to split components (used if
    from_file_name is True).
    - date_position (int): Position of the date component in the split file name (used if
    from_file_name is True).
    - date_format (str): Format of the date in the file name (used if from_file_name is True).
    - reverse (bool): If True, returns the file with the most recent date; if False,
    returns the oldest.

    Returns:
    - str: File name with the closest/furthest date in file_list.
    """

    # set dafault separator and date_format
    if separator is None:
        separator="_"

    if date_format is None:
        date_format='%Y%m%d'

    # initialize a dictionary to link file names to their dates
    file_date_link = {}

    # initialize a list to collect dates
    dates = []

    # iterate through the target files
    for f in file_list:

        # get date from file name or...
        if from_file_name:
            # split file name and collect the date
            f_date = datetime.datetime.strptime(f.split(separator)[date_position], date_format)

            # collect date into dates
            dates.append(f_date)

            # link date to original file name
            file_date_link[f_date]=f

        # ...get date from last modification date of the file
        else:
            # get the last modification date of the file
            f_date = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(directory_path,f)))

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


def default_multifile_name(file_list: list,
                            from_file_name: bool = False,
                            directory_path: str = os.PathLike,
                            separator: str | None = None,
                            date_position: int = 0,
                            date_format: str | None = None,
                            reverse: bool = True) -> list:
    """
    Given a list of file names, returns all files that share the most recent or oldest date,
    either from the file name (from_file_name=True) or from the last modification date
    (from_file_name=False).

    Args:
    - file_list (list): List of file names to evaluate.
    - from_file_name (bool): Whether to extract the date from the file name or from the last modification date.
    - directory_path (str): Directory path where the files are located (used if from_file_name=False).
    - separator (str): Separator used in the file name to split components (used if from_file_name=True).
    - date_position (int): Position of the date component in the split file name (used if from_file_name=True).
    - date_format (str): Format of the date in the file name (used if from_file_name=True).
    - reverse (bool): If True, considers the most recent date; if False, considers the oldest.

    Returns:
    - list: List of file names sharing the closest/furthest date in file_list.
    """

    if separator is None:
        separator = "_"

    if date_format is None:
        date_format = '%Y%m%d'

    # dictionary linking file names to their dates
    file_date_link = {}

    for f in file_list:
        if from_file_name:
            f_date = datetime.datetime.strptime(f.split(separator)[date_position], date_format)
        else:
            f_date = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(directory_path, f)))
        file_date_link[f] = f_date

    if not file_date_link:
        return []

    # get the extreme date
    extreme_date = max(file_date_link.values()) if reverse else min(file_date_link.values())

    # return all files that have this date
    return [f for f, d in file_date_link.items() if d == extreme_date]
