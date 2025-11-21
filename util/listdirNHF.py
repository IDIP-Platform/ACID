import os

def listdirNHF(path:os.PathLike,
               target:str|None=None,
               exclude:str|None=None) -> list:
    """
    Returns files in a given directory as a list, avoding hidden files. Hidden files are identified because the start with a "."

    Inputs:
    - path. os.PathLike. The path to the directory whose objects have to be listed.
    - target. str or None. Optional. If str, only files containing the indicated string will be returned.
    - exclude. str or None. Optional. If str, files containing the indicated string will be excluded.

    Output. List.  
    """
    # return an error if a variable different than None and different than a string is passed to target
    if target != None:
        assert isinstance(target, str), "target should be either None or a string"
    
    # return an error if a variable different than None and different than a string is passed to exclude
    if exclude != None:
        assert isinstance(exclude, str), "exclude should be either None or a string"

    # iterate through the files in the input directory. Exclude hidden files. Include files in a list
    file_list = [f for f in os.listdir(path) if not f.startswith(".")]

    # if target is provided, filter files for the presence of the string
    if target != None:
        target_files = [f2 for f2 in file_list if target in f2]
    else:
        target_files = file_list
    
    # if exclude is provided, filter files for the presence of the string
    if exclude != None:
        keep_files = [f3 for f3 in file_list if exclude not in f3]
    else:
        keep_files = file_list
    
    # combine target_list and keep_list
    combined_files = []
    for f4 in target_files:
        if f4 in keep_files:
            combined_files.append(f4)


    #return combined_files
    return combined_files
    
