import os

def listdirNHF(path:os.PathLike,
               target:str|list|None=None,
               exclude:str|list|None=None,
               hidden_file_signature:str=".") -> list:
    """
    Returns files in a given directory as a list, avoding hidden files. Hidden files are identified because the start with a "."

    Inputs:
    - path. os.PathLike. The path to the directory whose objects have to be listed.
    - target. str or None. Optional. If str, only files containing the indicated string will be returned.
    - exclude. str or None. Optional. If str, files containing the indicated string will be excluded.

    Output. List.  
    """
    # return an error if a variable different than None or a string or a list of strings is passed to target
    if target != None:
        assert (isinstance(target, str) or isinstance(target, list)), "target should be either None or a string or a list"
        
        if isinstance(target, list):
            assert all([isinstance(sub_target, str) for sub_target in target]), "if a list, target should be a list of strings"

    # return an error if a variable different than None or a string or a list of strings is passed to exclude
    if exclude != None:
        assert (isinstance(exclude, str) or isinstance(exclude, list)), "exclude should be either None or a string or a list"

        if isinstance(exclude, list):
            assert all([isinstance(sub_exclude, str) for sub_exclude in exclude]), "if a list, exclude should be a list of strings"

    # iterate through the files in the input directory. Exclude hidden files. Include files in a list
    file_list = [f for f in os.listdir(path) if not f.startswith(hidden_file_signature)]

    # if target is provided
    if target != None:
        # if a single string is provided as target, filter files for the presence of the string
        if isinstance(target, str):
            target_files = [f2 for f2 in file_list if target in f2]
        
        # if a list of strings is provided as target
        else:
            # initialize target_file list, to store files containing target string
            target_files = []

            # iterate through non-hidden files
            for f4 in file_list:

                # iterate through the target strings
                for tf in target:

                    # if the target string is in the non-hidden file
                    if tf in f4:

                        # add the non-hidden file to the collection list if it hasn't been added already
                        if f4 not in target_files:
                            target_files.append(f4)
    else:
        target_files = file_list
    
    # if exclude is provided, filter files for the presence of the string
    if exclude != None:

        # if a single string is provided as exclude, filter files for the presence of the string
        if isinstance(exclude, str):
            keep_files = [f3 for f3 in file_list if exclude not in f3]
        
        # if a list of strings is provided as exclude
        else:
            print("here")
            # initialiye keep_files list, to store files to keep as they don't contain the string to exclude
            keep_files = []

            # iterate through non-hidden files
            for f5 in file_list:
                
                # initialize a variable to decide whether to keep the file
                keep_this_file = True

                # iterate through the exclude strings
                for ef in exclude:
                    
                    # if the exclude string is in the non-hidden file, exclude the non-hidden file
                    if ef in f5:
                        keep_this_file = False
                    
                # keep the file if non ot the exclude string was in it
                if keep_this_file:
                    # add the non-hidden file to the collection list if it hasn't been added already
                    if f5 not in keep_files:
                        keep_files.append(f5)

    else:
        keep_files = file_list
    
    # combine target_list and keep_list
    combined_files = []
    for f6 in target_files:
        if f6 in keep_files:
            combined_files.append(f6)


    #return combined_files
    return combined_files
    
