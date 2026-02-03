import os

def mk_subdir(subdir_name:str|None=None,
              parent_dir:os.PathLike=os.getcwd(),
              return_exist:bool=False,
              overwrite_dir:bool=False):
    """
    creates a folder of a give name within a given directory. By default the folder is created in the current working directory.
    By default, the folder is only created if it does not already exist.

    Inputs:
    - subdir_name. String or None. Optional. Default my_subdirectory. The name of the folder to create.
    - parent_dir. PathLike. Optional. The default folder is the current working directory. The path to the directory where the new folder should be created.
    The path must exist.
    - return_exist. Boolean. Optional. Default False. If True, a boolean value is returned indicating whether subdir_name existed or not.
    - overwrite_dir. Boolean. Optiona. Default False. If True, when the folder already exists it is overwritten.

    Ouputs:
    - If return_exist is set to False (default), the function is fruitless.
        - If overwrite_dir is False (default)
            - If subdir_name does not pre-exist
                subdir_name is created within parent_dir if subdir_name.

            - If subdir_name pre-exists
                subdir_name is not overwritten.
        
        - If overwrite_dir is True
            - If subdir_name does not pre-exist
                subdir_name is created within parent_dir if subdir_name.

            - If subdir_name pre-exists
                subdir_name is overwritten.

    - If return_exist is set to True, the function returns a bool value.
        - If overwrite_dir is False (default)
            - If subdir_name does not pre-exist
                subdir_name is created within parent_dir if subdir_name.
                False is returned.

            - If subdir_name pre-exists
                subdir_name is not overwritten.
                True is returned.
        
        - If overwrite_dir is True
            - If subdir_name does not pre-exist
                subdir_name is created within parent_dir if subdir_name.
                False is returned.
                
            - If subdir_name pre-exists
                subdir_name is overwritten.
                True is returned.
    """
    
    # set default name for the sub-directory
    if subdir_name is None:
        subdir_name="my_subdirectory"

    # create the full directory of subdir_name within parent_dir
    subdir_full_path = os.path.join(parent_dir, subdir_name)
    
    # if subdir_name is present within parent_dir
    if os.path.exists(subdir_full_path):

        # make subdir_name if overwrite_dir is True
        if overwrite_dir:
            os.makedirs(subdir_full_path)

        # return True if return_exist is set to True, nothing otherwise
        if return_exist==True:
            return True
    
    # if subdir_name is not present within parent_dir
    else:

        # make subdir_name
        os.makedirs(subdir_full_path)

        # return False if return_exist is set to True, nothing otherwise
        if return_exist==True:
            return False

       
def mkdir_tree(secondary_output_name:str|None=None,
               secondary_output_parent:os.PathLike=os.getcwd(),
               secondary_output_return_exist:bool=False,
               seconary_output_overwrite_dir:bool=False,
               metadata_name:str|None=None,
               metadata_parent:os.PathLike=os.getcwd(),
               metadata_return_exist:bool=False,
               metadata_overwrite_dir:bool=False,
               fov_name:str|None=None,
               fov_parent:os.PathLike=os.getcwd(),
               fov_return_exist:bool=False,
               fov_overwrite_dir:bool=False)-> tuple:
    """
    Creates a series of specified folders. Returns all the folders which have been created
    as os.Path objects.

    This function was created in the context of the ACID project to create folders where to
    save output files. It allows to make the main notebook much more readable. It is probably only
    useful for the ACID project.

    Inputs:
    - secondary_output_name. str or None. Optional. Default 'secondary_output'. The parameter to pass to mk_subdir subdir_name
    argument for creating the secondary_output folder.

    - secondary_output_parent. os.PathLike. Optional. Default os.getcwd(). The parameter to pass to mk_subdir parent_dir
    argument for creating the secondary_output folder.

    - secondary_output_return_exist. bool. Optional. Default False. The parameter to pass to mk_subdir return_exist
    argument for creating the secondary_output folder.

    - seconary_output_overwrite_dir. bool. Optional. Default False. The parameter to pass to mk_subdir overwrite_dir
    argument for creating the secondary_output folder.

    - metadata_name. str or None. Optional. Default 'proc_metadata'. The parameter to pass to mk_subdir subdir_name
    argument for creating the metadata folder.
    
    - metadata_parent. os.PathLike. Optional. Default os.getcwd(). The parameter to pass to mk_subdir parent_dir
    argument for creating the metadata folder.
    
    - metadata_return_exist. bool. Optional. Default False. The parameter to pass to mk_subdir return_exist
    argument for creating the metadata folder.
    
    - metadata_overwrite_dir. bool. Optional. Default False. The parameter to pass to mk_subdir overwrite_dir
    argument for creating the metadata folder.
    
    - fov_name. str or None. Optional. Default 'fov'. The parameter to pass to mk_subdir subdir_name
    argument for creating the fov folder.
    
    - fov_parent. os.PathLike. Optional. Default os.getcwd(). The parameter to pass to mk_subdir parent_dir
    argument for creating the fov folder.
    
    - fov_return_exist. bool. Optional. Default False. The parameter to pass to mk_subdir return_exist
    argument for creating the fov folder.
    
    - fov_overwrite_dir. bool. Optional. Default False. The parameter to pass to mk_subdir overwrite_dir
    argument for creating the fov folder.

    Outputs.
    The following directory tree is created:
    - secondary_output_parent
        - secondary_output_name
    
    - metadata_parent
        - metadata_name
    
    - fov_parent
        - fov_name

    
    The logic of the creation depends on the parameters passed to overwrite_dir.

    IN ADDITION: a tuple is returned.
    - position 0. The full path of seconary_output as an os.Path object.
    - position 1. The full path of metadata as an os.Path object.
    - position 2. The full path of fov as an os.Path object.

    If return_exist is set to True, for the corresponding position a tuple is returned instead of the
    the directory. The tuple has the full path as os.Path object in position 0 and True/False in position
    1 indicating whether or not the directory already existed.
    """
    
    # set default sub-directory names
    if secondary_output_name is None:
        secondary_output_name='secondary_output'
    
    if metadata_name is None:
        metadata_name='proc_metadata'
    
    if fov_name is None:
        fov_name='fov'

    # Create secondary_output subdirectory - don't modify the following lines
    mk_subdir(subdir_name=secondary_output_name,
            parent_dir=secondary_output_parent,
            return_exist=secondary_output_return_exist,
            overwrite_dir=seconary_output_overwrite_dir)

    secondary_output_path=os.path.join(secondary_output_parent,secondary_output_name)

    # Create a metadata subfolder in output directory - don't modify the following lines
    mk_subdir(subdir_name=metadata_name,
            parent_dir=metadata_parent,
            return_exist=metadata_return_exist,
            overwrite_dir=metadata_overwrite_dir)

    metadata_directory = os.path.join(metadata_parent,metadata_name)


    # Create a fov (fields_of_views) subfolder in output directory - don't modify the following lines
    mk_subdir(subdir_name=fov_name,
            parent_dir=fov_parent,
            return_exist=fov_return_exist,
            overwrite_dir=fov_overwrite_dir)

    fov_directory = os.path.join(fov_parent,fov_name)
    
    return secondary_output_path, metadata_directory, fov_directory

    
