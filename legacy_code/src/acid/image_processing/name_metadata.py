import datetime
import pandas as pd

def try_string_to_integer(info_bit,default_return=None):
        try:
            subinfobit = int(info_bit)
        except:
            subinfobit = default_return
        return subinfobit

def extract_subinfo_bit(infobit:str,
                        separator:str,
                        subinfobit_position:int,
                        extraction_method:str|None=None,
                        default_return=None):

    try:
        # split infobit
        infobit_split = infobit.split(separator)

        # get subinfobit
        subinfobit = infobit_split[subinfobit_position]

        # apply method if necessary
        if extraction_method!=None:

            if extraction_method=='try_int':
                subinfobit = try_string_to_integer(subinfobit, default_return=subinfobit)

    except:
        subinfobit = default_return

    # return default if necessary
    return subinfobit


def extract_name_metadata(file_name:str,
                          separator:str|None=None,
                          infobits:dict|None=None,
                          replace_str:str|None=None,
                          infobit_position:int=0,
                          infobit_separator_position:int=1,
                          subinfobit_position_position:int=2,
                          extraction_method_position:int=3)->dict:
    """

    """

    # set default separator
    if separator is None:
        separator='_'

    # replace str in name if necessary
    if replace_str!=None:
        file_name = file_name.replace(' ', separator)

    # split_input_file_name
    file_name_split = file_name.split(sep=separator)

    # intitialize the output dictionary
    metadata_dict={}

    if infobits!=None:
        # iterate through the metadata to extract
        for meta_data in infobits:

            # get the information required to collect the infomation
            infobit_howto = infobits[meta_data]

            # collect the information bit in the splitted file_name if required
            if isinstance(infobit_howto,tuple):

                # get the information bit
                infobit = file_name_split[infobit_howto[infobit_position]]

                # get the infobit_separator
                infobit_separator = infobit_howto[infobit_separator_position]

                # get the subinfobit_position
                subinfobit_position = infobit_howto[subinfobit_position_position]

                # get the extraction_method
                extraction_method = infobit_howto[extraction_method_position]

                # get the subinfobit
                subinfobit = extract_subinfo_bit(infobit=infobit,
                                                separator=infobit_separator,
                                                subinfobit_position=subinfobit_position,
                                                extraction_method=extraction_method,
                                                default_return=infobit)

                # link subinfobit to meta_data in metadata_dict
                metadata_dict[meta_data]=subinfobit

            else:
                metadata_dict[meta_data]=infobit_howto

    return metadata_dict


# # def form_name_metadata_df(files_list:list,
# #                      original_well_string_1:str = "Well",
# #                      original_acquisition_settings_string_1:str = "Channel",
# #                      original_channel_separator_1: str = ",",
# #                      original_sequence_string_1:str = "Seq",
# #                      original_view_string_1:str = "_v",
# #                      original_imaged_channel_string_1:str = "_c",
# #                      original_projected_planes_string_start_1:str = "_z_",
# #                      original_projected_planes_string_end_1:str = ".tif",
# #                      original_plane_separator_1:str = "-") -> pd.DataFrame:
# #     """
# #     Extract metadata from the names of a list of files and organises them into a pandas data frame.
# #     The function iterates through the list of files and extracts metadata using extract_name_metadata function. Accumulates metadata to a collection dictionary which is used to form the output
# #     data frame.

# #     Inputs:
# #     Input:
# #     - files_list. iterable. List-like object with files names to extract metadata from. Elements are passed to extract_name_metadata function as file_name argument.

# #     - original_well_string_1. str. Optional. Default "Well". The string to be used in file_name to regognize the position of the well
# #     information bit. It is expected that the well information bit corresponds to the 3 digits directly following original_well_string:
# #     "original_well_string{CapitalLetter}{2-digits-number}" (default "Well{CapitalLetter}{2-digits-number}").
# #     The parameter is  passed to extract_name_metadata function as original_well_string argument.

# #     - original_acquisition_settings_string_1. str. Optional. Default "Channel". The string to be used in file_name to recognize the position of
# #     the acquisition_settings information bit. It is expected that the acquisition_settings information bit corresponds to the 11
# #     digits directly following original_acquisition_settings_string: "original_acquisition_settings_string{3-digits-number},{3-digits-number},{3-digits-number}"
# #     (default "Channel{3-digits-number},{3-digits-number},{3-digits-number}").
# #     The parameter is  passed to extract_name_metadata function as original_acquisition_settings_string.

# #     - original_channel_separator_1. str. The separator which is used in the acquisition_settings information bit to separate individual imaged channels.
# #     The parameter is  passed to extract_name_metadata function as original_channel_separator.

# #     - original_sequence_string_1. str. Optional. Default "Seq". The string to be used in file_name to regognize the position of
# #     the sequence information bit. It is expected that the sequence information bit corresponds to the 4
# #     digits directly following original_sequence_string: "original_sequence_string{4-digits-number}"
# #     (default "Seq{4-digits-number}").
# #     The parameter is  passed to extract_name_metadata function as original_sequence_string argument.

# #     - original_view_string_1. str. Optional. Default "_v". The string to be used in file_name to regognize the position of
# #     the view information bit. It is expected that the view information bit corresponds to the 2
# #     digits directly following original_view_string: "original_view_string{2-digits-number}"
# #     (default "_v{2-digits-number}").
# #     The parameter is  passed to extract_name_metadata function as original_view_string argument.

# #     - original_imaged_channel_string_1. str. Optional. Default "_c". The string to be used in file_name to regognize the position of
# #     the imaged_channel information bit. It is expected that the imaged_channel information bit corresponds to the 1
# #     digit directly following original_imaged_channel_string: "original_imaged_channel_string{1-digits-number}"
# #     (default "_c{1-digit-number}").
# #     The parameter is  passed to extract_name_metadata function as original_imaged_channel_string argument.

# #     - original_projected_planes_string_start_1. str. Optional. Default "_z_". The string to be used in file_name to regognize the start position of
# #     the projected_planes information bit. As the length of this information bit, as a string, is variable in the file names, a following argument
# #     (original_projected_planes_string_end) is needed to get it. It is expected that the projected_planes information bit within
# #     original_projected_planes_string_start and original_projected_planes_string_end is composed of 1-digit numbers separated by "-":
# #     "original_projected_planes_string_start{1-digit-number}-{1-digit-number}...original_projected_planes_string_end"
# #     (default "_z_{1-digit-number}-{1-digit-number}.tif").
# #     The parameter is  passed to extract_name_metadata function as original_projected_planes_string_start argument.

# #     - original_projected_planes_string_end_1. str. Optional. Default ".tif". The string to be used in file_name to regognize the emd position of
# #     the projected_planes information bit. As the length of this information bit, as a string, is variable in the file names, a previous argument
# #     (original_projected_planes_string_start) in combination with the present argument are needed to get it.
# #     It is expected that the projected_planes information bit within original_projected_planes_string_start and original_projected_planes_string_end
# #     is composed of 1-digit numbers separated by "-":
# #     "original_projected_planes_string_start{1-digit-number}-{1-digit-number}...original_projected_planes_string_end"
# #     (default "_z_{1-digit-number}-{1-digit-number}.tif").
# #     The parameter is  passed to extract_name_metadata function as original_projected_planes_string_end argument.

# #     - original_plane_separator_1. str. Optional. Default "-". The separator which is used in the projected_planes information bit to separate individual projected planes.
# #     The parameter is  passed to extract_name_metadata function as original_plane_separator argument.

# #     Output:
# #     Pandas Data Frame. A data frame with the metadata extracted from the file name. Each row correspond to a file in file_list. The following columns are present:
# #     - file_name: The name of the file. Str.
# #     - well: The well information bit. It is expected to be a string of 3 digits.
# #     - acquisition_settings: The acquisition_settings information bit. It is expected to be a string of 3 3-digits numbers separated by "_".
# #     - sequence: The sequence information bit. Integer.
# #     - view: The view information bit. Integer.
# #     - imaged_channel: The imaged_channel information bit. Integer.
# #     - projected_planes: The projected_planes information bit. It is expected to be a string of 2 to 3 1-digit numbers separated by "_".

# #     """

# #     # define a function to enter the metadata extracted from individual files (output of extract_name_metadata function) into a common collection dictionary
# #     def add_metadata_to_dict(collect_dictionary, individual_file_dict):

# #         """
# #         Updates a dictionary-B with the key-value pairs of a second, dictionary-A. Per each key-A in dictionary-A, if it is not present in dictionary-B, the key is added to dictionary-B (key-B)
# #         and paired to the corresponding value-A into a list (value-B). If key-A is already present in dictionary-B as key-B, value-B is updated by appending value-A to the list.

# #         Inputs:
# #         - collect_dictionary. The dictionary-B to update.
# #         - individual_file_dict. The dictionary-A to use to update dictionary-B.

# #         Outputs:
# #         dict. Dictionary-B.
# #         """

# #         # iterate through the keys of the individual_file_dict
# #         for k in individual_file_dict:
# #             # if k is not present in collect_dictionary
# #             if k not in collect_dictionary:

# #                 # add k to collect_dictionary and pair it with a list. Inside the list put the value linked to k in individual_file_dict
# #                 collect_dictionary[k]=[individual_file_dict[k]]

# #             # if k is already present in collect_dictionary
# #             else:
# #                 # update the value linked to k in collect_dictionary, but appending the value linked to k in individual_file_dict
# #                 collect_dictionary[k].append(individual_file_dict[k])

# #         return collect_dictionary


# #     # initialize a dictionary to collect metadata from all files. It will be used to form the output data frame
# #     collection_dictionary = {}

# #     # iterate through the files of file list
# #     for f in files_list:

# #         # extract metadata from the file
# #         file_metadata_dict = extract_name_metadata(file_name=f,
# #                                                    original_well_string= original_well_string_1,
# #                                                    original_acquisition_settings_string=original_acquisition_settings_string_1,
# #                                                    original_channel_separator = original_channel_separator_1,
# #                                                    original_sequence_string=original_sequence_string_1,
# #                                                    original_view_string=original_view_string_1,
# #                                                    original_imaged_channel_string=original_imaged_channel_string_1,
# #                                                    original_projected_planes_string_start=original_projected_planes_string_start_1,
# #                                                    original_projected_planes_string_end=original_projected_planes_string_end_1,
# #                                                    original_plane_separator=original_plane_separator_1)

# #         # update the collection dictionary
# #         add_metadata_to_dict(collection_dictionary, file_metadata_dict)

# #     # use collection dictionary to form a data frame
# #     filename_metadata_df = pd.DataFrame.from_dict(collection_dictionary)

# #     return filename_metadata_df
