def map_image_to_condition(condition_list, image_file_names, assert_file_number=True, number_of_files_expected=1):
        """
        Map the image files to the unique conditions in the metadata dataframe.
        
        Parameters:
        condition_list (list): list of unique conditions in the metadata dataframe
        image_file_names (list): list of image file names
        
        Returns:
        dict: a dictionary mapping each condition to its corresponding file name
        """
        # create a dictionary to store the mapping between conditions and files
        condition_dict = {}
        
        for condition in condition_list:
            # find the image file corresponding to the condition
            image_file = [f for f in image_file_names if condition in f]

            if assert_file_number:
                # assert that there is exactly one image file for the condition
                assert len(image_file) == number_of_files_expected, f"Expected exactly {number_of_files_expected} image file for condition {condition}, but found {len(image_file)}"
            
            # map the image file to the condition
            condition_dict[condition] = image_file
        
        return condition_dict