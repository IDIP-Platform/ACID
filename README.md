**Project name**: ACID

Authors: Alessandro Ulivi (alessandro.ulivi.89@gmail.com)

Creation (yyyy/mm/dd): 2025/11/21

Status: ongoing (2025/12/17)

# Description:
**Background and scope**\
The project is carried out at the Center for Integrative Infectious Disease Research in Heidelberg (abbreviated to CIID; https://ciid-heidelberg.de/).

The goal of the project is to build a model to classify cells non-infected with Dengue virus, infected with Dengue virus and non-treated, and infected with Dengue virus and treated with different compounds. The projects aims at establishing a fully automated pipeline of image processing, image quantification and data analysis.

Samples are (...) cells.

The following conditions were analysed:
- uninfected cells treated with (...). Negative control.
- infected cells treated with (...). Positive control.
- uninfected cells treated with NITD-688.
- uninfected cells treated with JNJ-A07.
- uninfected cells treated with JNJ-1802.
- infected cells treated with NITD-688.
- infected cells treated with JNJ-A07.
- infected cells treated with JNJ-1802.

Data collection was carried on under grant (...) as a collaboration between the Infectious Diseases Imaging Platform of CIID (abbreviated in IDIP; https://www.idip-heidelberg.org) and the group of Prof. Dr. Christian Klein of the Institute of Pharmacy and Molecular Biology (abbreviated in IPMB) at the Univeristy of Heidelberg (https://www.ipmb.uni-heidelberg.de/en/about-us/head-of-the-institute). The current project is carried out within the SHIELD consortium (Horizon grand agreement n. 101191794).

**Imaging strategy and raw file structure:**
Samples were imaged at a Nikon Ti2 microscope equipped with CSU-W1 module for spinning disk imaging using a (...) objective at IDIP (https://www.idip-heidelberg.org/crest). Samples were prepared in parallel into a multi-well plate and imaged on the same plate. The samples were imaged, live, for 48 hours, with an acquisition every (...). Per each condition, at each timepoint, 49 distinct fields of view were acquired. During the time-lapse imaging only transmitted light was acquired.

At the end of the time-lapse imaging, the samples were fixed and stained by adapting the Cell Painting protocol (Bray et al., Nature Protocols, 2016, "Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes", DOI https://doi.org/10.1038/nprot.2016.105) to the project. Stained samples were re-acquired at the same microscope and with the same imaging conditions. The same 49 fields of view, per each condition, were re-acquired.

The following structures (staining and imaged channel) were acquired:
- nucleus (Hoechst, channel 1).
- Endoplasmic reticulum (concanavalin A fused with 488 fluorophore, channel 2).
- Actin (phalloidin fused with fluorophore 568, channel 3).
- Nonstructural protein 3 (NS3, marker of cell infection) (anti-NS3 fused with fluorophore 647, channel 4).
- Full cells in transmitted light (channel 5).

Three independent experiments were carried out. Their names are:
- Experiment A07.2
- Experiment A07.3
- Experiment A07.4.

Each imaged field of view is a single planes. The pixel size is 0.325x0.325 micron. The frame size is 1024x1024 pixels.


# Project organization
**Analysis strategy**\
At the start of the project (2025/11/21) only the data from fixed samples are analysed.

The raw files are pre-processed to extract individual fields of view and save them as independent raw files in the open access ome.tif format, along with the relevant metadata. NOTE: as the raw files are multi-position, multi-channel, single-plane images, the original/raw xml metadata contain information about all the fields of view. For this reason the original/raw xml metadata are not propagated to each individually saved field of view as their information would be misleading. Instead the choice has been made to propagate to each individually saved field of view only the relevant information, whist saving a separate, open access file the global original/raw metadata.

A 70%-30% train-test split is then carried out. As conditions are balanced, no stratification is used for the split (as of 2025/11/25). All the following steps are implemented only using the 70% train data. As of 2025/11/25, it is foreseen only a possible exception concerning the estimation of the background function. It is foreseen as plausible that a suitable function can't be estimated using the the 30% test data and, if this will be the case, the background function calculated on the 70% train data will be used instead. This might simulate a future situation when a single background function is saved and applied to all data, rather than calculated per each experiment.

As of 2025/11/25 a train-validate sub-split is not foreseen for the development of the pipeline until feature extraction (included). This is done because the initial approach will attempt using relatively standard procedures for image processing, segmentation and quantification, without ad hoc fine-tuning. The result will be evaluated qualitatively and different steps tuned accordingly. This approach might change if, for example, a fine-tuning of segmentation models will be required. In such case, a validation set will be created for the fine-tuned model evaluation. Note that a train-validate sub-split is foreseen upfront for analyses downstream to feature extration.


**Input data**\
Input data are raw files collected from the Nikon Ti2 microscope. Files have .nd2 extension (proprietary Nikon format).

The following input data and input data organization is expected:

- input_data_directory:
    - file1.nd2
    - file2.nd2
    - file3.nd2
    - fileN.nd2
    ...

- output_data_directory

The input_data_directory can contain different files and directories, as long as their names don't contain the ".nd2" string.

The output_data_directory can contain files and sub-directories.

NOTE: the pipeline was conceptualized and built in a situation where:
- more than a .nd2 raw file was present in the input_data_directory.
- Each .nd2 raw file contained at least a field of view.
- Each field of view contained at least one segmentable cell.
- All field of view have 5 channels in the following order: [...] channel-638, channel-749 corresponding to F-actin staining, channel-488 corresponding to gfp signel, channel-405 corresponding to nuclear stainig, channel-DIA corresponding to images acquired using transmitted illumination. Different file structures should be compatible with the scripts, but haven't been tested.
- Situations different than the above, including edge cases of segmentation masks with no cells or images containing detrimental artifacts, haven't been evaluated.

**Output data**\
The pipeline generates the following directory tree and output files within the output_data_directory:

- output_data_directory:
    - [date]_nef_translocation_glob_measurements.csv
    - fov:
        - file1_selected_image1.ome.tif
        - file1_selected_image2.ome.tif
        - file1_selected_image3.ome.tif
        - file2_selected_image1.ome.tif
        - file2_selected_image2.ome.tif
        - file2_selected_image3.ome.tif
        - file3_selected_image1.ome.tif
        - file3_selected_image2.ome.tif
        - file3_selected_image3.ome.tif
        - fileN_selected_imageM.ome.tif
        ...
    - seg:
        - file1_selected_image1_Cl.ome.tif
        - file1_selected_image1_Ct.ome.tif
        - file1_selected_image1_Nc.ome.tif
        - file1_selected_image2_Cl.ome.tif
        - file1_selected_image2_Ct.ome.tif
        - file1_selected_image2_Nc.ome.tif
        - file1_selected_image3_Cl.ome.tif
        - file1_selected_image3_Ct.ome.tif
        - file1_selected_image3_Nc.ome.tif
        - file2_selected_image1_Cl.ome.tif
        - file2_selected_image1_Ct.ome.tif
        - file2_selected_image1_Nc.ome.tif
        - file2_selected_image2_Cl.ome.tif
        - file2_selected_image2_Ct.ome.tif
        - file2_selected_image2_Nc.ome.tif
        - file2_selected_image3_Cl.ome.tif
        - file2_selected_image3_Ct.ome.tif
        - file2_selected_image3_Nc.ome.tif
        - file3_selected_image1_Cl.ome.tif
        - file3_selected_image1_Ct.ome.tif
        - file3_selected_image1_Nc.ome.tif
        - file3_selected_image2_Cl.ome.tif
        - file3_selected_image2_Ct.ome.tif
        - file3_selected_image2_Nc.ome.tif
        - file3_selected_image3_Cl.ome.tif
        - file3_selected_image3_Ct.ome.tif
        - file3_selected_image3_Nc.ome.tif
        - fileN_selected_imageM_Cl.ome.tif
        - fileN_selected_imageM_Ct.ome.tif
        - fileN_selected_imageM_Nc.ome.tif
        ...
        
        NOTE: from 2025/11/07 an option is introduced to also save:
        - file1_selected_image1_Cl_F.ome.tif
        - file1_selected_image1_Nc_F.ome.tif
        - file1_selected_image2_Cl_F.ome.tif
        - file1_selected_image2_Nc_F.ome.tif
        - file1_selected_image3_Cl_F.ome.tif
        - file1_selected_image3_Nc_F.ome.tif
        - file2_selected_image1_Cl_F.ome.tif
        - file2_selected_image1_Nc_F.ome.tif
        - file2_selected_image2_Cl_F.ome.tif
        - file2_selected_image2_Nc_F.ome.tif
        - file2_selected_image3_Cl_F.ome.tif
        - file2_selected_image3_Nc_F.ome.tif
        - file3_selected_image1_Cl_F.ome.tif
        - file3_selected_image1_Nc_F.ome.tif
        - file3_selected_image2_Cl_F.ome.tif
        - file3_selected_image2_Nc_F.ome.tif
        - file3_selected_image3_Cl_F.ome.tif
        - file3_selected_image3_Nc_F.ome.tif
        - fileN_selected_imageM_Cl_F.ome.tif
        - fileN_selected_imageM_Nc_F.ome.tif
        ...

    - metadata:
        - original:
            - file1_string.xml
            - file2_string.xml
            - file3_string.xml
            - fileN_string.xml
        
        - proc_file_info:
            - [date]_NefCellLoc_metadata.csv

In addition, a directory named "secondary_output" is created inside the working directory. Inside secondary_output are saved hyperparameters per each run of the pipeline.


OUTPUT DATA DESCRIPTION
- [date]_nef_translocation_glob_measurements.csv. Comma-separated table with intensity measurements of nuclei and cytosols. The file includes all measurements per each field of view (also called scene) of each raw file. Rows are individual cells. Columns are:

    - raw_file_name. The name of the .nd2 raw file saved at the Nikon Ti2 microscope and the field of views (either multi-points or single positions). It is the input to the pipeline and it is saved in the input_data_directory.

    - scene_name. The name of the sub-file within the .nd2 raw file. The sub-file is the individual field of view to analyze. Some input files only contain a single field of view. Some others contain multiple fields of view (multi-positions).

    - acquisition_date_yyyymmdd. The date of raw file acquisition at the Nikon Ti2 microscope.

    - processing_date_yymmdd. The date of the extraction and saving as ome.tif of the field of view (also called scene) from the .nd2 raw file.

    - ome_tif_file_name. The name used for saving the field of view as an ome.tif file. This is done first main loop of the part 1 notebook (step 1.3 below).

    - location. The location where raw file acquisition took place.

    - microscope. The microscope used for raw file acquisition.

    - objective. The objective used for raw file acquisition.

    - donor. The code of the donor patient from which (...) was derived.

    - transfection. The gene samples have been transfected with (soluble gfp control or any of the NEF variants).

    - stiffness. The material onto which cells have been seeded.

    - stimulation. The (...).

    - time_of_stimulation. The duration of the stimulation.

    - channel_0. The reporter acquired at position 0 on the channel axis. This corresponds to the first channel acquired at the Nikon Ti2 microscope. NOTE: position numbering follows python convension (aka: starts at 0 and not at 1).

    - channel_1. The reporter acquired at position 1 on the channel axis. This corresponds to the second channel acquired at the Nikon Ti2 microscope. NOTE: position numbering follows python convension (aka: starts at 0 and not at 1). By default, this is assumed to be f-actin, and it is used for cell segmentation.

    - channel_2. The reporter acquired at position 2 on the channel axis. This corresponds to the third channel acquired at the Nikon Ti2 microscope. NOTE: position numbering follows python convension (aka: starts at 0 and not at 1).

    - channel_3. The reporter acquired at position 3 on the channel axis. This corresponds to the fourth channel acquired at the Nikon Ti2 microscope. NOTE: position numbering follows python convension (aka: starts at 0 and not at 1). By default, this is assumed to be a nucleid acid staining, and it is used for cell and nucleus segmentation.

    - channel_4. The reporter acquired at position 4 on the channel axis. This corresponds to the fourth channel acquired at the Nikon Ti2 microscope. NOTE: position numbering follows python convension (aka: starts at 0 and not at 1).

    - physical_size_unit_x. The unit used for the column 'physical_size_x'.

    - physical_size_unit_y. The unit used for the column 'physical_size_y'.

    - dtype. The .nd2 raw file data type.

    - size_t. The size of the time axis. In other words, the number of imaged timepoints.

    - size_c. The size of the channel axis. In other words, the number of imaged channels.

    - size_z. The size of the z axis. In other words, the number of imaged z planes.

    - size_y. The size of the y axis. In other words, the number of y pixels.

    - physical_size_y. The physical size of a single pixel on the y dimension. The unit is indicated in the column 'physical_size_unit_y'.

    - size_x. The size of the x axis. In other words, the number of x pixels.

    - physical_size_x. The physical size of a single pixel on the x dimension. The unit is indicated in the column 'physical_size_unit_x'.

    - dims_order. The order of the dimension in the ome.tif file. 'T'=time, 'C'='channel', 'Z'='z_dimension', 'Y'='y_dimension', 'X'='x_dimension'.

    - 'segmentation_date_yymmdd_Nc_Cl'. The date of the segmentation of nuclei and cells. This is done in the second main loop of part 1 notebook. Step 2.2 below.

    - 'cell_segmentation'. The name of the saved cell segmentation file.

    - 'nucleus_segmentation'. The name of the saved nucleus segmentation file.

    - 'cytosol_segmentation'. The name of the saved cytosol segmentation file.

    - 'segmentation_date_yymmdd_Ct'. The date of the segmentation of cytosol. This is done in the main loop of part 2 notebook. Step 3.2 below.

    - label. The value of the label for the cell whose measurements are in the row. The label unequivocally identifies the cell in the cytosol segmentation mask, while it might be different for the nucleus and cell masks.

    - (Nc/Ct)_area. The area of the nucleus/cytosol for a measured cell, in number of pixels.

    - (Nc/Ct)_centroid-N. The N-coordinate of the centroid of the nucleus/cytosol for a measured cell. [...TO BE CHECKED...When the segmentation mask is 2D as in the present project, this corresponds to the Y coordinate and starts from the top-left corner of the image...].

    - (Nc/Ct)_centroid-M. The M-coordinate of the centroid of the nucleus/cytosol for a measured cell. [...TO BE CHECKED...When the segmentation mask is 2D as in the present project, this corresponds to the X coordinate and starts from the top-left corner of the image...].

    - (Nc/Ct)_intensity_mean-(channel). The mean intensity value of the the nucleus/cytosol for a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - (Nc/Ct)_intensity_max-(channel). The max intensity value of the the nucleus/cytosol for a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - (Nc/Ct)_intensity_min-(channel). The min intensity value of the the nucleus/cytosol for a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - background_offset-(channel). The median value of the background pixels in a given field of view and for a given channel. Background pixels are all the pixels not present in the nucleus segmentation or in the cell segmentation masks (aka, pixels which are not present in neither or the masks). The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.


- fov. Directory containing individual fields of view extracted from the .nd2 raw file (Nikon proprietary file format) and saved as ome.tif files (free file format with standardized metadata). Fields of views from multiple .nd2 raw files are pooled together in the fov directory.

- seg. Directory containing the segmentation masks of each individual field of view extracted from the .nd2 raw file. 3 masks are always saved per each selected image:
    - the nuclei segementation from cellpose (suffix _Nc)
    - the cell segmentation from cellpose (suffix _Cl)
    - the cytosol segmentation obtained from the subtraction of the nuclei segmentation mask from the cell segmentation mask, after nuclei and cells preprocessing/filtering (suffix _Ct).

   From 2025/11/07 it as also possible to save:
   - the nuclei segementation after preprocessing/filtering (suffix _Nc_F)
   - the cell segementation after preprocessing/filtering (suffix _Cl_F)
   
   The segmentation masks of the selected images from multiple .nd2 files are pooled together in the seg directory.

- metadata
    - orginal. Directory containing the original xml metadata extracted from each .nd2 raw file and transformed into a string.
    - proc_file_info. Directory containg a csv file with information about the metadata and the processing of individual output files. The file has the following name structure {date_in_format_yymmdd}_nef_translocation_metadata.csv


**Analysis steps**:\
1) EXTRACTION OF SELECTED IMAGES AND METADATA

    1.1) extraction of origiginal xml metadata and saving in the output_data_directory->metadata->original directory.

    1.2) Extraction of the metadata contained in the raw file and individual fields of view (also called scene) names. Addition of information directly inputed by the user. These metadata are used to form a table which in the following steps is referred to as 'metadata_df'.

    1.3) Saving of individual field of views (also called scene) in the output_data_directory->fov directory as ome.tif file. Note that each ome.tif file contains the relative metadata.

NOTE: from here one the 'metadata_df' table is used to link field of views (also called scenes) to the relevant information.

2) NUCLEI AND CELL SEGMENTATION

    2.1) IMAGE PREPROCESSING
        2.1.1) The nuclear channel is selected. Output name: nucleus_arr.
        2.1.2) The nuclear channel and the f-actin channel are selected and stacked together in and individual array. The order of the signal on the stacked axis is: nuclear_channel, f-actin_channel. Output name: cell_arr.
        2.1.1) Median filtering. A 3x3 pixel kernel is used, independtly, for the nuclear and f-actin channels in the cell_arr. A 10x10 kernel is used for the nucleus_arr.
        2.1.2) downsampling. A factor of 2 is used. This means that images are downsampled at half of their original size. The downsampling method is local_mean (see https://scikit-image.org/docs/0.25.x/api/skimage.transform.html#skimage.transform.resize_local_mean). Note: for cell_arr, the axis onto which nucleus and f-actin channels are stacked, it is not down-sampled.
    
    2.2) IMAGE SEGMENTATION
        2.2.2) Cell segmentation (based on CellPoseSAM - version 4.0.6) using cell_arr.
        2.2.3) Cell segmentation mask upsampling (method nearest neightbour) to their original size. Note: the axis onto which nucleus and f-actin channels are stacked, it is not up-sampled.
        2.2.4) Nucleus segmentation (based on CellPoseSAM - version 4.0.6) using nucleus_arr.
        2.2.5) Nucleus segmentation mask upsampling (method nearest neightbour) to its original size.
        2.2.6) Saving of segmentation masks in the output_data_directory->seg directory as ome.tif file. Note that each ome.tif file contains the relative metadata.
    
    2.3) UPDATING AND SAVING OF THE METADATA
    The 'metadata_df' formed in step 1.2 is updated with the segmentation mask information and it is saved in the output_data_directory->metadata->proc_file_info directory as [date]_nef_translocation_metadata.csv file.

    2.4) SAVING OF HYPERPARAMETERS (in the repository->secondary_output directory)

3) EXTRACTION OF INTENSITY MEASUREMENTS
    
    3.1) IMAGE AND SEGMENTATION MASKS PREPROCESSING
        3.1.1) Fields of view channel axis reordering -> the channel axis is move to position -1, for compatibility with skimage.measure.regionprops.
        3.1.2) Filtering of labels:
                3.1.2.1) Cells with area too large or too small (the highpass and lowpass thresholds are inputed by the user) are removed.
                3.1.2.2) Cells with no nucleus are removed.
                3.1.2.3) Nuclei with no cell are removed.
                3.1.2.4) Cells touching the image edges are removed.
                3.1.2.5) Nuclei are forced to be completely contained into cell masks.
                3.1.2.6) Nuclei belonging to the same cell are assigned the same label.
                3.1.2.7) Pixels whose intensity value correspond to the min or the max value of the raw image data type are removed.
                3.1.2.8) The matching of the label values for cells and corresponding nuclei is double-checked.
        3.1.3) Invidual channels are independently smoothed using a gaussian kernel (by default of sigma size 0.6)
    
    3.2) CYTOSOL SEGMENTATION
        3.2.1) The cytosol segmentation mask is obtained by subtracting the filterd nuclei segmentation mask from the filered cell segmentation mask. This ensures that cytosol label values match the values of their corresponding nuclei.
        3.2.2) Saving of cytosol segmentation masks in the output_data_directory->seg directory as ome.tif file. Note that each ome.tif file contains the relative metadata.
    
    3.3) (optional) SAVING OF THE NUCLEUS AND CELL SEGMENTATION MASKS AFTER PREPROCESSING (point 3.1)
    
    3.4) UPDATING AND SAVING OF THE METADATA
    The 'metadata_df' formed in step 1.2 is updated with the cytosol segmentation mask information and it is saved in the output_data_directory->metadata->proc_file_info directory as [date]_nef_translocation_metadata.csv file. If date correspond in this step and step 2.3, the former metadata_df saved file is overwritten.
    NOTE: the metadata dataframe is not updated for the nucleu and cell segmentation masks saved in point 3.3.
    
    3.5) FEATURE EXTRACTION
    Features are calculated per each cell of each field of view. By default the following measurements are calculated: intensity mean, intensity max, intensity min, area. Features are collected in a single, global measurements file called 'glob_measurements_df' in the following steps.

    3.6) SAVE FEATURE AND METADATA
    'glob_measurements_df' is updated with metadata information and saved as '[date]_nef_translocation_glob_measurements.csv' in the output_data_directory

    3.7) SAVING OF HYPERPARAMETERS (in the repository->secondary_output directory)


# Run the analysis:
The part1_raw_to_segmentation.ipynb notebook can be used to run the analysis steps 1 and 2.
The part2_segmentation_to_measurement.ipynb notebook can be used to run the analysis step 3.


**Folders**\
The following folders are present:

- utils. Contains scripts of general interest, which are used across different projects and across different steps of the same project.

- image_preparation. Contains scripts used to extract metadata and prepare images for segmentation and feature extraction.

- image_processing. Contains scripts used to modify/process the images including filtering and resizing.

- feature_extraction. Contains scripts used to measure cell properties, including channels intensities.

- image_quality_control. Contains scripts used to perform quality controls on the cells to analyse or exclude.

- secondary_output. Contains saved analysis hyperparameters.


**Structure of file name**:\
__INPUT FILE NAME__\

.nd2 raw file: {donor}_{transfection}_{stiffness}_{stimulation}_{timeofstimulation}.extension



__PROCESSED FILE NAME__\

- extracted fields of view (also called scenes) saved in the fov directory

{donor}_{transfection}_{stiffness}_{stimulation}_{timeofstimulation}_{serial_series_number}.ome.tif

{serial_series_number} is a progressive numbers which is give to each sub-file within the raw .nd2 file. When a raw .nd2 file does not have multiple sub-files, a number 0 is still assigned.


- cell segmentation masks (saved in seg directory)
{donor}_{transfection}_{stiffness}_{stimulation}_{timeofstimulation}_{serial_series_number}_Cl.ome.tif

- nucleus segmentation masks (saved in seg directory)
{donor}_{transfection}_{stiffness}_{stimulation}_{timeofstimulation}_{serial_series_number}_Nc.ome.tif

- cytosol segmentation masks (saved in seg directory)
{donor}_{transfection}_{stiffness}_{stimulation}_{timeofstimulation}_{serial_series_number}_Ct.ome.tif



# Dependencies:
The file nef_translocation_develop.yml can be used for creating the environment containg script dependencies during the project development. The file nef_translocation_neo.yml file has been used to create the python environment on a computer workstation used for running the analysis, with gpu support, at the Infectious Diseases Imaging Platform of the CIID.

The following module versions are used:
- python==3.12.12
- jupyterlab==4.5.0
- pip==25.3
- numpy==2.2.6
- matplotlib==3.10.8
- scipy==1.16.3
- pandas==2.2.3
- dask==2025.11.0
- numba==0.62.1
- tifffile==2025.10.16
- statsmodels==0.15.5
- scikit-image==0.25.2
- scikit-learn==1.7.2
- seaborn==0.13.2
- pytorch==2.5.1
- torchvision==0.20.1
- torchaudio==2.5.1
- napari==0.6.6
- pycytominer==1.2.4
- roifile==2025.5.10
- cellpose==4.0.7
- bioio==3.0.0
- bioio-nd2==1.5.0


# Notes:



# To do list:



