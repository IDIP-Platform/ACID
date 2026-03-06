**Project name**: ACID

Authors: Alessandro Ulivi (alessandro.ulivi.89@gmail.com)

Creation (yyyy/mm/dd): 2025/11/21

Status: ongoing (2026/03/06)

# Description:
**Background and scope**\
The project is carried out at the Center for Integrative Infectious Disease Research in Heidelberg (abbreviated to CIID; https://ciid-heidelberg.de/).

The goal of the project is to build a model to classify cells non-infected with Dengue virus, infected with Dengue virus and non-treated, and infected with Dengue virus and treated with different compounds. The projects aims at establishing a Python-based pipeline of image processing, image quantification and data analysis.

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
- Nonstructural protein 3 (NS3, marker of cell infection) (anti-NS3 antibody fused with fluorophore 647, channel 4).
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

A 70%-30% train-test split is carried out. As conditions are balanced, no stratification is used for the split (as of 2025/11/25). All the following steps are implemented only using the 70% train data. As of 2025/11/25, it is foreseen only a possible exception concerning the estimation of the background function. It is foreseen as plausible that a suitable function can't be estimated using the the 30% test data and, if this will be the case, the background function calculated on the 70% train data will be used instead. This might simulate a future situation when a single background function is saved and applied to all data, rather than calculated per each experiment.

As of 2025/11/25 a train-validate sub-split is not foreseen for the development of the pipeline until feature extraction (included). This is done because the initial approach will attempt using relatively standard procedures for image processing, segmentation and quantification, without ad hoc fine-tuning. The result will be evaluated qualitatively and different steps tuned accordingly. This approach might change if, for example, a fine-tuning of segmentation models will be required. In such case, a validation set will be created for the fine-tuned model evaluation. Note that a train-validate sub-split is foreseen upfront for analyses downstream to feature extration.


**Input data**\
Input data are raw files collected from the Nikon Ti2 microscope. Files have .nd2 extension (proprietary Nikon format).

The following input data and input data organization is expected:

- input_data_directory:
    - experiment_directory:
        - file1.nd2
        - file2.nd2
        - file3.nd2
        - fileN.nd2
        ...
    - plate_layout.csv

- output_data_directory

The input_data_directory can contain different files, but expects only one or multiple experiment_directory as sub-directory.

The input_data_directory uses the last saved .csv file as default plate layout file.

The experiment_directory can contain different files and directories, as long as their names don't contain the ".nd2" string.

The output_data_directory can contain files and sub-directories.

NOTE: the pipeline was conceptualized and built in a situation where:
- more than one experiment_directory was present.
- more than a .nd2 raw file was present in the input_data_directory.
- Each .nd2 raw file contained at least a field of view.
- Each field of view contained at least one segmentable cell.
- All field of view have 5 channels in the following order: channel-405 corresponding to Hoechst staining, channel-488 corresponding to concanavalin A staining; channel-568 corresponding to phalloidin staining; channel-647 corresponding to anti-NS3 staining, channel-TL corresponding to transmitted light acquisition.

Situations different than the above, including edge cases of segmentation masks with no cells or images containing detrimental artifacts, haven't been evaluated (as of 2026/02/02).

**Output data**\
**NOTE: THE PRESENT STRUCTURE IS STILL UNDER DEVELOPMENT: AS THE PRESENT FILE WAS REFRACTORED FROM A DIFFERENT PROJECT, WHAT CURRENTLY DESCRIBED STEMS FROM SUCH PROJECT**

The pipeline generates the following directory tree and output files within the output_data_directory:

- output_data_directory:
    - [date]_ACID_glob_measurements.csv
    - fov:
        - file1_field_of_view1.ome.tif
        - file1_field_of_view2.ome.tif
        - file1_field_of_view3.ome.tif
        - file1_string.xml
        - file2_field_of_view1.ome.tif
        - file2_field_of_view2.ome.tif
        - file2_field_of_view3.ome.tif
        - file2_string.xml
        - file3_field_of_view1.ome.tif
        - file3_field_of_view2.ome.tif
        - file3_field_of_view3.ome.tif
        - file3_string.xml
        - fileN_field_of_viewM.ome.tif
        - fileN_string.xml
        ...
    - seg:
        - file1_field_of_view1.ome.tif
        - file1_field_of_view2.ome.tif
        - file1_field_of_view3.ome.tif
        - file2_field_of_view1.ome.tif
        - file2_field_of_view2.ome.tif
        - file2_field_of_view3.ome.tif
        - file3_field_of_view1.ome.tif
        - file3_field_of_view2.ome.tif
        - file3_field_of_view3.ome.tif
        - fileN_field_of_viewM.ome.tif
        ...
    - metadata:
        - [date]_ACID_metadata_part[notebook_progressive_number].csv
    -background:
        - [...]

In addition, a directory named "secondary_output" is created inside the working directory. Inside secondary_output are saved hyperparameters per each run of the pipeline.


OUTPUT DATA DESCRIPTION
- [date]_ACID_glob_measurements.csv. Comma-separated table with intensity measurements of nuclei and cytosols. The file includes all measurements per each field of view (also called scene) of each raw file. Rows are individual cells. Columns are:

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

    - 'segmentation_date_yymmdd'. The date of the segmentation. This is done in the second main loop of part [...] notebook. Step [...] below.

    - 'segmentation_name'. The name of the saved segmentation file.

    - label. The value of the label for the cell whose measurements are in the row. The label unequivocally identifies the cell in the cytosol segmentation mask, while it might be different for the nucleus and cell masks.

    - area. The area of a measured cell, in number of pixels.

    - centroid-N. The N-coordinate of the centroid a measured cell. [...TO BE CHECKED...When the segmentation mask is 2D as in the present project, this corresponds to the Y coordinate and starts from the top-left corner of the image...].

    - centroid-M. The M-coordinate of the centroid for a measured cell. [...TO BE CHECKED...When the segmentation mask is 2D as in the present project, this corresponds to the X coordinate and starts from the top-left corner of the image...].

    - intensity_mean-(channel). The mean intensity value of a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - intensity_max-(channel). The max intensity value of a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - intensity_min-(channel). The min intensity value of a measured cell in a given channel. The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.

    - background_offset-(channel). The median value of the background pixels in a given field of view and for a given channel. Background pixels are all the pixels not present in the nucleus segmentation or in the cell segmentation masks (aka, pixels which are not present in neither or the masks). The channel position correspond to the order of the channels in the input image. The structure corresponding to each channel can be found in the corresponding column.


- fov. Directory containing individual fields of view extracted from the .nd2 raw file (Nikon proprietary file format) and saved as ome.tif files (free file format with standardized metadata). Fields of views from multiple .nd2 raw files are pooled together in the fov directory.

- seg. Directory containing the segmentation masks of each individual field of view extracted from the .nd2 raw file.

- metadata. Directory containg a csv file with information about the metadata and the processing of individual output files. The file has the following name structure {date_in_format_yymmdd}_ACID_metadata_part{notebook_progressive_number}.csv


**Analysis steps**:\
**NOTE: THE PRESENT STRUCTURE IS STILL UNDER DEVELOPMENT: AS THE PRESENT FILE WAS REFRACTORED FROM A DIFFERENT PROJECT, WHAT CURRENTLY DESCRIBED STEMS FROM SUCH PROJECT**

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
**NOTE: THE PRESENT STRUCTURE IS STILL UNDER DEVELOPMENT: AS THE PRESENT FILE WAS REFRACTORED FROM A DIFFERENT PROJECT, WHAT CURRENTLY DESCRIBED STEMS FROM SUCH PROJECT**

The part1_raw_to_segmentation.ipynb notebook can be used to run the analysis steps 1 and 2.
The part2_segmentation_to_measurement.ipynb notebook can be used to run the analysis step 3.


**Folders**\
**NOTE: THE PRESENT STRUCTURE IS STILL UNDER DEVELOPMENT: AS THE PRESENT FILE WAS REFRACTORED FROM A DIFFERENT PROJECT, WHAT CURRENTLY DESCRIBED STEMS FROM SUCH PROJECT**

The following folders are present:

- utils. Contains scripts of general interest, which are used across different projects and across different steps of the same project.

- data_preparation. Contains scripts used to work on data frames and prepare them for processing and analyses. These include formatting data types, perform a train-test split, clean nan values etc.

- image_processing. Contains scripts used to modify/process the images including filtering and resizing.

- image_quality_control. Contains scripts used to calculate and evaluate quality control metrics at a whole-image level.

- feature_extraction. Contains scripts used to measure cell properties, including channels intensities, cells geometry and texture.

- secondary_output. Contains saved analysis hyperparameters.


**Structure of file name**:\
__INPUT FILE NAME__\

.nd2 raw file: {cell_line}_{virus}_{treatment}_{total_imaging_time}_{experimental_procedure}_{well}.extension



__PROCESSED FILE NAME__\

- extracted fields of view (also called scenes) saved in the fov directory

{cell_line}_{virus}_{treatment}_{total_imaging_time}_{experimental_procedure}_{well}_{experiment}_{scene}.ome.tif



# Dependencies:
The file acid_develop.yml can be used for creating the environment containg script dependencies during the project development. The file acid_neo.yml has been used to create the python environment on a computer workstation used for running the analysis, with gpu support, at the Infectious Diseases Imaging Platform of the CIID (the workstation is ...).

The following module versions are used for acid_develop environment:
- python==3.12.12
- jupyterlab==4.5.3
- pip==25.3
- numpy==2.3.5
- matplotlib==3.10.8
- scipy==1.17.0
- pandas==2.2.3
- dask==2026.1.1
- numba==0.63.1
- tifffile==2026.1.28
- statsmodels==0.14.6
- scikit-image==0.26.0
- scikit-learn==1.8.0
- seaborn==0.13.2
- mahotas==1.4.18
- pytorch==2.5.1
- torchvision==0.20.1
- torchaudio==2.5.1
- napari==0.6.6
- pycytominer==1.2.4
- roifile==2026.1.29
- cellpose==4.0.8
- opencv-python==4.13.0.90
- bioio==3.2.0
- bioio-nd2==1.6.2

The file 20260202_acid_develop.txt is the explicit list of the acid_develop environment created on the 2026/02/02.

As of 2026/02/02 the acid_neo environment hasn't been tested yet.

# Notes:



# To do list:



