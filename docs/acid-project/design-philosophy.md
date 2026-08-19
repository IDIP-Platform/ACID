# Design Philosophy

ACID is being developed as a practical analysis pipeline for microscopy data.
The README describes a staged approach: begin with fixed-sample data, establish
robust image processing and feature extraction, and then use those outputs for
downstream analyses.

## Fixed-Sample Focus

At the start of the project, only data from fixed samples are analyzed. The
fixed samples are stained using a Cell Painting-inspired protocol and then
processed through field-of-view extraction, quality control, background
correction, segmentation, and feature extraction.

## Metadata Strategy

Raw ND2 files can contain multiple positions, channels, and associated global
metadata. ACID extracts individual fields of view as OME-TIFF files, but does
not simply copy the original raw XML metadata into each extracted field.

Instead, the workflow keeps relevant per-field metadata with the extracted file
and stores global raw metadata separately. This avoids attaching metadata for
all fields of view to one individual field-of-view file.

## Train/Test Strategy

The README describes a 70/30 train/test split. Because the described conditions
are balanced, no stratification is currently used.

Pipeline development up to and including feature extraction is described as
using the training data. The README notes a possible exception for background
function estimation, where a single function estimated from training data may
need to be reused when a suitable test-data function cannot be estimated.

## Qualitative Tuning

The initial approach is to use relatively standard image-processing,
segmentation, and quantification procedures without heavy ad hoc tuning. Results
are evaluated qualitatively and individual steps are adjusted as needed.

!!! note "Validation split"
    The README states that a train/validation split is not initially planned for
    pipeline development through feature extraction. A validation set may be
    introduced if the project begins fine-tuning segmentation models or other
    model-like components.

## Downstream Analysis

A train/validation split is expected for analyses downstream of feature
extraction. Those downstream modeling steps are conceptually separate from the
image-processing workflow documented here.
