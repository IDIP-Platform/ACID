# Project Overview

ACID is an ongoing Python project for microscopy image processing, image
quantification, and data-analysis workflows. The current repository focuses on
the analysis of fixed microscopy samples from a Dengue virus Cell
Painting-inspired study.

## Our Mission

Our mission is to provide a reproducible Python workflow that turns microscopy
image data into reliable, analysis-ready cellular measurements.

ACID currently emphasizes:

- extracting fields of view from microscopy files
- preserving useful metadata while avoiding misleading per-field metadata
- preparing train/test splits for image-analysis development
- assessing image quality
- correcting background signal
- segmenting nuclei and cells
- extracting cellular measurements and features

For the current Dengue use case, these steps support classification of cells
across infection and treatment conditions.

## Our Vision

Our vision is for ACID to become a maintainable image-analysis foundation for
infectious-disease microscopy projects: transparent enough for researchers to
inspect, modular enough for developers to improve, and documented enough for
new contributors to run and validate the workflow with confidence.

In practice, this means keeping scientific assumptions visible, separating
package code from notebook orchestration, preserving metadata carefully, and
documenting known uncertainties while the workflow continues to mature.

## Collaborating Context

The project is carried out at the Center for Integrative Infectious Disease
Research in Heidelberg (CIID) and is connected to work from the Infectious
Diseases Imaging Platform (IDIP).

Data collection was performed through a collaboration between IDIP and the group
of Prof. Dr. Christian Klein at the Institute of Pharmacy and Molecular Biology
(IPMB), University of Heidelberg. The current scientific direction is part of
the SHIELD consortium.

!!! note "Repository status"
    The README contains dated notes from project development. This documentation
    keeps those notes visible when they affect how users should interpret the
    current workflow.
