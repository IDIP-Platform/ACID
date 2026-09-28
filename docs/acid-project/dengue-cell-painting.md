# Dengue Cell Painting

This page describes the current Dengue virus microscopy use case that motivates
the ACID workflow.

## Dengue Cell Classification

The project aims to support classification of cells across infection and
treatment states:

- non-infected cells
- Dengue-infected, untreated cells
- Dengue-infected cells treated with different compounds

The described treatment and control groups are:

- uninfected negative-control cells
- infected positive-control cells
- uninfected cells treated with NITD-688
- uninfected cells treated with JNJ-A07
- uninfected cells treated with JNJ-1802
- infected cells treated with NITD-688
- infected cells treated with JNJ-A07
- infected cells treated with JNJ-1802

## Imaging Strategy

Samples were prepared in parallel in a multi-well plate. The workflow describes
live transmitted-light time-lapse imaging followed by fixation, Cell
Painting-style staining, and re-acquisition of the same fields of view.

The fixed-sample acquisition includes:

- nucleus: Hoechst, channel 1
- endoplasmic reticulum: concanavalin A with 488 fluorophore, channel 2
- actin: phalloidin with 568 fluorophore, channel 3
- NS3 infection marker: anti-NS3 antibody with 647 fluorophore, channel 4
- full-cell transmitted light, channel 5

## Experiments

The project currently describes three independent experiments:

- Experiment A07.2
- Experiment A07.3
- Experiment A07.4

Each field of view is described as a single plane with a pixel size of
0.325 x 0.325 micron and a frame size of 1024 x 1024 pixels.

!!! note "Incomplete experimental details"
    Some sample, grant, objective, and timing details are still placeholders.
    Those fields should be completed before this page is treated as a full
    experimental methods description.
