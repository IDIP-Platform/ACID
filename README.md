# ACID - pipeline

A pipeline for DIC timelapse microscopy data of cell experiments. It
takes the raw microscopy data and carries it through four stages —
**processing → segmentation → tracking → feature extraction**.


```mermaid
flowchart LR
%%{init: {'flowchart': {'nodeSpacing': 25, 'rankSpacing': 40}, 'themeVariables': {'fontSize': '13px'}}}%%
    subgraph c1[" "]
        direction TB
        raw["<b>Raw data</b>"] --> proc["<b>Processing</b>
        1) Illumination correction
        2) Equalization
        3) Median subtraction
        4) Denoising
        5) Contrast enhancement"]
        proc --> seg["<b>Segmentation</b>
        1) cpsam
        2) cpsam_v2
        3) cpdino
        4) cpdino_vitb"]
        seg --> track["<b>Tracking</b>
        1) Overlap
        2) SimpleSparseLAP
        3) SparseLAP
        4) Kalman
        5) AdvancedKalman"]
    end
 
    feat["<b>Feature extraction</b>
    1) size
    2) shape
    3) texture
    4) movement"]
    analysis["<b>Analysis</b>"]
 
    c1 -->|texture| feat
    c1 -->|masks| feat
    c1 -->|movement| feat
    feat --> analysis
 
    classDef rawStyle fill:#F5CCE8,stroke:#C77DB1,color:#000
    classDef procStyle fill:#D9F2A8,stroke:#7AAE3A,color:#000
    classDef segStyle fill:#B3F0F0,stroke:#4FBFBF,color:#000
    classDef trackStyle fill:#D9C2F0,stroke:#9B72C7,color:#000
    classDef featStyle fill:#F5DFB8,stroke:#D0A860,color:#000
 
    class raw,analysis rawStyle
    class proc procStyle
    class seg segStyle
    class track trackStyle
    class feat featStyle
 
    style c1 fill:none,stroke:#333,stroke-width:1px
```

## The four stages

### 1. Processing
Cleans the raw DIC frames so they can be segmented and compared. Per-frame
shadow correction removes uneven illumination, equalization matches every frame
to a shared intensity reference (so cells are comparable across the whole
dataset), and median subtraction removes fixed background dirt. Optional
denoising and contrast steps can be added. Output filenames record exactly which
steps and parameters were applied.

### 2. Segmentation
Finds the cells in every frame. Several Cellpose version 4.2.1.1 models can be compared
side-by-side first, then the chosen model is run over the whole timelapse to
produce a mask for each cell in each frame.

### 3. Tracking
Links the masks across frames so each cell keeps a consistent identity over
time (TrackMate using CellPhe). The result is a trajectory per cell where it is, frame by
frame plus its outline (ROI) at every timepoint.

### 4. Feature extraction
Measures each tracked cell at every frame (CellPhe): shape, texture and movement
features. The output is one row per cell per frame, a time series of
measurements for every cell that can then be plotted and analysed.

## Running code

| File | Stage | What it does | Conda environment
|---|---|---|---|
| `processing` | 1 | Preprocessing: shadow correction, equalization, median subtraction, denoising, contract enhancement | processing |
| `segmentation_tracking_features` | 2-4 | Segment, track and extract features in one workflow | cellphepy |

Two conda environments are used, because Cellpose / CellPhe / Java versions don't coexist cleanly.

## Detailed documentation

Each stage has its own document with a full description of what the code does:

| Stage | Detailed description |
|---|---|
| 1. Processing | [`processing.md`](processing.md) |
| 2-4. Segmentation, tracking & feature extraction | [`segmentation_tracking_features.md`](segmentation_tracking_features.md) |

See those files for the details of each stage.
