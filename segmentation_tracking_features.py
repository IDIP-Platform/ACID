"""
Segmentation, Tracking and Feature Extraction
─────────────────────────────────────────────
Takes preprocessed timelapse stacks (the output of processing.py) and runs,
for each file in INPUT_FILES:

    Cellpose segmentation  ->  TrackMate tracking  ->  CellPhe features

Every file gets its own output folder named after the file, so results are
traceable back to exactly which processed stack (and which model/tracker)
produced them. Edit the SETTINGS block, then run:

    python cellphe.py

Steps read from disk, so DO_SEGMENT / DO_TRACK / DO_FEATURES can be toggled
to rerun only part of the pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import tifffile as tiff
import re  

# ═══════════════════════════════════════════════════════════════════
# SETTINGS — edit these, then run
# ═══════════════════════════════════════════════════════════════════

# Preprocessed stacks to process. Each file is one position, (T, Y, X).
# Add as many as you like — they are processed one after another.
INPUT_FILES = [
    r"C:\Users\ACID\Desktop\DENV\programs\fine_tuning\A07.3_pos11_pos11_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2389.tif",
    #r"C:\Users\ACID\Desktop\DENV\programs\fine_tuning\A07.3_pos34_pos34_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2439.tif",
    # r"...\A02.1\A02.1_pos13_sc_gaussian8_eq40_med2487.tif",
    # r"...\A03.1\A03.1_pos15_sc_gaussian8_eq40_med2801.tif",
]

OUTPUT_DIR = r"..\output\segmentation_output"

# ── Which steps to run ──
DO_SEGMENT  = True
DO_TRACK    = True
DO_FEATURES = True

# ── Segmentation (Cellpose) ──
CP_MODEL              = "cpsam_v2"    # "cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"
CP_GPU                = True
CP_DIAMETER           = 60         # cell diameter in px; None = auto-estimate
CP_FLOW_THRESHOLD     = 0.7        # higher = separates touching cells better
CP_CELLPROB_THRESHOLD = -3.0       # lower = finds more / fainter cells
CP_MIN_SIZE           = 3
CP_NORMALIZE          = True
SEGMENT_FRAMES        = "all"      # "all", "last", or a list like [0, 40, 79]

# ── Tracking (TrackMate) ──
# SimpleSparseLAP, SparseLAP, Kalman, AdvancedKalman, NearestNeighbor, Overlap
TRACKER   = "Overlap"
MINFRAMES = 30        # keep only cells tracked for at least this many frames

# ── Feature extraction (CellPhe) ──
FRAMERATE = 1         # 1 = scaleless velocity units

# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════

def unique_dir(path):
    """Return `path` if free, else `path_2`, `path_3`, ... (never overwrites)."""
    if not os.path.exists(path):
        return path
    n = 2
    while os.path.exists(f"{path}_{n}"):
        n += 1
    return f"{path}_{n}"


def experiment_of(stem):
    """Experiment = the leading A07.3 / A02.1 part of the file name."""
    m = re.match(r"([A-Z]\d+\.\d+)", stem)
    return m.group(1) if m else "unknown"


def paths_for(input_file):
    stem = os.path.splitext(os.path.basename(input_file))[0]
    exp = experiment_of(stem)
    target = os.path.join(OUTPUT_DIR, exp, f"{stem}_{CP_MODEL}")   # <-- exp ubačen

    if DO_SEGMENT:
        base = unique_dir(target)      # nova segmentacija -> svjež folder
    else:
        base = target                  # reusing masks -> postojeći folder
        if not os.path.isdir(base):
            raise SystemExit(f"DO_SEGMENT is False but no existing folder: {base}")

    return {
        "stem":     stem,
        "base":     base,
        "frames":   os.path.join(base, "frames"),
        "masks":    os.path.join(base, "masks"),
        "params":   os.path.join(base, "segmentation_parameters.txt"),
        "csv":      os.path.join(base, f"{stem}_tracked_{TRACKER}.csv"),
        "rois":     os.path.join(base, f"{stem}_rois_{TRACKER}.zip"),
        "features": os.path.join(base, f"{stem}_features_{TRACKER}_min{MINFRAMES}.csv"),
    }


def write_params(p, input_file):
    """Record the segmentation settings used, in the output folder."""
    os.makedirs(p["base"], exist_ok=True)
    with open(p["params"], "w") as f:
        f.write("Segmentation parameters\n")
        f.write("=======================\n")
        f.write(f"input_file            : {input_file}\n")
        f.write(f"model                 : {CP_MODEL}\n")
        f.write(f"gpu                   : {CP_GPU}\n")
        f.write(f"diameter              : {CP_DIAMETER}\n")
        f.write(f"flow_threshold        : {CP_FLOW_THRESHOLD}\n")
        f.write(f"cellprob_threshold    : {CP_CELLPROB_THRESHOLD}\n")
        f.write(f"min_size              : {CP_MIN_SIZE}\n")
        f.write(f"normalize             : {CP_NORMALIZE}\n")
        f.write(f"segment_frames        : {SEGMENT_FRAMES}\n")
        f.write(f"tracker               : {TRACKER}\n")
        f.write(f"minframes             : {MINFRAMES}\n")
    print(f"    parameters -> {p['params']}")


# ── STEP 1: segmentation ──
def segment(input_file, p):
    """Segment the chosen frames; write frames and masks in CellPhe's layout.

    Frames  -> exp-0001.tif   (1-indexed, TrackMate convention)
    Masks   -> frame_000_mask.tif  (0-indexed)
    """
    from cellpose import models

    os.makedirs(p["frames"], exist_ok=True)
    os.makedirs(p["masks"], exist_ok=True)
    write_params(p, input_file)

    stack = tiff.imread(input_file)
    T = stack.shape[0]
    if SEGMENT_FRAMES == "all":
        frame_indices = list(range(T))
    elif SEGMENT_FRAMES == "last":
        frame_indices = [T - 1]
    else:
        frame_indices = list(SEGMENT_FRAMES)

    print(f"    {len(frame_indices)} frame(s), model={CP_MODEL}")
    model = models.CellposeModel(gpu=CP_GPU, pretrained_model=CP_MODEL)

    for t in frame_indices:
        frame = stack[t]
        # eval gets a COPY — with normalize=True Cellpose rescales its input
        # in place, which would otherwise corrupt the frame we save (a slice of
        # `stack` is a view, so the normalized values would leak back).
        masks, flows, styles = model.eval(
            frame.copy(),
            diameter=CP_DIAMETER,
            flow_threshold=CP_FLOW_THRESHOLD,
            cellprob_threshold=CP_CELLPROB_THRESHOLD,
            normalize=CP_NORMALIZE,
            min_size=CP_MIN_SIZE,
        )
        tiff.imwrite(f"{p['frames']}/exp-{t + 1:04d}.tif", frame.astype("float32"))
        tiff.imwrite(f"{p['masks']}/frame_{t:03d}_mask.tif", masks.astype("uint16"))
        print(f"      frame {t + 1}/{T}: {masks.max()} cells", flush=True)


# ── STEP 2: tracking ──
def track(p):
    """Link masks across frames with TrackMate. Writes the CSV and ROI zip."""
    from cellphe import track_images

    print(f"    tracker={TRACKER} (first run downloads TrackMate, ~10 min)")
    track_images(p["masks"], p["csv"], p["rois"], TRACKER)


# ── STEP 3: feature extraction ──
def extract_features(p):
    """Compute the 74 per-frame CellPhe features for ALL tracked cells and save."""
    from cellphe import import_data, cell_features

    df = import_data(p["csv"], "Trackmate_auto", minframes=MINFRAMES)
    n_cells = df["CellID"].nunique()
    print(f"    cells tracked >={MINFRAMES} frames: {n_cells}")
    if n_cells == 0:
        print("    ! no cells survived the minframes filter — lower MINFRAMES")
        return

    report_tracking(df)

    start = time.time()
    features = cell_features(df, p["rois"], p["frames"], framerate=FRAMERATE)

    # safety net: drop any accidental duplicate cell-frame rows
    before = len(features)
    features = features.drop_duplicates(subset=["FrameID", "CellID"]).reset_index(drop=True)
    if len(features) != before:
        print(f"    removed {before - len(features)} duplicate rows")

    features.to_csv(p["features"], index=False)
    print(f"    features {features.shape} -> {p['features']}")
    print(f"    took {(time.time() - start) / 60:.1f} min")


def report_tracking(df):
    """Print how well tracking held cell identity across frames."""
    counts = df.groupby("CellID")["FrameID"].count()
    print(f"      unique IDs: {df['CellID'].nunique()}")
    print(f"      median frames/ID: {counts.median():.0f}")
    for n in (10, 30, 60):
        print(f"      cells >={n} frames: {(counts >= n).sum()}")


def main():
    total = len(INPUT_FILES)
    ok, failed = 0, []

    for i, input_file in enumerate(INPUT_FILES, 1):
        p = paths_for(input_file)
        print(f"\n=== [{i}/{total}] {p['stem']} ===")
        os.makedirs(p["base"], exist_ok=True)

        try:
            if DO_SEGMENT:
                print("  segmenting...")
                segment(input_file, p)
            if DO_TRACK:
                print("  tracking...")
                track(p)
            if DO_FEATURES:
                print("  extracting features...")
                extract_features(p)
            ok += 1
        except Exception as e:
            # one bad file shouldn't kill the whole batch
            print(f"  ! FAILED: {type(e).__name__}: {e}")
            failed.append((p["stem"], str(e)))

    print(f"\nDone — {ok}/{total} succeeded.")
    if failed:
        print("Failed files:")
        for stem, err in failed:
            print(f"  {stem}: {err}")


if __name__ == "__main__":
    main()