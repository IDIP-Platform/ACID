"""
DIC Timelapse Preprocessing
───────────────────────────
Runs the preprocessing pipeline on selected positions from one or more
ND2 files:

    per-frame shadow correction  →  equalization  →  median
    subtraction  →  N2V denoising → contrast enhancement

Edit the SETTINGS block at the top, then run:

    python processing.py

Each enabled step can be toggled on/off. Output filenames encode which
file, which position, and which steps were applied.
"""

import os
import numpy as np
import nd2
import tifffile as tiff
from scipy.ndimage import gaussian_filter, median_filter
from skimage.exposure import match_histograms


# ═══════════════════════════════════════════════════════════════════
# SETTINGS — edit these, then run
# ═══════════════════════════════════════════════════════════════════

# Which files, and which positions from each.
# Give a list of position indices, or the string "all" for every position.
# Numbers start from 0
FILES_AND_POSITIONS = {
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250812\A09.1\H7_DENV2_MOI-dilution_40h.nd2": [35, 37, 38, 40, 4, 10, 29, 30, 31, 32, 26, 27, 24, 16, 17],
    r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250812\A09.1\H7_DENV2_MOI-dilution_40h.nd2": [28],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250729\A07.3\H7_DENV2_MOI1_40h.nd2": [4, 5, 6, 9, 11, 32, 34],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250708\A07.1\H7_DENV2_MOI1_40h.nd2": [1, 4, 5, 9, 10, 11, 14, 15, 31, 35, 30, 21],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250618\A04.2\H7_DENV2_MOI1_40h_live.nd2": [8, 9, 26, 28, 29, 23, 24],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250513\A02.1\H7_DENV2_MOI0.2_40h_live.nd2": [2, 5, 19, 24],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250515\A03.1\H7_DENV2_MOI1_30h_live.nd2": [2, 6, 19],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250520\A03.2\H7_DENV2_MOI1_40h_live.nd2": [10, 22],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250626\A06.1\H7_DENV2_MOI1_30h.nd2": [6, 15, 26, 30],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250820\A10.1\H7_DMSO-dilution_40h.nd2": [12, 36],
    #r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\260128\A09.4\H7_MOI-dilution_40h.nd2": [6, 10, 18, 20],

    # r"C:\path\to\another_file.nd2": "all",
    # r"C:\path\to\third_file.nd2": [0, 3, 15],
}

OUTPUT_DIR = r"..\output\processing_output"

# Save the result after every step (with cumulative names), not just the final.
SAVE_INTERMEDIATES = True

# ── Which steps to run ──
DO_SHADING   = True
DO_EQUALIZE  = True
DO_MEDIAN    = True
DO_DENOISE   = False        # slow; uses GPU if available
DO_CONTRAST  = False

# ── Step parameters ──
BG_SIGMA        = 8            # blur radius (px)
BG_METHOD       = "gaussian"        # "gaussian", "median", "basic"
# How to build the equalization reference:
#   "saved"  = load a reference image from disk (built once — most reproducible;
#              every position, in every run, matches the identical target)
#   "frame"  = one frame from a reference position, rebuilt each run
#   "local"  = a frame from within each file being processed (no cross-file target)
EQUALIZE_MODE   = "saved"        # "saved"
EQUALIZE_REF_T  = 40             # reference frame index (used by "frame" and "local")

# Only if EQUALIZE_MODE = "saved":
# the .tif written by processing.ipynb (or the notebook's build cell)
EQUALIZE_REF_PATH = r"equalization_references/ref_A07.3_avg41pos_x81t_sc_gaussian8.tif"

# Only if EQUALIZE_GLOBAL = "frame":
EQUALIZE_REF_FILE     = r"//Hive2004/ag_idip/ZeljkaBaca_Data/ACID/250729/A07.3/H7_DENV2_MOI1_40h.nd2"
EQUALIZE_REF_POSITION = 6        # which position from EQUALIZE_REF_FILE

# ── Median subtraction (dirt removal) ──
MEDIAN_OFFSET   = "auto"           # constant kept after median subtraction

# ── Denoising ──
N2V_EPOCHS      = 100             # denoising training epochs (NoiseToVoid)
N2V_PATCH       = 64
N2V_BATCH       = 16

# ── Contrast enhancement ──
CONTRAST_METHOD = "gamma"        # "clahe", "gamma", "percentile", "sigmoid"

CLAHE_CLIP      = 0.03           # higher = stronger local contrast (0.01-0.05)
CLAHE_KERNEL    = 128            # tile size in px; roughly 2-3x a cell diameter
 
GAMMA           = 0.7            # <1 brightens mid-tones, >1 darkens them
 
PCT_LOW         = 1.0            # percentile stretch: clip below this percentile
PCT_HIGH        = 99.0           # and above this one
 
SIGMOID_CUTOFF  = 0.5            # midpoint of the S-curve (0-1)
SIGMOID_GAIN    = 10             # steepness; higher = harder contrast

# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════

# ── STEP 1: per-frame shadow correction ──
def shadow_correction(stack, sigma=8, method="gaussian"):
    """Per-frame additive shadow correction.

    Estimates each frame's own low-frequency illumination (by blurring
    away the cells) and subtracts it, keeping the mean level so the
    image doesn't go dark. Adapts to inconsistent shading frame-to-frame.

    stack: (T, Y, X)  ->  corrected (T, Y, X)
    """
    if method in ("gaussian", "median"):
        out = np.empty_like(stack)
        for t in range(stack.shape[0]):
            frame = stack[t]
            if method == "gaussian":
                background = gaussian_filter(frame, sigma=sigma)
            elif method == "median":
                background = median_filter(frame, size=sigma)
            out[t] = frame - background + background.mean()
        return out
 
    elif method == "basic":
        from basicpy import BaSiC
        basic = BaSiC()
        basic.fit(stack)                                   # fit across all frames
        flatfield = basic.flatfield[np.newaxis, :, :]      # (1, Y, X)
        return (stack / flatfield).astype(np.float32)      # flatfield-only correction
 
    else:
        raise ValueError(f"Unknown BG_METHOD: {method}")

# ── STEP 2: intensity equalization to a reference frame ──
def equalize_intensity(stack, ref_t=40, reference=None):
    """Histogram-match every frame to a reference.

    If `reference` is given (global mode), every frame in every position
    is matched to that external frame.
    If None, falls back to matching within the stack (frame ref_t of that position).

    stack: (T, Y, X)  ->  equalized (T, Y, X)
    """
    ref_t = min(ref_t, stack.shape[0] - 1)
    if reference is None:
        reference = stack[ref_t]

    out = np.empty_like(stack)
    for t in range(stack.shape[0]):
        out[t] = match_histograms(stack[t], reference)
    return out

# ── STEP 3: median background subtraction ──
def median_subtract(stack, offset=1300):
    """Subtract the temporal-median background
    offset="auto" keeps the smallest possible floor: one less than the
    darkest pixel of the median image.


    stack: (T, Y, X)  ->  corrected (T, Y, X)
    """
    median_img = np.median(stack, axis=0)          # (Y, X)
    if offset == "auto":
        offset = float(median_img.min()) - 1
        print(median_img.min())
    return stack - (median_img - offset), offset

# ── STEP 4: N2V denoising ──
def denoise_n2v(stack, epochs=100, patch=64, batch=16):
    """Noise2Void denoising via CAREamics.

    stack: (T, Y, X)  ->  denoised (T, Y, X)
    """
    from careamics import CAREamist
    from careamics.config import create_n2v_config

    config = create_n2v_config(
        experiment_name="n2v",
        data_type="array",
        axes="SYX",
        patch_size=[patch, patch],
        batch_size=batch,
        num_epochs=epochs,
    )
    careamist = CAREamist(config)
    careamist.train(train_data=stack.astype(np.float32))

    result = careamist.predict(stack.astype(np.float32), axes="SYX")
    predictions = result[0] if isinstance(result, tuple) else result
    return np.array(predictions).squeeze()

# ── STEP 5: contrast enhancement ──
def enhance_contrast(stack, method="clahe"):
    """Enhance contrast, frame by frame.
 
    The stack is normalised to [0, 1] using the min and max of the WHOLE
    stack, so every frame is on the same scale — normalising per frame
    would make the timelapse flicker. The result is mapped back to the
    original value range.
 
    stack: (T, Y, X)  ->  enhanced (T, Y, X)
    """
    from skimage import exposure
 
    lo, hi = float(stack.min()), float(stack.max())
    if hi <= lo:
        return stack
 
    out = np.empty_like(stack, dtype=np.float32)
 
    for t in range(stack.shape[0]):
        norm = np.clip((stack[t] - lo) / (hi - lo), 0, 1)
 
        if method == "clahe":
            # local histogram equalisation — strongest on low-contrast DIC,
            # but it also amplifies noise in empty background
            e = exposure.equalize_adapthist(
                norm, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
 
        elif method == "gamma":
            # global and monotonic: gamma < 1 lifts faint detail
            e = exposure.adjust_gamma(norm, gamma=GAMMA)
 
        elif method == "percentile":
            # linear stretch between two percentiles, clipping the tails
            p_lo, p_hi = np.percentile(norm, (PCT_LOW, PCT_HIGH))
            e = exposure.rescale_intensity(norm, in_range=(p_lo, p_hi))
 
        elif method == "sigmoid":
            # S-curve: expands contrast around the cutoff, compresses the tails
            e = exposure.adjust_sigmoid(norm, cutoff=SIGMOID_CUTOFF, gain=SIGMOID_GAIN)
 
        else:
            raise ValueError(f"Unknown CONTRAST_METHOD: {method}")
 
        out[t] = e * (hi - lo) + lo
 
    return out

# ── Output naming ──
def shading_tag():
    """The tag fragment for the shading step."""
    if BG_METHOD == "basic":
        return "sc_basic"
    return f"sc_{BG_METHOD}{BG_SIGMA}"

def eq_tag():
    """The tag fragment for the equalization step, recording the reference used."""
    if EQUALIZE_MODE == "saved":
        # the reference file's own name, so the output records exactly what it matched to
        ref_id = os.path.splitext(os.path.basename(EQUALIZE_REF_PATH))[0].replace("ref_", "")
        return f"_eq({ref_id})"
    if EQUALIZE_MODE == "frame":
        ref_experiment = os.path.basename(os.path.dirname(EQUALIZE_REF_FILE))
        return f"_eq({ref_experiment}_pos{EQUALIZE_REF_POSITION:02d}_t{EQUALIZE_REF_T})"
    # "local"
    return f"_eq{EQUALIZE_REF_T}"

def contrast_tag():
    """The tag fragment for the contrast step, including its parameters."""
    if CONTRAST_METHOD == "clahe":
        return f"ce_clahe{CLAHE_CLIP}"
    if CONTRAST_METHOD == "gamma":
        return f"ce_gamma{GAMMA}" 
    if CONTRAST_METHOD == "percentile":
        return f"ce_pct{PCT_LOW}-{PCT_HIGH}"
    if CONTRAST_METHOD == "sigmoid":
        return f"ce_sig{SIGMOID_CUTOFF}g{SIGMOID_GAIN}"
    return CONTRAST_METHOD
 
def save_stack(stack, experiment, tag):
    """Save a stack into the experiment folder, named '<experiment>_<tag>.tif'."""
    out_subdir = os.path.join(OUTPUT_DIR, experiment)
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"{experiment}_{tag}.tif")
    tiff.imwrite(out_path, stack.astype(np.float32),
                 imagej=True, metadata={"axes": "TYX"})
    print(f"    saved: {out_path}")

def process_position(data, position, experiment, global_reference=None):
    """Run the enabled steps on one position.
 
    The name is built up cumulatively as steps are applied. If
    SAVE_INTERMEDIATES is on, the stack is saved after every step;
    otherwise only the final result is saved.
    """
    stack = data[:, position].astype(np.float32)      # (T, Y, X)
    tag = f"pos{position+1:02d}"                          # grows with each step
 
    if DO_SHADING:
        print(f"    - shadow correction ({BG_METHOD}, sigma={BG_SIGMA})")
        stack = shadow_correction(stack, sigma=BG_SIGMA, method=BG_METHOD)
        tag += f"_{shading_tag()}"
        if SAVE_INTERMEDIATES:
            save_stack(stack, experiment, tag)
 
    if DO_EQUALIZE:
        print(f"    - equalization ({EQUALIZE_MODE})", flush=True)
        stack = equalize_intensity(stack, ref_t=EQUALIZE_REF_T,
                                reference=global_reference)
        tag += eq_tag()

        if SAVE_INTERMEDIATES:
            save_stack(stack, experiment, tag)
 
    if DO_MEDIAN:
        print(f"    - median subtraction (offset={MEDIAN_OFFSET})")
        stack, offset_used = median_subtract(stack, offset=MEDIAN_OFFSET)
        tag += f"_med{int(offset_used)}"
        if SAVE_INTERMEDIATES:
            save_stack(stack, experiment, tag)
 
    if DO_DENOISE:
        print(f"    - N2V denoising ({N2V_EPOCHS} epochs)")
        stack = denoise_n2v(stack, epochs=N2V_EPOCHS,
                            patch=N2V_PATCH, batch=N2V_BATCH)
        tag += f"_n2v_{N2V_EPOCHS}_{N2V_PATCH}_{N2V_BATCH}"
        if SAVE_INTERMEDIATES:
            save_stack(stack, experiment, tag)

    if DO_CONTRAST:
        print(f"    - contrast enhancement ({CONTRAST_METHOD})", flush=True)
        stack = enhance_contrast(stack, method=CONTRAST_METHOD)
        tag += f"_{contrast_tag()}"
        if SAVE_INTERMEDIATES:
            save_stack(stack, experiment, tag)
 
    # Always save the final result (if intermediates were on, the last
    # save above already wrote this exact file, so skip the duplicate).
    if not SAVE_INTERMEDIATES:
        save_stack(stack, experiment, tag)

def main():
    # Load the equalization reference once, according to EQUALIZE_MODE.
    # "local" leaves it None, so each stack matches a frame within itself.
    global_reference = None

    if DO_EQUALIZE and EQUALIZE_MODE == "saved":
        print(f"Loading saved equalization reference:")
        print(f"  {EQUALIZE_REF_PATH}")
        global_reference = tiff.imread(EQUALIZE_REF_PATH).astype(np.float32)
        print(f"  shape {global_reference.shape}, mean {global_reference.mean():.1f}")

    elif DO_EQUALIZE and EQUALIZE_MODE == "frame":
        print(f"Building equalization reference "
              f"(pos {EQUALIZE_REF_POSITION}, frame {EQUALIZE_REF_T})...")
        ref_data = nd2.imread(EQUALIZE_REF_FILE).astype(np.float32)
        ref_stack = ref_data[:, EQUALIZE_REF_POSITION]    # (T, Y, X)
        if DO_SHADING:
            print(f"  applying shadow correction to reference position...")
            ref_stack = shadow_correction(ref_stack, sigma=BG_SIGMA, method=BG_METHOD)
        global_reference = ref_stack[EQUALIZE_REF_T]
        print(f"  reference shape: {global_reference.shape}, "
              f"mean: {global_reference.mean():.1f}")
        
    for nd2_path, positions in FILES_AND_POSITIONS.items():
        # Experiment name = the name of the folder the ND2 file sits in.
        # e.g. ".../ACID/250729/A07.3/H7_DENV2_MOI1_40h.nd2"  ->  "A07.3"
        experiment = os.path.basename(os.path.dirname(nd2_path))
        print(f"\n=== {experiment} ===")
 
        print("  loading...")
        data = nd2.imread(nd2_path).astype(np.float32)     # (T, P, Y, X)
        T, P, Y, X = data.shape
        print(f"  shape {data.shape}")
 
        # resolve "all" to every position
        if positions == "all":
            positions = range(P)
 
        for pos in positions:
            if pos >= P:
                print(f"  ! position {pos+1} out of range (file has {P}) — skipping")
                continue
            print(f"  position {pos+1}:")
            process_position(data, pos, experiment, global_reference)
 
    print("\nDone.")

if __name__ == "__main__":
    main()