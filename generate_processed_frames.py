"""
Generate differently-processed frame sets for feature extraction
────────────────────────────────────────────────────────────────
For each segmented video in batch_output, builds extra frame folders that
share the SAME base processing (shadow correction + equalization + median
subtraction) but VARY the later steps (denoising, contrast enhancement).
Masks / ROIs / tracking are untouched — only the pixels change — so texture
features can be extracted on each variant and compared.

Uses the EXACT functions from processing.py (imported), so the maths and the
saved equalization reference match your real pipeline.

    batch_output/<stem>/
        frames/                       (original — untouched)
        masks/  results/              (untouched)
        frames_base/                  <- shadow + eq + median only
        frames_gamma0.7/              <- base + gamma contrast
        frames_clahe0.01/             <- base + CLAHE
        frames_n2v/                   <- base + denoising
        frames_n2v_gamma0.7/          <- base + denoising + gamma
        ...

Each folder holds exp-0001.tif … (same naming as frames/), built from the
ORIGINAL ND2 so processing isn't stacked on already-processed frames.

Edit SETTINGS, then run:  python generate_processed_frames.py
"""

import os
import re
import glob
import numpy as np
import nd2
import tifffile as tiff

# import your REAL pipeline functions + settings, so nothing drifts.
# REQUIRED: processing.py must end its last line with
#     if __name__ == "__main__":
#         main()
# (not a bare `main()`), otherwise importing it re-runs the whole pipeline.
import processing as P

# fast (training-free) denoising methods — much quicker than N2V
from scipy.ndimage import median_filter as _median_filter, gaussian_filter as _gaussian_filter
from skimage.restoration import (denoise_tv_chambolle, denoise_nl_means,
                                 denoise_bilateral, estimate_sigma)


def denoise_fast(stack, method):
    """Classical, training-free denoising (per frame). Much faster than N2V.

    method: "median"    — median filter (removes speckle, keeps edges)
            "gaussian"  — light Gaussian blur (fastest, softens everything)
            "tv"        — total-variation (edge-preserving, smooths flat areas)
            "nlmeans"   — non-local means (best quality, slower but no training)
            "bilateral" — edge-preserving smoothing
    """
    out = np.empty_like(stack, dtype=np.float32)
    for t in range(stack.shape[0]):
        f = stack[t].astype(np.float32)
        if method == "median":
            out[t] = _median_filter(f, size=3)
        elif method == "gaussian":
            out[t] = _gaussian_filter(f, sigma=1)
        elif method == "tv":
            out[t] = denoise_tv_chambolle(f, weight=0.1)
        elif method == "nlmeans":
            sig = float(np.mean(estimate_sigma(f)))
            out[t] = denoise_nl_means(f, h=0.8 * sig, sigma=sig,
                                      patch_size=5, patch_distance=6, fast_mode=True)
        elif method == "bilateral":
            out[t] = denoise_bilateral(f, sigma_color=None, sigma_spatial=2)
        else:
            raise ValueError(f"Unknown fast denoise method: {method}")
    return out


# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

BATCH_ROOT = r"C:\Users\ACID\Desktop\DENV\programs\batch_output"

# Original ND2 per experiment (position is read from each folder name).
ND2_FILES = {
    "A09.1": r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250812\A09.1\H7_DENV2_MOI-dilution_40h.nd2",
    "A04.2": r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250618\A04.2\H7_DENV2_MOI1_40h_live.nd2",
    # add the rest as you process them
}

# ── FIXED base steps (same for every variant) ──
BASE_SHADOW = True     # shadow correction
BASE_EQ     = True     # equalization (uses processing.py's saved reference)
BASE_MEDIAN = True     # median subtraction

# ── VARIED steps: each entry becomes its own frames_<tag> folder ──
# "denoise": False | "n2v" | "median" | "gaussian" | "tv" | "nlmeans" | "bilateral"
#            (n2v is the slow trained one; the rest are fast & training-free)
# "contrast": None | "gamma" | "clahe" | "percentile" | "sigmoid"
#VARIANTS = {
#    "base":            {"denoise": False,      "contrast": None},
#    "gamma0.7":        {"denoise": False,      "contrast": "gamma"},
#    "clahe0.01":       {"denoise": False,      "contrast": "clahe"},
#    "sigmoid":         {"denoise": False,      "contrast": "sigmoid"},
#    # fast denoising (no training):
#    "median":          {"denoise": "median",   "contrast": None},
#    "tv":              {"denoise": "tv",        "contrast": None},
#    "nlmeans":         {"denoise": "nlmeans",   "contrast": None},
#    "nlmeans_gamma0.7":{"denoise": "nlmeans",   "contrast": "gamma"},
    # slow trained denoiser (optional — comment out to skip):
    # "n2v":           {"denoise": "n2v",       "contrast": None},
    # contrast params (GAMMA, CLAHE_CLIP, ...) come from processing.py's settings
#}
VARIANTS = {
    "base":        {"denoise": False,     "contrast": None},   # control
    "nlmeans":     {"denoise": "nlmeans", "contrast": None},   # best denoise
    "tv":          {"denoise": "tv",      "contrast": None},   # alt denoise
    "gamma":       {"denoise": False,     "contrast": "gamma"},# contrast only
    "nlmeans_gamma":{"denoise": "nlmeans","contrast": "gamma"},# denoise + contrast
}


# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════

def base_tag():
    """Folder-name fragment recording the FIXED base parameters, so every
    variant folder says exactly which base produced it."""
    parts = []
    if BASE_SHADOW:
        parts.append("sc_basic" if P.BG_METHOD == "basic" else f"sc_{P.BG_METHOD}{P.BG_SIGMA}")
    if BASE_EQ:
        if P.EQUALIZE_MODE == "saved":
            ref_id = os.path.splitext(os.path.basename(P.EQUALIZE_REF_PATH))[0].replace("ref_", "")
            parts.append(f"eq({ref_id})")
        elif P.EQUALIZE_MODE == "frame":
            parts.append(f"eq_pos{P.EQUALIZE_REF_POSITION:02d}_t{P.EQUALIZE_REF_T}")
        else:
            parts.append(f"eq{P.EQUALIZE_REF_T}")
    if BASE_MEDIAN:
        parts.append(f"med{P.MEDIAN_OFFSET}")
    return "_".join(parts)


def experiment_of(stem):
    m = re.match(r"([A-Z]\d+\.\d+)", stem)
    return m.group(1) if m else None

def position_of(stem):
    m = re.search(r"_pos(\d+)", stem)
    return int(m.group(1)) - 1 if m else None      # 1-indexed name -> 0-indexed ND2

def load_reference():
    """The saved equalization reference, loaded the same way processing.py does."""
    if BASE_EQ and P.EQUALIZE_MODE == "saved":
        return tiff.imread(P.EQUALIZE_REF_PATH).astype(np.float32)
    return None

def base_process(raw, reference):
    """Fixed base: shadow -> equalization -> median, using processing.py functions."""
    stack = raw.astype(np.float32)
    if BASE_SHADOW:
        stack = P.shadow_correction(stack, sigma=P.BG_SIGMA, method=P.BG_METHOD)
    if BASE_EQ:
        stack = P.equalize_intensity(stack, ref_t=P.EQUALIZE_REF_T, reference=reference)
    if BASE_MEDIAN:
        stack, _ = P.median_subtract(stack, offset=P.MEDIAN_OFFSET)
    return stack

def apply_variant(base_stack, spec):
    """Apply the varied steps (denoise, contrast) on top of the base."""
    stack = base_stack
    d = spec["denoise"]
    if d == "n2v":
        stack = P.denoise_n2v(stack, epochs=P.N2V_EPOCHS,
                              patch=P.N2V_PATCH, batch=P.N2V_BATCH)
    elif d:                      # any fast method name
        stack = denoise_fast(stack, d)
    if spec["contrast"]:
        stack = P.enhance_contrast(stack, method=spec["contrast"])
    return stack

def write_frames(stack, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for t in range(stack.shape[0]):
        tiff.imwrite(os.path.join(out_dir, f"exp-{t+1:04d}.tif"),
                     stack[t].astype("float32"))

def main():
    reference = load_reference()
    if reference is not None:
        print(f"equalization reference: {os.path.basename(P.EQUALIZE_REF_PATH)} "
              f"(shape {reference.shape})")

    stems = sorted(os.path.basename(p) for p in glob.glob(os.path.join(BATCH_ROOT, "*"))
                   if os.path.isdir(p))
    print(f"{len(stems)} video folder(s)\n")

    for stem in stems:
        base_dir = os.path.join(BATCH_ROOT, stem)
        exp = experiment_of(stem)
        pos = position_of(stem)
        print(f"=== {stem} ===  (experiment {exp}, position {pos})")

        nd2_path = ND2_FILES.get(exp)
        if not nd2_path or pos is None:
            print("  ! no ND2 mapping or position — skipping\n")
            continue

        # load raw stack for this position, build the fixed base once
        data = nd2.imread(nd2_path).astype(np.float32)      # (T, P, Y, X)
        raw = data[:, pos]                                  # (T, Y, X)
        print(f"  raw {raw.shape} -> building base (shadow+eq+median)...")
        base_stack = base_process(raw, reference)

        # each variant folder is named  frames_<base params>__<variant>
        btag = base_tag()
        for tag, spec in VARIANTS.items():
            out_dir = os.path.join(base_dir, f"frames_{btag}__{tag}")
            if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "*.tif")):
                print(f"    frames_{tag}: exists — skipping")
                continue
            stack = apply_variant(base_stack, spec)
            write_frames(stack, out_dir)
            print(f"    frames_{tag}: {stack.shape[0]} frames written")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
