"""
Feature extraction for selected cells, across every processing variant
──────────────────────────────────────────────────────────────────────
For each labelled cell (infected / uninfected), extracts CellPhe features
from EVERY frames_* folder of its video — so you can compare which processing
variant best separates infected from uninfected in the texture features.

Masks/ROIs/tracking are shared; only the frame pixels differ per variant.

Output: one CSV per position per variant, saved in that video's own results
folder, with the selected cells' per-frame features plus their infection label.

Run in the cellphepy environment:  python extract_features_variants.py
"""

import os
import glob
import time
import pandas as pd
from cellphe import import_data, cell_features


# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

BATCH_ROOT = r"C:\Users\ACID\Desktop\DENV\programs\batch_output"
TRACKER    = "Overlap"
FRAMERATE  = 1

# The cells you care about, per video stem.
# stem : {"label": "infected"|"uninfected", "cells": [CellIDs]}
CELLS = {
    "A09.1_pos29_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2389": {"label": "infected",   "cells": [1]},
    "A09.1_pos20_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2420": {"label": "infected",   "cells": [88, 93, 96]},
    "A09.1_pos37_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2407": {"label": "infected",   "cells": [97, 98, 43]},
    "A09.1_pos40_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2439": {"label": "infected",   "cells": [37, 74, 146]},
    "A09.1_pos05_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2362": {"label": "uninfected", "cells": [24]},
    "A04.2_pos10_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2597": {"label": "uninfected", "cells": [21, 25, 102]},
    "A04.2_pos16_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2527": {"label": "uninfected", "cells": [5, 20, 35, 43, 49, 64]},
}


# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════

def variant_of(frames_dir):
    """The variant tag = the frames folder name minus the leading 'frames_'."""
    return os.path.basename(frames_dir).replace("frames_", "", 1)

def main():
    t_start = time.time()
    for stem, info in CELLS.items():
        t_video = time.time()
        base_dir = os.path.join(BATCH_ROOT, stem)
        results  = os.path.join(base_dir, "results")
        tracked_csv = os.path.join(results, f"{stem}_tracked_{TRACKER}.csv")
        rois_zip    = os.path.join(results, f"{stem}_rois_{TRACKER}.zip")

        if not os.path.exists(tracked_csv):
            print(f"! {stem}: no tracking CSV — skipping")
            continue

        # tracking table, filtered to the cells we care about
        ft = import_data(tracked_csv, "Trackmate_auto", minframes=1)
        ft = ft.drop_duplicates(subset=["CellID", "FrameID"])
        ft_sel = ft[ft["CellID"].isin(info["cells"])].copy()
        got = sorted(ft_sel["CellID"].unique())
        print(f"\n=== {stem} [{info['label']}] ===")
        print(f"  requested cells {info['cells']} -> found {got}")

        # each variant -> its OWN CSV in this video's results folder
        variant_dirs = sorted(glob.glob(os.path.join(base_dir, "frames_*")))
        for frames_dir in variant_dirs:
            tag = variant_of(frames_dir)
            try:
                t0 = time.time()
                feats = cell_features(ft_sel, rois_zip, frames_dir, framerate=FRAMERATE)
                feats["infected"] = (info["label"] == "infected")
                feats["label"]    = info["label"]
                out = os.path.join(results, f"{stem}_features_{tag}.csv")
                feats.to_csv(out, index=False)
                print(f"    {tag}: {feats.shape} -> {os.path.basename(out)}  ({time.time()-t0:.1f}s)")
            except Exception as e:
                print(f"    {tag}: FAILED {type(e).__name__}: {e}")

        print(f"  ({(time.time()-t_video)/60:.1f} min for this position)")

    print(f"\nDone — total {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()