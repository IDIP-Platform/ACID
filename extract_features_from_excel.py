"""
Feature extraction driven by the Experiments_overview Excel
────────────────────────────────────────────────────────────
Reads the 'CellPhePy' sheet, which lists — per tracked file — the cells to
follow and whether each is infected / not infected. For every one of those
cells it extracts CellPhe features from each processing variant of that file,
tagging every row with the infection label.

Sheet layout it expects (CellPhePy sheet):
  - a row whose column D ends in "_tracked_Overlap.csv"  → starts a file block
  - within a block, a row with column A = "infected"/"not infected"
    and column D beginning "Cell N:"                     → a labelled cell

Output: one CSV per position per variant, in that video's results/ folder,
        with `infected` (bool) + `label` columns.

Run in the cellphepy environment:  python extract_features_from_excel.py
"""

import os
import re
import glob
import time
import openpyxl
import pandas as pd
from cellphe import import_data, cell_features


# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

EXCEL_PATH = r"C:\Users\ACID\Desktop\DENV\programs\Experiments_overview2.xlsx"
SHEET      = "CellPhePy"

BATCH_ROOT = r"C:\Users\ACID\Desktop\DENV\programs\batch_output"
TRACKER    = "Overlap"
FRAMERATE  = 1

# which frame folders to extract from: "all" = every frames_* folder,
# or a list of variant tags e.g. ["...__base", "...__nlmeans"]
VARIANTS = "all"


# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════

def parse_excel(path, sheet):
    """-> {file_stem: {"infected":[ids], "not infected":[ids]}}"""
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    cells = {}
    current = None
    for r in range(1, ws.max_row + 1):
        a = ws.cell(r, 1).value
        d = ws.cell(r, 4).value
        if d and str(d).strip().endswith("_tracked_Overlap.csv"):
            current = str(d).strip().replace("_tracked_Overlap.csv", "")
            cells[current] = {"infected": [], "not infected": []}
            continue
        if current and a and d:
            label = str(a).strip().lower()
            m = re.match(r"Cell\s+(\d+)", str(d).strip())
            if m and label in ("infected", "not infected"):
                cells[current][label].append(int(m.group(1)))
    # drop empty blocks
    return {k: v for k, v in cells.items() if v["infected"] or v["not infected"]}


def variant_of(frames_dir):
    return os.path.basename(frames_dir).replace("frames_", "", 1)


def main():
    t_start = time.time()
    cohort = parse_excel(EXCEL_PATH, SHEET)
    n_inf = sum(len(v["infected"]) for v in cohort.values())
    n_uni = sum(len(v["not infected"]) for v in cohort.values())
    print(f"Excel: {len(cohort)} files, {n_inf} infected + {n_uni} not infected = {n_inf+n_uni} cells\n")

    for stem, groups in cohort.items():
        base_dir = os.path.join(BATCH_ROOT, stem)
        results  = os.path.join(base_dir, "results")
        tracked  = os.path.join(results, f"{stem}_tracked_{TRACKER}.csv")
        rois_zip = os.path.join(results, f"{stem}_rois_{TRACKER}.zip")

        # map each CellID -> its label
        label_of = {}
        for cid in groups["infected"]:      label_of[cid] = "infected"
        for cid in groups["not infected"]:  label_of[cid] = "not infected"
        want = set(label_of)

        print(f"=== {stem} ===")
        print(f"  {len(groups['infected'])} infected + {len(groups['not infected'])} not infected")

        if not os.path.exists(tracked):
            print(f"  ! no tracking CSV — skipping\n")
            continue

        ft = import_data(tracked, "Trackmate_auto", minframes=1)
        ft = ft.drop_duplicates(subset=["CellID", "FrameID"])
        ft_sel = ft[ft["CellID"].isin(want)].copy()
        found = sorted(ft_sel["CellID"].unique())
        missing = sorted(want - set(found))
        if missing:
            print(f"  ! cells not in tracking: {missing}")

                # which variant folders
        if VARIANTS == "all":
            variant_dirs = sorted(glob.glob(os.path.join(base_dir, "frames")))
        else:
            variant_dirs = [os.path.join(base_dir, f"frames_{v}") for v in VARIANTS]

        variant_dirs = [d for d in variant_dirs if os.path.isdir(d)]
        print(f"  {len(variant_dirs)} folder(s), {len(found)} cells found", flush=True)

        t_video = time.time()
        for vi, frames_dir in enumerate(variant_dirs, start=1):
            tag = variant_of(frames_dir)
            print(f"    [{vi}/{len(variant_dirs)}] extracting from '{os.path.basename(frames_dir)}' ...", flush=True)
            try:
                t0 = time.time()
                feats = cell_features(ft_sel, rois_zip, frames_dir, framerate=FRAMERATE)
                feats["label"]    = feats["CellID"].map(label_of)
                feats["infected"] = feats["label"] == "infected"
                out = os.path.join(results, f"{stem}_features_{tag}.csv")
                feats.to_csv(out, index=False)
                print(f"        done {feats.shape} in {time.time()-t0:.1f}s -> {os.path.basename(out)}", flush=True)
            except Exception as e:
                print(f"        FAILED {type(e).__name__}: {e}", flush=True)
        print(f"  file done in {(time.time()-t_video)/60:.1f} min "
              f"(total {(time.time()-t_start)/60:.1f} min)\n", flush=True)

    print(f"Done — total {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()