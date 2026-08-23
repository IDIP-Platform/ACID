"""
Export one cell's ROIs (Fiji) + fixed-size square crops (model input)
─────────────────────────────────────────────────────────────────────
Given a tracked video and a CellID, exports two things for that cell:

  1. a Fiji-readable ROI zip  — the cell's outlines across all its frames
     (CellPhe ROIs are already ImageJ format; just selected + repackaged)

  2. fixed-size square image crops — one PxP TIFF per frame, centered on the
     cell, following it as it moves. Uniform size = ready for model input.

Run in the cellphepy environment:  python export_cell_rois_fiji.py
"""

import os
import zipfile
import numpy as np
import tifffile as tiff
from cellphe import import_data
from cellphe.input import read_rois


# ═══════════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════════

BASE = r"batch_output\A04.2_pos16_sc_gaussian8_eq(A07.3_avg41pos_x81t_sc_gaussian8)_med2527"

CELL_IDS  = [5, 20, 35, 43, 49, 64]  # which tracked CellIDs to export (each -> its own zip)
TRACKER   = "Overlap"
 
# square crop settings
CROP_SIZE   = 96           # square side in pixels (fixed for every frame)
FRAMES_DIR  = "frames"     # which frame folder to crop from (e.g. "frames",
                           #   or a variant like "frames_..._nlmeans")
MASK_CROP   = False        # True = zero out pixels outside the cell mask in the crop
 
OUT_DIR = ""               # defaults to BASE/results if left as ""
 
 
# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════
 
def square_crop(frame, cx, cy, size):
    """Fixed-size square crop centered on (cx, cy), zero-padded at image edges."""
    half = size // 2
    H, W = frame.shape
    cx, cy = int(round(cx)), int(round(cy))
    out = np.zeros((size, size), dtype=frame.dtype)
    x0, y0 = cx - half, cy - half
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + size), min(H, y0 + size)
    dx0, dy0 = sx0 - x0, sy0 - y0
    out[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = frame[sy0:sy1, sx0:sx1]
    return out
 
 
def export_cell(cell_id, df, rois, roi_zip, frames_dir, out_dir, stem):
    """Export one cell: its ROI zip + fixed-size square crops."""
    cell = df[df["CellID"] == cell_id].sort_values("FrameID")
    if cell.empty:
        print(f"\n! Cell {cell_id}: not found — skipping")
        return
    wanted = set(cell["ROI_filename"].astype(str))
    frames = cell["FrameID"].tolist()
    print(f"\n=== Cell {cell_id}: {len(frames)} frames ({frames[0]}–{frames[-1]}) ===")
 
    # 1) ROI zip for Fiji
    out_roi = os.path.join(out_dir, f"{stem}_cell{cell_id}_ROIs.zip")
    copied = 0
    with zipfile.ZipFile(roi_zip, "r") as zin, \
         zipfile.ZipFile(out_roi, "w", zipfile.ZIP_DEFLATED) as zout:
        for entry in zin.namelist():
            s = entry[:-4] if entry.lower().endswith(".roi") else entry
            if s in wanted or entry in wanted:
                zout.writestr(entry, zin.read(entry))
                copied += 1
    print(f"  ROIs: {copied} -> {os.path.basename(out_roi)}")
 
    # 2) fixed-size square crops (centroid from the ROI polygon)
    crop_dir = os.path.join(out_dir, f"{stem}_cell{cell_id}_crops{CROP_SIZE}")
    os.makedirs(crop_dir, exist_ok=True)
    made = 0
    for _, row in cell.iterrows():
        fr  = int(row["FrameID"])
        rn  = str(row["ROI_filename"])
        poly = rois.get(rn) or rois.get(rn + ".roi")
        if poly is None:
            print(f"    frame {fr}: ROI '{rn}' not in archive — skipping")
            continue
        poly = np.asarray(poly)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
 
        frame_path = os.path.join(frames_dir, f"exp-{fr:04d}.tif")
        if not os.path.exists(frame_path):
            print(f"    frame {fr}: {os.path.basename(frame_path)} missing — skipping")
            continue
        frame = tiff.imread(frame_path).astype(np.float32)
 
        if MASK_CROP:
            m = np.zeros_like(frame, dtype=bool)
            xs = np.clip(poly[:, 0].astype(int), 0, frame.shape[1] - 1)
            ys = np.clip(poly[:, 1].astype(int), 0, frame.shape[0] - 1)
            from skimage.draw import polygon as sk_polygon
            rr, cc = sk_polygon(ys, xs, frame.shape)
            m[rr, cc] = True
            frame = np.where(m, frame, 0)
 
        crop = square_crop(frame, cx, cy, CROP_SIZE)
        tiff.imwrite(os.path.join(crop_dir, f"cell{cell_id}_frame{fr:04d}.tif"),
                     crop.astype("float32"))
        made += 1
    print(f"  crops: {made} × {CROP_SIZE}px -> {os.path.basename(crop_dir)}/")
 
 
def main():
    stem = os.path.basename(BASE.rstrip("\\/"))
    results   = os.path.join(BASE, "results")
    csv_path  = os.path.join(results, f"{stem}_tracked_{TRACKER}.csv")
    roi_zip   = os.path.join(results, f"{stem}_rois_{TRACKER}.zip")
    frames_dir = os.path.join(BASE, FRAMES_DIR)
 
    for p in (csv_path, roi_zip):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
 
    out_dir = OUT_DIR or results
    os.makedirs(out_dir, exist_ok=True)
 
    # load tracking table + ROIs ONCE, reuse for every cell
    df = import_data(csv_path, "Trackmate_auto", minframes=1)
    df = df.drop_duplicates(subset=["CellID", "FrameID"])
    rois = read_rois(roi_zip)
 
    for cell_id in CELL_IDS:
        export_cell(cell_id, df, rois, roi_zip, frames_dir, out_dir, stem)
 
    print("\nFiji: ROI Manager → More ▸▸ → Open → any of the ROI zips")
 
 
if __name__ == "__main__":
    main()
 