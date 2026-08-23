"""
Mask / Fluorescence Alignment
─────────────────────────────
Loads a segmentation mask and a 4-channel fluorescence image into napari
and lets you shift the mask over the fluorescence until they line up.

    python align_napari.py

Controls (with the napari window focused):

    arrow keys        nudge mask by STEP px
    Shift + arrows    nudge by 1 px (fine)
    r                 reset offset to zero
    s                 save offset to a .json next to the mask
    p                 print current offset

The offset is in (y, x) pixels: positive y = down, positive x = right.
"""

import os
import json
import numpy as np
import tifffile as tiff
import napari
from napari.utils import DirectLabelColormap
import nd2


# ═══════════════════════════════════════════════════════════════════
# SETTINGS — edit these, then run
# ═══════════════════════════════════════════════════════════════════

PROCESSED_FILE = r"processed_positions\A07.3\A07.3_pos15_sc_gaussian8_eq40_med3131__ce_pct1.0-99.0.tif"
MASK_FILE = r"segmentation_output\A07.3_pos15_sc_gaussian8_eq40_med3131__ce_pct1.0-99.0_cpsam_v2\masks\frame_080_mask.tif"

# 4-channel fluorescence. Expected shape (4, Y, X) — if the channel axis
# is somewhere else, set CHANNEL_AXIS to its index.
FLUO_FILE    = r"\\Hive2004\ag_idip\ZeljkaBaca_Data\ACID\250729\A07.3\H7_DENV2_MOI1_40h_fixed_stained.nd2"
FLUO_POSITION = 14        # must match the position the mask came from, it starts from 0

# Names and colours per channel, in file order.
CHANNEL_NAMES  = ["DAPI", "membrane", "actin", "viral"]
CHANNEL_COLORS = ["blue", "green", "red", "magenta"]

STEP = 5          # px per arrow press

# ═══════════════════════════════════════════════════════════════════
# (DO NOT EDIT BELOW)
# ═══════════════════════════════════════════════════════════════════


def load_fluorescence(path, position):
    """Read one position from a multi-position, multi-channel ND2."""
    with nd2.ND2File(path) as f:
        print(f"  nd2 sizes: {f.sizes}")
        img = f.asarray()

    img = img[position]                    # -> (C, Y, X) if order is (P, C, Y, X)
    return [img[c] for c in range(img.shape[0])]


def load_mask(path):
    """Read the mask. If it's a stack, take the last frame."""
    m = tiff.imread(path)
    if m.ndim == 3:
        print(f"  mask is a stack {m.shape}; using last frame")
        m = m[-1]
    return m.astype("uint16")


def main():
    print("loading...")
    mask = load_mask(MASK_FILE)
    channels = load_fluorescence(FLUO_FILE, FLUO_POSITION)
    print(f"  mask {mask.shape}, {len(channels)} fluorescence channel(s)")

    if channels[0].shape != mask.shape:
        print(f"  ! shapes differ: mask {mask.shape} vs fluo {channels[0].shape}")
        print("    a pure shift will not align these — check the source files")

    viewer = napari.Viewer(title="Mask / fluorescence alignment")

    processed = tiff.imread(PROCESSED_FILE)
    if processed.ndim == 3:
        processed = processed[-1]        # last frame
    viewer.add_image(processed, name="DIC (last frame)", colormap="gray")

    # Fluorescence underneath, one layer per channel
    for i, chan in enumerate(channels):
        name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else f"ch{i}"
        color = CHANNEL_COLORS[i] if i < len(CHANNEL_COLORS) else "gray"
        viewer.add_image(
            chan,
            name=name,
            colormap=color,
            blending="additive",
            visible=(i == 0),        # start with just the first channel on
        )

    # Mask on top — this is the layer we move
    n = int(mask.max())
    yellow = {i: [1.0, 1.0, 0.0, 1.0] for i in range(1, n + 1)}
    yellow[None] = [0, 0, 0, 0]        # background transparent
    mask_layer = viewer.add_labels(
        mask,
        name="mask",
        opacity=1.0,
        colormap=DirectLabelColormap(color_dict=yellow),
    )
    mask_layer.contour = 2

    offset = [0, 0]      # (y, x)

    dic_layer = viewer.add_image(processed, name="DIC (last frame)", colormap="gray")
    moving = [dic_layer, mask_layer]      # these move together

    def apply():
        for l in moving:
            l.translate = tuple(offset)
        viewer.status = f"offset (y, x) = ({offset[0]}, {offset[1]})"

    def shift(dy, dx):
        offset[0] += dy
        offset[1] += dx
        apply()

    @viewer.bind_key("Up")
    def _up(v): shift(-STEP, 0)

    @viewer.bind_key("Down")
    def _down(v): shift(STEP, 0)

    @viewer.bind_key("Left")
    def _left(v): shift(0, -STEP)

    @viewer.bind_key("Right")
    def _right(v): shift(0, STEP)

    @viewer.bind_key("Shift-Up")
    def _fup(v): shift(-1, 0)

    @viewer.bind_key("Shift-Down")
    def _fdown(v): shift(1, 0)

    @viewer.bind_key("Shift-Left")
    def _fleft(v): shift(0, -1)

    @viewer.bind_key("Shift-Right")
    def _fright(v): shift(0, 1)

    @viewer.bind_key("r")
    def _reset(v):
        offset[0] = offset[1] = 0
        apply()

    @viewer.bind_key("p")
    def _print(v):
        print(f"offset (y, x) = ({offset[0]}, {offset[1]})")

    @viewer.bind_key("s")
    def _save(v):
        out = os.path.splitext(MASK_FILE)[0] + "_offset.json"
        with open(out, "w") as f:
            json.dump({
                "mask_file": MASK_FILE,
                "fluo_file": FLUO_FILE,
                "offset_yx": offset,
            }, f, indent=2)
        print(f"saved offset {offset} -> {out}")
        viewer.status = f"saved -> {os.path.basename(out)}"

    apply()
    print("\narrows = nudge, Shift+arrows = fine, r = reset, s = save, p = print")
    napari.run()


if __name__ == "__main__":
    main()