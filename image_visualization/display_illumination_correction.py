import numpy as np
import matplotlib.pyplot as plt

def plot_image_channels_and_diagonals(
    image,
    corrected_image,
    channel_axis,
    axes,
    title_1="Original",
    title_2="Corrected",
    title_3="Main Diagonal",
    title_4="Anti Diagonal",
    x_axis="Pixel index",
    y_axis="Intensity",
    normalize=False
):
    """
    Visualize channels and diagonal intensity profiles.

    Parameters
    ----------
    image : np.ndarray
        Original image (3D).
    corrected_image : np.ndarray
        Corrected image (same shape as image).
    channel_axis : int
        Axis corresponding to channels.
    axes : np.ndarray
        Matplotlib axes array of shape (n_channels, 4).
    normalize : bool
        Whether to normalize intensity profiles to [0, 1].
    """

    # Move channel axis to front → (C, H, W)
    img = np.moveaxis(image, channel_axis, 0)
    corr = np.moveaxis(corrected_image, channel_axis, 0)

    n_channels = img.shape[0]

    for c in range(n_channels):
        ax_img = axes[c, 0]
        ax_corr = axes[c, 1]
        ax_diag1 = axes[c, 2]
        ax_diag2 = axes[c, 3]

        channel_img = img[c]
        channel_corr = corr[c]

        # --- Column 1: Original ---
        ax_img.imshow(channel_img, cmap="gray")
        ax_img.axis("off")

        # --- Column 2: Corrected ---
        ax_corr.imshow(channel_corr, cmap="gray")
        ax_corr.axis("off")

        # --- Extract diagonals ---
        diag1_img = np.diag(channel_img)
        diag1_corr = np.diag(channel_corr)

        diag2_img = np.diag(np.fliplr(channel_img))
        diag2_corr = np.diag(np.fliplr(channel_corr))

        # --- Normalize if requested ---
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-8)

        if normalize:
            diag1_img = norm(diag1_img)
            diag1_corr = norm(diag1_corr)
            diag2_img = norm(diag2_img)
            diag2_corr = norm(diag2_corr)

        # --- Column 3: Main diagonal ---
        ax_diag1.plot(diag1_img, label="Original")
        ax_diag1.plot(diag1_corr, label="Corrected")
        ax_diag1.set_xlabel(x_axis)
        ax_diag1.set_ylabel(y_axis)

        # --- Column 4: Anti-diagonal ---
        ax_diag2.plot(diag2_img, label="Original")
        ax_diag2.plot(diag2_corr, label="Corrected")
        ax_diag2.set_xlabel(x_axis)
        ax_diag2.set_ylabel(y_axis)

        # Add legend only once per row (optional: could restrict to first row)
        ax_diag1.legend()
        ax_diag2.legend()

    # --- Column titles (top row only) ---
    axes[0, 0].set_title(title_1)
    axes[0, 1].set_title(title_2)
    axes[0, 2].set_title(title_3)
    axes[0, 3].set_title(title_4)

    plt.tight_layout()

