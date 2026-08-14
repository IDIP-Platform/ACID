import numpy as np

def make_natural_like_image(size=256, seed=0):
    """
    Generate a synthetic natural-like image with a 1/f amplitude spectrum.
    """
    rng = np.random.default_rng(seed)

    # Frequency grid
    fx = np.fft.fftfreq(size)
    fy = np.fft.fftfreq(size)
    FX, FY = np.meshgrid(fx, fy)
    freq = np.sqrt(FX**2 + FY**2)
    freq[0, 0] = np.inf  # avoid division by zero

    # Random phase
    phase = rng.uniform(0, 2*np.pi, (size, size))

    # 1/f amplitude spectrum
    amplitude = 1.0 / freq

    # Complex spectrum
    spectrum = amplitude * np.exp(1j * phase)

    # Inverse FFT
    image = np.fft.ifft2(spectrum).real

    # Normalize
    image -= image.min()
    image /= image.max()

    return image
