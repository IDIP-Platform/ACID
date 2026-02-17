import numba
import numpy as np


@numba.njit(cache=True)
def graycoprops(P: np.ndarray, prop="contrast"):
    """Calculate texture properties of a GLCM.

    Compute a feature of a gray level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1}P_{i,j}|i-j|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]
    - 'mean': :math:`\\sum_{i=0}^{levels-1} i*P_{i}`
    - 'variance': :math:`\\sum_{i=0}^{levels-1} P_{i}*(i-mean)^2`
    - 'std': :math:`\\sqrt{variance}`
    - 'entropy': :math:`\\sum_{i,j=0}^{levels-1} -P_{i,j}*log(P_{i,j})`

    Each GLCM is normalized to have a sum of 1 before the computation of
    texture properties.

    .. versionchanged:: 0.19
           `greycoprops` was renamed to `graycoprops` in 0.19.

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the gray-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that gray-level j
        occurs at a distance d and at an angle theta from
        gray-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', \
            'correlation', 'ASM', 'mean', 'variance', 'std', 'entropy'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] M. Hall-Beyer, 2007. GLCM Texture: A Tutorial v. 1.0 through 3.0.
           The GLCM Tutorial Home Page,
           https://prism.ucalgary.ca/handle/1880/51900
           DOI:`10.11575/PRISM/33280`

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = graycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = graycoprops(g, 'contrast')
    >>> contrast
    array([[0.58333333, 1.        ],
           [1.25      , 2.75      ]])

    """

    num_level, num_level2, num_dist, num_angle = np.int64(P.shape)
    if num_level != num_level2:
        raise ValueError("num_level and num_level2 must be equal.")
    if num_dist <= 0:
        raise ValueError("num_dist must be positive.")
    if num_angle <= 0:
        raise ValueError("num_angle must be positive.")

    npone = np.int64(1)

    I_ = np.arange(num_level, dtype=np.float64).reshape(
        (num_level, npone, npone, npone)
    )
    tmp_ = I_ * P
    mean = np.sum(tmp_, axis=0)
    mean = np.sum(mean, axis=0)

    # normalize each GLCM
    P = P.astype(np.float64)
    glcm_sums_1 = np.sum(P, axis=0)
    glcm_sums = np.sum(glcm_sums_1, axis=0)
    glcm_sums = np.expand_dims(glcm_sums, axis=0)
    glcm_sums = np.expand_dims(glcm_sums, axis=0)
    shape = glcm_sums.shape
    glcm_sums = glcm_sums.flatten()
    glcm_sums[glcm_sums == 0] = 1
    glcm_sums = glcm_sums.reshape(shape)
    P /= glcm_sums

    # create weights for specified property
    # I, J = np.ogrid[0:num_level, 0:num_level]
    I = np.arange(num_level, dtype=np.float64).reshape((num_level, npone))
    J = I.T
    weights = np.empty((num_level, num_level, npone, npone), dtype=np.float64)
    if prop == "contrast":
        weights = ((I - J) ** 2).reshape((num_level, num_level, npone, npone))
    elif prop == "dissimilarity":
        weights = (np.abs(I - J)).reshape((num_level, num_level, npone, npone))
    elif prop == "homogeneity":
        weights = (1.0 / (1.0 + (I - J) ** 2)).reshape(
            (num_level, num_level, npone, npone)
        )
    elif prop in ["ASM", "energy", "correlation", "entropy", "variance", "mean", "std"]:
        pass
    else:
        raise ValueError(f"{prop} is an invalid property")

    # compute property for each GLCM
    if prop == "energy":
        asm_1 = np.sum(P**2, axis=0)
        asm = np.sum(asm_1, axis=0)
        results = np.expand_dims(np.sqrt(asm), axis=0)
    elif prop == "ASM":
        asm_1 = np.sum(P**2, axis=0)
        asm = np.sum(asm_1, axis=0)
        results = np.expand_dims(asm, axis=0)
    elif prop == "mean":
        results = mean
        results = np.atleast_3d(mean)
    elif prop == "variance":
        I = I_
        var_1 = np.sum(P * ((I - mean) ** 2), axis=0)
        var = np.sum(var_1, axis=0)
        results = np.atleast_3d(var)
    elif prop == "std":
        I = I_
        var_1 = np.sum(P * ((I - mean) ** 2), axis=0)
        var = np.sum(var_1, axis=0)
        results = np.atleast_3d(np.sqrt(var))
    elif prop == "entropy":
        shape_p = P.shape
        ln = -np.log(P)
        ln = ln.flatten()
        ln[ln == np.inf] = 0
        ln.reshape(shape_p)

        ent_1 = np.sum(P * ln, axis=0)
        ent = np.sum(ent_1, axis=0)
        results = np.atleast_3d(ent)

    elif prop == "correlation":
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.arange(num_level, dtype=np.float64).reshape(
            (num_level, npone, npone, npone)
        )
        J = np.arange(num_level, dtype=np.float64).reshape(
            (npone, num_level, npone, npone)
        )
        diff_i_1 = np.sum(I * P, axis=0)
        diff_i = I - np.sum(diff_i_1, axis=0)
        diff_j_1 = np.sum(J * P, axis=0)
        diff_j = J - np.sum(diff_j_1, axis=0)

        std_i_1 = np.sum(P * (diff_i) ** 2, axis=0)
        std_i = np.sqrt(np.sum(std_i_1, axis=0))
        std_j_1 = np.sum(P * (diff_j) ** 2, axis=0)
        std_j = np.sqrt(np.sum(std_j_1, axis=0))
        cov_1 = np.sum(P * (diff_i * diff_j), axis=0)
        cov = np.sum(cov_1, axis=0)

        # handle the special case of standard deviations near zero
        mask_0i = std_i < 1e-15
        mask_0j = std_j < 1e-15
        mask_0i = mask_0i.flatten()
        mask_0j = mask_0j.flatten()
        mask_0i[mask_0j] = True

        results_shape = results.shape
        results_f = results.flatten()
        results_f[mask_0i] = 1

        # mask_0 = mask_0i.reshape(m0i_shape)

        # handle the standard case
        mask_1i = ~mask_0i
        cov = cov.flatten()
        std_i = std_i.flatten()
        std_j = std_j.flatten()
        results_f[mask_1i] = cov[mask_1i] / (std_i[mask_1i] * std_j[mask_1i])
        results_ff = results_f.reshape(results_shape)
        return np.expand_dims(results_ff, axis=0)

    elif prop in ["contrast", "dissimilarity", "homogeneity"]:
        # weights = weights.reshape((num_level, num_level, npone, npone))
        # shape = np.array([num_level, num_level, npone, npone], dtype=int)

        w = weights
        m = P * w

        results_1 = np.sum(m, axis=0)
        return np.expand_dims(np.sum(results_1, axis=0), axis=0)

    return results
