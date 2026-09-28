# test_my_correction.py

from acid.image_processing.correct_background import correct_background_nd
from acid.image_processing.bgcorrection_validation import validate_correction

try:
    results = validate_correction(correct_background_nd, correction_kwargs={'method': "division", 'rescale_background':"max"})
    for bg_name, r in results.items():
        print(f"{bg_name}: std_observed={r['std_observed']:.6f}  std_corrected={r['std_corrected']:.6f}")
except AssertionError as e:
    print(e)


