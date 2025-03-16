# Configuration settings for the brain skull stripping pipeline

# Default parameters for image processing
DEFAULT_SIGMA = 1.5  # Standard deviation for Gaussian smoothing
DEFAULT_INTENSITY_NORMALIZATION = "z-score"  # Options: 'z-score', 'min-max'

# Paths to the MNI atlas files
ATLAS_PATH = "data/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii"
MASK_PATH = "data/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii"
