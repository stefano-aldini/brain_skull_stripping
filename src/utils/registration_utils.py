import nibabel as nib
import numpy as np
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr


def load_atlas_mask(atlas_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    @brief Loads the MNI atlas image.

    Loads the MNI atlas image from the specified file path using nibabel.

    @param atlas_path (str): Path to the MNI atlas NIFTI file.

    @return tuple: A tuple containing the nibabel image object and the image data as a NumPy array.
    """
    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata()

    # Load the brain mask
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()
    return atlas_data, mask_data


def register_image(input_img: np.ndarray, atlas_img: np.ndarray) -> np.ndarray:
    """
    @brief Registers an input image to the MNI atlas space.

    Resizes the input image to match the dimensions of the MNI atlas image.

    @param input_img (numpy.ndarray): The input image data as a NumPy array.
    @param atlas_img (numpy.ndarray): The MNI atlas image data as a NumPy array.

    @return numpy.ndarray: The resized input image registered to the atlas space.
    """
    # Resize input image to match atlas dimensions
    input_resized = resize(
        input_img, atlas_img.shape, mode="reflect", anti_aliasing=True
    )
    return input_resized


def apply_brain_mask(input_img: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    @brief Applies a brain mask to an input image.

    Applies the provided brain mask to the input image to extract the brain region.

    @param input_img (numpy.ndarray): The input image data as a NumPy array.
    @param brain_mask (numpy.ndarray): The brain mask data as a NumPy array.

    @return numpy.ndarray: The masked image with only the brain region visible.
    """
    # Apply the brain mask to the input image
    masked_img = input_img * brain_mask
    return masked_img


def calculate_quality_metrics(
    original_image: np.ndarray, skull_stripped_image: np.ndarray
) -> dict:
    """
    @brief Calculates quality assessment metrics to evaluate the skull stripping results.

    Calculates Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR),
    and Pearson Correlation Coefficient (PCC) between the original and skull-stripped images.

    @param original_image (numpy.ndarray): The original brain image data as a NumPy array.
    @param skull_stripped_image (numpy.ndarray): The skull-stripped brain image data as a NumPy array.

    @return dict: A dictionary containing the calculated quality metrics.
    """
    # Ensure the images have the same dynamic range
    original_image = original_image.astype(np.float64)
    skull_stripped_image = skull_stripped_image.astype(np.float64)

    # Calculate SSIM
    ssim_score = ssim(
        original_image,
        skull_stripped_image,
        data_range=skull_stripped_image.max() - skull_stripped_image.min(),
    )

    # Calculate PSNR
    psnr_score = psnr(
        original_image,
        skull_stripped_image,
        data_range=skull_stripped_image.max() - skull_stripped_image.min(),
    )

    # Calculate PCC
    pcc, _ = pearsonr(original_image.flatten(), skull_stripped_image.flatten())

    return {"SSIM": ssim_score, "PSNR": psnr_score, "PCC": pcc}
