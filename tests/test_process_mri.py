import pytest
import nibabel as nib
import numpy as np
import pydicom
import os
from src.process_mri import main, setup_logging, create_pdf_report
from src.utils.image_utils import (
    normalize_image,
    gaussian_smoothing,
    visualize_results,
    ImageHandler,
)
from src.utils.registration_utils import (
    load_atlas_mask,
    register_image,
    apply_brain_mask,
    calculate_quality_metrics,
)
from src.config import (
    DEFAULT_SIGMA,
    ATLAS_PATH,
    MASK_PATH,
    DEFAULT_INTENSITY_NORMALIZATION,
)
from matplotlib import pyplot as plt
from contextlib import contextmanager
import logging


@contextmanager
def assert_logging_level(level):
    """
    @brief A context manager to assert the logging level within a block of code.

    @param level (int): The expected logging level.
    """
    original_level = logging.root.level
    logging.disable(level - 1)
    yield
    logging.disable(original_level - 1)


def test_normalize_image() -> None:
    """
    @brief Tests the normalize_image function.

    Tests the normalization of an image using both z-score and min-max methods.
    """
    image = np.array([[1, 2], [3, 4]], dtype=float)
    normalized_image = normalize_image(image, method="z-score")
    assert np.allclose(normalized_image.mean(), 0)
    assert np.allclose(normalized_image.std(), 1)

    normalized_image = normalize_image(image, method="min-max")
    assert np.allclose(normalized_image.min(), 0)
    assert np.allclose(normalized_image.max(), 1)


def test_gaussian_smoothing() -> None:
    """
    @brief Tests the gaussian_smoothing function.

    Tests the application of Gaussian smoothing to an image.
    """
    image = np.array([[1, 2], [3, 4]], dtype=float)
    smoothed_image = gaussian_smoothing(image, sigma=1.0)
    assert smoothed_image.shape == image.shape


def test_load_atlas_mask() -> None:
    """
    @brief Tests the load_atlas_mask function.

    Tests the loading of the MNI atlas and mask.
    """
    atlas, mask = load_atlas_mask(ATLAS_PATH, MASK_PATH)
    assert isinstance(atlas, np.ndarray)
    assert isinstance(mask, np.ndarray)


def test_register_image() -> None:
    """
    @brief Tests the register_image function.

    Tests the registration of an input image to the MNI atlas space.
    """
    image = np.random.rand(10, 10, 10)
    atlas = np.random.rand(10, 10, 10)
    registered_image = register_image(image, atlas)
    assert registered_image.shape == image.shape


def test_apply_brain_mask() -> None:
    """
    @brief Tests the apply_brain_mask function.

    Tests the application of a brain mask to an input image.
    """
    image = np.random.rand(10, 10, 10)
    mask = np.ones((10, 10, 10))
    masked_image = apply_brain_mask(image, mask)
    assert np.array_equal(masked_image, image)


def create_dummy_nifti(tmp_path, data):
    """
    @brief Creates a dummy NIFTI file for testing.

    @param tmp_path (pathlib.Path): Temporary directory for test files.
    @param data (numpy.ndarray): Image data.
    """
    input_file = tmp_path / "input.nii.gz"
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, input_file)
    return str(input_file)


def test_load_image_nifti(tmp_path):
    """
    @brief Tests the load_image function with a NIFTI file.
    """
    data = np.random.rand(10, 10, 10)
    input_file = create_dummy_nifti(tmp_path, data)
    image_handler = ImageHandler(input_file)
    img_data = image_handler.img_data
    file_type = image_handler.file_type
    assert np.array_equal(img_data, data)
    assert file_type == "nii"


def test_visualize_results(tmp_path):
    """
    @brief Tests the visualize_results function.
    """
    data = np.random.rand(10, 10, 10)
    output_file_base = str(tmp_path / "test_output")
    visualize_results(data, data, output_file_base)
    assert os.path.exists(output_file_base + "_original.png")
    assert os.path.exists(output_file_base + "_skull_stripped.png")
    assert os.path.exists(output_file_base + "_checkerboard.png")


def test_main_invalid_normalization_method(tmp_path):
    """
    @brief Tests the main function with an invalid normalization method.
    """
    data = np.random.rand(10, 10, 10)
    input_file = create_dummy_nifti(tmp_path, data)
    output_file = tmp_path / "output.nii.gz"

    setup_logging()
    with pytest.raises(ValueError):
        main(
            str(input_file),
            str(output_file),
            DEFAULT_SIGMA,
            "invalid_method",
            True,  # Generate PDF
        )


def test_main_file_not_found(tmp_path):
    """
    @brief Tests the main function when the input file is not found.
    """
    input_file = tmp_path / "nonexistent_file.nii.gz"
    output_file = tmp_path / "output.nii.gz"

    setup_logging()
    with assert_logging_level(logging.ERROR):
        main(
            str(input_file),
            str(output_file),
            DEFAULT_SIGMA,
            DEFAULT_INTENSITY_NORMALIZATION,
            True,  # Generate PDF
        )

    assert not output_file.exists()


def test_calculate_quality_metrics():
    """
    @brief Tests the calculate_quality metrics function.
    """
    # Create two sample 3D numpy arrays
    image1 = np.random.rand(64, 64, 64)
    image2 = np.random.rand(64, 64, 64)

    # Call the function
    metrics = calculate_quality_metrics(image1, image2)

    # Assert that the returned object is a dictionary
    assert isinstance(metrics, dict)

    # Assert that the dictionary contains the expected keys
    assert "PCC" in metrics
    assert "PSNR" in metrics
    assert "SSIM" in metrics

    # Assert that the values are floats
    assert isinstance(metrics["PCC"], float)
    assert isinstance(metrics["PSNR"], float)
    assert isinstance(metrics["SSIM"], float)


def test_image_handler_nifti(tmp_path):
    """
    @brief Tests the ImageHandler class with a NIFTI file.
    """
    data = np.random.rand(10, 10, 10)
    input_file = create_dummy_nifti(tmp_path, data)
    image_handler = ImageHandler(input_file)
    assert np.array_equal(image_handler.img_data, data)
    assert image_handler.file_type == "nii"
    assert image_handler.affine.shape == (4, 4)


def test_image_handler_save_nifti(tmp_path):
    """
    @brief Tests the save_image method of ImageHandler with NIFTI format.
    """
    data = np.random.rand(10, 10, 10)
    input_file = create_dummy_nifti(tmp_path, data)
    image_handler = ImageHandler(input_file)
    output_file = str(tmp_path / "output.nii.gz")
    image_handler.save_image(output_file, data)
    assert os.path.exists(output_file)
    loaded_img = nib.load(output_file)
    assert np.array_equal(loaded_img.get_fdata(), data)
