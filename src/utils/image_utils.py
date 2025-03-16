import numpy as np
from scipy.ndimage import gaussian_filter
import logging
import matplotlib.pyplot as plt
import nibabel as nib
from pydicom import dcmread, dcmwrite
import gzip
import shutil
import os


class ImageHandler:
    def __init__(self, input_file: str):
        self.file_type = None
        self.affine = None
        self.input_file = input_file
        self.img_data = self._load_image(input_file)

    def _load_image(self, input_file: str) -> np.ndarray:
        """
        @brief Loads image data from either NIFTI or DICOM format.

        Handles loading of different image formats and returns the image data, affine transformation, and the file type.

        @param input_file (str): Path to the input image file.

        @return np.ndarray: The image data as a NumPy array.
        """
        try:
            if input_file.lower().endswith((".nii", ".nii.gz")):
                logging.info(f"Loading NIFTI file: {input_file}")
                img = nib.load(input_file)
                img_data = img.get_fdata()
                self.affine = img.affine
                self.file_type = "nii"
            elif input_file.lower().endswith(".dcm.gz"):
                logging.info(f"Decompressing and loading DICOM file: {input_file}")
                decompressed_file = input_file[:-3]  # Remove .gz extension
                with gzip.open(input_file, "rb") as f_in:
                    with open(decompressed_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                dicom = dcmread(decompressed_file)
                img_data = dicom.pixel_array
                self.affine = None  # DICOM doesn't have a direct affine, needs more complex handling
                self.file_type = "dcm"
                os.remove(decompressed_file)  # Clean up decompressed file
            elif input_file.lower().endswith(".dcm"):
                logging.info(f"Loading DICOM file: {input_file}")
                dicom = dcmread(input_file)
                img_data = dicom.pixel_array
                self.affine = None  # DICOM doesn't have a direct affine, needs more complex handling
                self.file_type = "dcm"
            else:
                raise ValueError(
                    "Unsupported image format. Please provide a NIFTI or DICOM file."
                )

            if not isinstance(img_data, np.ndarray):
                raise TypeError("Image data is not a numpy array.")

            return img_data

        except Exception as e:
            logging.error(f"Error loading image file: {e}")
            raise

    def save_image(self, output_file: str, image_data: np.ndarray) -> None:
        """
        @brief Saves image data to either NIFTI or DICOM format.

        Handles saving of different image formats and writes the image data to the specified output file.

        @param output_file (str): Path to save the output image file.
        @param image_data (np.ndarray): The image data as a NumPy array.
        """
        try:
            if self.file_type == "nii":
                if output_file.endswith(".gz"):
                    nib.save(nib.Nifti1Image(image_data, self.affine), output_file[:-3])
                    with open(output_file[:-3], "rb") as f_in:
                        with gzip.open(output_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(output_file[:-3])
                else:
                    nib.save(nib.Nifti1Image(image_data, self.affine), output_file)
            elif self.file_type == "dcm":
                # Load the original DICOM file to copy metadata
                original_dicom = dcmread(self.input_file)

                # Create a new DICOM dataset
                ds = original_dicom.copy()

                # Update pixel data
                ds.PixelData = image_data.astype(
                    original_dicom.pixel_array.dtype
                ).tobytes()
                ds.Rows, ds.Columns = image_data.shape

                if output_file.endswith(".gz"):
                    # Save the new DICOM file
                    dcmwrite(output_file[:-3], ds)
                    with open(output_file[:-3], "rb") as f_in:
                        with gzip.open(output_file, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(output_file[:-3])
                else:
                    # Save the new DICOM file
                    dcmwrite(output_file, ds)
            else:
                raise ValueError(
                    "Unsupported image format. Please provide a NIFTI or DICOM file."
                )

        except Exception as e:
            logging.error(f"Error saving image file: {e}")
            raise


def normalize_image(image: np.ndarray, method: str = "z-score") -> np.ndarray:
    """
    @brief Normalizes the intensity values of an image.

    Normalizes the image using either z-score normalization or min-max scaling.

    @param image (numpy.ndarray): The input image as a NumPy array.
    @param method (str): The normalization method to use. Options are 'z-score' or 'min-max'. Default is 'z-score'.

    @return numpy.ndarray: The normalized image.

    @raises ValueError: If an invalid normalization method is specified.
    """
    if method == "z-score":
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
    elif method == "min-max":
        min_val = image.min()
        max_val = image.max()
        normalized_image = (image - min_val) / (max_val - min_val)
    else:
        raise ValueError("Normalization method must be 'z-score' or 'min-max'.")

    return normalized_image


def gaussian_smoothing(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    @brief Applies Gaussian smoothing to an image.

    Applies a Gaussian filter to smooth the image.

    @param image (numpy.ndarray): The input image as a NumPy array.
    @param sigma (float): The standard deviation for the Gaussian kernel. Default is 1.0.

    @return numpy.ndarray: The smoothed image.
    """
    smoothed_image = gaussian_filter(image, sigma=sigma)
    return smoothed_image


def visualize_results(
    original_image: np.ndarray,
    registered_image: np.ndarray,
    skull_stripped_image: np.ndarray,
    output_file_base: str,
) -> None:
    """
    @brief Visualizes the skull stripping results by plotting middle slices in each dimension.

    @param original_image (numpy.ndarray): The original brain image data as a NumPy array.
    @param registered_image (numpy.ndarray): The registered brain image data as a NumPy array.
    @param skull_stripped_image (numpy.ndarray): The skull-stripped brain image data as a NumPy array.
    @param output_path (str): The path to save the visualization image.
    """
    # Get middle slices
    slice_x = original_image.shape[0] // 2
    slice_y = original_image.shape[1] // 2
    slice_z = original_image.shape[2] // 2

    # Create the figure and axes
    fig_original = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    ax_z = fig_original.add_subplot(121)  # Z axis on the left
    ax_x = fig_original.add_subplot(243)  # X axis on the right, top
    ax_y = fig_original.add_subplot(247)  # Y axis on the right, bottom

    # Plot Z axis
    ax_z.imshow(original_image[:, :, slice_z].T, cmap="gray", origin="lower")
    ax_z.set_title("Z Axis")
    ax_z.axis("off")

    # Plot X axis (scaled down)
    ax_x.imshow(original_image[slice_x, :, :].T, cmap="gray", origin="lower")
    ax_x.set_title("X Axis")
    ax_x.axis("off")

    # Plot Y axis (scaled down)
    ax_y.imshow(original_image[:, slice_y, :].T, cmap="gray", origin="lower")
    ax_y.set_title("Y Axis")
    ax_y.axis("off")

    fig_original.suptitle("Middle slices of original image")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the image
    plt.savefig(output_file_base + "_original.png")

    # Show the plot
    plt.show()

    # Close the figure
    plt.close()

    # Get middle slices of registered and skull-striped images
    slice_x = registered_image.shape[0] // 2
    slice_y = registered_image.shape[1] // 2
    slice_z = registered_image.shape[2] // 2

    # Create the figure and axes
    fig_registered = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    ax_z = fig_registered.add_subplot(121)  # Z axis on the left
    ax_x = fig_registered.add_subplot(243)  # X axis on the right, top
    ax_y = fig_registered.add_subplot(247)  # Y axis on the right, bottom

    # Plot Z axis
    ax_z.imshow(registered_image[:, :, slice_z].T, cmap="gray", origin="lower")
    ax_z.set_title("Z Axis")
    ax_z.axis("off")

    # Plot X axis (scaled down)
    ax_x.imshow(registered_image[slice_x, :, :].T, cmap="gray", origin="lower")
    ax_x.set_title("X Axis")
    ax_x.axis("off")

    # Plot Y axis (scaled down)
    ax_y.imshow(registered_image[:, slice_y, :].T, cmap="gray", origin="lower")
    ax_y.set_title("Y Axis")
    ax_y.axis("off")

    fig_registered.suptitle("Middle slices of registered image")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Save the image
    plt.savefig(output_file_base + "_registered.png")

    # Show the plot
    plt.show()

    # Close the figure
    plt.close()

    # Create the figure and axes
    fig_stripped_skull = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    ax_z = fig_stripped_skull.add_subplot(121)  # Z axis on the left
    ax_x = fig_stripped_skull.add_subplot(243)  # X axis on the right, top
    ax_y = fig_stripped_skull.add_subplot(247)  # Y axis on the right, bottom

    # Plot Z axis
    ax_z.imshow(skull_stripped_image[:, :, slice_z].T, cmap="gray", origin="lower")
    ax_z.set_title("Z Axis")
    ax_z.axis("off")

    # Plot X axis (scaled down)
    ax_x.imshow(skull_stripped_image[slice_x, :, :].T, cmap="gray", origin="lower")
    ax_x.set_title("X Axis")
    ax_x.axis("off")

    # Plot Y axis (scaled down)
    ax_y.imshow(skull_stripped_image[:, slice_y, :].T, cmap="gray", origin="lower")
    ax_y.set_title("Y Axis")
    ax_y.axis("off")

    fig_stripped_skull.suptitle("Middle slices of skull stripped image")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the skull-stripped image visualization
    skull_stripped_output_path = output_file_base + "_skull_stripped.png"
    plt.savefig(skull_stripped_output_path)

    # Show the plot
    plt.show()

    # Close the figure
    plt.close()

    # Create checkerboard comparison
    fig_checkerboard = plt.figure(figsize=(12, 6))  # Adjust figure size as needed
    ax_checkerboard_z = fig_checkerboard.add_subplot(121)  # Z axis on the left
    ax_checkerboard_x = fig_checkerboard.add_subplot(243)  # X axis on the right, top
    ax_checkerboard_y = fig_checkerboard.add_subplot(247)  # Y axis on the right, bottom

    # Create checkerboard pattern for Z axis
    checkerboard_z = np.zeros_like(registered_image[:, :, slice_z])
    checkerboard_z[::2, ::2] = registered_image[::2, ::2, slice_z]
    checkerboard_z[1::2, 1::2] = skull_stripped_image[1::2, 1::2, slice_z]

    # Plot checkerboard comparison for Z axis
    ax_checkerboard_z.imshow(checkerboard_z.T, cmap="gray", origin="lower")
    ax_checkerboard_z.set_title("Checkerboard Z Axis")
    ax_checkerboard_z.axis("off")

    # Create checkerboard pattern for X axis
    checkerboard_x = np.zeros_like(registered_image[slice_x, :, :])
    checkerboard_x[::2, ::2] = registered_image[slice_x, ::2, ::2]
    checkerboard_x[1::2, 1::2] = skull_stripped_image[slice_x, 1::2, 1::2]

    # Plot checkerboard comparison for X axis
    ax_checkerboard_x.imshow(checkerboard_x.T, cmap="gray", origin="lower")
    ax_checkerboard_x.set_title("Checkerboard X Axis")
    ax_checkerboard_x.axis("off")

    # Create checkerboard pattern for Y axis
    checkerboard_y = np.zeros_like(registered_image[:, slice_y, :])
    checkerboard_y[::2, ::2] = registered_image[::2, slice_y, ::2]
    checkerboard_y[1::2, 1::2] = skull_stripped_image[1::2, slice_y, 1::2]

    # Plot checkerboard comparison for Y axis
    ax_checkerboard_y.imshow(checkerboard_y.T, cmap="gray", origin="lower")
    ax_checkerboard_y.set_title("Checkerboard Y Axis")
    ax_checkerboard_y.axis("off")

    fig_checkerboard.suptitle(
        "Checkerboard Comparison of Original and Skull-Stripped Images"
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the checkerboard comparison visualization
    checkerboard_output_path = output_file_base + "_checkerboard.png"
    plt.savefig(checkerboard_output_path)

    # Show the plot
    plt.show()

    # Close the figure
    plt.close()
