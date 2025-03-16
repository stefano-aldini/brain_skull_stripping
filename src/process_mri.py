import argparse
import logging
import numpy as np
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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
import os


def setup_logging() -> None:
    """
    @brief Sets up basic logging configuration.

    Configures the logging system to output log messages to the console.
    The log level is set to INFO, and the log messages include a timestamp,
    the log level, and the message itself.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def create_pdf_report(report: dict, pdf_output_path: str = "") -> None:
    """
    @brief Generates a PDF report summarizing the processing results.

    @param report (dict): A dictionary containing the report data.
    @param pdf_output_path (str): The path to save the PDF report.
    """

    def split_text(text, max_length):
        """
        Splits text into multiple lines if it exceeds the max_length.
        """
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 <= max_length:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    max_line_length = 100  # Maximum length of a line before splitting

    with PdfPages(pdf_output_path) as pdf:
        # First page: Report summary
        plt.figure(figsize=(8.5, 11))
        plt.axis("off")

        # Title
        plt.text(
            0.5,
            0.95,
            "Brain Skull Stripping Report",
            fontsize=20,
            ha="center",
            weight="bold",
        )

        # Date and time
        plt.text(
            0.5,
            0.9,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=10,
            ha="center",
        )

        # Horizontal line
        plt.axhline(y=0.88, color="black", linestyle="-", linewidth=1)

        # Report content
        y_pos = 0.85
        for key, value in report.items():
            if key == "quality_metrics" and isinstance(value, dict):
                plt.text(0.01, y_pos, f"{key}:", fontsize=8, weight="bold")
                y_pos -= 0.03
                for metric_key, metric_value in value.items():
                    plt.text(
                        0.05, y_pos, f"{metric_key}: {metric_value:.4f}", fontsize=8
                    )
                    y_pos -= 0.03
            elif (
                key != "visualization_output"
            ):  # Skip visualization path in text report
                lines = split_text(f"{key}: {value}", max_line_length)
                if len(lines) == 1:
                    key_part, value_part = lines[0].split(":", 1)
                    plt.text(0.01, y_pos, f"{key_part}:", fontsize=8, weight="bold")
                    y_pos -= 0.03
                    plt.text(0.05, y_pos, value_part.strip(), fontsize=8)
                    y_pos -= 0.03
                else:
                    for i, line in enumerate(lines):
                        if i == 0:
                            plt.text(0.01, y_pos, line, fontsize=8, weight="bold")
                        else:
                            plt.text(0.05, y_pos, line, fontsize=8)
                        y_pos -= 0.03

        # Status indication with color
        if "status" in report:
            if report["status"] == "success":
                status_color = "green"
            else:
                status_color = "red"
            plt.text(
                0.5,
                0.1,
                f"Processing Status: {report['status'].upper()}",
                fontsize=14,
                ha="center",
                weight="bold",
                color=status_color,
            )

        pdf.savefig()
        plt.close()

        # Second page: Visualization
        if "visualization_output" in report and report["visualization_output"]:
            try:
                if isinstance(report["visualization_output"], str):
                    # Create a figure for the visualizations
                    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))

                    # Original image
                    original_image_path = (
                        report["visualization_output"] + "_original.png"
                    )
                    if os.path.exists(original_image_path):
                        img = plt.imread(original_image_path)
                        axes[0].imshow(img)
                        axes[0].set_title("Original Brain Image")
                        axes[0].axis("off")
                    else:
                        logging.error(
                            f"Original image not found: {original_image_path}"
                        )
                        axes[0].text(0.5, 0.5, "Original image not found", ha="center")

                    # Skull-stripped image
                    skull_stripped_image_path = (
                        report["visualization_output"] + "_skull_stripped.png"
                    )
                    if os.path.exists(skull_stripped_image_path):
                        img = plt.imread(skull_stripped_image_path)
                        axes[1].imshow(img)
                        axes[1].set_title("Skull-Stripped Brain Image")
                        axes[1].axis("off")
                    else:
                        logging.error(
                            f"Skull-stripped image not found: {skull_stripped_image_path}"
                        )
                        axes[1].text(
                            0.5, 0.5, "Skull-stripped image not found", ha="center"
                        )

                    # Add the figure to the PDF
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Checkerboard comparison image on a new page
                    fig_checkerboard = plt.figure(figsize=(8.5, 11))
                    checkerboard_image_path = (
                        report["visualization_output"] + "_checkerboard.png"
                    )
                    if os.path.exists(checkerboard_image_path):
                        img = plt.imread(checkerboard_image_path)
                        plt.imshow(img)
                        plt.title("Checkerboard Comparison")
                        plt.axis("off")
                    else:
                        logging.error(
                            f"Checkerboard image not found: {checkerboard_image_path}"
                        )
                        plt.text(0.5, 0.5, "Checkerboard image not found", ha="center")

                    # Add the checkerboard figure to the PDF
                    plt.tight_layout()
                    pdf.savefig(fig_checkerboard)
                    plt.close(fig_checkerboard)
            except Exception as e:
                logging.error(f"Error adding visualization to PDF: {e}")
                fig = plt.figure(figsize=(8.5, 11))
                plt.text(
                    0.5,
                    0.5,
                    f"Error adding visualization: {e}",
                    fontsize=10,
                    ha="center",
                )
                pdf.savefig(fig)
                plt.close(fig)


def main(
    input_file: str,
    output_file: str,
    sigma: float,
    normalization_method: str,
    generate_pdf: bool,
) -> None:
    """
    @brief Main function to perform atlas-based skull stripping on a 3D brain scan.

    Loads a NIFTI or DICOM image, preprocesses it, registers it to an atlas, applies a brain mask,
    and saves the skull-stripped brain to an output file.

    @param input_file (str): Path to the input NIFTI or DICOM file.
    @param output_file (str): Path to save the skull-stripped NIFTI file.
    @param sigma (float): Standard deviation for Gaussian smoothing.
    @param normalization_method (str): The normalization method to use. Options are 'z-score' or 'min-max'.
    @param generate_pdf (bool): Whether to generate a PDF report.
    """
    # Load and validate the input volume
    report = {}  # Initialize a dictionary to store report data
    report["input_file"] = input_file
    report["output_file"] = output_file
    report["sigma"] = sigma
    report["normalization_method"] = normalization_method
    try:
        image_handler = ImageHandler(input_file)
        img_data = image_handler.img_data
        file_type = image_handler.file_type
        report["file_type"] = file_type

        # Validate the image data
        if img_data is None:
            logging.error("Image data is empty.")
            report["error"] = "Image data is empty."
            return

        if img_data.ndim != 3:
            logging.error("Image data is not 3-dimensional.")
            report["error"] = "Image data is not 3-dimensional."
            return

        if np.any(np.isnan(img_data)):
            logging.error("Image data contains NaN values.")
            report["error"] = "Image data contains NaN values."
            return
    except Exception as e:
        logging.error(f"Error loading input file: {e}")
        report["error"] = str(e)
        print_report(report)
        return

    # Preprocessing: Normalize and smooth the image
    logging.info("Normalizing image intensities")
    normalized_img = normalize_image(img_data, method=normalization_method)

    logging.info("Applying Gaussian smoothing")
    smoothed_img = gaussian_smoothing(normalized_img, sigma)

    # Atlas-based skull stripping
    try:
        logging.info("Loading MNI atlas and creating atlas data array")
        atlas, mask = load_atlas_mask(ATLAS_PATH, MASK_PATH)

        logging.info("Registering input image to atlas")
        registered_img = register_image(smoothed_img, atlas)

        logging.info("Applying brain mask to extract brain region")
        brain_extracted = apply_brain_mask(registered_img, mask)

        # Calculate quality assessment metrics
        metrics = calculate_quality_metrics(registered_img, brain_extracted)
        logging.info(f"Quality Assessment Metrics: {metrics}")
        report["quality_metrics"] = metrics

        # Generate visualization
        if input_file.endswith(".nii.gz"):
            visualization_output_base = input_file[:-7]
        elif input_file.endswith(".nii") or input_file.endswith(".dcm"):
            visualization_output_base = input_file[:-4]
        elif input_file.endswith(".dcm.gz"):
            visualization_output_base = input_file[:-7]
        visualize_results(registered_img, brain_extracted, visualization_output_base)
        visualization_output_dir = os.path.dirname(visualization_output_base)
        logging.info(f"Visualization saved directory: {visualization_output_dir}")
        report["visualization_output"] = visualization_output_base

        # Save the segmented brain to an output file
        logging.info(f"Saving output file: {output_file}")

        image_handler.save_image(output_file, brain_extracted)

        report["status"] = "success"
        logging.info("Processing complete")
    except Exception as e:
        logging.error(f"Error during skull stripping: {e}")
        report["status"] = "failure"
        report["error"] = str(e)
    finally:
        print_report(report)
        if generate_pdf:
            pdf_file = output_file[:-3] if output_file.endswith(".gz") else output_file
            pdf_output = pdf_file.replace(".nii", "_report.pdf").replace(
                ".dcm", "_report.pdf"
            )
            create_pdf_report(report, pdf_output)
            logging.info(f"PDF report saved to {pdf_output}")


def print_report(report: dict) -> None:
    """
    @brief Prints a summary report of the processing results.

    @param report (dict): A dictionary containing the report data.
    """
    print("-------------------- Processing Report --------------------")
    for key, value in report.items():
        print(f"{key}: {value}")
    print("-----------------------------------------------------------")


if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Atlas-based skull stripping of 3D brain scans."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input NIFTI or DICOM file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA,
        help="Gaussian smoothing parameter",
    )
    parser.add_argument(
        "--normalization_method",
        type=str,
        default=DEFAULT_INTENSITY_NORMALIZATION,
        help="Normalization method ('z-score' or 'min-max')",
    )
    parser.add_argument(
        "--generate_pdf",
        action="store_true",
        help="Generate a PDF report",
    )

    args = parser.parse_args()
    main(
        args.input,
        args.output,
        args.sigma,
        args.normalization_method,
        args.generate_pdf,
    )
