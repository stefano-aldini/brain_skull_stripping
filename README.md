# Brain Skull Stripping Pipeline

This project implements a preprocessing pipeline for atlas-based skull stripping of 3D brain scans. The pipeline is designed to assist in the analysis of neurological MRI scans, providing a streamlined approach for extracting brain regions from volumetric data.

## Overview

The pipeline consists of several key steps:

1.  **Load and Validate Input Volume**: The tool accepts 3D brain scans in DICOM or NIFTI format and ensures that the input data is valid for processing.

2.  **Basic Preprocessing**:
    *   **Normalize Image Intensities**: The image intensities can be normalized using either Z-score normalization or min-max scaling.
    *   **Gaussian Smoothing**: A Gaussian filter is applied to the image to reduce noise and improve the quality of the skull stripping process.

3.  **Atlas-Based Skull Stripping**:
    *   **Load MNI Brain Atlas**: The MNI brain atlas template and corresponding mask are loaded.
    *   **Register Input Image**: The input image is registered to the MNI atlas template to align the anatomical structures.
    *   **Apply Brain Mask**: The transformed brain mask is applied to extract the brain region from the input image.

4.  **Save Segmented Brain**:
    *   **Save Resulting Brain Data**: The resulting segmented brain volume is saved to an output file in the format specified by the output name.
    *   **Save Images**: The images generated during the pipeline are saved in PNG format, for easy access.
    *   **Generate PDF with Result Summary**: A PDF report is created. It contains the metrics and images generated while processing the brain image.

## Installation/Setup Instructions

1.  **Extract the folder from the zip file provided and navigate to the directory:**

    ```bash
    cd brain_skull_stripping
    ```


2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage Examples

To run the command-line tool, use the following syntax:

```bash
python -m src.process_mri --input <input_file> --output <output_file> --sigma <smoothing_parameter> --normalization_method <normalization_method> --generate_pdf
```

For example:

```bash
python -m src.process_mri --input sub-0002_ses-01_T1w.nii.gz --output sub-0002_ses-0001_T1w_defaced_out.nii.gz --sigma 1.5 --normalization_method z-score --generate_pdf
```

## Test The Script

To test the script, the following subject has been used:
https://s3.amazonaws.com/openneuro.org/ds000247/sub-0002/ses-01/anat/sub-0002_ses-01_T1w.nii.gz?versionId=71.XAnuxtjw6ITyFLSPZeH_lAayTeyvq

With Atlas and mask from:
http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip



## Assumptions Made

The pipeline makes the following assumptions:

1. **Valid Input Data:** The input file exists, is a valid NIFTI or DICOM file, and contains a 3D brain scan without NaN values.
2. **Atlas and Mask Availability:** The MNI brain atlas (ATLAS_PATH) and corresponding mask (MASK_PATH) exist and can be loaded.
3. **Registration Accuracy:** The input image can be accurately registered to the MNI atlas using affine registration.
4. **Normalization Method:** The specified normalization method (normalization_method) is either 'z-score' or 'min-max'.
5. **Image Intensities:** The image intensities are suitable for the chosen normalization method and Gaussian smoothing.
6. **Output File Handling:** The code has write permissions to the specified output file location.
7. **Visualization Dependencies:** If generate_pdf is True, the necessary visualization files can be created and saved.

## Potential Improvements

1. Add --save_image as an argument to the Python module.
2. Check that the shape of the atlas is reasonable.
3. Check that the input data is reasonable.
4. Improve tests. For example test for DICOM, by using the dicomgenerator library.
5. Make code more robust.
6. PdfGenerator could be in a separate file/class.
7. Depending on the user, a simple GUI could be created to avoid using the command line.
8. Better logging. Include warnings. For example, if the image becomes too blurred after smoothing, a warning pointing to that phase of the pipeline could be useful.
9. Resulting images could be cropped/resized to be consistent in size.
10. The generated PDF could be improved in formatting and presented data.
11. Pass the atlas and mask data folder as an argument from the command line
