# Veriserum (Ongoing)

**Veriserum**: A dual-plane fluoroscopic dataset with implant phantoms for deep learning in medical imaging.

## Overview

Veriserum is an open-source dataset specifically designed for advancing deep learning in medical imaging, particularly in fluoroscopic image analysis. This dataset provides dual-plane fluoroscopic images with implant phantoms, enabling researchers to explore applications in biomechanics, computer vision, and pose estimation. Veriserum is freely available to support research communities in developing and testing algorithms in areas like image registration, calibration, segmentation, and 3D reconstruction.

### Key Features
- **Dual-Plane Fluoroscopy**: Paired images from two perspectives facilitate 3D reconstruction and pose estimation.
- **Implant Phantoms**: Synthetic implant structures allow for detailed testing of algorithms.
- **Pre-calibrated Images**: The images are standardized in size and intensity, ready for direct application with common deep learning models.
- **Open Source**: Free for academic and research use, promoting transparency and collaboration.

## Directory Structure

The project directory is organized as follows:

```plaintext
project_directory/
├── model.py                  # Defines the PoseEstimationModel
├── train.py                  # Contains data loading and training logic
├── veriserum_dataset.py      # Defines the custom Veriserum_calibrated dataset
├── sampler.py                # Defines the custom NonObsoleteSampler
├── csv_files/
│   ├── veriserum_pose_reloaded_refined_all.csv
│   ├── distortion_calibration_reloaded.csv
│   └── source_int_calibration_reloaded.csv
```

### Directory and File Descriptions

- **model.py**: Contains `PoseEstimationModel`, a model that utilizes a pre-trained ResNeXt backbone to perform pose estimation on fluoroscopic images.
- **train.py**: Includes data loading, augmentation, and model training using PyTorch Lightning. Configures training/validation splits, logging, and model checkpointing.
- **veriserum_dataset.py**: Defines `Veriserum_calibrated`, a custom PyTorch dataset class that loads dual-plane fluoroscopic images, calibrates them if needed, and retrieves corresponding pose data.
- **sampler.py**: Implements `NonObsoleteSampler`, which filters data based on specific conditions (e.g., removing obsolete entries) to ensure high-quality samples.
- **csv_files/**: Contains calibration and pose data files:
  - `veriserum_pose_reloaded_refined_all.csv`: Pose data for each image.
  - `distortion_calibration_reloaded.csv`: Distortion calibration data for correcting images.
  - `source_int_calibration_reloaded.csv`: Calibration data for source intensifiers.

## Setup and Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/yourusername/veriserum.git
    cd veriserum
    ```

2. **Install dependencies**: This project requires Python 3.8+ and PyTorch. Install required packages with:

    ```bash
    pip install -r requirements.txt
    ```

3. **Data Preparation**: Ensure that the dataset and calibration files are correctly placed in the `csv_files/` directory. Modify `veriserum_dataset.py` paths if using a custom directory structure.

## Usage

### Training the Model

To train the model on Veriserum:

1. Configure training parameters (e.g., batch size, learning rate) in `train.py`.
2. Run the training script:

    ```bash
    python train.py
    ```

This will train the `PoseEstimationModel` and save the best model checkpoint based on validation loss.

### Customizing Data Loading and Transformations

The `Veriserum_calibrated` class in `veriserum_dataset.py` can be customized to apply specific transformations or adjust the calibration on-the-fly.
`train.py` also provides a template for augmenting the dataset with additional transformations, using the PyTorch `torchvision.transforms` module.

## Contributing

Contributions are welcome! If you find any issues or have ideas for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgements

Special thanks to collaborators and institutions supporting open data initiatives in medical imaging and biomechanics.
