# INDECS: Infra-Red Detection Classification System

INDECS is a specialized tool developed to count sage grouse in lek environments using infrared drone footage and machine learning. The system leverages YOLOv4 object detection to identify and count birds in thermal video frames.

## About the Project

INDECS (Infra-Red Detection Classification System) was developed to help wildlife researchers monitor sage grouse populations using drone-captured infrared videos. The system processes video frames through a trained YOLOv4 model to detect and count birds, providing valuable population data with minimal disturbance to wildlife.

## Pre-trained Model

A pre-trained model is freely available on Hugging Face:  
**Repository**: [lyulyok1/yolov4-grouse](https://huggingface.co/lyulyok1/yolov4-grouse)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/biasintelligence/INDECS.git
cd INDECS
```

### 2. Create a Conda Environment
```bash
conda env create -f environment.yml
conda activate indecs_yolov4
```

### 3. Install Darknet

Darknet is the framework used to run the YOLOv4 model.

1. **Download Darknet:**
   - Clone from the official repository: `git clone https://github.com/AlexeyAB/darknet`
   - Visit the [Darknet FAQ](https://www.ccoderun.ca/programming/darknet_faq/) for helpful information

2. **Compile Darknet:**
   - Navigate to the darknet directory: `cd darknet`
   - Check the [build requirements](https://github.com/AlexeyAB/darknet?tab=readme-ov-file#requirements-for-windows-linux-and-macos)
   - For GPU support (recommended): Edit the `Makefile` to set `GPU=1` and `CUDNN=1`
   - Run `make` to compile

3. **Add Darknet to PATH:**
   - **Windows**: Add the darknet directory to your Path environment variable
   - **Linux/macOS**: Add to your `~/.bashrc` or `~/.zshrc`:
     ```bash
     export PATH=$PATH:/full/path/to/darknet
     ```

### 4. Install CUDA and cuDNN (for GPU acceleration)

1. **Install CUDA Toolkit:**
   - Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation instructions for your operating system

2. **Install cuDNN:**
   - Download cuDNN from [NVIDIA's website](https://developer.nvidia.com/cudnn) (requires account)
   - Ensure the version is compatible with your CUDA installation
   - Copy the files to your CUDA installation directory

### 5. Set Up the `.env` File

Create a `.env` file in the project root with the following configuration:

```ini
# Hugging Face configuration for model download
REPO_ID=lyulyok1/yolov4-grouse
WEIGHTS_FILE=sagegrouse.weights
MODEL_DIR=./model_weights
DARKNET_LIB_PATH=./darknet
```

### 6. Download the Model Weights

The model weights will be automatically downloaded from Hugging Face when you first run the detection script. You can also download them explicitly with:

```bash
python -m src.indecs.process_images /path/to/image_dir /path/to/output_dir --download
```

## Usage

### Process Images with Object Detection

To detect sage grouse in images:

```bash
python -m src.indecs.process_images /path/to/image_dir /path/to/output_dir --threshold 0.3 --nms 0.45
```

Parameters:
- `/path/to/image_dir`: Directory containing input images
- `/path/to/output_dir`: Directory where processed images will be saved
- `--threshold`: Detection confidence threshold (default: 0.25)
- `--nms`: Non-maximum suppression threshold (default: 0.45)
- `--download`: Add this flag to download the model before processing

## Project Structure

```
indecs/
├── src/
│   └── indecs/
│       ├── config.py       # Configuration and model management
│       ├── detector.py     # YOLOv4 detector implementation
│       ├── process_images.py # Image processing script
│       └── data/
│           └── model/
│               ├── yolov4-grouse.cfg  # Model configuration
│               ├── sagegrouse.data    # Data file for YOLO
│               └── sagegrouse.names   # Class names
├── model_weights/          # Downloaded model weights
└── .env                    # Environment configuration
```

## How It Works

1. **Configuration Management**: The `config.py` file handles loading environment variables and manages model paths. It includes functionality to automatically download model weights from Hugging Face when needed.

2. **Detection System**: The `detector.py` file implements the YOLOv4-based object detection system. It initializes the model and provides methods for detecting sage grouse in images.

3. **Image Processing**: The `process_images.py` script processes images from a specified directory, applies the detector, and saves the results with bounding boxes around detected birds.

## Troubleshooting

1. **CUDA/cuDNN Issues**:
   - Ensure compatible versions of CUDA and cuDNN
   - Check if your GPU is CUDA-compatible
   - Verify proper installation of drivers

2. **Darknet Problems**:
   - Ensure Darknet is compiled with the correct options
   - Verify paths in the `.env` file are correct
   - Make sure the Darknet library can be found in your PATH

3. **Model Download Issues**:
   - Check your internet connection
   - Verify you have specified the correct repository ID in the `.env` file
   - Ensure you have adequate disk space for the model weights

4. **Python Import Errors**:
   - Make sure you're running commands from the project root directory
   - Verify the conda environment is activated

## License

This project is released under the GNU General Public License v3.0 (GPL-3.0).

## Acknowledgments

- The YOLOv4 model is based on [Darknet](https://github.com/AlexeyAB/darknet)
- Pre-trained model hosted on [Hugging Face](https://huggingface.co/lyulyok1/yolov4-grouse)