# INDECS
This repository contains the INDECS project, which uses a YOLOv4 model to detect sage-grouse in infrared drone videos.

## Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/biasintelligence/INDECS.git)
cd INDECS
```

### 2. Create a Conda Environment

* Create a conda environment with the required packages:

```bash
conda env create -f environment.yml
conda activate indecs_yolov4
```

### 3. Install Darknet

* **Download Darknet:**
    * Download the Darknet source code from the official repository: https://github.com/AlexeyAB/darknet
    * Helpful FAQ here https://www.ccoderun.ca/programming/darknet_faq/

* **Compile Darknet:**
    * Open a terminal or command prompt and navigate to the `darknet` directory.
    * Check the building options here: https://github.com/AlexeyAB/darknet?tab=readme-ov-file#requirements-for-windows-linux-and-macos
    
* **Add Darknet to PATH:**
  
    * Add the full path to the `darknet` directory to your system's PATH environment variable. This allows Python to find `darknet.dll`.
    
    * **Windows:**
      
        * Add the full path to the `darknet` directory to your Path environment variable
        
    * **Linux/macOS:**
        * Add the following line to your `~/.bashrc` or `~/.zshrc` file, replacing `/path/to/darknet` with the actual path:
        ```bash
        export PATH=$PATH:/path/to/darknet
        ```
        
    * Close and reopen your terminal or command prompt to apply the changes.

### 4. Install cuDNN

* **Download cuDNN:**
    * Download cuDNN from the NVIDIA website: https://developer.nvidia.com/cudnn
    * Ensure that the cuDNN version is compatible with your CUDA version.
* **Extract cuDNN:**
    * Extract the downloaded cuDNN files.
* **Copy DLLs:**
    * Copy the cuDNN DLLs (e.g., `cudnn64_8.dll`) to the CUDA bin directory. For example, to a directory `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` if you are targeting that version. Find the match based on the version of CUDA you have.

### 5. Set Up the `.env` File

* **Create `.env`:**
    * Create a `.env` file in the root directory of the project.
* **Add Configuration:**
    * Add the following lines to the `.env` file, replacing the placeholders with your actual values:

    ```ini
    REPO_ID=your_username/your_model_repo
    WEIGHTS_FILE=your_weights_file.weights
    MODEL_DIR=./model_weights
    DARKNET_LIB_PATH=./darknet
    ```

### 6. Download the Model Weights

* **Run Download Script:**
    * Run the following command to download the model weights from Hugging Face Hub:

    ```bash
    python process_images.py /path/to/your/image_dir /path/to/your/output_dir --download
    ```

## Usage

* **Process Images:**
    * To process images and perform object detection, use the following command:

    ```bash
    python cli/process_images.py /path/to/your/image_dir /path/to/your/output_dir --threshold 0.3 --nms 0.5
    ```

    * Replace `/path/to/your/image_dir` with the directory containing your input images.
    * Replace `/path/to/your/output_dir` with the directory where you want to save the output images.
    * Adjust `--threshold` and `--nms` as needed.

## Notes

* **Sage-Grouse Data:**
    * The `sagegrouse.data` file is located in `src/indecs/data/model`.
    * It contains the path to the `sagegrouse.names` file, which lists the class names.
* **Troubleshooting:**
    * If you encounter issues, refer to the troubleshooting steps in the comments of the Python scripts.



