# Project Methodology

## Object Detection and Counting Methodology

INDECS leverages the YOLO (You Only Look Once) [^1]: object detection algorithm for its core bird identification capability. YOLO is a state-of-the-art real-time object detection system that offers a good balance between speed and accuracy. Unlike traditional object detection methods that involve region proposal and classification steps, YOLO performs detection as a single regression problem, directly predicting bounding boxes and class probabilities from the input image.

## YOLO Algorithm

YOLO divides the input image into a grid. Each grid cell is responsible for predicting a fixed number of bounding boxes and class probabilities for objects that fall within that cell. The predictions are encoded as a tensor, where each bounding box is represented by its center coordinates, width, height, and confidence score. The confidence score reflects the model's certainty that the box contains an object and how accurate it believes the predicted box is.

## Training Process

To train the YOLO model for sage grouse detection in infrared imagery, we curated a dataset of about 5600 images captured from lek environments in Montana. These images were manually annotated with bounding boxes around each individual sage grouse. The annotation process involved carefully examining each image and precisely marking the location of each bird, ensuring accurate ground truth data for training.

We utilized the Darknet[^2][^3] framework, an open-source neural network framework written in C and CUDA, for implementing and training the YOLO model. The training process involved feeding the annotated images to the network and optimizing its parameters to minimize the difference between predicted bounding boxes and the ground truth annotations. This optimization was achieved through a loss function that penalizes both localization errors (incorrect bounding box predictions) and classification errors (incorrect object identification).

## Inference and Bird Counting

Once the YOLO model was trained, we used it to perform inference on new, unseen infrared video footage. The process can broken down into the following steps:

1. **Frame Extraction**: The video is processed frame-by-frame extracting individual frames for analysis.

2. **Object Detection**: Each frame is fed to the trained YOLO model, which predicts the bounding boxes of potential sage grouse within the image.

* **Coordinate Transformation**: The bounding box coordinates predicted by YOLO are normalized to the original image space.

* **Bird Count per Frame**: The number of detected birds is determined for each frame. This gives us a time series of bird counts throughout the video.

* **Peak Detection**: To avoid overcounting due to multiple detections of the same bird in consecutive frames or minor fluctuations in bird positions, a peak detection algorithm is applied. This algorithm identifies local maxima in the bird count time series, representing moments of actual bird gatherings.

* **Windowing for Accuracy**: A configurable window size is used in the peak detection process. This window helps to ensure that only distinct peaks, representing actual gatherings, are counted, and prevents spurious peaks from being included in the final count. The window size is determined based on factors like the video's duration and frame rate.

* **Data Filtering**: The peak detection results are then processed to remove any duplicate peaks that might occur due to very similar bird counts in consecutive frames. This ensures that each peak represents a distinct gathering of birds.

By combining YOLO object detection with this peak detection and filtering process, INDECS achieves a more robust and accurate count of sage grouse in infrared video footage. This approach addresses the challenges of individual bird identification and movement within the lek environment, providing valuable data for population monitoring and conservation efforts.

------

[^1]: "You Only Look Once," *Wikipedia*, Wikimedia Foundation, October 2024,  https://en.wikipedia.org/wiki/You_Only_Look_Once.
[^2]: Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. Retrieved from [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)

[^3]: YOLOv4 GitHub Repository: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
