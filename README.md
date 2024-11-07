# YOLOv11-FaceAnonymization
## Face Blurring with YOLOv11 and MTCNN
## Introduction
The Face Blurring with **YOLOv11** and **MTCNN model** is a real-time solution for face detection and blurring in video streams. This system integrates the YOLOv11 object detection model for identifying human subjects and the MTCNN (Multi-task Cascaded Convolutional Networks) for precise face detection, enabling autonomous systems to automatically anonymize faces in videos. This model is ideal for applications requiring privacy protection, such as surveillance, content moderation, and public video use.

By leveraging both YOLOv11 for fast person detection and MTCNN for accurate face localization, this model offers high efficiency and performance, allowing seamless integration into real-world scenarios.

## Table of Contents
- [Model_Architecture](#model-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Inference](#model-inference)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Model Architecture
The model consists of two key components for face blurring:

**YOLOv11 for Object Detection:** YOLOv11 is used to detect the "person" class in the video frames. This allows the system to identify regions of interest (i.e., human subjects) for further processing.

**MTCNN for Face Detection:** Once the "person" region is identified by YOLOv11, MTCNN is used to detect faces within the bounding box, ensuring that faces are accurately localized before applying the blur effect.

**Blurring Mechanism:** The face regions detected by MTCNN are blurred using a kernel-based blurring technique (e.g., cv2.blur), ensuring privacy by anonymizing individuals in video streams.

### Model Use
**YOLOv11:** The YOLOv11 model used here has been pre-trained on various datasets, including COCO and VOC.

**MTCNN:** This model performs face detection based on key facial landmarks and ensures that detected faces are blurred effectively.

**Post-processing:** After detecting faces and applying blur, the original image is updated with the modified regions for continuous video processing.

## Features
**Real-time Face Detection & Blurring:** Achieves real-time processing of video streams to anonymize faces while maintaining the integrity of the rest of the video.
YOLOv11 for Efficient Person Detection: Detects human subjects with high speed and accuracy in complex environments.

**MTCNN for Precise Face Localization:** Provides accurate face bounding boxes for precise blurring, even in diverse facial poses and lighting conditions.

**High Compatibility:** Designed to work on both CPU and GPU setups, with support for CUDA-enabled devices for enhanced performance.

**Versatile Deployment:** Ready for use in surveillance, privacy protection, and content moderation applications.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/krishnapriya-nynaru/YOLOv11-FaceAnonymization
2. Install required packages :
    ```bash
    pip install -r requirements.txt
3. Change to Project Directory
    ```bash
    cd YOLOv11_faceanonymization

## Usage
Run the script with Python
```bash
python main.py
```

## Model Inference
The model processes input videos as follows:

**Preprocessing:** 
- Input video frames are captured and processed in real-time.
- Each frame is passed through YOLOv11 to detect human subjects (persons).

**Face Detection:**
- Once a person is detected, MTCNN detects faces within the bounding box of each person.

**Blurring:**
- Detected faces are blurred using a kernel (defined by blur_ratio), ensuring privacy.

**Post-processing:**
- The blurred faces are merged back into the original video frame.

**Display and Save:**
- The processed video is displayed in a window and saved as an output file.

## Results

## Contributing
Contributions are welcome! To contribute to this project:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and ensure the code passes all tests.
4. Submit a pull request with a detailed description of your changes.

If you have any suggestions for improvements or features, feel free to open an issue!

## Acknowledgments
- [**YOLOv11 for object detection.**](https://github.com/ultralytics/yolov11)
- [**MTCNN for face detection.**](https://github.com/haroonshakeel/mtcnn)
- [**OpenCV for computer vision functionalities.**](https://opencv.org/)