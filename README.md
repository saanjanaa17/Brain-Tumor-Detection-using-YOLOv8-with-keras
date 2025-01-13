# Brain Tumor Detection Using YOLOv8

## 1. Introduction

Brain tumor detection is a critical task in the field of medical imaging, as it plays a significant role in early diagnosis and treatment. Traditionally, radiologists manually analyze medical images such as MRI scans to detect and diagnose tumors, which can be time-consuming and error-prone. The advancement of machine learning, particularly deep learning models like YOLO (You Only Look Once), offers the potential to automate this process with high accuracy and efficiency.

In this project, we used YOLOv8, a state-of-the-art object detection model, to detect and classify brain tumors in medical images. The model was trained on a dataset of brain tumor images annotated with bounding box information and class labels, where the goal is to predict tumor locations and identify their classes (e.g., benign or malignant).

This report presents a detailed explanation of the dataset, data preprocessing, model architecture, training process, and evaluation.

## 2. Dataset Description

The dataset used in this project is derived from a collection of brain tumor images and their corresponding annotations. The images were collected as part of a medical image dataset for brain tumor detection.

### 2.1 Image Format and Annotations

- **Image Dimensions**: The images in the dataset are high-resolution medical images containing brain scans. Each image is represented in JPEG format with 3 color channels (RGB).
- **Annotation Format**: Each image has an associated annotation file in `.txt` format that specifies the bounding box coordinates for tumors within the image. The annotation is in the following format:

```arduino
class_id x_center y_center width height
```
Where:

* class_id: Integer representing the tumor class (e.g., benign, malignant).
* x_center, y_center: Normalized coordinates of the center of the bounding box (relative to the image width and height).
* width, height: Normalized width and height of the bounding box (relative to the image width and height).

2.2 Data Splitting
* The dataset is split into three subsets:

- **Training Set** : Used to train the model.
- **Validation Set** : Used to validate the model’s performance during training.
- **Test Set** : Used to evaluate the final trained model.
* The dataset includes tumor images from different patients, ensuring the model generalizes well to unseen data.

3. Data Preprocessing
* Data preprocessing is crucial for preparing raw data into a format suitable for training a machine learning model. The following preprocessing steps were applied to the images and annotations:

3.1 Image Preprocessing
```python
# code for image resizing and normalization
import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)  # Load the image
    image = cv2.resize(image, (640, 640))  # Resize to 640x640
    image = image / 255.0  # Normalize the image
    return image
```
3.2 Bounding Box Conversion
* The bounding box annotations were provided in a normalized format (x_center, y_center, width, height).
* These values were transformed into pixel-based coordinates (xmin, ymin, xmax, ymax) to ensure the model can accurately predict bounding box locations.
```
python

def convert_bbox(x_center, y_center, width, height, image_width, image_height):
    xmin = (x_center - width / 2) * image_width
    ymin = (y_center - height / 2) * image_height
    xmax = (x_center + width / 2) * image_width
    ymax = (y_center + height / 2) * image_height
    return xmin, ymin, xmax, ymax
```

3.3 Dataset Preparation
```python
import tensorflow as tf

def create_tf_dataset(image_paths, annotations):
    # Create a TensorFlow dataset with image paths and annotation files
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotations))
    return dataset
```

3.4 Data Augmentation
* Data augmentation techniques such as random scaling, cropping, and flipping were applied to images during training to artificially increase the size of the dataset and reduce overfitting.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
```

4. Model Architecture
* The model used for this project is YOLOv8, a state-of-the-art object detection model optimized for real-time performance.

4.1 Backbone
* The backbone of the model was initialized with the pre-trained YOLOv8 XS backbone, trained on the COCO dataset. This backbone contains convolutional layers that learn to extract features from input images.

4.2 YOLOv8 Detector

* YOLOv8 consists of:

- **FPN (Feature Pyramid Network)**: Handles objects of different sizes.
- **Bounding Box Prediction**: Predicts bounding boxes in the xyxy format.
- **Class Prediction**: Classifies objects into predefined categories (e.g., benign, malignant).
  
4.3 Loss Function: 
The model uses a combination of:

- **Binary Cross-Entropy Loss**: Used for classification tasks to calculate the difference between predicted and true class labels.
- **CIoU (Complete Intersection over Union) Loss**: Used for bounding box regression to measure the overlap between predicted and ground-truth bounding boxes.
```python
import tensorflow as tf
#  loss function in TensorFlow
def custom_loss(y_true, y_pred):
    classification_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    bbox_loss = ciou_loss(y_true, y_pred)  # Define CIoU loss function
    return classification_loss + bbox_loss
```

4.4 Optimizer
* The AdamW optimizer with a learning rate of 0.0001 and weight decay of 0.004 was used for model optimization.

```python
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.004)
```

### 5. Training Process
* The model was trained for 120 epochs, using the following configuration:

- **Batch Size**: 4
- **Learning Rate**: 0.0001
- **Epochs**: 120
```python
history = model.fit(train_dataset, epochs=120, validation_data=val_dataset)
```
### 6. Results and Evaluation
* The model's performance was evaluated based on:

- **Loss** : Overall loss, classification loss, and bounding box loss were tracked during training.
- **1. Training Loss**: The training loss steadily decreased, indicating successful learning of both classification and bounding box predictions.
- **2. Bounding Box Loss**: The box loss decreased, indicating the model’s increasing ability to predict bounding box locations accurately.
- **3. Classification Loss**: The classification loss also decreased, showing that the model was learning to predict the correct tumor class.

### 7. Test Predictions and Visualization
* To evaluate the model’s performance, predictions were made on the test dataset.

7.1 Visualization Results
```python
import matplotlib.pyplot as plt

# visualization code
def visualize_predictions(image, predicted_boxes, true_boxes):
    plt.imshow(image)
    for box in predicted_boxes:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='yellow'))
    for box in true_boxes:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red'))
    plt.show()
```

### 8. Conclusion
* In this project, a YOLOv8-based model was successfully trained and evaluated for the task of brain tumor detection. 
* The model demonstrated good performance in both classification and localization tasks.
* Despite good performance, further improvements can be made with techniques like:

- **Ensemble Methods**: Combining predictions from multiple models.
- **Fine-tuning**: Fine-tuning the model on a larger dataset.

### 9. Future Work
Future improvements could include:

* Expanding the dataset to include a wider variety of brain tumor types.
* Experimenting with transfer learning from other medical imaging datasets.
* Testing the model in a real-time clinical environment.
* Exploring advanced techniques like attention mechanisms.

### 10. Installation
To run the project, ensure you have the following dependencies installed:

```bash
pip install tensorflow keras-cv opencv-python matplotlib numpy pandas tqdm
```

### 11. References
YOLOv8: YOLOv8 documentation and research paper.
TensorFlow: Official TensorFlow documentation for model training and building.
Keras CV: Keras CV documentation for computer vision utilities and tools.
Medical Image Datasets: Brain tumor detection datasets for research, such as those used in this project.
```css

This `README.md` is now a comprehensive guide that includes all the project details, explanations, and code for setting up and running the brain tumor detection using YOLOv8.
```
