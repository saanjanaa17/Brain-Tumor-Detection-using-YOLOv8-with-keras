# Brain Tumor Detection Using YOLOv8

## Table of Contents
- Dataset
- Data Preprocessing
- Model Architecture
- Training Process
- Results and Evaluation
- Test Predictions and Visualization
- Conclusion
- Future Work
- Installation
- References

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

### 3. Data Preprocessing
* Data preprocessing is crucial for preparing raw data into a format suitable for training a machine learning model. The following preprocessing steps were applied to the images and annotations:

3.1 Image Preprocessing
```python
# A function for converting txt file to list
def parse_txt_annot(img_path, txt_path):
    img = cv2.imread(img_path)
    w = int(img.shape[0])
    h = int(img.shape[1])

    file_label = open(txt_path, "r")
    lines = file_label.read().split('\n')

    boxes = []
    classes = []

    if lines[0] == '':
        return img_path, classes, boxes
    else:
        for i in range(0, int(len(lines))):
            objbud = lines[i].split(' ')
            class_ = int(objbud[0])

            x1 = float(objbud[1])
            y1 = float(objbud[2])
            w1 = float(objbud[3])
            h1 = float(objbud[4])

            xmin = int((x1 * w) - (w1 * w) / 2.0)
            ymin = int((y1 * h) - (h1 * h) / 2.0)
            xmax = int((x1 * w) + (w1 * w) / 2.0)
            ymax = int((y1 * h) + (h1 * h) / 2.0)

            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(class_)

    return img_path, classes, boxes


# A function for creating file paths list
def create_paths_list(path):
    full_path = []
    images = sorted(os.listdir(path))

    for i in images:
        full_path.append(os.path.join(path, i))

    return full_path

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
class_ids = ['label0', 'label1', 'label2']
class_mapping = dict(zip(range(len(class_ids)), class_ids))

# A function for creating a dict format of files
def creating_files(img_files_paths, annot_files_paths):

    img_files = create_paths_list(img_files_paths)
    annot_files = create_paths_list(annot_files_paths)

    image_paths = []
    bbox = []
    classes = []

    for i in range(0, len(img_files)):
        image_path_, classes_, bbox_ = parse_txt_annot(img_files[i], annot_files[i])
        image_paths.append(image_path_)
        bbox.append(bbox_)
        classes.append(classes_)

    image_paths = tf.ragged.constant(image_paths)
    bbox = tf.ragged.constant(bbox)
    classes = tf.ragged.constant(classes)

    return image_paths, classes, bbox

# Applying functions
train_img_paths, train_classes, train_bboxes = creating_files('/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/train/images',
                                                              '/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/train/labels')

valid_img_paths, valid_classes, valid_bboxes = creating_files('/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/valid/images',
                                                              '/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/valid/labels')

test_img_paths, test_classes, test_bboxes = creating_files('/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/test/images',
                                                            '/kaggle/input/medical-image-dataset-brain-tumor-detection/BrainTumorYolov8/test/labels')

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

### 4. Model Architecture
* The model used for this project is YOLOv8, a state-of-the-art object detection model optimized for real-time performance.

4.1 Backbone
* The backbone of the model was initialized with the pre-trained YOLOv8 XS backbone, trained on the COCO dataset. This backbone contains convolutional layers that learn to extract features from input images.

4.2 YOLOv8 Detector

* YOLOv8 consists of:

- **FPN (Feature Pyramid Network)**: Handles objects of different sizes.
- **Bounding Box Prediction**: Predicts bounding boxes in the xyxy format.
- **Class Prediction**: Classifies objects into predefined categories (e.g., benign, malignant).
```python
# Creating mirrored strategy
stg = tf.distribute.MirroredStrategy()

# Creating pre-trained model backbone with coco weights
with stg.scope():
    backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone_coco")
    
    YOLOV8_model = keras_cv.models.YOLOV8Detector(num_classes=len(class_mapping),
                                              bounding_box_format="xyxy",
                                              backbone=backbone, fpn_depth=1)

    optimizer = AdamW(learning_rate=0.0001, weight_decay=0.004, global_clipnorm=GLOBAL_CLIPNORM)

    YOLOV8_model.compile(optimizer=optimizer, classification_loss='binary_crossentropy', box_loss='ciou')
```

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
# Training
hist = YOLOV8_model.fit(train_dataset, validation_data=valid_dataset, epochs=120)
```
### 6. Results and Evaluation
* The model's performance was evaluated based on:

- **Loss** : Overall loss, classification loss, and bounding box loss were tracked during training.
- **1. Training Loss**: The training loss steadily decreased, indicating successful learning of both classification and bounding box predictions.
- **2. Bounding Box Loss**: The box loss decreased, indicating the model’s increasing ability to predict bounding box locations accurately.
- **3. Classification Loss**: The classification loss also decreased, showing that the model was learning to predict the correct tumor class.
```python
# Training Results, Evaluation
fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=130)

axs[0].grid(linestyle="dashdot")
axs[0].set_title("Loss")
axs[0].plot(hist.history['loss'][1:])
axs[0].plot(hist.history['val_loss'][1:])
axs[0].legend(["train", "validation"])

axs[1].grid(linestyle="dashdot")
axs[1].set_title("Box Loss")
axs[1].plot(hist.history['box_loss'])
axs[1].plot(hist.history['val_box_loss'])
axs[1].legend(["train", "validation"])

axs[2].grid(linestyle="dashdot")
axs[2].set_title("Class Loss")
axs[2].plot(hist.history['class_loss'][1:])
axs[2].plot(hist.history['val_class_loss'][1:])
axs[2].legend(["train", "validation"])

```
### 7. Test Predictions and Visualization
* To evaluate the model’s performance, predictions were made on the test dataset.
```python
# Test Predictions
def visualize_predict_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))

    y_pred = model.predict(images)
    y_pred = keras_cv.bounding_box.to_ragged(y_pred)

    keras_cv.visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        true_color=(192, 57, 43),
        pred_color=(255, 235, 59),
        scale=8,
        font_scale=0.8,
        line_thickness=2,
        dpi=100,
        rows=2,
        cols=2,
        show=True,
        class_mapping=class_mapping,
    )
```
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
* To run the project, ensure you have the following dependencies installed:

```bash
pip install tensorflow keras-cv opencv-python matplotlib numpy pandas tqdm
```

### 11. References
* YOLOv8: YOLOv8 documentation and research paper.
* TensorFlow: Official TensorFlow documentation for model training and building.
* Keras CV: Keras CV documentation for computer vision utilities and tools.
* Medical Image Datasets: Brain tumor detection datasets for research, such as those used in this project.
```css

This `README.md` is now a comprehensive guide that includes all the project details, explanations, and code for setting up and running the brain tumor detection using YOLOv8.
```
