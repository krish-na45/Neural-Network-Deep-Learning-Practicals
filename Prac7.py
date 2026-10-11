# Install necessary libraries
!pip install opencv-python tensorflow_hub matplotlib

import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained SSD MobileNet V2 model
print("Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")
print("Model loaded successfully!")

# Function to load image from URL
def load_image_from_url(image_url):
    image_path = tf.keras.utils.get_file(origin=image_url)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.array(image_rgb)

# Sample image URL
image_url = "https://feeds.abplive.com/onecms/images/uploaded-images/2022/06/08/cba50dc1f15b89efcce65f6a9ec7350b_original.JPG"
image_np = load_image_from_url(image_url)

# Display original image
plt.figure(figsize=(10, 8))
plt.imshow(image_np)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Prepare input tensor
input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

# Perform inference
result = detector(input_tensor)
result = {key: value.numpy() for key, value in result.items()}

# Extract detection results
boxes = result['detection_boxes'][0]
class_ids = result['detection_classes'][0].astype(int)
scores = result['detection_scores'][0]
num_detections = int(result['num_detections'][0])

# Example category index (extend as needed)
category_index = {
    15: {'id': 15, 'name': 'cat'},
    16: {'id': 16, 'name': 'dog'},
    17: {'id': 17, 'name': 'horse'},
    18: {'id': 18, 'name': 'sheep'},
    19: {'id': 19, 'name': 'cow'},
    20: {'id': 20, 'name': 'elephant'},
    21: {'id': 21, 'name': 'bear'}
}

# Function to draw boxes and labels
def draw_boxes_and_labels(image, boxes, class_ids, scores, num_detections, category_index, threshold=0.5):
    img_copy = image.copy()
    h, w, _ = img_copy.shape

    for i in range(num_detections):
        if scores[i] >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            left, right, top, bottom = int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h)

            cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2)
            class_name = category_index.get(class_ids[i], {'name': 'N/A'})['name']
            label = f"{class_name}: {scores[i]:.2f}"
            cv2.putText(img_copy, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img_copy

# Draw detection results
image_with_detections = draw_boxes_and_labels(image_np, boxes, class_ids, scores, num_detections, category_index)

# Display detections
plt.figure(figsize=(10, 8))
plt.imshow(image_with_detections)
plt.axis('off')
plt.title('Object Detections')
plt.show()
