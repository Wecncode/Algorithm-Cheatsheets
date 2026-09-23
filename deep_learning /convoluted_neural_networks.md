# Convolutional Neural Networks (CNN) Cheatsheet

**Type:** Deep Learning  
**Output:** Classification (Probabilities) or Continuous (Bounding box coordinates for Object Detection)

## Intuition
If you feed a standard Artificial Neural Network (ANN) a high-resolution image, you have to flatten the 2D pixel grid into a single, massive 1D line of numbers. When you do this, **you destroy all spatial relationships**. The network forgets that a pixel was next to another pixel.

**Convolutional Neural Networks (CNNs)** solve this by processing the image in its original 2D (or 3D, if color) shape. Instead of looking at the whole image at once, a CNN acts like a magnifying glass, sliding a small window over the image to look for specific patterns—first simple edges, then shapes, then complex objects like a face or a car.

## The Core Layers

**1. Convolutional Layer (Conv2D):**  
The workhorse of the CNN. It slides a mathematical matrix called a **Filter** (or Kernel) across the image. The filter multiplies pixel values and sums them up to detect features like vertical lines, horizontal lines, or curves. This creates a "Feature Map."

**2. Activation Function (ReLU):**  
Immediately after convolution, the output is passed through a ReLU (Rectified Linear Unit) function. It simply replaces all negative pixel values with 0. This introduces non-linearity, allowing the network to learn complex patterns without colors/pixels canceling each other out.

**3. Pooling Layer (Max Pooling):**  
A down-sampling technique. It slides a window across the Feature Map and only keeps the maximum value in that window. 
*   *Why?* It drastically reduces the size of the image (saving memory and computation) and makes the network **Translation Invariant** (it can recognize a cat whether it's in the top-left or bottom-right of the image).

**4. Flattening & Fully Connected (Dense) Layers:**  
Once the image is broken down into high-level features, it is flattened into a 1D array and fed into a standard Neural Network (Dense layers) to make the final prediction (e.g., Dog vs. Cat).

## Key Hyperparameters
*   **Number of Filters:** How many different patterns the layer is looking for (e.g., 32, 64, 128). Deep layers have more filters.
*   **Kernel Size:** The size of the sliding window (usually $3 \times 3$ or $5 \times 5$).
*   **Stride:** How many pixels the filter shifts over at a time. A stride of 1 means it moves one pixel over.
*   **Padding:** 
    *   *Valid:* No padding. The image shrinks as it passes through filters.
    *   *Same:* Adds a border of 0s around the image so the output feature map stays the exact same size as the input.

## Pros and Cons

| 🟢 Pros | 🔴 Cons |
| :--- | :--- |
| **Spatial Awareness:** Preserves the 2D geometry of images perfectly. | **Data Hungry:** Requires massive amounts of labeled image data to train from scratch without overfitting. |
| **Parameter Efficiency:** Because filters are reused across the whole image (weight sharing), CNNs have vastly fewer parameters than standard ANNs. | **Computationally Expensive:** Training requires powerful GPUs, especially for deep architectures (ResNet, VGG). |
| **Translation Invariance:** Can detect objects regardless of where they are positioned in the frame. | **Black Box:** It is notoriously difficult to understand *exactly* why a CNN made a specific prediction (though tools like Grad-CAM help). |

## Evaluation Metrics
*   **Classification:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
*   **Object Detection:** Intersection over Union (IoU), Mean Average Precision (mAP).

## Real-World Projects & Use Cases
1.  **Medical Imaging:** Detecting tumors, pneumonia, or fractures in X-ray and MRI scans with superhuman accuracy.
2.  **Autonomous Vehicles:** Self-driving cars using CNNs to detect pedestrians, stop signs, and lane boundaries in real-time (YOLO algorithms).
3.  **Facial Recognition:** Unlocking your phone using FaceID.
4.  **Agriculture:** Drones scanning crop fields to detect plant diseases or assess harvest readiness.

## Sample Python Implementation (TensorFlow / Keras)

```python
# 1. Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 2. Initialize the CNN
model = Sequential()

# 3. Add Convolutional & Pooling Layers
# Input shape: 64x64 pixel images with 3 color channels (RGB)
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a second Conv layer for deeper feature extraction
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 4. Flattening
model.add(Flatten())

# 5. Fully Connected (Dense) Layers
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5)) # Dropout prevents overfitting by randomly turning off neurons

# 6. Output Layer (Binary Classification: e.g., Cat vs Dog)
model.add(Dense(units=1, activation='sigmoid'))

# 7. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. View the architecture summary
model.summary()

# (Training would use model.fit() with an ImageDataGenerator)
