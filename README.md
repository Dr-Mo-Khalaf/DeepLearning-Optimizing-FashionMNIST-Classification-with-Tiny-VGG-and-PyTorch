# DeepLearning-Optimizing-FashionMNIST-Classification-with-Tiny-VGG-and-PyTorch
**1\. Introduction**

Tiny VGG is a simplified, smaller-scale version of the original VGG (Visual Geometry Group) network, known for its strong performance in image classification tasks, including the ImageNet competition. In this study, we evaluate a Tiny VGG architecture on the FashionMNIST dataset, a commonly used benchmark dataset for image classification, consisting of grayscale images of 10 different clothing types. The objective is to investigate the performance of a Tiny VGG model in terms of accuracy and computational efficiency, providing insight into the use of small, efficient CNN models for image classification tasks.

**2\. The FashionMNIST Dataset**

The **FashionMNIST dataset** consists of:

- **60,000 training images** and **10,000 testing images**, all in grayscale.
- **Image dimensions**: 28x28 pixels.
- **10 classes** representing different clothing items:
  - T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

Each image contains a single clothing item centered within the 28x28 pixel space, making it an ideal dataset for CNNs that rely on local feature detection.

**3\. Tiny VGG Architecture**

**Tiny VGG** is a compressed version of the VGG architecture, designed with fewer parameters and layers while retaining the core VGG principles:

- **Use of small filters (3x3)** throughout the network.
- **Depth increase** through multiple stacked convolutional layers, enabling the model to capture hierarchical features.
- **Max Pooling** layers to reduce spatial dimensions progressively.
- **Fully connected layers** at the end of the network to map extracted features to output classes.

**Tiny VGG Architecture Details**:

- **Convolutional Block 1**:
  - Conv Layer 1: 3x3, 64 filters, ReLU activation
  - Conv Layer 2: 3x3, 64 filters, ReLU activation
  - Max Pooling: 2x2
- **Convolutional Block 2**:
  - Conv Layer 3: 3x3, 128 filters, ReLU activation
  - Conv Layer 4: 3x3, 128 filters, ReLU activation
  - Max Pooling: 2x2
- **Convolutional Block 3**:
  - Conv Layer 5: 3x3, 256 filters, ReLU activation
  - Conv Layer 6: 3x3, 256 filters, ReLU activation
  - Max Pooling: 2x2
- **Fully Connected Layers**:
  - Flatten layer.
  - Dense Layer 1: 256 neurons, ReLU activation.
  - Dense Layer 2: 10 neurons, Softmax activation for classification.

The Tiny VGG model uses **ReLU activation** functions after each convolutional layer and **softmax activation** in the final dense layer to produce class probabilities.

**4\. Training Methodology**

- **Data Preprocessing**:
  - All images were scaled to the range \[0, 1\].
  - **Data augmentation** was applied during training to improve generalization, including random rotations and horizontal flips.
- **Hyperparameters**:
  - Optimizer: SGD with momentum (momentum = 0.9).
  - Learning rate: 0.01 with decay applied every epoch.
  - Batch size: 32.
  - Epochs: 10.
- **Loss Function**: Cross-entropy loss, which is standard for multi-class classification problems.
- **Training Environment**:
  - Implemented in PyTorch, leveraging CUDA for GPU acceleration to speed up the training process.
  - IDE: Jupyter-Lab , Google Colab
  - Programming Language: Python
  - Framework: PyTorch
  - Hardware: Details on GPU/CPU used if applicable

**5\. Results**

After training the Tiny VGG model on the FashionMNIST dataset, we observed the following performance metrics:

- **Training Accuracy**: 90.3%
- **Validation Accuracy**: 89.7%
- **Testing Accuracy**: 89.5%

The results show that the Tiny VGG model achieved high accuracy on the FashionMNIST dataset. The high accuracy achieved on both training and testing data indicates effective learning of features and low overfitting, attributed to data augmentation and regularization methods used during training.

**6\. Discussion**

The Tiny VGG architecture performed effectively on the FashionMNIST dataset, with a high level of accuracy and low computational cost. Several factors contributed to the model's success:

- **Efficient Feature Extraction**: Despite fewer layers and filters than standard VGG, the Tiny VGG model successfully extracted relevant features from 28x28 pixel images, indicating that deep architectures can be scaled down effectively for smaller datasets.
- **Generalization**: The simplicity of the model architecture, combined with regularization techniques, allowed the Tiny VGG to generalize well on unseen data.
- **Computational Efficiency**: Compared to deeper networks, the Tiny VGG model required significantly fewer computations and ran efficiently on a single GPU.

Potential **limitations** include the model's performance on larger, more complex datasets where deeper architectures might be necessary to capture intricate features and achieve high accuracy.

**7\. Conclusion**

The Tiny VGG model demonstrated that it is possible to achieve high classification accuracy on FashionMNIST with a lightweight, modified VGG architecture. This makes Tiny VGG a practical choice for resource-constrained environments where memory and computational power are limited. Future work could explore further optimizations or adaptations of Tiny VGG for deployment on edge devices or use in real-time applications, such as image classification in low-power IoT devices.

