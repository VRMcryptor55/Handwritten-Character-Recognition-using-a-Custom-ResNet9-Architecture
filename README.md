# MNIST and Bengali  Character Recognition using ResNet-9

This project demonstrates the training of a ResNet-9 model for character recognition on two different datasets: the classic MNIST dataset of handwritten digits and a custom dataset of Telugu characters. The project also explores data augmentation and transfer learning with fine-tuning.

## Features

-   **Dual Dataset Training:** The model is trained and evaluated on both the MNIST and a custom Bengali character dataset.
-   **ResNet-9 Architecture:** A custom implementation of the ResNet-9 neural network is used as the model.
-   **Data Augmentation:** For the Bengali dataset, data augmentation techniques like random resized cropping and random rotation are applied to increase the diversity of the training data.
-   **Checkpointing:** The training progress, including model state and optimizer state, is saved as checkpoints to Google Drive, allowing for resumption of training.
-   **Evaluation:** The model's performance is evaluated using accuracy metrics and a confusion matrix.
-   **Transfer Learning:** An example of transfer learning with fine-tuning is included, where a pre-trained model is adapted for a new task.

## Requirements

-   Python 3
-   PyTorch
-   TorchVision
-   NumPy
-   Matplotlib
-   Scikit-learn
-   OpenCV-Python
-   Pillow (PIL)

You can install the required Python libraries using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn opencv-python Pillow
