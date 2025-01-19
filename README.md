# SpectraNet: FFT-assisted Deep Learning Classifier for Deepfake Detection

## Introduction

Detecting deepfake images is crucial in mitigating misinformation and addressing security challenges posed by synthetic media. **SpectraNet** presents a robust and lightweight deep learning solution for binary classification of real and deepfake images. This project combines the EfficientNet-B6 architecture with advanced preprocessing and optimization techniques to achieve high accuracy and generalization.

## Features
- **EfficientNet-B6 Architecture**: A lightweight and scalable convolutional neural network fine-tuned for binary classification.
- **Class Imbalance Mitigation**: Oversampling and augmentation techniques to address dataset imbalances.
- **Fourier Transform Integration**: Leveraging phase and amplitude analysis for enhanced feature representation.
- **Robust Preprocessing**: Normalization, resizing, and data augmentation for improved training stability.
- **Advanced Optimization**: Employing Adam optimizer, ReduceLROnPlateau scheduler, and mixed precision training.

## Dataset
This project utilizes state-of-the-art datasets curated for deepfake detection, including:
- **Celeb-DF v1 and v2**: Large-scale datasets containing realistic deepfake videos.
- **Deepfake Detection Challenge (DFDC)**: Comprehensive dataset released by Facebook AI.
- **FaceForensics++**: A popular benchmark dataset for detecting facial manipulations.

The dataset consists of **262,160 images**, with:
- Real images: **42,690**
- Fake images: **219,470**

To address class imbalance, we applied oversampling and augmentation techniques, ensuring a balanced representation of real and fake samples during training.

## Methodology

### Model
- Adapted **EfficientNet-B6**:
  - Fine-tuned pretrained weights from ImageNet.
  - Added dropout layers for regularization.
  - Modified the final dense layer for binary classification.

### Training
- **Loss Function**: Binary Cross-Entropy with Logits Loss.
- **Optimizer**: Adam with L2 regularization.
- **Scheduler**: ReduceLROnPlateau for dynamic learning rate adjustment.
- **Mixed Precision Training**: Accelerated computation using PyTorch's AMP toolkit.

## Results

| Metric             | EfficientNet-B6 | Hybrid Model |
|--------------------|----------------|--------------|
| **AUC**            | 0.9104         | 0.8984       |
| **Accuracy**       | 0.9102         | 0.8981       |
| **F1 Score**       | 0.9074         | 0.8946       |
| **Precision**      | 0.9435         | 0.9346       |
| **Recall**         | 0.8740         | 0.8579       |
| **Evaluation Time**| 2.55 sec       | 3.48 sec     |

EfficientNet-B6 demonstrated superior performance, achieving **91.02% accuracy** with reduced evaluation time, making it suitable for real-time applications.

## Key Findings
- Fourier Transform, while theoretically beneficial, showed limited practical impact on performance.
- EfficientNet-B6 effectively balances computational efficiency with classification accuracy.
- The framework provides a scalable and accessible solution for deepfake detection.

## Acknowledgments
This project was conducted as part of an academic effort by the Department of Electronic and Telecommunications Engineering, University of Moratuwa, and the School of Electrical and Data Engineering, University of Technology Sydney.
