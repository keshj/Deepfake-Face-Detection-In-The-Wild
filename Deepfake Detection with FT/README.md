# PyTorch Project: Evaluating Fourier Transform Features for Detection of AI-Generated Images

## Overview
This project implements a neural network model using PyTorch to evaluate the effectiveness of Fourier Transform features in detecting AI-generated images. The model architecture combines convolutional neural networks (CNNs) with Fourier Transform information to enhance image classification tasks.

## Project Structure
```
pytorch-project
├── src
│   ├── main.py          # Entry point for the application
│   ├── models.py        # Neural network model definition
│   ├── train.py         # Training loop and checkpoint management
│   ├── test.py          # Model testing and evaluation
│   ├── utils.py         # Utility functions for preprocessing and transformations
│   ├── datasets.py      # Dataset classes and data loaders
│   └── tensorboard_utils.py # Logging metrics and visualizations to TensorBoard
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore            # Files and directories to ignore by Git
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd pytorch-project
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
- To train the model, run:
  ```
  python src/main.py --mode train
  ```

- To test the model, run:
  ```
  python src/main.py --mode test
  ```

## Model Architecture
The model architecture consists of:
- Convolutional layers for feature extraction.
- A fully connected layer for classification.
- Integration of Fourier Transform features to enhance performance.

## Experiments
The project includes experiments to compare the performance of the model with and without Fourier Transform features. Results will be logged for analysis.

## License
This project is licensed under the MIT License - see the LICENSE file for details.