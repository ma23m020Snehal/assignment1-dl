# Overview 
This repository contains code for training and evaluating a neural network model for image classification using various optimization techniques and hyperparameters tuning. 

# Key Features:

Implements a Multilayer Perceptron (MLP) for classification.

Supports multiple optimization techniques including SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam.

Logs training and evaluation metrics using wandb.

Configurable via command-line arguments.


# Table of Contents

1. Dataset & Preprocessing

2. Model Architecture

3. Optimization Algorithms

4. Training Process

5. Hyperparameter Tuning

6. Weights & Biases (wandb) Integration

7. How to Run

8. Example Command-line Usage

9. Results & Evaluation

10. Code organisation


# Dataset & Preprocessing

The model is trained on Fashion-MNIST consisting of:

60,000 training images and 10,000 test images.

28×28 grayscale images belonging to 10 categories.



# Preprocessing Steps:

Each 28×28 image is converted into a 784-dimensional vector , Pixel values are scaled to [0,1] and  labels are converted to one-hot vectors of length 10.

 
# Model Architecture

The neural network consists of:

Input Layer: 784 neurons (one for each pixel).

Hidden Layers: Configurable number of layers and nodes.

Activation Functions:

Hidden layers: ReLU, Sigmoid, or Tanh.

Output Layer: 10 neurons (one for each class).

Weight Initialization

Random Initialization: Weights are initialized randomly with small values.

Xavier (Glorot) Initialization: Used for better gradient propagation.


# Optimization Algorithms

The model supports multiple optimization algorithms, which update weights during training:

Stochastic Gradient Descent (SGD)

Momentum-based Gradient Descent

Nesterov Accelerated Gradient Descent (NAG)

RMSProp

Adam

Nadam (Adam with Nesterov momentum)

Each optimizer is implemented in Python using NumPy, and can be selected using command-line arguments.


# Training Process

The training process includes:

Forward Propagation: Computes activations at each layer.

Loss Computation:Uses Cross-Entropy Loss or Mean Squared Error (MSE).

Backward Propagation: Computes gradients and updates weights using the selected optimizer.

Evaluation: Computes validation accuracy and logs metrics.


# Hyperparameter Tuning 

Learning rate: values= [1e-3, 1e-4] ; default: 0.001

Number of hidden layers: values = [3, 4, 5] ; default: 3

Number of nodes per layer: values = [32, 64, 128] ; default: 128

Activation function: values = ['sigmoid', 'tanh', 'relu'] ; default :  'relu'

Optimizer: values= ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']; default ='nadam'

Batch size: values=[16, 32, 64];  default: 64

Number of epochs: values : [5, 10] ; default 10

Weight initialization: values : [Random, Xavier] ; default ='xavier'

Weight_decay : values = [0, 0.0005, 0.5] ; default = 0

Users can modify hyperparameters using command-line arguments


# Weights & Biases (wandb) Integration

The script logs all training metrics to Weights & Biases for experiment tracking.

## How to Enable wandb Logging:

Create an account at wandb.ai.

Login using:

```bash
wandb login
```

Run the script with 
```bash
--wandb_entity and --wandb_project set.
```

## Logged Metrics:

Training Loss

Validation Loss

Validation Accuracy

Test Accuracy

# How to Run

## Install Dependencies

Ensure you have Python 3.6+ and install required libraries:

```bash
pip install numpy wandb tensorflow scikit-learn
```

## Clone the Repository
```bash
git clone https://github.com/ma23m020Snehal/assignment1-dl.git
cd assignment1-dl
```

## Run the Training Script
```bash
python strain.py --wandb_entity snehalma23m020-iit-madras --wandb_project DLassigment1
```

# Example Command-line Usage

Run the Training with Default Parameters

python strain.py --wandb_entity my_wandb_account --wandb_project DL_project

# Custom Training Configuration

```bash
python train.py --wandb_entity snehalma23m020-iit-madras --wandb_project DLassigment1 --dataset mnist --epochs 10 --batch_size 64 --loss cross_entropy --optimizer nadam --learning_rate 0.001 --num_layers 3 --hidden_size 128 --activation relu
```

# View Available Arguments
```bash
python train.py --help
```
# Results & Evaluation
The detailled results and their visualisations via wandb is available in the report. The results are also available in ipynb file .

# Code Organization

- **train.py**  
  This is the main training script. It:
   Loads the chosen dataset (MNIST or Fashion‑MNIST), pre-processes the data, and splits it into training, validation, and test sets defines feedforward neural network (MLP) and various optimizer classes.Trains the model while logging metrics to wandb.

- **Assignment1.ipynb**  
  This contains the proper step by step solutions to all the 10 questions .
  
- **README.md**  
  This file provides a high level overview about the project , It also contains instructions on how to run the code and a link to the wandb report.


- **wandb/**  
  Directory containing logs and run artifacts generated by wandb (automatically created).

# View wandb Dashboard

Check the wandb report for visualizations:


## Conclusion

This repository demonstrates how to configure and train a feedforward neural network using a fully configurable Python script with wandb integration. The script logs all relevant training metrics to wandb, making it easy to track experiments, compare hyperparameter settings, and visualize results in real-time.
