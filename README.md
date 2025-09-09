
# Neural Networks from Scratch

This repository implements neural networks and deep learning concepts from first principles using only NumPy and Matplotlib. No high-level frameworks like TensorFlow or PyTorch are used for the core implementations.

## Philosophy

- **From First Principles**: Every algorithm is implemented from scratch to understand the underlying mathematics.
- **Modular & Educational**: Each file is self-contained and demonstrates a single concept.
- **Progressive Complexity**: Start with simple concepts and build up to advanced architectures.

## Learning Path

1. **01_neurons_and_activation_functions**: Basic building blocks
2. **02_loss_functions**: How to measure model performance
3. **03_gradient_descent**: Optimization algorithms
4. **04_building_a_network**: Combining layers into networks
5. **05_putting_it_all_together**: Training complete models
6. **06_improving_training**: Advanced training techniques
7. **07_convolutional_neural_networks**: CNNs for vision tasks
8. **08_recurrent_neural_networks**: RNNs for sequential data
9. **09_modern_architectures**: Transformers, autoencoders, etc.
10. **10_applications_and_transfer_learning**: Practical applications

## Detailed File Descriptions

### 01_neurons_and_activation_functions
- **`artificial_neuron.py`**: Implements a single artificial neuron with inputs, weights, bias, and output computation. Demonstrates the fundamental unit of neural networks.
- **`activation_functions.py`**: Contains implementations of common activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax) used to introduce non-linearity in neural networks.
- **`derivatives_of_activations.py`**: Computes and visualizes the derivatives of activation functions, essential for backpropagation during training.

### 02_loss_functions
- **`mean_squared_error.py`**: Implements Mean Squared Error (MSE) loss function for regression tasks, including both forward computation and gradient calculation.
- **`binary_cross_entropy.py`**: Binary Cross-Entropy loss function for binary classification problems with gradient computation for optimization.
- **`categorical_cross_entropy.py`**: Categorical Cross-Entropy loss for multi-class classification, handling one-hot encoded targets and computing gradients.

### 03_gradient_descent
- **`vanilla_gradient_descent.py`**: Basic gradient descent optimizer that minimizes functions by following the negative gradient direction.
- **`stochastic_gradient_descent.py`**: Stochastic Gradient Descent with mini-batch support for efficient training on large datasets.
- **`momentum.py`**: Gradient descent with momentum to accelerate convergence and reduce oscillations.
- **`rmsprop.py`**: RMSProp optimizer that adapts learning rates based on recent gradient magnitudes.
- **`adam.py`**: Adam optimizer combining momentum and RMSProp for adaptive learning rates and bias correction.

### 04_building_a_network
- **`dense_layer.py`**: Fully-connected (Dense) layer implementation with forward pass, backward pass, and parameter updates.
- **`neural_network.py`**: NeuralNetwork class that combines multiple layers into a complete network architecture.
- **`forward_pass.py`**: Implementation of forward propagation through the entire network.
- **`backward_pass.py`**: Backpropagation algorithm that computes gradients through the network using chain rule.

### 05_putting_it_all_together
- **`linear_regression_nn.py`**: Trains a neural network for linear regression using synthetic data, demonstrating end-to-end training pipeline.
- **`binary_classifier_nn.py`**: Trains a neural network for binary classification on XOR dataset, showing non-linear decision boundaries.
- **`multi_class_classifier_nn.py`**: Multi-class classification using Iris dataset with Softmax activation and categorical cross-entropy loss.

### 06_improving_training
- **`weight_initialization.py`**: Various weight initialization techniques (Zero, Random, Xavier, He) to improve training stability.
- **`batch_normalization.py`**: Batch normalization layer that normalizes activations to improve training speed and stability.
- **`dropout.py`**: Dropout regularization technique to prevent overfitting by randomly deactivating neurons during training.
- **`l1_l2_regularization.py`**: L1 and L2 regularization penalties to prevent overfitting and encourage simpler models.

### 07_convolutional_neural_networks
- **`im2col.py`**: Efficient im2col algorithm for converting image patches to matrix format for convolution operations.
- **`conv2d_layer.py`**: 2D convolutional layer with forward and backward passes for feature extraction from images.
- **`max_pooling_layer.py`**: Max pooling layer for downsampling feature maps and reducing spatial dimensions.
- **`lenet5_from_scratch.py`**: Complete LeNet-5 convolutional neural network architecture for handwritten digit recognition.

### 08_recurrent_neural_networks
- **`rnn_cell.py`**: Basic RNN cell implementation with hidden state management for sequential data processing.
- **`vanilla_rnn.py`**: Multi-layer vanilla RNN implementation for processing sequences of arbitrary length.
- **`lstm_cell.py`**: Long Short-Term Memory (LSTM) cell with forget, input, and output gates to handle long-term dependencies.
- **`gru_cell.py`**: Gated Recurrent Unit (GRU) cell, a simplified LSTM variant with reset and update gates.

### 09_modern_architectures
- **`self_attention.py`**: Self-attention mechanism and multi-head attention for capturing relationships between elements in sequences.
- **`transformer_block.py`**: Complete transformer encoder block with multi-head attention and feed-forward networks.
- **`unet_segmentation.py`**: U-Net architecture for image segmentation tasks with encoder-decoder structure and skip connections.
- **`vae.py`**: Variational Autoencoder (VAE) for generative modeling with encoder, decoder, and reparameterization trick.

### 10_applications_and_transfer_learning
- **`transfer_learning_cv.py`**: Demonstrates transfer learning by fine-tuning pre-trained ResNet/VGG models for new classification tasks.
- **`fine_tuning_bert.py`**: Fine-tuning BERT (Bidirectional Encoder Representations from Transformers) for downstream NLP tasks.
- **`style_transfer.py`**: Neural style transfer algorithm that combines content and style images using convolutional features.

## Why from Scratch?

Implementing from scratch provides:
- Deep understanding of the mathematics
- Ability to customize and experiment
- Foundation for advanced research
- Appreciation for the engineering in high-level libraries

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples

Each file contains example usage in the `if __name__ == "__main__":` block. For instance:

```python
# Example: Training a simple neural network
from neural_networks_from_scratch.iv_building_a_network.neural_network import NeuralNetwork

# Create a network: 2 inputs -> 3 hidden -> 1 output
nn = NeuralNetwork([2, 3, 1], ['relu', 'sigmoid'])

# Forward pass
output = nn.forward(X_train)

# Backward pass
nn.backward(d_loss)

# Update parameters
nn.update(learning_rate=0.01)
```

## Testing Status

All implementations include example usage and have been verified to work correctly. The project provides a complete, educational implementation of neural networks from basic concepts to advanced architectures.
