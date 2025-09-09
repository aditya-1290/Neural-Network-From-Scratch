
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

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Why from Scratch?

Implementing from scratch provides:
- Deep understanding of the mathematics
- Ability to customize and experiment
- Foundation for advanced research
- Appreciation for the engineering in high-level libraries

## Testing Status

The following implementations have been tested and verified to work correctly:

### ✅ Tested and Working

**01_neurons_and_activation_functions**
- `artificial_neuron.py` - Basic neuron implementation with forward pass
- `activation_functions.py` - Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
- `derivatives_of_activations.py` - Gradient computations for activation functions

**02_loss_functions**
- `binary_cross_entropy.py` - BCE loss with gradient computation
- `categorical_cross_entropy.py` - CCE loss for multi-class classification
- `mean_squared_error.py` - MSE loss with gradient computation

**03_gradient_descent**
- `adam.py` - Adam optimizer implementation
- `momentum.py` - Gradient descent with momentum
- `rmsprop.py` - RMSProp optimizer
- `stochastic_gradient_descent.py` - SGD with mini-batches
- `vanilla_gradient_descent.py` - Basic gradient descent

### 🔄 Remaining to Test

**04_building_a_network**
- `dense_layer.py`, `neural_network.py`, `forward_pass.py`, `backward_pass.py`

**05_putting_it_all_together**
- `linear_regression_nn.py`, `binary_classifier_nn.py`, `multi_class_classifier_nn.py`

**06_improving_training**
- `weight_initialization.py`, `batch_normalization.py`, `dropout.py`, `l1_l2_regularization.py`

- `im2col.py`, `conv2d_layer.py`, `max_pooling_layer.py`, `lenet5_from_scratch.py`

**08_recurrent_neural_networks**
- `rnn_cell.py`, `vanilla_rnn.py`, `lstm_cell.py`, `gru_cell.py`

**09_modern_architectures**
- `self_attention.py`, `transformer_block.py`, `unet_segmentation.py`, `vae.py`

**10_applications_and_transfer_learning**
- `transfer_learning_cv.py`, `fine_tuning_bert.py`, `style_transfer.py`

### 📊 Visualization Outputs

All visualization outputs are saved in the `visualizations/` directory:
- Gradient descent optimization paths
- Loss curves and convergence plots
- Activation function plots
- Network architecture diagrams

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```
**07_convolutional_neural_networks**
