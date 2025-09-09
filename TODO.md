# TODO List for Neural Networks from Scratch

## 01_neurons_and_activation_functions
- [x] artificial_neuron.py: Implement single neuron with inputs, weights, bias, output
- [x] activation_functions.py: Sigmoid, Tanh, ReLU, Leaky ReLU, Softmax
- [x] derivatives_of_activations.py: Compute and plot gradients

## 02_loss_functions
- [x] mean_squared_error.py: MSE for regression
- [x] binary_cross_entropy.py: BCE for binary classification
- [x] categorical_cross_entropy.py: CCE for multi-class classification

## 03_gradient_descent
- [x] vanilla_gradient_descent.py: Basic GD on simple function
- [x] stochastic_gradient_descent.py: SGD with mini-batches
- [x] momentum.py: GD with momentum
- [x] rmsprop.py: RMSProp optimizer
- [x] adam.py: Adam optimizer

## 04_building_a_network
- [x] dense_layer.py: Fully-connected layer
- [x] neural_network.py: Class combining multiple layers
- [x] forward_pass.py: Full forward propagation
- [x] backward_pass.py: Backpropagation with chain rule

## 05_putting_it_all_together
- [x] linear_regression_nn.py: Train network for linear regression
- [x] binary_classifier_nn.py: Train for binary classification (XOR)
- [x] multi_class_classifier_nn.py: Train on Iris/MNIST with Softmax

## 06_improving_training
- [x] weight_initialization.py: Zero, Random, Xavier, He init
- [x] batch_normalization.py: BatchNorm layer
- [x] dropout.py: Dropout layer
- [x] l1_l2_regularization.py: L1/L2 penalty

## 07_convolutional_neural_networks
- [x] im2col.py: im2col algorithm
- [x] conv2d_layer.py: Convolutional layer
- [x] max_pooling_layer.py: Max pooling layer
- [x] lenet5_from_scratch.py: LeNet-5 on MNIST

## 08_recurrent_neural_networks
- [x] rnn_cell.py: Core RNN cell
- [x] vanilla_rnn.py: Multi-step RNN layer
- [x] lstm_cell.py: LSTM cell with gates
- [x] gru_cell.py: GRU cell

## 09_modern_architectures
- [x] self_attention.py: Single and multi-head attention
- [x] transformer_block.py: Transformer encoder block
- [x] unet_segmentation.py: U-Net for segmentation
- [x] vae.py: Variational Autoencoder

## 10_applications_and_transfer_learning
- [x] transfer_learning_cv.py: Fine-tune ResNet/VGG
- [x] fine_tuning_bert.py: Fine-tune BERT
- [ ] style_transfer.py: Neural style transfer

## General
- [x] Create directory structure
- [x] Create README.md
- [x] Create requirements.txt
- [x] Install dependencies
- [ ] Test implementations
