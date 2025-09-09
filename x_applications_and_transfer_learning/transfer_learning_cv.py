import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import Conv2DLayer
from neural_networks_from_scratch import DenseLayer
from neural_networks_from_scratch import categorical_cross_entropy, cce_derivative

class PretrainedFeatureExtractor:
    """
    Simulated pre-trained feature extractor (like ResNet or VGG backbone).
    In practice, this would load actual pre-trained weights.
    """

    def __init__(self, input_channels=3, feature_dim=512):
        """
        Initialize pre-trained feature extractor.

        Args:
            input_channels (int): Number of input channels
            feature_dim (int): Output feature dimension
        """
        self.input_channels = input_channels
        self.feature_dim = feature_dim

        # Simulate pre-trained convolutional layers
        self.conv1 = Conv2DLayer(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = Conv2DLayer(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2DLayer(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = Conv2DLayer(256, feature_dim, kernel_size=3, stride=2, padding=1)

        # Mark as pre-trained (freeze weights)
        self.frozen = True

    def forward(self, x):
        """
        Forward pass through feature extractor.

        Args:
            x (np.array): Input images, shape (batch, channels, height, width)

        Returns:
            np.array: Extracted features
        """
        x = np.maximum(0, self.conv1.forward(x))  # ReLU
        x = np.maximum(0, self.conv2.forward(x))
        x = np.maximum(0, self.conv3.forward(x))
        x = np.maximum(0, self.conv4.forward(x))

        # Global average pooling
        x = np.mean(x, axis=(2, 3))  # Shape: (batch, feature_dim)
        return x

    def backward(self, d_output):
        """
        Backward pass (only if not frozen).

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        if self.frozen:
            return np.zeros_like(d_output)  # No gradients for frozen layers

        # Expand gradients for global average pooling
        batch_size, feature_dim = d_output.shape
        d_expanded = d_output[:, :, np.newaxis, np.newaxis] / (32 * 32)  # Assuming 32x32 final feature map

        d_expanded = self.conv4.backward(d_expanded)
        d_expanded = self.conv3.backward(d_expanded)
        d_expanded = self.conv2.backward(d_expanded)
        d_expanded = self.conv1.backward(d_expanded)

        return d_expanded

    def update(self, learning_rate):
        """
        Update parameters (only if not frozen).

        Args:
            learning_rate (float): Learning rate
        """
        if not self.frozen:
            self.conv1.update(learning_rate)
            self.conv2.update(learning_rate)
            self.conv3.update(learning_rate)
            self.conv4.update(learning_rate)

class TransferLearningClassifier:
    """
    Transfer learning classifier that uses pre-trained features.
    """

    def __init__(self, num_classes=10, feature_dim=512, hidden_dim=256):
        """
        Initialize transfer learning classifier.

        Args:
            num_classes (int): Number of output classes
            feature_dim (int): Feature dimension from pre-trained extractor
            hidden_dim (int): Hidden layer dimension
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Pre-trained feature extractor
        self.feature_extractor = PretrainedFeatureExtractor(feature_dim=feature_dim)

        # Task-specific layers (trainable)
        self.classifier = DenseLayer(feature_dim, hidden_dim, activation='relu')
        self.output_layer = DenseLayer(hidden_dim, num_classes, activation='linear')

    def forward(self, x):
        """
        Forward pass through classifier.

        Args:
            x (np.array): Input images, shape (batch, channels, height, width)

        Returns:
            np.array: Class logits, shape (batch, num_classes)
        """
        # Extract features
        features = self.feature_extractor.forward(x)

        # Classify
        hidden = self.classifier.forward(features)
        output = self.output_layer.forward(hidden)

        return output

    def backward(self, d_output):
        """
        Backward pass through classifier.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Backward through classifier
        d_hidden = self.output_layer.backward(d_output)
        d_features = self.classifier.backward(d_hidden)

        # Backward through feature extractor
        d_input = self.feature_extractor.backward(d_features)

        return d_input

    def update(self, learning_rate):
        """
        Update trainable parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.classifier.update(learning_rate)
        self.output_layer.update(learning_rate)
        self.feature_extractor.update(learning_rate)

    def freeze_feature_extractor(self):
        """Freeze the pre-trained feature extractor."""
        self.feature_extractor.frozen = True

    def unfreeze_feature_extractor(self):
        """Unfreeze the pre-trained feature extractor for fine-tuning."""
        self.feature_extractor.frozen = False

def create_synthetic_dataset(num_samples=1000, num_classes=10, image_size=224):
    """
    Create synthetic dataset for demonstration.

    Args:
        num_samples (int): Number of samples
        num_classes (int): Number of classes
        image_size (int): Image size (square)

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
    # Generate random images
    X = np.random.randn(num_samples, 3, image_size, image_size).astype(np.float32)

    # Generate random labels
    y = np.random.randint(0, num_classes, num_samples)
    y_onehot = np.zeros((num_samples, num_classes))
    y_onehot[np.arange(num_samples), y] = 1

    # Split into train/test
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_onehot[:split_idx], y_onehot[split_idx:]

    return X_train, y_train, X_test, y_test

def train_transfer_learning_model():
    """
    Train transfer learning model on synthetic dataset.
    """
    print("Creating synthetic dataset...")
    X_train, y_train, X_test, y_test = create_synthetic_dataset(
        num_samples=1000, num_classes=10, image_size=224
    )

    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

    # Create model
    model = TransferLearningClassifier(num_classes=10, feature_dim=512, hidden_dim=256)

    # Phase 1: Train only classifier (feature extractor frozen)
    print("\nPhase 1: Training classifier with frozen feature extractor...")
    model.freeze_feature_extractor()

    learning_rate = 0.01
    epochs_phase1 = 10

    for epoch in range(epochs_phase1):
        # Forward pass
        y_pred = model.forward(X_train)

        # Compute loss
        loss = categorical_cross_entropy(y_train, y_pred)

        # Backward pass
        d_loss = cce_derivative(y_train, y_pred)
        model.backward(d_loss)

        # Update
        model.update(learning_rate)

        if epoch % 2 == 0:
            print(f"Phase 1 - Epoch {epoch}, Loss: {loss:.4f}")

    # Phase 2: Fine-tune entire model
    print("\nPhase 2: Fine-tuning entire model...")
    model.unfreeze_feature_extractor()

    learning_rate_finetune = 0.001  # Lower learning rate for fine-tuning
    epochs_phase2 = 10

    for epoch in range(epochs_phase2):
        # Forward pass
        y_pred = model.forward(X_train)

        # Compute loss
        loss = categorical_cross_entropy(y_train, y_pred)

        # Backward pass
        d_loss = cce_derivative(y_train, y_pred)
        model.backward(d_loss)

        # Update
        model.update(learning_rate_finetune)

        if epoch % 2 == 0:
            print(f"Phase 2 - Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = model.forward(X_test)
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_test_pred_labels == y_test_labels)
    print(f"Test accuracy: {accuracy:.4f}")

    return model

if __name__ == "__main__":
    print("Transfer Learning for Computer Vision")
    print("=" * 40)

    # Train model
    model = train_transfer_learning_model()

    print("\nTransfer learning demonstration completed!")
    print("In practice, you would:")
    print("1. Load actual pre-trained weights (e.g., from torchvision)")
    print("2. Use real datasets (e.g., ImageNet, CIFAR)")
    print("3. Implement data augmentation")
    print("4. Use proper optimizers and learning rate schedules")
