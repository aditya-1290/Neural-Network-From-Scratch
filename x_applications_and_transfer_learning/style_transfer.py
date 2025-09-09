import numpy as np

import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import Conv2DLayer
from neural_networks_from_scratch import DenseLayer

class VGGFeatureExtractor:
    """
    Simplified VGG-like feature extractor for style transfer.
    """

    def __init__(self):
        """
        Initialize VGG feature extractor.
        """
        # Convolutional layers (simplified VGG-like architecture)
        self.conv1_1 = Conv2DLayer(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = Conv2DLayer(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = Conv2DLayer(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Conv2DLayer(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = Conv2DLayer(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Conv2DLayer(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = Conv2DLayer(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = Conv2DLayer(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = Conv2DLayer(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = Conv2DLayer(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = Conv2DLayer(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = Conv2DLayer(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = Conv2DLayer(512, 512, kernel_size=3, padding=1)

    def forward(self, x, layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']):
        """
        Forward pass through VGG, extracting features from specified layers.

        Args:
            x (np.array): Input image, shape (batch, channels, height, width)
            layers (list): List of layer names to extract features from

        Returns:
            dict: Dictionary of feature maps for each requested layer
        """
        features = {}

        # Block 1
        x = np.maximum(0, self.conv1_1.forward(x))  # ReLU
        if 'conv1_1' in layers:
            features['conv1_1'] = x
        x = np.maximum(0, self.conv1_2.forward(x))
        if 'conv1_2' in layers:
            features['conv1_2'] = x
        x = self._max_pool(x)

        # Block 2
        x = np.maximum(0, self.conv2_1.forward(x))
        if 'conv2_1' in layers:
            features['conv2_1'] = x
        x = np.maximum(0, self.conv2_2.forward(x))
        if 'conv2_2' in layers:
            features['conv2_2'] = x
        x = self._max_pool(x)

        # Block 3
        x = np.maximum(0, self.conv3_1.forward(x))
        if 'conv3_1' in layers:
            features['conv3_1'] = x
        x = np.maximum(0, self.conv3_2.forward(x))
        if 'conv3_2' in layers:
            features['conv3_2'] = x
        x = np.maximum(0, self.conv3_3.forward(x))
        if 'conv3_3' in layers:
            features['conv3_3'] = x
        x = self._max_pool(x)

        # Block 4
        x = np.maximum(0, self.conv4_1.forward(x))
        if 'conv4_1' in layers:
            features['conv4_1'] = x
        x = np.maximum(0, self.conv4_2.forward(x))
        if 'conv4_2' in layers:
            features['conv4_2'] = x
        x = np.maximum(0, self.conv4_3.forward(x))
        if 'conv4_3' in layers:
            features['conv4_3'] = x
        x = self._max_pool(x)

        # Block 5
        x = np.maximum(0, self.conv5_1.forward(x))
        if 'conv5_1' in layers:
            features['conv5_1'] = x
        x = np.maximum(0, self.conv5_2.forward(x))
        if 'conv5_2' in layers:
            features['conv5_2'] = x
        x = np.maximum(0, self.conv5_3.forward(x))
        if 'conv5_3' in layers:
            features['conv5_3'] = x

        return features

    def _max_pool(self, x, kernel_size=2, stride=2):
        """
        Simple max pooling.

        Args:
            x (np.array): Input, shape (batch, channels, height, width)
            kernel_size (int): Pooling kernel size
            stride (int): Pooling stride

        Returns:
            np.array: Pooled output
        """
        batch, channels, height, width = x.shape
        out_height = height // stride
        out_width = width // stride

        output = np.zeros((batch, channels, out_height, out_width))

        for b in range(batch):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride
                        h_end = h_start + kernel_size
                        w_start = j * stride
                        w_end = w_start + kernel_size
                        output[b, c, i, j] = np.max(x[b, c, h_start:h_end, w_start:w_end])

        return output

class NeuralStyleTransfer:
    """
    Neural Style Transfer implementation.
    """

    def __init__(self, content_layers=['conv4_2'], style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']):
        """
        Initialize Neural Style Transfer.

        Args:
            content_layers (list): Layers to extract content features from
            style_layers (list): Layers to extract style features from
        """
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.feature_extractor = VGGFeatureExtractor()

        # Style and content weights
        self.content_weight = 1.0
        self.style_weight = 1000.0

    def gram_matrix(self, feature_map):
        """
        Compute Gram matrix for style representation.

        Args:
            feature_map (np.array): Feature map, shape (batch, channels, height, width)

        Returns:
            np.array: Gram matrix
        """
        batch, channels, height, width = feature_map.shape
        features = feature_map.reshape(batch, channels, -1)  # (batch, channels, height*width)

        # Compute Gram matrix: (channels, channels)
        gram = np.zeros((batch, channels, channels))
        for b in range(batch):
            gram[b] = np.dot(features[b], features[b].T) / (channels * height * width)

        return gram

    def compute_content_loss(self, generated_features, content_features):
        """
        Compute content loss.

        Args:
            generated_features (dict): Features from generated image
            content_features (dict): Features from content image

        Returns:
            float: Content loss
        """
        loss = 0.0
        for layer in self.content_layers:
            gen_feat = generated_features[layer]
            cont_feat = content_features[layer]
            loss += np.sum((gen_feat - cont_feat) ** 2)
        return loss / len(self.content_layers)

    def compute_style_loss(self, generated_features, style_features):
        """
        Compute style loss.

        Args:
            generated_features (dict): Features from generated image
            style_features (dict): Features from style image

        Returns:
            float: Style loss
        """
        loss = 0.0
        for layer in self.style_layers:
            gen_gram = self.gram_matrix(generated_features[layer])
            style_gram = self.gram_matrix(style_features[layer])
            loss += np.sum((gen_gram - style_gram) ** 2)
        return loss / len(self.style_layers)

    def total_variation_loss(self, generated_image):
        """
        Compute total variation loss for smoothness.

        Args:
            generated_image (np.array): Generated image

        Returns:
            float: Total variation loss
        """
        # Horizontal variation
        h_var = np.sum((generated_image[:, :, :, 1:] - generated_image[:, :, :, :-1]) ** 2)

        # Vertical variation
        v_var = np.sum((generated_image[:, :, 1:, :] - generated_image[:, :, :-1, :]) ** 2)

        return (h_var + v_var) / (generated_image.size)

    def transfer_style(self, content_image, style_image, num_iterations=100, learning_rate=0.01):
        """
        Perform neural style transfer.

        Args:
            content_image (np.array): Content image, shape (1, 3, height, width)
            style_image (np.array): Style image, shape (1, 3, height, width)
            num_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimization

        Returns:
            np.array: Generated image
        """
        print("Starting neural style transfer...")

        # Initialize generated image as content image
        generated_image = content_image.copy()

        # Extract features from content and style images
        print("Extracting content features...")
        content_features = self.feature_extractor.forward(content_image, self.content_layers)

        print("Extracting style features...")
        style_features = self.feature_extractor.forward(style_image, self.style_layers)

        # Optimization loop
        for i in range(num_iterations):
            # Extract features from generated image
            generated_features = self.feature_extractor.forward(generated_image,
                                                              self.content_layers + self.style_layers)

            # Compute losses
            content_loss = self.compute_content_loss(generated_features, content_features)
            style_loss = self.compute_style_loss(generated_features, style_features)
            tv_loss = self.total_variation_loss(generated_image)

            # Total loss
            total_loss = (self.content_weight * content_loss +
                         self.style_weight * style_loss +
                         0.001 * tv_loss)  # Small weight for TV loss

            # Compute gradients (simplified - in practice would use automatic differentiation)
            # For demonstration, we'll use finite differences
            grad = self.compute_gradient(generated_image, content_features, style_features)

            # Update generated image
            generated_image -= learning_rate * grad

            # Clip to valid range
            generated_image = np.clip(generated_image, 0, 1)

            if i % 10 == 0:
                print(f"Iteration {i}: Total Loss = {total_loss:.4f}, "
                      f"Content Loss = {content_loss:.4f}, Style Loss = {style_loss:.4f}")

        return generated_image

    def compute_gradient(self, generated_image, content_features, style_features, eps=1e-8):
        """
        Compute gradient of total loss w.r.t. generated image (simplified).

        Args:
            generated_image (np.array): Generated image
            content_features (dict): Content features
            style_features (dict): Style features
            eps (float): Small epsilon for numerical gradient

        Returns:
            np.array: Gradient
        """
        # Simplified gradient computation using finite differences
        grad = np.zeros_like(generated_image)

        # Sample a few pixels for gradient computation (simplified)
        for c in range(min(3, generated_image.shape[1])):
            for h in range(min(10, generated_image.shape[2])):
                for w in range(min(10, generated_image.shape[3])):
                    # Forward difference
                    generated_image[0, c, h, w] += eps
                    gen_features = self.feature_extractor.forward(generated_image,
                                                                self.content_layers + self.style_layers)

                    content_loss_pos = self.compute_content_loss(gen_features, content_features)
                    style_loss_pos = self.compute_style_loss(gen_features, style_features)

                    # Backward difference
                    generated_image[0, c, h, w] -= 2 * eps
                    gen_features = self.feature_extractor.forward(generated_image,
                                                                self.content_layers + self.style_layers)

                    content_loss_neg = self.compute_content_loss(gen_features, content_features)
                    style_loss_neg = self.compute_style_loss(gen_features, style_features)

                    # Reset
                    generated_image[0, c, h, w] += eps

                    # Compute gradient
                    grad[0, c, h, w] = ((self.content_weight * (content_loss_pos - content_loss_neg) +
                                       self.style_weight * (style_loss_pos - style_loss_neg)) / (2 * eps))

        return grad

def create_sample_images():
    """
    Create sample content and style images for demonstration.

    Returns:
        tuple: (content_image, style_image)
    """
    # Create simple synthetic images
    content_image = np.random.rand(1, 3, 128, 128).astype(np.float32)
    style_image = np.random.rand(1, 3, 128, 128).astype(np.float32)

    # Add some structure to make it more interesting
    # Content image: add a square in the center
    content_image[0, :, 48:80, 48:80] = 0.8

    # Style image: add diagonal patterns
    for i in range(128):
        for j in range(128):
            if (i + j) % 20 < 10:
                style_image[0, :, i, j] = 0.9
            else:
                style_image[0, :, i, j] = 0.1

    return content_image, style_image

def run_style_transfer_demo():
    """
    Run neural style transfer demonstration.
    """
    print("Neural Style Transfer Demo")
    print("=" * 30)

    # Create sample images
    print("Creating sample images...")
    content_image, style_image = create_sample_images()

    print(f"Content image shape: {content_image.shape}")
    print(f"Style image shape: {style_image.shape}")

    # Create style transfer model
    style_transfer = NeuralStyleTransfer()

    # Perform style transfer
    generated_image = style_transfer.transfer_style(
        content_image, style_image,
        num_iterations=50,  # Reduced for demo
        learning_rate=0.1
    )

    print(f"\nGenerated image shape: {generated_image.shape}")
    print("Style transfer completed!")

    return generated_image

if __name__ == "__main__":
    # Run demonstration
    result = run_style_transfer_demo()

    print("\nNeural style transfer demonstration completed!")
    print("In practice, you would:")
    print("1. Load actual pre-trained VGG weights")
    print("2. Use real images (content and style)")
    print("3. Implement proper automatic differentiation")
    print("4. Use L-BFGS optimizer for better convergence")
    print("5. Add more sophisticated loss functions")
