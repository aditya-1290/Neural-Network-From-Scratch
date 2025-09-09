import numpy as np
from self_attention import MultiHeadAttention

class LayerNorm:
    """
    Layer normalization.
    """

    def __init__(self, d_model, eps=1e-6):
        """
        Initialize layer norm.

        Args:
            d_model (int): Model dimension
            eps (float): Small value for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones((d_model,))
        self.beta = np.zeros((d_model,))

        # Gradients
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (np.array): Input, shape (batch_size, seq_len, d_model)

        Returns:
            np.array: Normalized output
        """
        self.input = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std = np.sqrt(var + self.eps)
        self.x_hat = (x - mean) / self.std

        output = self.gamma * self.x_hat + self.beta
        return output

    def backward(self, d_output):
        """
        Backward pass.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        batch_size, seq_len, d_model = d_output.shape

        # Gradients w.r.t. gamma and beta
        self.d_gamma = np.sum(d_output * self.x_hat, axis=(0, 1))
        self.d_beta = np.sum(d_output, axis=(0, 1))

        # Gradient w.r.t. normalized input
        d_x_hat = d_output * self.gamma

        # Gradient w.r.t. input
        d_var = np.sum(d_x_hat * (self.input - np.mean(self.input, axis=-1, keepdims=True)) * (-0.5) * (self.std ** -3), axis=-1, keepdims=True)
        d_mean = np.sum(d_x_hat * (-1 / self.std), axis=-1, keepdims=True) + d_var * np.mean(-2 * (self.input - np.mean(self.input, axis=-1, keepdims=True)), axis=-1, keepdims=True)

        d_input = d_x_hat / self.std + d_var * 2 * (self.input - np.mean(self.input, axis=-1, keepdims=True)) / d_model + d_mean / d_model

        return d_input

    def update(self, learning_rate):
        """
        Update parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.gamma -= learning_rate * self.d_gamma
        self.beta -= learning_rate * self.d_beta

class FeedForward:
    """
    Feed-forward network.
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize feed-forward network.

        Args:
            d_model (int): Input/output dimension
            d_ff (int): Hidden dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff

        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.b1 = np.zeros((d_ff,))
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b2 = np.zeros((d_model,))

        # Gradients
        self.d_W1 = np.zeros_like(self.W1)
        self.d_b1 = np.zeros_like(self.b1)
        self.d_W2 = np.zeros_like(self.W2)
        self.d_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (np.array): Input, shape (batch_size, seq_len, d_model)

        Returns:
            np.array: Output
        """
        self.input = x
        batch_size, seq_len, d_model = x.shape

        # Flatten for matrix multiplication
        x_flat = x.reshape(-1, d_model)

        # First linear layer + ReLU
        self.hidden = np.maximum(0, np.dot(x_flat, self.W1) + self.b1)  # ReLU

        # Second linear layer
        output_flat = np.dot(self.hidden, self.W2) + self.b2

        # Reshape back
        output = output_flat.reshape(batch_size, seq_len, d_model)

        return output

    def backward(self, d_output):
        """
        Backward pass.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        batch_size, seq_len, d_model = d_output.shape

        # Flatten
        d_output_flat = d_output.reshape(-1, d_model)

        # Gradient w.r.t. W2 and b2
        self.d_W2 = np.dot(self.hidden.T, d_output_flat)
        self.d_b2 = np.sum(d_output_flat, axis=0)

        # Gradient w.r.t. hidden
        d_hidden = np.dot(d_output_flat, self.W2.T)

        # Gradient w.r.t. ReLU
        d_hidden[self.hidden <= 0] = 0

        # Gradient w.r.t. W1 and b1
        x_flat = self.input.reshape(-1, d_model)
        self.d_W1 = np.dot(x_flat.T, d_hidden)
        self.d_b1 = np.sum(d_hidden, axis=0)

        # Gradient w.r.t. input
        d_input_flat = np.dot(d_hidden, self.W1.T)
        d_input = d_input_flat.reshape(batch_size, seq_len, d_model)

        return d_input

    def update(self, learning_rate):
        """
        Update parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.W1 -= learning_rate * self.d_W1
        self.b1 -= learning_rate * self.d_b1
        self.W2 -= learning_rate * self.d_W2
        self.b2 -= learning_rate * self.d_b2

class TransformerBlock:
    """
    Transformer encoder block.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize transformer block.

        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
            d_ff (int): Feed-forward hidden dimension
            dropout (float): Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (np.array): Input, shape (batch_size, seq_len, d_model)
            mask (np.array): Attention mask

        Returns:
            np.array: Output
        """
        # Multi-head attention
        attn_output, _ = self.attention.forward(x, mask)

        # Add & norm
        x = self.norm1.forward(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward.forward(x)

        # Add & norm
        x = self.norm2.forward(x + ff_output)

        return x

    def backward(self, d_output):
        """
        Backward pass.

        Args:
            d_output (np.array): Gradient w.r.t. output

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Backward through second add & norm
        d_norm2 = self.norm2.backward(d_output)
        d_ff = d_norm2
        d_residual2 = d_norm2

        # Backward through feed-forward
        d_ff_input = self.feed_forward.backward(d_ff)

        # Backward through first add & norm
        d_norm1 = self.norm1.backward(d_residual2 + d_ff_input)
        d_attn = d_norm1
        d_residual1 = d_norm1

        # Backward through attention
        d_x = self.attention.backward(d_attn)

        return d_x + d_residual1

    def update(self, learning_rate):
        """
        Update all parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.attention.update(learning_rate)
        self.norm1.update(learning_rate)
        self.norm2.update(learning_rate)
        self.feed_forward.update(learning_rate)

# Example usage
if __name__ == "__main__":
    # Sample input
    batch_size, seq_len, d_model = 2, 4, 8
    x = np.random.randn(batch_size, seq_len, d_model)

    # Create transformer block
    transformer = TransformerBlock(d_model=8, num_heads=2, d_ff=16)

    # Forward pass
    output = transformer.forward(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    # Backward pass (dummy gradient)
    d_output = np.random.randn(batch_size, seq_len, d_model)
    d_input = transformer.backward(d_output)
    print("Gradient w.r.t. input shape:", d_input.shape)

    # Update
    transformer.update(0.01)
    print("Parameters updated successfully")
