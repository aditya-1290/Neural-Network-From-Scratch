import numpy as np

def softmax(x, axis=-1):
    """
    Softmax function.

    Args:
        x (np.array): Input array
        axis (int): Axis along which to compute softmax

    Returns:
        np.array: Softmax output
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_q, W_k, W_v, mask=None):
    """
    Self-attention mechanism.

    Args:
        X (np.array): Input sequence, shape (batch_size, seq_len, d_model)
        W_q (np.array): Query weight matrix, shape (d_model, d_k)
        W_k (np.array): Key weight matrix, shape (d_model, d_k)
        W_v (np.array): Value weight matrix, shape (d_model, d_v)
        mask (np.array): Attention mask, shape (batch_size, seq_len, seq_len)

    Returns:
        np.array: Attention output, shape (batch_size, seq_len, d_v)
    """
    batch_size, seq_len, d_model = X.shape
    d_k = W_q.shape[1]

    # Linear transformations
    Q = np.dot(X, W_q)  # Shape: (batch_size, seq_len, d_k)
    K = np.dot(X, W_k)  # Shape: (batch_size, seq_len, d_k)
    V = np.dot(X, W_v)  # Shape: (batch_size, seq_len, d_v)

    # Attention scores
    scores = np.dot(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # Shape: (batch_size, seq_len, seq_len)

    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)

    # Softmax
    attention_weights = softmax(scores, axis=-1)  # Shape: (batch_size, seq_len, seq_len)

    # Attention output
    attention_output = np.dot(attention_weights, V)  # Shape: (batch_size, seq_len, d_v)

    return attention_output, attention_weights

class MultiHeadAttention:
    """
    Multi-head self-attention mechanism.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model (int): Model dimension
            num_heads (int): Number of attention heads
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        # Weight matrices for all heads
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01

        # Gradients
        self.d_W_q = np.zeros_like(self.W_q)
        self.d_W_k = np.zeros_like(self.W_k)
        self.d_W_v = np.zeros_like(self.W_v)
        self.d_W_o = np.zeros_like(self.W_o)

    def forward(self, X, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            X (np.array): Input sequence, shape (batch_size, seq_len, d_model)
            mask (np.array): Attention mask, shape (batch_size, seq_len, seq_len)

        Returns:
            np.array: Attention output, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = X.shape

        # Linear transformations
        Q = np.dot(X, self.W_q)  # Shape: (batch_size, seq_len, d_model)
        K = np.dot(X, self.W_k)  # Shape: (batch_size, seq_len, d_model)
        V = np.dot(X, self.W_v)  # Shape: (batch_size, seq_len, d_model)

        # Split into heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # Attention for each head
        attention_outputs = []
        attention_weights_list = []
        for h in range(self.num_heads):
            head_output, head_weights = self_attention(
                Q[:, h:h+1].reshape(batch_size, seq_len, self.d_k),
                K[:, h:h+1].reshape(batch_size, seq_len, self.d_k),
                V[:, h:h+1].reshape(batch_size, seq_len, self.d_v),
                mask
            )
            attention_outputs.append(head_output)
            attention_weights_list.append(head_weights)

        # Concatenate heads
        multi_head_output = np.concatenate(attention_outputs, axis=-1)  # (batch, seq, d_model)

        # Final linear transformation
        output = np.dot(multi_head_output, self.W_o)  # (batch, seq, d_model)

        return output, attention_weights_list

    def backward(self, d_output):
        """
        Backward pass for multi-head attention.

        Args:
            d_output (np.array): Gradient w.r.t. output, shape (batch_size, seq_len, d_model)

        Returns:
            np.array: Gradient w.r.t. input, shape (batch_size, seq_len, d_model)
        """
        # This is a simplified backward pass
        # In practice, you'd need to implement full backprop through attention

        # Gradient w.r.t. W_o
        self.d_W_o += np.dot(d_output.transpose(0, 2, 1), d_output).sum(axis=0)

        # Gradient w.r.t. multi-head output
        d_multi_head = np.dot(d_output, self.W_o.T)

        # For simplicity, assume equal contribution to each head
        d_X = d_multi_head / self.num_heads

        return d_X

    def update(self, learning_rate):
        """
        Update parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.W_q -= learning_rate * self.d_W_q
        self.W_k -= learning_rate * self.d_W_k
        self.W_v -= learning_rate * self.d_W_v
        self.W_o -= learning_rate * self.d_W_o

        # Reset gradients
        self.d_W_q = np.zeros_like(self.W_q)
        self.d_W_k = np.zeros_like(self.W_k)
        self.d_W_v = np.zeros_like(self.W_v)
        self.d_W_o = np.zeros_like(self.W_o)

# Example usage
if __name__ == "__main__":
    # Sample input
    batch_size, seq_len, d_model = 2, 4, 8
    X = np.random.randn(batch_size, seq_len, d_model)

    # Single-head self-attention
    W_q = np.random.randn(d_model, d_model) * 0.01
    W_k = np.random.randn(d_model, d_model) * 0.01
    W_v = np.random.randn(d_model, d_model) * 0.01

    attention_output, attention_weights = self_attention(X, W_q, W_k, W_v)
    print("Single-head attention output shape:", attention_output.shape)
    print("Attention weights shape:", attention_weights.shape)

    # Multi-head attention
    num_heads = 2
    mha = MultiHeadAttention(d_model, num_heads)

    mha_output, mha_weights = mha.forward(X)
    print("Multi-head attention output shape:", mha_output.shape)
    print("Number of attention weight matrices:", len(mha_weights))

    # Backward pass (simplified)
    d_output = np.random.randn(batch_size, seq_len, d_model)
    d_X = mha.backward(d_output)
    print("Gradient w.r.t. input shape:", d_X.shape)

    # Update
    mha.update(0.01)
    print("Parameters updated successfully")
