import numpy as np

class RNNCell:
    """
    Basic RNN cell implementation.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the RNN cell.

        Args:
            input_size (int): Size of input vector
            hidden_size (int): Size of hidden state
            output_size (int): Size of output vector
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))

        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        self.b_y = np.zeros((1, output_size))

        # Gradients
        self.d_W_xh = np.zeros_like(self.W_xh)
        self.d_W_hh = np.zeros_like(self.W_hh)
        self.d_b_h = np.zeros_like(self.b_h)
        self.d_W_hy = np.zeros_like(self.W_hy)
        self.d_b_y = np.zeros_like(self.b_y)

        # Cache for backpropagation
        self.x_cache = []
        self.h_cache = []

    def forward(self, x, h_prev=None):
        """
        Forward pass for one time step.

        Args:
            x (np.array): Input at current time step, shape (batch_size, input_size)
            h_prev (np.array): Previous hidden state, shape (batch_size, hidden_size)

        Returns:
            tuple: (output, hidden_state)
                - output: shape (batch_size, output_size)
                - hidden_state: shape (batch_size, hidden_size)
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))

        # Hidden state
        h = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)

        # Output
        y = np.dot(h, self.W_hy) + self.b_y

        # Cache for backprop
        self.x_cache.append(x)
        self.h_cache.append(h_prev)

        return y, h

    def forward_sequence(self, x_sequence):
        """
        Forward pass for a sequence.

        Args:
            x_sequence (np.array): Input sequence, shape (batch_size, seq_len, input_size)

        Returns:
            tuple: (outputs, hidden_states)
                - outputs: shape (batch_size, seq_len, output_size)
                - hidden_states: shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x_sequence.shape
        outputs = []
        hidden_states = []

        h_prev = np.zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            y, h = self.forward(x_sequence[:, t, :], h_prev)
            outputs.append(y)
            hidden_states.append(h)
            h_prev = h

        return np.stack(outputs, axis=1), np.stack(hidden_states, axis=1)

    def backward(self, d_outputs, d_h_next=None):
        """
        Backward pass for one time step.

        Args:
            d_outputs (np.array): Gradient w.r.t. outputs, shape (batch_size, output_size)
            d_h_next (np.array): Gradient from next time step, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x, d_h_prev)
                - d_x: gradient w.r.t. input, shape (batch_size, input_size)
                - d_h_prev: gradient w.r.t. previous hidden, shape (batch_size, hidden_size)
        """
        # Get cached values
        x = self.x_cache.pop()
        h_prev = self.h_cache.pop()

        # Gradient w.r.t. output weights
        h = np.tanh(np.dot(x, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        self.d_W_hy += np.dot(h.T, d_outputs)
        self.d_b_y += np.sum(d_outputs, axis=0, keepdims=True)

        # Gradient w.r.t. hidden
        d_h = np.dot(d_outputs, self.W_hy.T)
        if d_h_next is not None:
            d_h += d_h_next

        # Gradient w.r.t. tanh
        d_h *= (1 - h**2)

        # Gradient w.r.t. weights and biases
        self.d_W_xh += np.dot(x.T, d_h)
        self.d_W_hh += np.dot(h_prev.T, d_h)
        self.d_b_h += np.sum(d_h, axis=0, keepdims=True)

        # Gradient w.r.t. input and previous hidden
        d_x = np.dot(d_h, self.W_xh.T)
        d_h_prev = np.dot(d_h, self.W_hh.T)

        return d_x, d_h_prev

    def backward_sequence(self, d_outputs, d_h_next=None):
        """
        Backward pass for a sequence.

        Args:
            d_outputs (np.array): Gradients w.r.t. outputs, shape (batch_size, seq_len, output_size)
            d_h_next (np.array): Gradient from next time step, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x_sequence, d_h_prev)
                - d_x_sequence: gradients w.r.t. inputs, shape (batch_size, seq_len, input_size)
                - d_h_prev: gradient w.r.t. initial hidden, shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = d_outputs.shape
        d_x_sequence = []

        d_h = d_h_next
        for t in reversed(range(seq_len)):
            d_x, d_h = self.backward(d_outputs[:, t, :], d_h)
            d_x_sequence.insert(0, d_x)

        return np.stack(d_x_sequence, axis=1), d_h

    def update(self, learning_rate):
        """
        Update parameters using gradients.

        Args:
            learning_rate (float): Learning rate
        """
        self.W_xh -= learning_rate * self.d_W_xh
        self.W_hh -= learning_rate * self.d_W_hh
        self.b_h -= learning_rate * self.d_b_h
        self.W_hy -= learning_rate * self.d_W_hy
        self.b_y -= learning_rate * self.d_b_y

        # Reset gradients
        self.d_W_xh = np.zeros_like(self.W_xh)
        self.d_W_hh = np.zeros_like(self.W_hh)
        self.d_b_h = np.zeros_like(self.b_h)
        self.d_W_hy = np.zeros_like(self.W_hy)
        self.d_b_y = np.zeros_like(self.b_y)

# Example usage
if __name__ == "__main__":
    # Create RNN cell
    rnn = RNNCell(input_size=4, hidden_size=8, output_size=2)

    # Sample sequence: batch of 2, sequence length 3, input size 4
    x_seq = np.random.randn(2, 3, 4)

    # Forward pass
    outputs, hidden_states = rnn.forward_sequence(x_seq)
    print("Input sequence shape:", x_seq.shape)
    print("Outputs shape:", outputs.shape)
    print("Hidden states shape:", hidden_states.shape)

    # Backward pass (dummy gradients)
    d_outputs = np.random.randn(2, 3, 2)
    d_x_seq, d_h_prev = rnn.backward_sequence(d_outputs)
    print("Gradient w.r.t. input sequence shape:", d_x_seq.shape)
    print("Gradient w.r.t. initial hidden shape:", d_h_prev.shape)

    # Update
    rnn.update(0.01)
    print("Parameters updated successfully")
