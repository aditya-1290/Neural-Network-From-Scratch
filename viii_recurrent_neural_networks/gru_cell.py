import numpy as np

class GRUCell:
    """
    GRU cell implementation with reset and update gates.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize the GRU cell.

        Args:
            input_size (int): Size of input vector
            hidden_size (int): Size of hidden state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update gate: z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_z = np.zeros((1, hidden_size))

        # Reset gate: r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_r = np.zeros((1, hidden_size))

        # Candidate hidden: tilde{h}_t = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_h = np.zeros((1, hidden_size))

        # Gradients
        self.d_W_z = np.zeros_like(self.W_z)
        self.d_b_z = np.zeros_like(self.b_z)
        self.d_W_r = np.zeros_like(self.W_r)
        self.d_b_r = np.zeros_like(self.b_r)
        self.d_W_h = np.zeros_like(self.W_h)
        self.d_b_h = np.zeros_like(self.b_h)

        # Cache for backpropagation
        self.x_cache = []
        self.h_cache = []
        self.z_cache = []
        self.r_cache = []
        self.tilde_h_cache = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, x):
        return x * (1 - x)

    def forward(self, x, h_prev=None):
        """
        Forward pass for one time step.

        Args:
            x (np.array): Input at current time step, shape (batch_size, input_size)
            h_prev (np.array): Previous hidden state, shape (batch_size, hidden_size)

        Returns:
            np.array: Hidden state, shape (batch_size, hidden_size)
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))

        # Concatenate h_prev and x
        concat = np.concatenate((h_prev, x), axis=1)

        # Update gate
        z = self._sigmoid(np.dot(concat, self.W_z) + self.b_z)

        # Reset gate
        r = self._sigmoid(np.dot(concat, self.W_r) + self.b_r)

        # Reset hidden
        r_h_prev = r * h_prev

        # Concatenate r_h_prev and x
        concat_reset = np.concatenate((r_h_prev, x), axis=1)

        # Candidate hidden
        tilde_h = np.tanh(np.dot(concat_reset, self.W_h) + self.b_h)

        # Hidden state
        h = (1 - z) * h_prev + z * tilde_h

        # Cache for backprop
        self.x_cache.append(x)
        self.h_cache.append(h_prev)
        self.z_cache.append(z)
        self.r_cache.append(r)
        self.tilde_h_cache.append(tilde_h)

        return h

    def forward_sequence(self, x_sequence):
        """
        Forward pass for a sequence.

        Args:
            x_sequence (np.array): Input sequence, shape (batch_size, seq_len, input_size)

        Returns:
            np.array: Hidden states, shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x_sequence.shape
        hidden_states = []

        h_prev = np.zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            h = self.forward(x_sequence[:, t, :], h_prev)
            hidden_states.append(h)
            h_prev = h

        return np.stack(hidden_states, axis=1)

    def backward(self, d_h, d_h_next=None):
        """
        Backward pass for one time step.

        Args:
            d_h (np.array): Gradient w.r.t. hidden state, shape (batch_size, hidden_size)
            d_h_next (np.array): Gradient from next time step, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x, d_h_prev)
                - d_x: gradient w.r.t. input, shape (batch_size, input_size)
                - d_h_prev: gradient w.r.t. previous hidden, shape (batch_size, hidden_size)
        """
        # Get cached values
        x = self.x_cache.pop()
        h_prev = self.h_cache.pop()
        z = self.z_cache.pop()
        r = self.r_cache.pop()
        tilde_h = self.tilde_h_cache.pop()

        # Gradient from next time step
        if d_h_next is not None:
            d_h += d_h_next

        # Gradient w.r.t. candidate hidden
        d_tilde_h = d_h * z * (1 - tilde_h**2)

        # Gradient w.r.t. reset hidden
        r_h_prev = r * h_prev
        concat_reset = np.concatenate((r_h_prev, x), axis=1)
        d_concat_reset = np.dot(d_tilde_h, self.W_h.T)

        # Split gradients
        d_r_h_prev = d_concat_reset[:, :self.hidden_size]
        d_x_from_h = d_concat_reset[:, self.hidden_size:]

        # Gradient w.r.t. reset gate
        d_r = d_r_h_prev * h_prev * self._dsigmoid(r)

        # Gradient w.r.t. update gate
        d_z = d_h * (tilde_h - h_prev) * self._dsigmoid(z)

        # Concatenate for weight gradients
        concat = np.concatenate((h_prev, x), axis=1)

        # Update weight gradients
        self.d_W_z += np.dot(concat.T, d_z)
        self.d_b_z += np.sum(d_z, axis=0, keepdims=True)
        self.d_W_r += np.dot(concat.T, d_r)
        self.d_b_r += np.sum(d_r, axis=0, keepdims=True)
        self.d_W_h += np.dot(concat_reset.T, d_tilde_h)
        self.d_b_h += np.sum(d_tilde_h, axis=0, keepdims=True)

        # Gradient w.r.t. concat
        d_concat = np.dot(d_z, self.W_z.T) + np.dot(d_r, self.W_r.T)

        # Split gradients
        d_h_prev = d_concat[:, :self.hidden_size] + d_r_h_prev * r
        d_x = d_concat[:, self.hidden_size:] + d_x_from_h

        return d_x, d_h_prev

    def backward_sequence(self, d_hidden, d_h_next=None):
        """
        Backward pass for a sequence.

        Args:
            d_hidden (np.array): Gradients w.r.t. hidden states, shape (batch_size, seq_len, hidden_size)
            d_h_next (np.array): Gradient from next time step, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x_sequence, d_h_prev)
                - d_x_sequence: gradients w.r.t. inputs, shape (batch_size, seq_len, input_size)
                - d_h_prev: gradient w.r.t. initial hidden, shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = d_hidden.shape
        d_x_sequence = []

        d_h = d_h_next
        for t in reversed(range(seq_len)):
            d_x, d_h = self.backward(d_hidden[:, t, :], d_h)
            d_x_sequence.insert(0, d_x)

        return np.stack(d_x_sequence, axis=1), d_h

    def update(self, learning_rate):
        """
        Update parameters using gradients.

        Args:
            learning_rate (float): Learning rate
        """
        self.W_z -= learning_rate * self.d_W_z
        self.b_z -= learning_rate * self.d_b_z
        self.W_r -= learning_rate * self.d_W_r
        self.b_r -= learning_rate * self.d_b_r
        self.W_h -= learning_rate * self.d_W_h
        self.b_h -= learning_rate * self.d_b_h

        # Reset gradients
        self.d_W_z = np.zeros_like(self.W_z)
        self.d_b_z = np.zeros_like(self.b_z)
        self.d_W_r = np.zeros_like(self.W_r)
        self.d_b_r = np.zeros_like(self.b_r)
        self.d_W_h = np.zeros_like(self.W_h)
        self.d_b_h = np.zeros_like(self.b_h)

# Example usage
if __name__ == "__main__":
    # Create GRU cell
    gru = GRUCell(input_size=4, hidden_size=8)

    # Sample sequence: batch of 2, sequence length 3, input size 4
    x_seq = np.random.randn(2, 3, 4)

    # Forward pass
    hidden_states = gru.forward_sequence(x_seq)
    print("Input sequence shape:", x_seq.shape)
    print("Hidden states shape:", hidden_states.shape)

    # Backward pass (dummy gradients)
    d_hidden = np.random.randn(2, 3, 8)
    d_x_seq, d_h_prev = gru.backward_sequence(d_hidden)
    print("Gradient w.r.t. input sequence shape:", d_x_seq.shape)
    print("Gradient w.r.t. initial hidden shape:", d_h_prev.shape)

    # Update
    gru.update(0.01)
    print("Parameters updated successfully")
