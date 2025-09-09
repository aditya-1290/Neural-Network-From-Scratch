import numpy as np

class LSTMCell:
    """
    LSTM cell implementation with gates.
    """

    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM cell.

        Args:
            input_size (int): Size of input vector
            hidden_size (int): Size of hidden state and cell state
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Forget gate: f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_f = np.zeros((1, hidden_size))

        # Input gate: i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_i = np.zeros((1, hidden_size))

        # Candidate values: tilde{C}_t = tanh(W_c * [h_{t-1}, x_t] + b_c)
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_c = np.zeros((1, hidden_size))

        # Output gate: o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.b_o = np.zeros((1, hidden_size))

        # Gradients
        self.d_W_f = np.zeros_like(self.W_f)
        self.d_b_f = np.zeros_like(self.b_f)
        self.d_W_i = np.zeros_like(self.W_i)
        self.d_b_i = np.zeros_like(self.b_i)
        self.d_W_c = np.zeros_like(self.W_c)
        self.d_b_c = np.zeros_like(self.b_c)
        self.d_W_o = np.zeros_like(self.W_o)
        self.d_b_o = np.zeros_like(self.b_o)

        # Cache for backpropagation
        self.x_cache = []
        self.h_cache = []
        self.C_cache = []
        self.f_cache = []
        self.i_cache = []
        self.tilde_C_cache = []
        self.o_cache = []

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _dsigmoid(self, x):
        return x * (1 - x)

    def forward(self, x, h_prev=None, C_prev=None):
        """
        Forward pass for one time step.

        Args:
            x (np.array): Input at current time step, shape (batch_size, input_size)
            h_prev (np.array): Previous hidden state, shape (batch_size, hidden_size)
            C_prev (np.array): Previous cell state, shape (batch_size, hidden_size)

        Returns:
            tuple: (h, C)
                - h: hidden state, shape (batch_size, hidden_size)
                - C: cell state, shape (batch_size, hidden_size)
        """
        if h_prev is None:
            h_prev = np.zeros((x.shape[0], self.hidden_size))
        if C_prev is None:
            C_prev = np.zeros((x.shape[0], self.hidden_size))

        # Concatenate h_prev and x
        concat = np.concatenate((h_prev, x), axis=1)

        # Forget gate
        f = self._sigmoid(np.dot(concat, self.W_f) + self.b_f)

        # Input gate
        i = self._sigmoid(np.dot(concat, self.W_i) + self.b_i)

        # Candidate values
        tilde_C = np.tanh(np.dot(concat, self.W_c) + self.b_c)

        # Cell state
        C = f * C_prev + i * tilde_C

        # Output gate
        o = self._sigmoid(np.dot(concat, self.W_o) + self.b_o)

        # Hidden state
        h = o * np.tanh(C)

        # Cache for backprop
        self.x_cache.append(x)
        self.h_cache.append(h_prev)
        self.C_cache.append(C_prev)
        self.f_cache.append(f)
        self.i_cache.append(i)
        self.tilde_C_cache.append(tilde_C)
        self.o_cache.append(o)

        return h, C

    def forward_sequence(self, x_sequence):
        """
        Forward pass for a sequence.

        Args:
            x_sequence (np.array): Input sequence, shape (batch_size, seq_len, input_size)

        Returns:
            tuple: (hidden_states, cell_states)
                - hidden_states: shape (batch_size, seq_len, hidden_size)
                - cell_states: shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x_sequence.shape
        hidden_states = []
        cell_states = []

        h_prev = np.zeros((batch_size, self.hidden_size))
        C_prev = np.zeros((batch_size, self.hidden_size))

        for t in range(seq_len):
            h, C = self.forward(x_sequence[:, t, :], h_prev, C_prev)
            hidden_states.append(h)
            cell_states.append(C)
            h_prev = h
            C_prev = C

        return np.stack(hidden_states, axis=1), np.stack(cell_states, axis=1)

    def backward(self, d_h, d_C_next=None):
        """
        Backward pass for one time step.

        Args:
            d_h (np.array): Gradient w.r.t. hidden state, shape (batch_size, hidden_size)
            d_C_next (np.array): Gradient from next time step w.r.t. cell state, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x, d_h_prev, d_C_prev)
                - d_x: gradient w.r.t. input, shape (batch_size, input_size)
                - d_h_prev: gradient w.r.t. previous hidden, shape (batch_size, hidden_size)
                - d_C_prev: gradient w.r.t. previous cell, shape (batch_size, hidden_size)
        """
        # Get cached values
        x = self.x_cache.pop()
        h_prev = self.h_cache.pop()
        C_prev = self.C_cache.pop()
        f = self.f_cache.pop()
        i = self.i_cache.pop()
        tilde_C = self.tilde_C_cache.pop()
        o = self.o_cache.pop()

        C = f * C_prev + i * tilde_C

        # Gradient w.r.t. output gate
        d_o = d_h * np.tanh(C) * self._dsigmoid(o)

        # Gradient w.r.t. cell state
        d_C = d_h * o * (1 - np.tanh(C)**2)
        if d_C_next is not None:
            d_C += d_C_next

        # Gradient w.r.t. forget gate
        d_f = d_C * C_prev * self._dsigmoid(f)

        # Gradient w.r.t. input gate
        d_i = d_C * tilde_C * self._dsigmoid(i)

        # Gradient w.r.t. candidate values
        d_tilde_C = d_C * i * (1 - tilde_C**2)

        # Concatenate for weight gradients
        concat = np.concatenate((h_prev, x), axis=1)

        # Update weight gradients
        self.d_W_f += np.dot(concat.T, d_f)
        self.d_b_f += np.sum(d_f, axis=0, keepdims=True)
        self.d_W_i += np.dot(concat.T, d_i)
        self.d_b_i += np.sum(d_i, axis=0, keepdims=True)
        self.d_W_c += np.dot(concat.T, d_tilde_C)
        self.d_b_c += np.sum(d_tilde_C, axis=0, keepdims=True)
        self.d_W_o += np.dot(concat.T, d_o)
        self.d_b_o += np.sum(d_o, axis=0, keepdims=True)

        # Gradient w.r.t. concat
        d_concat = (np.dot(d_f, self.W_f.T) + np.dot(d_i, self.W_i.T) +
                    np.dot(d_tilde_C, self.W_c.T) + np.dot(d_o, self.W_o.T))

        # Split gradients
        d_h_prev = d_concat[:, :self.hidden_size]
        d_x = d_concat[:, self.hidden_size:]

        # Gradient w.r.t. previous cell state
        d_C_prev = d_C * f

        return d_x, d_h_prev, d_C_prev

    def backward_sequence(self, d_hidden, d_C_next=None):
        """
        Backward pass for a sequence.

        Args:
            d_hidden (np.array): Gradients w.r.t. hidden states, shape (batch_size, seq_len, hidden_size)
            d_C_next (np.array): Gradient from next time step w.r.t. cell state, shape (batch_size, hidden_size)

        Returns:
            tuple: (d_x_sequence, d_h_prev, d_C_prev)
                - d_x_sequence: gradients w.r.t. inputs, shape (batch_size, seq_len, input_size)
                - d_h_prev: gradient w.r.t. initial hidden, shape (batch_size, hidden_size)
                - d_C_prev: gradient w.r.t. initial cell, shape (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = d_hidden.shape
        d_x_sequence = []

        d_C = d_C_next
        for t in reversed(range(seq_len)):
            d_x, d_h, d_C = self.backward(d_hidden[:, t, :], d_C)
            d_x_sequence.insert(0, d_x)

        return np.stack(d_x_sequence, axis=1), d_h, d_C

    def update(self, learning_rate):
        """
        Update parameters using gradients.

        Args:
            learning_rate (float): Learning rate
        """
        self.W_f -= learning_rate * self.d_W_f
        self.b_f -= learning_rate * self.d_b_f
        self.W_i -= learning_rate * self.d_W_i
        self.b_i -= learning_rate * self.d_b_i
        self.W_c -= learning_rate * self.d_W_c
        self.b_c -= learning_rate * self.d_b_c
        self.W_o -= learning_rate * self.d_W_o
        self.b_o -= learning_rate * self.d_b_o

        # Reset gradients
        self.d_W_f = np.zeros_like(self.W_f)
        self.d_b_f = np.zeros_like(self.b_f)
        self.d_W_i = np.zeros_like(self.W_i)
        self.d_b_i = np.zeros_like(self.b_i)
        self.d_W_c = np.zeros_like(self.W_c)
        self.d_b_c = np.zeros_like(self.b_c)
        self.d_W_o = np.zeros_like(self.W_o)
        self.d_b_o = np.zeros_like(self.b_o)

# Example usage
if __name__ == "__main__":
    # Create LSTM cell
    lstm = LSTMCell(input_size=4, hidden_size=8)

    # Sample sequence: batch of 2, sequence length 3, input size 4
    x_seq = np.random.randn(2, 3, 4)

    # Forward pass
    hidden_states, cell_states = lstm.forward_sequence(x_seq)
    print("Input sequence shape:", x_seq.shape)
    print("Hidden states shape:", hidden_states.shape)
    print("Cell states shape:", cell_states.shape)

    # Backward pass (dummy gradients)
    d_hidden = np.random.randn(2, 3, 8)
    d_x_seq, d_h_prev, d_C_prev = lstm.backward_sequence(d_hidden)
    print("Gradient w.r.t. input sequence shape:", d_x_seq.shape)
    print("Gradient w.r.t. initial hidden shape:", d_h_prev.shape)
    print("Gradient w.r.t. initial cell shape:", d_C_prev.shape)

    # Update
    lstm.update(0.01)
    print("Parameters updated successfully")
