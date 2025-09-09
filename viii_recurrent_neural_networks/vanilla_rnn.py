import numpy as np
from rnn_cell import RNNCell

class VanillaRNN:
    """
    Vanilla RNN implementation using RNNCell.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the Vanilla RNN.

        Args:
            input_size (int): Size of input vector
            hidden_size (int): Size of hidden state
            output_size (int): Size of output vector
            num_layers (int): Number of RNN layers
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Create RNN layers
        self.layers = []
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            out_size = output_size if i == num_layers - 1 else hidden_size
            self.layers.append(RNNCell(in_size, hidden_size, out_size))

    def forward(self, x_sequence):
        """
        Forward pass for a sequence.

        Args:
            x_sequence (np.array): Input sequence, shape (batch_size, seq_len, input_size)

        Returns:
            tuple: (outputs, hidden_states)
                - outputs: shape (batch_size, seq_len, output_size)
                - hidden_states: list of hidden states for each layer
        """
        batch_size, seq_len, _ = x_sequence.shape
        hidden_states = []

        # Process through layers
        current_input = x_sequence
        for layer in self.layers:
            outputs, layer_hidden = layer.forward_sequence(current_input)
            hidden_states.append(layer_hidden)
            current_input = outputs  # Output of this layer is input to next

        return outputs, hidden_states

    def backward(self, d_outputs, hidden_states):
        """
        Backward pass for a sequence.

        Args:
            d_outputs (np.array): Gradients w.r.t. outputs, shape (batch_size, seq_len, output_size)
            hidden_states (list): Hidden states from forward pass

        Returns:
            np.array: Gradients w.r.t. input sequence, shape (batch_size, seq_len, input_size)
        """
        # Backward through layers
        d_current = d_outputs
        for i in reversed(range(self.num_layers)):
            layer = self.layers[i]
            if i > 0:
                # For intermediate layers, d_current is gradient w.r.t. output of this layer
                d_current, _ = layer.backward_sequence(d_current)
            else:
                # For first layer, d_current is gradient w.r.t. final output
                d_current, _ = layer.backward_sequence(d_current)

        return d_current

    def update(self, learning_rate):
        """
        Update all layers.

        Args:
            learning_rate (float): Learning rate
        """
        for layer in self.layers:
            layer.update(learning_rate)

# Example usage
if __name__ == "__main__":
    # Create Vanilla RNN: 2 layers
    rnn = VanillaRNN(input_size=4, hidden_size=8, output_size=2, num_layers=2)

    # Sample sequence: batch of 2, sequence length 3, input size 4
    x_seq = np.random.randn(2, 3, 4)

    # Forward pass
    outputs, hidden_states = rnn.forward(x_seq)
    print("Input sequence shape:", x_seq.shape)
    print("Outputs shape:", outputs.shape)
    print("Number of hidden state layers:", len(hidden_states))
    print("Hidden states shape for layer 0:", hidden_states[0].shape)

    # Backward pass (dummy gradients)
    d_outputs = np.random.randn(2, 3, 2)
    d_x_seq = rnn.backward(d_outputs, hidden_states)
    print("Gradient w.r.t. input sequence shape:", d_x_seq.shape)

    # Update
    rnn.update(0.01)
    print("Parameters updated successfully")
