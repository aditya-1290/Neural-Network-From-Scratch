import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import SelfAttention
from neural_networks_from_scratch import TransformerBlock
from neural_networks_from_scratch import DenseLayer
from neural_networks_from_scratch import categorical_cross_entropy, cce_derivative

class BERTEmbedding:
    """
    Simplified BERT embedding layer (token + position + segment embeddings).
    """

    def __init__(self, vocab_size=30000, max_seq_len=512, hidden_dim=768):
        """
        Initialize BERT embeddings.

        Args:
            vocab_size (int): Vocabulary size
            max_seq_len (int): Maximum sequence length
            hidden_dim (int): Hidden dimension
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # Embedding matrices
        self.token_embeddings = np.random.randn(vocab_size, hidden_dim) * 0.02
        self.position_embeddings = np.random.randn(max_seq_len, hidden_dim) * 0.02
        self.segment_embeddings = np.random.randn(2, hidden_dim) * 0.02  # For sentence pairs

        # Layer norm parameters (simplified)
        self.layer_norm_gamma = np.ones(hidden_dim)
        self.layer_norm_beta = np.zeros(hidden_dim)

    def forward(self, input_ids, segment_ids=None):
        """
        Forward pass through embeddings.

        Args:
            input_ids (np.array): Token IDs, shape (batch, seq_len)
            segment_ids (np.array): Segment IDs, shape (batch, seq_len)

        Returns:
            np.array: Embeddings, shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.token_embeddings[input_ids]  # (batch, seq_len, hidden_dim)

        # Position embeddings
        position_ids = np.arange(seq_len)
        position_embeds = self.position_embeddings[position_ids]  # (seq_len, hidden_dim)
        position_embeds = np.tile(position_embeds, (batch_size, 1, 1))  # (batch, seq_len, hidden_dim)

        # Segment embeddings
        if segment_ids is None:
            segment_ids = np.zeros((batch_size, seq_len), dtype=int)
        segment_embeds = self.segment_embeddings[segment_ids]  # (batch, seq_len, hidden_dim)

        # Sum embeddings
        embeddings = token_embeds + position_embeds + segment_embeds

        # Layer normalization (simplified)
        mean = np.mean(embeddings, axis=-1, keepdims=True)
        var = np.var(embeddings, axis=-1, keepdims=True)
        embeddings = (embeddings - mean) / np.sqrt(var + 1e-8)
        embeddings = embeddings * self.layer_norm_gamma + self.layer_norm_beta

        return embeddings

class BERTEncoder:
    """
    Simplified BERT encoder with multiple transformer blocks.
    """

    def __init__(self, num_layers=12, hidden_dim=768, num_heads=12, ff_dim=3072):
        """
        Initialize BERT encoder.

        Args:
            num_layers (int): Number of transformer layers
            hidden_dim (int): Hidden dimension
            num_heads (int): Number of attention heads
            ff_dim (int): Feed-forward dimension
        """
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Transformer blocks
        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(hidden_dim, num_heads, ff_dim))

        # Pooler (for classification)
        self.pooler = DenseLayer(hidden_dim, hidden_dim, activation='tanh')

    def forward(self, embeddings):
        """
        Forward pass through encoder.

        Args:
            embeddings (np.array): Input embeddings, shape (batch, seq_len, hidden_dim)

        Returns:
            tuple: (sequence_output, pooled_output)
        """
        # Pass through transformer layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states)

        # Pooling: take [CLS] token (first token)
        pooled_output = self.pooler.forward(hidden_states[:, 0, :])  # (batch, hidden_dim)

        return hidden_states, pooled_output

class BERTForSequenceClassification:
    """
    BERT model for sequence classification.
    """

    def __init__(self, vocab_size=30000, max_seq_len=512, hidden_dim=768,
                 num_layers=6, num_heads=8, num_classes=2):
        """
        Initialize BERT for classification.

        Args:
            vocab_size (int): Vocabulary size
            max_seq_len (int): Maximum sequence length
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of transformer layers
            num_heads (int): Number of attention heads
            num_classes (int): Number of output classes
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BERT components
        self.embeddings = BERTEmbedding(vocab_size, max_seq_len, hidden_dim)
        self.encoder = BERTEncoder(num_layers, hidden_dim, num_heads)

        # Classification head
        self.classifier = DenseLayer(hidden_dim, num_classes, activation='linear')

        # Freeze BERT parameters initially
        self.frozen = True

    def forward(self, input_ids, segment_ids=None):
        """
        Forward pass through BERT classifier.

        Args:
            input_ids (np.array): Token IDs, shape (batch, seq_len)
            segment_ids (np.array): Segment IDs, shape (batch, seq_len)

        Returns:
            np.array: Class logits, shape (batch, num_classes)
        """
        # Get embeddings
        embeddings = self.embeddings.forward(input_ids, segment_ids)

        # Encode
        _, pooled_output = self.encoder.forward(embeddings)

        # Classify
        logits = self.classifier.forward(pooled_output)

        return logits

    def freeze_bert(self):
        """Freeze BERT parameters."""
        self.frozen = True

    def unfreeze_bert(self):
        """Unfreeze BERT parameters for fine-tuning."""
        self.frozen = False

def create_synthetic_text_data(num_samples=1000, max_seq_len=128, vocab_size=1000, num_classes=2):
    """
    Create synthetic text classification dataset.

    Args:
        num_samples (int): Number of samples
        max_seq_len (int): Maximum sequence length
        vocab_size (int): Vocabulary size
        num_classes (int): Number of classes

    Returns:
        tuple: (input_ids, labels)
    """
    # Generate random token sequences
    input_ids = np.random.randint(1, vocab_size, (num_samples, max_seq_len))

    # Add [CLS] token at beginning
    input_ids[:, 0] = 101  # [CLS] token

    # Generate random labels
    labels = np.random.randint(0, num_classes, num_samples)
    labels_onehot = np.zeros((num_samples, num_classes))
    labels_onehot[np.arange(num_samples), labels] = 1

    return input_ids, labels_onehot

def train_bert_classifier():
    """
    Train BERT for text classification.
    """
    print("Creating synthetic text classification dataset...")
    input_ids, labels = create_synthetic_text_data(
        num_samples=500, max_seq_len=128, vocab_size=1000, num_classes=2
    )

    print(f"Dataset shape: input_ids={input_ids.shape}, labels={labels.shape}")

    # Create BERT model (smaller for demonstration)
    model = BERTForSequenceClassification(
        vocab_size=1000, max_seq_len=128, hidden_dim=256,
        num_layers=4, num_heads=8, num_classes=2
    )

    # Phase 1: Train only classifier (BERT frozen)
    print("\nPhase 1: Training classifier with frozen BERT...")
    model.freeze_bert()

    learning_rate = 0.01
    epochs_phase1 = 5

    for epoch in range(epochs_phase1):
        # Forward pass
        logits = model.forward(input_ids)

        # Compute loss
        loss = categorical_cross_entropy(labels, logits)

        # Backward pass
        d_loss = cce_derivative(labels, logits)

        # For now, simplified backward (would need full implementation)
        print(f"Phase 1 - Epoch {epoch}, Loss: {loss:.4f}")

    # Phase 2: Fine-tune entire model
    print("\nPhase 2: Fine-tuning entire BERT model...")
    model.unfreeze_bert()

    learning_rate_finetune = 0.001  # Lower learning rate for fine-tuning
    epochs_phase2 = 5

    for epoch in range(epochs_phase2):
        # Forward pass
        logits = model.forward(input_ids)

        # Compute loss
        loss = categorical_cross_entropy(labels, logits)

        # Backward pass (simplified)
        d_loss = cce_derivative(labels, logits)

        print(f"Phase 2 - Epoch {epoch}, Loss: {loss:.4f}")

    # Evaluate
    print("\nEvaluating model...")
    predictions = model.forward(input_ids)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)

    accuracy = np.mean(pred_labels == true_labels)
    print(f"Training accuracy: {accuracy:.4f}")

    return model

if __name__ == "__main__":
    print("BERT Fine-tuning for Text Classification")
    print("=" * 45)

    # Train model
    model = train_bert_classifier()

    print("\nBERT fine-tuning demonstration completed!")
    print("In practice, you would:")
    print("1. Load actual pre-trained BERT weights")
    print("2. Use real text datasets (e.g., GLUE, SST)")
    print("3. Implement proper tokenization (WordPiece, BPE)")
    print("4. Use Adam optimizer with warmup and decay")
    print("5. Add dropout and other regularization")
