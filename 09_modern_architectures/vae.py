import numpy as np
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from neural_networks_from_scratch import DenseLayer

class VAE:
    """
    Variational Autoencoder implementation.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize VAE.

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            latent_dim (int): Latent space dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = DenseLayer(input_dim, hidden_dim, activation='relu')
        self.enc_mu = DenseLayer(hidden_dim, latent_dim, activation='linear')
        self.enc_log_var = DenseLayer(hidden_dim, latent_dim, activation='linear')

        # Decoder
        self.dec1 = DenseLayer(latent_dim, hidden_dim, activation='relu')
        self.dec_output = DenseLayer(hidden_dim, input_dim, activation='sigmoid')  # For binary data

    def encode(self, x):
        """
        Encode input to latent space.

        Args:
            x (np.array): Input, shape (batch_size, input_dim)

        Returns:
            tuple: (mu, log_var) of latent distribution
        """
        h = self.enc1.forward(x)
        mu = self.enc_mu.forward(h)
        log_var = self.enc_log_var.forward(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick.

        Args:
            mu (np.array): Mean of latent distribution
            log_var (np.array): Log variance of latent distribution

        Returns:
            np.array: Sampled latent vector
        """
        std = np.exp(0.5 * log_var)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def decode(self, z):
        """
        Decode from latent space.

        Args:
            z (np.array): Latent vector, shape (batch_size, latent_dim)

        Returns:
            np.array: Reconstructed output
        """
        h = self.dec1.forward(z)
        output = self.dec_output.forward(h)
        return output

    def forward(self, x):
        """
        Forward pass through VAE.

        Args:
            x (np.array): Input, shape (batch_size, input_dim)

        Returns:
            tuple: (reconstruction, mu, log_var)
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    def loss(self, x, reconstruction, mu, log_var):
        """
        Compute VAE loss.

        Args:
            x (np.array): Original input
            reconstruction (np.array): Reconstructed output
            mu (np.array): Mean of latent distribution
            log_var (np.array): Log variance of latent distribution

        Returns:
            float: Total loss
        """
        # Reconstruction loss (binary cross-entropy for binary data)
        recon_loss = -np.sum(x * np.log(reconstruction + 1e-8) + (1 - x) * np.log(1 - reconstruction + 1e-8), axis=1)
        recon_loss = np.mean(recon_loss)

        # KL divergence
        kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)
        kl_loss = np.mean(kl_loss)

        return recon_loss + kl_loss

    def backward(self, x, reconstruction, mu, log_var):
        """
        Backward pass through VAE.

        Args:
            x (np.array): Original input
            reconstruction (np.array): Reconstructed output
            mu (np.array): Mean of latent distribution
            log_var (np.array): Log variance of latent distribution

        Returns:
            np.array: Gradient w.r.t. input
        """
        # Simplified backward pass
        batch_size = x.shape[0]

        # Gradient w.r.t. reconstruction
        d_recon = (reconstruction - x) / (reconstruction * (1 - reconstruction) + 1e-8)

        # Backward through decoder
        d_dec_output = self.dec_output.backward(d_recon)
        d_dec1 = self.dec1.backward(d_dec_output)

        # Gradient w.r.t. latent z
        d_z = d_dec1

        # Gradient w.r.t. mu and log_var (from KL divergence)
        d_mu = d_z + mu  # From KL: -mu
        d_log_var = 0.5 * d_z * np.exp(0.5 * log_var) * np.random.randn(*log_var.shape) - 0.5 * (1 - np.exp(log_var))  # Simplified

        # Backward through encoder
        d_enc_mu = self.enc_mu.backward(d_mu)
        d_enc_log_var = self.enc_log_var.backward(d_log_var)
        d_enc1 = self.enc1.backward(d_enc_mu + d_enc_log_var)

        return d_enc1

    def update(self, learning_rate):
        """
        Update all parameters.

        Args:
            learning_rate (float): Learning rate
        """
        self.enc1.update(learning_rate)
        self.enc_mu.update(learning_rate)
        self.enc_log_var.update(learning_rate)
        self.dec1.update(learning_rate)
        self.dec_output.update(learning_rate)

    def sample(self, num_samples):
        """
        Sample from the learned latent space.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            np.array: Generated samples
        """
        z = np.random.randn(num_samples, self.latent_dim)
        return self.decode(z)

# Example usage
if __name__ == "__main__":
    # Sample input: batch of 4, 784 features (flattened 28x28 MNIST-like)
    x = np.random.rand(4, 784)

    # Create VAE
    vae = VAE(input_dim=784, hidden_dim=256, latent_dim=20)

    # Forward pass
    reconstruction, mu, log_var = vae.forward(x)
    print("Input shape:", x.shape)
    print("Reconstruction shape:", reconstruction.shape)
    print("Mu shape:", mu.shape)
    print("Log var shape:", log_var.shape)

    # Compute loss
    total_loss = vae.loss(x, reconstruction, mu, log_var)
    print("Total loss:", total_loss)

    # Backward pass
    d_input = vae.backward(x, reconstruction, mu, log_var)
    print("Gradient w.r.t. input shape:", d_input.shape)

    # Update
    vae.update(0.01)
    print("Parameters updated successfully")

    # Sample new data
    samples = vae.sample(4)
    print("Generated samples shape:", samples.shape)
