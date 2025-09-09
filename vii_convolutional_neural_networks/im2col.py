import numpy as np

def im2col(input_data, kernel_h, kernel_w, stride=1, padding=0):
    """
    Transform input image to column matrix for efficient convolution.

    Args:
        input_data (np.array): Input of shape (C, H, W)
        kernel_h (int): Kernel height
        kernel_w (int): Kernel width
        stride (int): Stride
        padding (int): Padding

    Returns:
        np.array: Column matrix of shape (C * kernel_h * kernel_w, H_out * W_out)
    """
    C, H, W = input_data.shape

    # Pad the input
    if padding > 0:
        input_padded = np.pad(input_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        input_padded = input_data

    H_padded, W_padded = input_padded.shape[1], input_padded.shape[2]

    # Output dimensions
    H_out = (H_padded - kernel_h) // stride + 1
    W_out = (W_padded - kernel_w) // stride + 1

    # Initialize the column matrix
    col = np.zeros((C * kernel_h * kernel_w, H_out * W_out))

    # Fill the column matrix
    col_idx = 0
    for i in range(H_out):
        for j in range(W_out):
            # Extract patch
            patch = input_padded[:, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w]
            # Flatten and store
            col[:, col_idx] = patch.flatten()
            col_idx += 1

    return col

def col2im(col, input_shape, kernel_h, kernel_w, stride=1, padding=0):
    """
    Transform column matrix back to image for gradient computation.

    Args:
        col (np.array): Column matrix of shape (C * kernel_h * kernel_w, H_out * W_out)
        input_shape (tuple): Original input shape (C, H, W)
        kernel_h (int): Kernel height
        kernel_w (int): Kernel width
        stride (int): Stride
        padding (int): Padding

    Returns:
        np.array: Reconstructed image of shape (C, H, W)
    """
    C, H, W = input_shape

    # Output dimensions
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding
    H_out = (H_padded - kernel_h) // stride + 1
    W_out = (W_padded - kernel_w) // stride + 1

    # Initialize the reconstructed image
    img = np.zeros((C, H_padded, W_padded))

    # Fill the image
    col_idx = 0
    for i in range(H_out):
        for j in range(W_out):
            # Extract column
            patch_flat = col[:, col_idx]
            # Reshape to patch
            patch = patch_flat.reshape((C, kernel_h, kernel_w))
            # Add to image
            img[:, i*stride:i*stride+kernel_h, j*stride:j*stride+kernel_w] += patch
            col_idx += 1

    # Remove padding
    if padding > 0:
        img = img[:, padding:H_padded-padding, padding:W_padded-padding]

    return img

# Example usage
if __name__ == "__main__":
    # Sample input: 3 channels, 5x5 image
    input_data = np.random.randn(3, 5, 5)
    print("Input shape:", input_data.shape)

    # im2col with 3x3 kernel, stride 1, padding 1
    kernel_h, kernel_w = 3, 3
    stride = 1
    padding = 1

    col = im2col(input_data, kernel_h, kernel_w, stride, padding)
    print("Column matrix shape:", col.shape)

    # Expected: C * KH * KW = 3 * 3 * 3 = 27
    # H_out = (5 + 2*1 - 3) // 1 + 1 = 5
    # W_out = 5
    # So 27 x 25

    # Reconstruct
    reconstructed = col2im(col, input_data.shape, kernel_h, kernel_w, stride, padding)
    print("Reconstructed shape:", reconstructed.shape)
    print("Reconstruction error:", np.max(np.abs(input_data - reconstructed)))

    # Test with different parameters
    print("\nTesting with different parameters:")
    col2 = im2col(input_data, 2, 2, stride=2, padding=0)
    print("Kernel 2x2, stride 2, padding 0 - Col shape:", col2.shape)
    # H_out = (5 - 2) // 2 + 1 = 2
    # W_out = 2
    # 3*2*2 = 12 x 4

    reconstructed2 = col2im(col2, input_data.shape, 2, 2, stride=2, padding=0)
    print("Reconstruction error:", np.max(np.abs(input_data - reconstructed2)))
