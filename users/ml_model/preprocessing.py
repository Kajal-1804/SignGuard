import tensorflow as tf
import numpy as np
import cv2

def preprocess_image_from_variable(image):
    """
    Preprocess the input image for comparison.
    Args:
        image (numpy array or tensor): Input image as a numpy array or TensorFlow tensor.
    Returns:
        np.array: Preprocessed image ready for the model.
    """
    # Convert TensorFlow tensor to numpy array if needed
    if isinstance(image, tf.Tensor):
        image = image.numpy()

    # Validate input dimensions
    if image.ndim not in [2, 3]:
        raise ValueError("Input image must be 2D (grayscale) or 3D (RGB).")

    # Convert to grayscale if it's a 3-channel image
    if image.ndim == 3 and image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 128x128
    image_resized = cv2.resize(image, (128, 128))

    # Apply Canny edge detection (optional, depends on training)
    image_uint8 = image_resized.astype(np.uint8)
    image_edges = cv2.Canny(image_resized, 20, 220)

    # Normalize pixel values to [0, 1]
    image_normalized = image_edges.astype(np.float32) / 255.0

    # Add a channel dimension for compatibility with Conv2D layers
    image_final = np.expand_dims(image_normalized, axis=-1)  # Shape: (128, 128, 1)

    return image_final

