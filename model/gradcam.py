"""
gradcam.py — GradCAM heatmap generation for NeuroScan AI
"""
import numpy as np
import cv2
import tensorflow as tf


def generate_gradcam(model, img_array: np.ndarray,
                     original_path: str, output_path: str) -> str:
    """
    Generate a GradCAM heatmap overlay and save to output_path.

    Args:
        model:         Loaded Keras model
        img_array:     Preprocessed image array (1, H, W, 3)
        original_path: Path to the original uploaded image
        output_path:   Path to save the heatmap overlay image

    Returns:
        output_path string
    """
    # ── Find last conv layer ──
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    # ── Build grad model ──
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # ── Normalize ──
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    # ── Overlay on original image ──
    original = cv2.imread(original_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image: {original_path}")

    h, w = original.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )

    # Blend
    superimposed = cv2.addWeighted(original, 0.55, heatmap_colored, 0.45, 0)
    cv2.imwrite(output_path, superimposed)

    return output_path