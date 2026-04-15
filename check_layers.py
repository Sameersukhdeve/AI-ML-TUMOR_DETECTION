# check_layers.py
import tensorflow as tf

model = tf.keras.models.load_model('model/tumor_model.h5')

print("\n--- All layer names ---")
for layer in model.layers:
    print(layer.name)