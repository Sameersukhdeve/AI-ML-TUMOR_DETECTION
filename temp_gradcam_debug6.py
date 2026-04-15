import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

path = glob.glob('model/dataset/Testing/*/*')[0]
model = load_model('model/tumor_model.h5')
img = image.load_img(path, target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)
img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
last_conv = model.get_layer('conv2d_1').output
grad_model = tf.keras.models.Model(inputs=model.input, outputs=[last_conv, model.output])
print('grad_model inputs', grad_model.inputs)
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    tape.watch(conv_outputs)
    loss = tf.reduce_mean(predictions[:,0])
    print('loss', loss)
grads = tape.gradient(loss, conv_outputs)
print('grads', grads)
print('grads shape', None if grads is None else grads.shape)
