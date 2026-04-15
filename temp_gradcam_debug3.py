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
last_conv = model.get_layer('conv2d_1').output
print('last_conv', last_conv)
grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv, model.outputs[0]])
print('grad_model outputs', grad_model.outputs)
with tf.GradientTape() as tape:
    tape.watch(img_array)
    conv_outputs, predictions = grad_model(img_array)
    loss = tf.reduce_mean(predictions[:,0])
    print('conv_outputs', conv_outputs.shape, type(conv_outputs))
    print('predictions', predictions.shape, type(predictions))
    print('loss', loss)
grads_img = tape.gradient(loss, img_array)
print('grads_img', type(grads_img), None if grads_img is None else grads_img.shape)
with tf.GradientTape() as tape2:
    tape2.watch(conv_outputs)
    loss2 = tf.reduce_mean(predictions[:,0])
grads_conv = tape2.gradient(loss2, conv_outputs)
print('grads_conv', type(grads_conv), grads_conv)
