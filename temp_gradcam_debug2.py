import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from model.gradcam import generate_gradcam

path = glob.glob('model/dataset/Testing/*/*')[0]
print('PATH', path)
model = load_model('model/tumor_model.h5')
print('MODEL inputs', model.inputs)
print('MODEL outputs', model.outputs)
img = image.load_img(path, target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

from tensorflow.keras import backend as K
from tensorflow import GradientTape

# build grad model
import tensorflow as tf
last_conv = model.get_layer('conv2d_1').output
print('last_conv', last_conv)

try:
    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv, model.outputs[0]])
    print('grad_model inputs', grad_model.inputs)
    print('grad_model outputs', grad_model.outputs)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        print('conv_outputs shape', conv_outputs.shape)
        print('predictions shape', predictions.shape)
        tape.watch(conv_outputs)
        loss = tf.reduce_mean(predictions[:,0])
        print('loss', loss)
    grads = tape.gradient(loss, conv_outputs)
    print('grads', grads)
    print('grads shape', None if grads is None else grads.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
