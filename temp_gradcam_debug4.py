import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

path = glob.glob('model/dataset/Testing/*/*')[0]
model = load_model('model/tumor_model.h5')
img = image.load_img(path, target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)
img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
last_conv = model.get_layer('conv2d_1').output
grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[last_conv, model.outputs[0]])
conv_outputs, predictions = grad_model(img_tensor)
loss = tf.reduce_mean(predictions[:,0])
print('conv_outputs', conv_outputs.shape)
print('predictions', predictions.shape)
print('loss', loss)
try:
    grads = K.gradients(loss, conv_outputs)
    print('K grads', grads)
except Exception as e:
    import traceback; traceback.print_exc()
