import glob
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from model.gradcam import generate_gradcam

path = glob.glob('model/dataset/Testing/*/*')[0]
model = load_model('model/tumor_model.h5')
img = image.load_img(path, target_size=(150,150))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)
try:
    hm = generate_gradcam(model, img_array, 'conv2d_1')
    print('HEATMAP', type(hm), getattr(hm, 'shape', None), hm.min(), hm.max())
except Exception as e:
    import traceback
    traceback.print_exc()
