from tensorflow.keras.preprocessing.image import ImageDataGenerator

g = ImageDataGenerator().flow_from_directory('model/dataset/Training')
print(g.class_indices)
