################################################### to get rid of warnings
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().warning('test')
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')
###################################################


import tensorflow
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Dropout


###################################################


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

###################################################

mobile = tensorflow.keras.applications.mobilenet.MobileNet()

x = mobile.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)
for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

model.load_weights('model.h5')

###################################################

class_labels = ["psoriasis","measles","melanoma","ringworm"]

def loadImages(path):
  img = image.load_img(path,target_size=(224, 224))
  img_data = image.img_to_array(img)
  img_data = np.expand_dims(img_data, axis=0)
  img_data = tensorflow.keras.applications.mobilenet.preprocess_input(img_data)
  features = np.array(model.predict(img_data))
  y_classes = features.argmax(axis=-1)
  return y_classes


###################################################


# x = loadImages("ring.jpeg")

# print(class_labels[x[0]])

# x = loadImages("measles.jpg")

# print(class_labels[x[0]])

# x = loadImages("melanoma.jpg")

# print(class_labels[x[0]])

# x = loadImages("psoriasis.jpeg")

# print(class_labels[x[0]])
