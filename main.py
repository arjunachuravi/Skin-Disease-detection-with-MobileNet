import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


train_path                  = 'train'
valid_path                  = 'val'
test_path                   = 'test'

num_train_samples           = len(os.listdir(train_path+'/'+"measles")) + len(os.listdir(train_path+'/'+"melanoma")) + len(os.listdir(train_path+'/'+"Psoriasis")) + len(os.listdir(train_path+'/'+"ringworm")) 
num_val_samples             = len(os.listdir(test_path+'/'+"measles")) + len(os.listdir(test_path+'/'+"melanoma")) + len(os.listdir(test_path+'/'+"Psoriasis")) + len(os.listdir(test_path+'/'+"ringworm")) 


image_size = 224


datagen = ImageDataGenerator(
    rotation_range          = 180,
    width_shift_range       = 0.1,
    height_shift_range      = 0.1,
    zoom_range              = 0.1,
    horizontal_flip         = True,
    vertical_flip           = True,
    fill_mode               = 'nearest',
    preprocessing_function  = tensorflow.keras.applications.mobilenet.preprocess_input
    )

train_batches = datagen.flow_from_directory(train_path,
                                            target_size=(image_size,image_size),
                                            batch_size=80)

valid_batches = datagen.flow_from_directory(valid_path,
                                            target_size=(image_size,image_size),
                                            batch_size=10)

test_batches = datagen.flow_from_directory(test_path,
                                            target_size=(image_size,image_size),
                                            batch_size=10,
                                            shuffle=False)

#######################################################################################

mobile = tensorflow.keras.applications.mobilenet.MobileNet()
x = mobile.layers[-6].output
x = Dropout(0.25)(x)
predictions = Dense(4, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=predictions)
for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

class_weights={
    0: 3.0,
    1: 1.5,
    2: 2.0,
    3: 2.4,
}

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, 
                             save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
                              
                              
callbacks_list = [checkpoint, reduce_lr]

history = model.fit_generator(train_batches, 
                                class_weight=class_weights,
                                validation_data=valid_batches,
                                epochs=30, verbose=1,
                                callbacks=callbacks_list)


#######################################################################################


val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

model.load_weights('model.h5')

val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches)

print('val_loss:', val_loss)
print('val_cat_acc:', val_cat_acc)
print('val_top_2_acc:', val_top_2_acc)
print('val_top_3_acc:', val_top_3_acc)

predictions = model.predict_generator(test_batches, verbose=1)

#######################################################################################