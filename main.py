import tensorflow
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy

def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


# SETTING UP THE GENERATORS 

# IMAGE-GENERATORS ACCEPTS THE ORIGINAL DATA,RANDOMLY TRANSFORMING IT AND RETURN ONLY THE NEW TRANSFORMED DATA. [AUGMENTATION]

train_path                  = 'train'
valid_path                  = 'val'
test_path                   = 'test'


# DIMENSION OF IMAGE 
image_size = 224

# THE IMAGE GENERATOR [KERAS]
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

# MOBILENET ARCHITECTURE 
# MOBILENET IS A CNN ARCHITECTURE FOR IMAGE CLASSIFICATION . IT IS FAST AND CUSTOMIZABLE . 
# USES DEPTH WISE SEPERABLE CONVOLUTION

mobile = tensorflow.keras.applications.mobilenet.MobileNet()

# MODIFY THE MOBILENET

#--EXCLUDE LAST 6 LAYERS
x = mobile.layers[-6].output
#--CREATE A DENSE LAYER FOR PREDICTION OF 4 CLASSES 
x = Dropout(0.25)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)

#--LAST 23 LAYERS WILL BE TRAINED AND FREEZING THE WEIGHTS OF REST
for layer in model.layers[:-23]:
    layer.trainable = False

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
                             metrics=[categorical_accuracy, top_2_accuracy, top_3_accuracy])

# CLASS SENSITIVITY , HERE PSORIASIS CLASS IS THE MOST SENSITIVE

class_weights={
    0: 3.0,   # PSORIASIS
    1: 1.5,   # MEASLES
    2: 2.0,   # MELANOMA
    3: 2.4,   # RINGWORM
}

filepath = "model.h5"

#######################################################################################

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

#---- TESTING THE MODEL ---- #

# model.load_weights('model.h5')

# val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = model.evaluate_generator(test_batches)

# print('val_loss:', val_loss)
# print('val_cat_acc:', val_cat_acc)
# print('val_top_2_acc:', val_top_2_acc)
# print('val_top_3_acc:', val_top_3_acc)

# predictions = model.predict_generator(test_batches, verbose=1)

#######################################################################################
