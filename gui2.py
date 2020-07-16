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

import sys,getopt
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

###################################################
import os
import tkinter as tk
from PIL import ImageTk,Image 
import tkinter.filedialog as fd

class MyApp():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("sKiN DiSeAsE DeTecTion")
        self.cbutton= tk.Button(self.root, text="STEP#2 SHOW", command=self.imgfn)
        self.bbutton= tk.Button(self.root, text="STEP#1 IMPORT", command=self.browsefn)
        self.bbutton.grid(row=0, column=0, padx=(50, 50), pady=(25, 10) )
        self.cbutton.grid(row=1, column=0, padx=(50, 50), pady=(10, 10))
        l2 = tk.Label(self.root, fg='Red' , text="## The Result is : ##")
        self.dbutton= tk.Button(self.root, text="STEP#3 PREDICT", command=self.loadImages)
        self.dbutton.grid(row=3, column=0, padx=(50, 50), pady=(25, 10) )
        l2.grid(row=4, column=0, padx=(100, 100), pady=(10, 10))
        
    
    def browsefn(self):
        self.path= fd.askopenfilename(filetypes=[("Image File",'.png'),("Image File",'.jpg')])
        if len(self.path) is 0:
            os._exit(os.EX_OK)

    def imgfn(self):
        self.image = Image.open(self.path)
        self.image.thumbnail((115,115), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.image)
        self.canvas = tk.Canvas(self.root,width = 150, height = 150)
        self.canvas.grid(row=2, column=0, padx=(100, 100), pady=(10,10))
        self.canvas.create_image(20,20,anchor=tk.NW, image=self.img)
        self.canvas.image = self.img
    
    def loadImages(self):
        self.img = image.load_img(self.path,target_size=(224, 224))
        self.img_data = image.img_to_array(self.img)
        self.img_data = np.expand_dims(self.img_data, axis=0)
        self.img_data = tensorflow.keras.applications.mobilenet.preprocess_input(self.img_data)
        self.features = np.array(model.predict(self.img_data))
        self.indice_var = self.features.argmax(axis=-1)
        self.result = class_labels[self.indice_var[0]]
        l3 = tk.Label(self.root, text=self.result)
        l3.grid(row=5, column=0, padx=(100, 100), pady=(5, 25))

app = MyApp()
app.root.mainloop()