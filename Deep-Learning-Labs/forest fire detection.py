import tensorflow as tf
from tensorflow.keras.oreprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten , Dense, Dropout

import tkinter as tk
from tkinter import filedialog
from PIL import Image,ImageTk
import numpy as np

train_dir = r"c:\User\user\Desktop\download\datasets\forest fire classification data\train"
valid_dir= r"c:\User\user\Desktop\download\datasets\forest fire classification data\valid"
test_dir=  r"c:\User\user\Desktop\download\datasets\forest fire classification data\test"


train_datagenm= ImageDataGenerator(rescale=1./255)
valid_datagenm= ImageDataGenerator(rescale=1./255)
test_datagenm= ImageDataGenerator(rescale=1./255)

train_generator= train_datagenm.flow_from_directory(train_dir, target_size=(64,64),batch_size=32,class_mode='binary')
valid_generator= valid_datagenm.flow_from_directory(valid_dir, target_size=(64,64),batch_size=32,class_mode='binary')
test_generator= test_datagenm.flow_from_directory(test_dir, target_size=(64,64),batch_size=32,class_mode='binary')

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(train_generator,validation_data=valid_generator,epochs=10,verbose=1)

#function to load and predict an image

def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200,200))
        img = ImageTK.PhotoImage(img)
        image_label.configure(Image=img)
        image_label.image = img

        img_for_model = Image.open(file_path).resize((64,64))
        img_array = np.array(img_for_model) / 255.0
        img_array=np.expand_dims(img_array,axis=0)

        prediction = model.predict(img_array)[0][0]
        result = "wildfire" if prediction > 0.5 else "No Wildfire"
        result_label.config(text="Prediction: "+result)


root = tk.TK()
root.title("forest fire detection")
root.geometry("400x400")

btn = tk.Button(root,text="upload image",command=predict_image)
btn.pack(pady=20)

image_label = tk.Label(root)
image_label.pack()

result_label =tk.Label(root,text="prediction: ",font=("Helvetica",16))
result_label.pack(pady=20)

root.mainloop()
