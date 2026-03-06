import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import warnings
warnings.filterwarnings('ignore')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from tensorflow.keras.utils import plot_model
from glob import glob


train_path = "c:/user/Dell/downloads/waste/DATASET/TRAIN" 
test_path ="c:/user/Dell/download/waste/dATASET/TEST"

x_data= []
y_data=[]

for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array=cv2.imread(file)
        img_array=cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split("/")[-1])

data=pd.DataFrame({'image':x_data,'label':y_data})

from collections import Counter
Counter(y_data)

colors = ['#a0d157','#c48bb8']
counts = data['label'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    counts,
    labels=counts.index,
    startangle=90,
    explode=[0.05]*len(counts),
    autopct='%0.2f%%'
)
plt.show()

plt.figure(figsize=(20,15))
for i in range(9):
    plt.subplot(4,3,(i%12)+1)
    index=np.random.randint(15000)
    plt.title('This image is of {0}'.format(data.label.iloc[index]),
                                             fontdict={'size':20,'weight':'bold'})
    plt.imshow(data.iamge[index])
    plt.tight_layout()

scaler= StanderScaler()
x_train=scaler.fit_transform(x_train)

x_test= scaler.transform(x_test)

className=glob(train_path+"/*")
numberOfClass =len(classNmae)
print("number of class: ",numberOfClass)

model=Sequential()

model.add((Conv2D(32,(3,3),input_shape=(224,224,3))))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Conv2D(128,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass))
model.add(Activation("sigmoid"))


model.compile(loss="binary_crossestropy",optimizer="adam",metrics=["accuracy"])
batch_size=256

model.summary

train_dategen =ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)

train_generator =train_datagen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

test_generator =test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

plt.figure(figsize=[10,6])
plt.plot(hist.history["accuracy"], label="Train acc")
plt.plot(hist.history["val_accuracy"], label="Validation accuracy")
plt.legend()
plt.show()

plt.figure(figsize=[10,6])
plt.plot(hist.history["loss"], label="loss")
plt.plot(hist.history["val_loss"], label="Validation loss")
plt.legend()
plt.show()

def predict_fun(img):
    plt.figure(frigsize=(6,4))
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img =cv2.resize(img,(244,244))
    img =np.reshape(img,[-1,224,224,3])
    result=np.argmax(model.predict(img))
    if result == 0:
        print("\033[94m"+"This image -> Recyclable"+"\033[0m]")
    elif result == 1 :
        print("\033[94"+"This image -> Organic"+"\033[0m")
test_img=cv2.imread("c:/user/Dell/download/waste/dATASET/TEST/0/0_12573.jpg")
predict_func(test_img)

test_img=cv2.imread("c:/user/Dell/download/waste/dATASET/TEST/R/R_10753.jpg")
predict_func(test_img)
