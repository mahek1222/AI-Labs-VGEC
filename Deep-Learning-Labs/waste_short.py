
import tesnsorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#set up paths to the train and test directories

base_dir= r"c:\User\user\Desktop\content sessions\computer vision\Datasets\waste Classification data\DATASET"
train_dir = base_dir+ r"\train"
test_dir = base_dir + r"\test"

#Use ImageDataGenerator to preapre the data for training and testing 
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale =1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

test_data = train_datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size = 32,
    class_mode ='binary'
)

#Display sample images from each class
plt.figure(figureSize=(10,5))
for i in range(4):
    image,label=train_data.next()
    plt.subplot(1,4,i + 1)
    plt.imshow(image[0])
    plt.title("class :" + ("oraganic" if label[0] == 0 else "Recyclable"))
    plt.axis('off')
plt.suptitle("sample Images from each class")
plt.show()

mobilenet_model = tf.keras.applications.MobileNetvV2(input_shape=(244,244,3),include_top=False,weights='imagenet')
mobilenet_model.trainable=False

#add custom layers for classification
model = tf.keras.Sequential([
    mobilenet_model,
    tf.keras.layer.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(Optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

history=model.fit(
    train_data,
    validation_data=test_data,
    epochs=2
)

model.save('mobilenet_waste_classifier.h5')

loss,accuracy = model.evaluate(test_data)
print("Model Accuracy:",accuracy)

from sklearn.metrics import classification_report, confusion_matrix

#Gnerate Predictioms
y_pred_prob=model.predict(test_data).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)
y_true = test_data.classes

print("Classification report: \n",classification_report(y_true,y_pred,target_names=['Organic','Recyclable']))

conf_matrix= confusion_matrix(y_true,y_pred_prob)
print("confusion Matrix:\n",conf_matrix)

#ROC Curve and AUC
fpr,tpr, _ =roc_curve(y_true,y_pred_prob)
roc_auc = auc(fpr,tpr)

plt.figure()
plt.plot(fpr,tpr,color='darkorange', lw =2, label=f'ROC curve (AUC={roc_auc:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.show()

import tesnsorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

#path to the trained model and test image 

model_path='mobilenet_waste_classifier.h5'
test_image_path = r"c:\User\user\Desktop\Edunet_contet_Usha\\computer Vision\Datasets\waste Classification data\DATASET\DATASET\TEST\R\R_27"

model= tf.keras.models.load_model(model_path)

test_img=  load_img(test_image_path,target_size=(224,224))
test_img_array=img_to_array(test_img)/255.0
test_img_array=np.expand_dims(test_img_array,axis=0)

predicted_prob=model.predict(test_img_array)[0][0]
predicted_class='Organic' if predicted_prob < 0.5 else 'No Organic'

plt.imshow(test_img)
plt.title(f"Predicted :{predicted_class}")
plt.axis('off')
plt.show()
