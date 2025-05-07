import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
import cv2
mnist=tf.keras.datasets.mnist ## handwritten character size image 28x28 0 to 9
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape

plt.imshow (x_train[0])
plt.show() #to show graph
plt.imshow (x_train[0],cmap = plt.cm.binary) #to show in binary -- white 0 black 255
print (x_train [0])
x_train = tf.keras.utils.normalize(x_train, axis= 1)
x_test = tf.keras.utils.normalize(x_test, axis= 1)
plt.imshow (x_train[0],cmap = plt.cm.binary)
print (x_train[0])
print (y_train[0])

IMG_SIZE=28
x_trainr=np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE,1)
x_testr=np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE,1)
print ("training Samples dimension",x_trainr.shape)
print ("testing Samples dimension",x_testr.shape)

model = Sequential()

## First Layer
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

## 2nd Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

## 3rd Layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

## Fully Connected Layer 1
model.add (Flatten())
model.add (Dense(64))
model.add(Activation("relu")) 

## Fully Connected Layer 2
model.add (Dense(32))
model.add(Activation("relu")) 

## Fully Connected Layer 3
model.add(Dense(10))
model.add(Activation('softmax')) 
model.summary()
print ("total training sample",len(x_trainr))
model.fit (x_trainr,y_train,epochs=5, validation_split= 0.3,batch_size=1)
test_loss, test_acc = model.evaluate(x_testr, y_test,batch_size=1)
print ("Test Loss on 10000 test samples", test_loss)
print ("Validation Accuracy on 10000 test samples",test_acc)
predicions = model.predict([x_testr],batch_size=1)
print (predicions)
print (np.argmax(predicions[65]))
plt.imshow(x_test[565])

img = cv2.imread('C:/abc/four.png')
img = Image.open('C:/abc/four.png')
plt.imshow(img)
img_array = np.array(img)
img_shape = img_array.shape
print("Shape of the image:", img_shape)
img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
gray_shape = gray.shape
print("Shape of the image:", gray_shape)
resize = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

plt.imshow(resize, cmap='gray')
plt.show()


resize_shape = resize.shape
print("Shape of the resized image:", resize_shape)
newing = np.array(resize).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print("Shape of the reshaped image:", newing.shape)
predicions=model.predict(newing)
print(np.argmax(predicions))
