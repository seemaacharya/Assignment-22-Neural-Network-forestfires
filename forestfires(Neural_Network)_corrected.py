# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 19:58:16 2021

@author: DELL
"""
#Importing the libraries
import pandas as pd
import numpy as np

#loading the dataset
forest=pd.read_csv("forestfires.csv")
forest.head()


#removing month and day columns
forest.pop('month')
forest.head()
forest.pop('day')
forest.head()

#checking the missing valaues
forest.isna().sum()
#There are no missing values in the dataset

#preprocessing (to convert the 'size_category' column into 0 and 1)
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
forest['size_category'] = label_encoder.fit_transform(forest['size_category'])

forest.head()

#setting x(independent features) and y (target)
x=forest.iloc[:,0:28]
y=forest.iloc[:,28]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#using StnadardScaler 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)

#using Neural Network-ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#initializing ANN
model=Sequential()

#Adding input and 1st hidden layer
model.add(Dense(units=10,activation="relu",kernel_initializer='he_uniform',input_dim=28 ))
#adding 2nd hidden layer
model.add(Dense(units=8,activation='relu',kernel_initializer='he_uniform'))
#adding 3rd hidden layer
model.add(Dense(units=1,activation='sigmoid',kernel_initializer='glorot_uniform'))

#Compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the model
history=model.fit(x_train,y_train,validation_split=0.33,epochs=150,batch_size=10)

#predict the model
y_pred=model.predict(x_test)
y_pred

#evaluating the model
scores=model.evaluate(x_test,y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#accuracy: 88.46%

#Visualize training history
#list all the data in history
model.history.history.keys()

#summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

#Summarize the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()












