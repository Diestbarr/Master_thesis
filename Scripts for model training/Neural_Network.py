# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:03:54 2022

@author: diest
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from tensorflow import keras
from tensorflow.keras import layers, Sequential, callbacks, regularizers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


#Loads the dataframe
Path = pathlib.Path("/directory")#Searches for the dataset pickle file located in /directoru

Dataframe = pd.read_pickle(Path) #Loads the dataframe

Name =" " #String containing the name of the selected observable


#Setting of the hyperparameters

activation = "sigmoid" #Activation function
alpha = 10**(-4) #Size of the regularization parameter
batch_size = 60 #The actual size of every batch will be N*(1-validation_split)/batch_size
epochs = 50 #Number of epochs
loss_func = "mean_squared_error" #Loss function
optimizer = "adam" #Optimizer algorithm
validation_split=0.3 #Proportion of the data used for validation

np.random.seed(42) #Sets the seed for replicability

#Defines the neural network
Net=Sequential() 
Net.add(Dense(100,input_shape=(1,),activation=activation,
              kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha))) 


Net.add(Dense(100,input_shape=(1,),activation=activation, 
              kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))


Net.add(Dense(100,input_shape=(1,),activation=activation, 
              kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))


Net.add(Dense(100,input_shape=(1,),activation=activation, 
              kernel_regularizer=l2(alpha), bias_regularizer=l2(alpha)))

#Net.add(Dense(30,activation="sigmoid")) # second hidden layer: 10 neurons
Net.add(Dense(1,activation="linear")) # output layer: 1 neuron "relu"

# Compile network: (randomly initialize weights, choose advanced optimizer, set up everything!)
Net.compile(loss=loss_func,
              optimizer=optimizer,
		metrics=['mse'])


#Imports the variables
X = Dataframe["q2"]
Y = Dataframe["Diff"]

#Training...
History = Net.fit(X, Y, validation_split=validation_split, epochs=epochs,batch_size=batch_size, verbose=1)


#Shows the validation and training curves
Epochs = np.linspace(0,epochs-1, epochs)


Loss = np.array(History.history['loss'])
Validation = np.array(History.history['val_loss'])


plt.plot(Epochs, Loss, label='Loss', color="blue")
plt.plot(Epochs, Validation, label='Validation', color="orange", linestyle="dashed")
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend(loc='best')

plt.title("Loss functions(reg %s)" %alpha)

#plt.savefig("Loss_vs_validation.jpg", dpi=300) #Uncomment to save the loss vs validation graph

plt.show()


#Predicts and plots the trained model

q2 = np.linspace(0.1,8,2000)


Predictions = Net.predict(q2)


plt.plot(Dataframe["q2"], Dataframe["Diff"], 'bo', label="Synth data", markersize=0.2)
plt.plot(q2, Predictions, label=r"NN fit $\alpha=%s$" %alpha, linewidth=2, color="orange")

plt.ylabel(r'$F_i(q^2)$')
plt.xlabel(r'$q^2(GeV^2)$')
plt.legend(loc='lower right')
plt.title(r'%s' %Name)

#plt.savefig("Predictions.jpg", dpi=300) #Uncomment to save the predictions plot

plt.show()

#Net.save('NN.h5') #Saves the trained model as a .h5 file



