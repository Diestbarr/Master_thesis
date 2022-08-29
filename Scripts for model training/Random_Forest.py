# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 12:16:31 2022

@author: diest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor

#Loads the dataframe
Dataframe = pd.read_pickle("dataframe_name") #Loads the dataframe named 'dataframe_name'

Name =" " #String containing the name of the selected observable. Fill with the observable's name


#Splits into training data and testing data. Obs stands for the selected observable


q_train, q_test, Obs_train, Obs_test = train_test_split(Dataframe["q2"], Dataframe["Diff"], test_size=0.3)


#Reshapes the q values so that the regressor can read them
q_train = np.array(q_train)
q_train = q_train.reshape(-1,1)

q_test = np.array(q_test)
q_test = q_test.reshape(-1,1)

#Defines the regressor
Regressor = RandomForestRegressor(random_state=42, max_depth=6)

#Training...

Regressor.fit(q_train,Obs_train)

#Evaluates the training

Score = Regressor.score(q_test,Obs_test)

print("The score for the regressor was", Score)

#Predicts and plots the trained model

q2 = np.linspace(0.1,8,2000)

q2= q2.reshape(-1,1)


Predictions = Regressor.predict(q2)



plt.plot(q2, Predictions, label=r'%s' %Name)
plt.ylabel(r'$F_i(q^2)$')
plt.xlabel(r'$q^2(GeV^2)$')
#plt.legend(loc='lower right')
plt.title("Random forest")

#plt.savefig("RF_predictions.png", dpi=200, bbox_inches="tight") #Uncomment to save the predictions plot

plt.show()




plt.plot(Dataframe["q2"], Dataframe["Diff"], 'o', label=r'%s' %Name, markersize=0.2)
plt.plot(q2, Predictions, label="%s" %Name, color='orange')



plt.ylabel(r'$F_i(q^2)$', fontsize=18)
plt.xlabel(r'$q^2(GeV^2)$', fontsize=18)
#plt.legend(loc='best')
#plt.title(label="%s fit vs. Hybrid data" %Name)

#plt.savefig("RF_vs_Data.png", dpi=200, bbox_inches="tight") #Uncomment to save the predictions overlayed with the dataset plot

plt.show()




