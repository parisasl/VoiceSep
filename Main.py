
"""
# Introduction to Gender Voice Recognation with Logistic Regression

# Index of Contents

* [Read Data and Check Features](#1)
* [Adjustment of Label values (male = 1, female = 0)](#2)
* [Data Normalization](#3)
* [Split Operation for Train and Test Data](#4)
* [Matrix creation function for initial weight values](#5)
* [Sigmoid function declaration](#6)
* [Forward and Backward Propogation](#7)
* [Updating Parameters](#8)
* [Prediction with Test Data](#9)
* [Logistic Regression Implementation](#10)
* [Logistic Regression with sklearn](#11)
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#import os
#print(os.listdir("input"))

"""
Read Data and Check Features
"""

data = pd.read_csv("voice.csv")

# Get some information about our data
#data.info()

"""
***Adjustment of Label values (male = 1, female = 0***
* After getting information about data we'll call male as 1 and female as 0***
"""

data.label = [1 if each == "male" else 0 for each in data.label]

#data.info() # now we have label as integer

"""
***Data Normalization***
"""

y = data.label.values # main results male or female
x_data = data.drop(["label"], axis = 1) # prediction components

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values # all data evaluated from 1 to 0

"""
***Split Operation for Train and Test Data***
* Data is splitted for training and testing operations. We'll have %20 of data for test and %80 of data for train after split operation.
"""

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Data Shapes
#print("x_train.shape : ", x_train.shape)
#print("x_test.shape : ", x_test.shape)
#print("y_train.shape : ", y_train.shape)
#print("y_test.shape : ", y_test.shape)

# Transform features to rows (Transpose)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

"""
***Matrix creation function for initial weight values***
"""

def initializeWeightsAndBias(dimension): # according to our data dimension will be 20
    w = np.full((dimension, 1), 0.01) 
    b = 0.0
    return w,b

"""
***Sigmoid function declaration***
"""

def sigmoid(z):
    y_head = (1 / (1 + np.exp(-z)))
    return y_head

"""
***Forward and Backward Propogation***
* Get z values from sigmoid function and calculate loss and cost. 
"""

#x_train.shape[1]

def forward_backward_propogation(w, b, x_train, y_train):
    
    #forward propogation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1] # x_train.shape[1] is for scaling
    
    #backward propogation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] # x_train.shape[1] is for scaling
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # x_train.shape[1] is for scaling
    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
    
    return cost, gradients

"""
***Updating parameters***
* Our purpose is find to optimum weight and bias values using derivative of these values.
"""

def update(w, b, x_train, y_train, learningRate, numberOfIteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(numberOfIteration):
        # make forward and backward propogation and find costs and gradients
        cost,gradients = forward_backward_propogation(w, b, x_train, y_train)
        cost_list.append(cost)
        #lets update
        w = w - learningRate * gradients["derivative_weight"]
        b = b - learningRate * gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) paramters weights and bias
    parameters = {"weight" : w, "bias" : b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation = 'vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

"""
***Prediction with Test Data***
* Prediction using test data which is splitted first.
"""

def predict(w,b, x_test):
    # x_test is an input for forward propogation
    z = sigmoid(np.dot(w.T, x_test) + b)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is Male (y_head = 1)
    # if z is smaller than 0.5, our prediction is Female (y_head = 0)
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1
    
    return Y_prediction

"""
***Logistic Regression Implementation***
"""

def logistic_regression(x_train, y_train, x_test, y_test, learningRate, numberOfIterations):
    dimension = x_train.shape[0] # that is 20 (feature count of data)
    w,b = initializeWeightsAndBias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learningRate, numberOfIterations)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    #print("test accuracy for logistic regression: {} %.".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

#Let's try our model and check costs and prediction results.
logistic_regression(x_train, y_train, x_test, y_test, learningRate = 1, numberOfIterations = 100)

logistic_regression(x_train, y_train, x_test, y_test, learningRate = 1, numberOfIterations = 1000)

"""As you see above, when the iteration is increased, accuracy increasing too.

***Logistic Regression with sklearn***
* Logistic Regression Classification can be done with sklearn library. All codes which are written above correspond to the codes below.
"""

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#lr.fit(x_train.T, y_train.T)
#print("test accuracy logistic using sklearn {}".format(lr.score(x_test.T, y_test.T)))



#########################################################################################################################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', 
                    solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.001,
                    power_t=0.5, max_iter=100, shuffle=True, random_state=None,
                    tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                    nesterovs_momentum=True, early_stopping=False,
                    validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                    epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

#mlp.fit(x_train.T, y_train.T)
#print("test accuracy mlp using sklearn {}".format(mlp.score(x_test.T, y_test.T)))

##########################################################################################################################
from keras.models import Sequential
from keras.layers import Dense
import tensorflow_core.estimator

def mplClassifier(x_train,y_train,x_test,y_test):
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=20, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train.T, y_train.T, epochs=100, batch_size=1000)
    _, accuracy = model.evaluate(x_test.T, y_test.T)
    print('Accuracy: %.2f' % (accuracy*100))
    

#mplClassifier(x_train,y_train,x_test,y_test)

def mplClassifierNeuron(x_train,y_train,x_test,y_test,Neuron):
    # define the keras model
    for i in range(len(Neuron)):
        model = Sequential()
        model.add(Dense(12, input_dim=20, activation='relu'))
        model.add(Dense(Neuron[i], activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train.T, y_train.T, epochs=100, batch_size=1000)
        _, accuracy = model.evaluate(x_test.T, y_test.T)
        print('Accuracy with %f neuron in second layer: %.2f' %(Neuron[i],accuracy*100))
        
    
Neuron = [1,2,4,8,16,32,64]
mplClassifierNeuron(x_train,y_train,x_test,y_test,Neuron)  

from keras.layers import Dropout    
def mplClassifierDropout(x_train,y_train,x_test,y_test,dropout):
    # define the keras model
    for i in dropout:
        model = Sequential()
        model.add(Dropout(i, input_shape=(20,)))
        model.add(Dense(12, input_dim=20, activation='tanh'))
        model.add(Dense(8, activation='tanh'))
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train.T, y_train.T, epochs=100, batch_size=1000)
        _, accuracy = model.evaluate(x_test.T, y_test.T)
        print('Accuracy with %f dropout in input layer: %.2f' %(i,accuracy*100))
    
dropout = [0.0,0.1,0.2,0.3,0.4]
mplClassifierDropout(x_train,y_train,x_test,y_test,dropout)




