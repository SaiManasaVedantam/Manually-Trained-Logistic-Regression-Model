"""
This program builds Linear Regression Model for Titanic Dataset without using the Python's LogisticRegression() function.
This is typically helpful to understand the internal optimization techniques rather than using a go-and-grab code from Python.
"""

import math
import warnings
import numpy as np
import random as rd
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# For Stochastic Gradient Descent, this helps to pick samples at random and returns that
def sampleBatches(X, y):
    sample_batch_X = []
    sample_batch_y = []
    
    for i in range(100):
        index = math.ceil(rd.randint(0, len(X)-1))
        sample_batch_X.append(X[index])
        sample_batch_y.append(y[index])
        i += 1
    
    sample_batch_X = np.array(sample_batch_X)
    sample_batch_y = np.array(sample_batch_y)
    
    return sample_batch_X, sample_batch_y


# Finds theta.Xi to get predicted values
def findH(Xi, theta):
    return sigmoid(np.dot(theta.T, Xi))


# Finds sigmoid of X.theta
def sigmoid(XdotTheta):
    return 1 / (math.exp(XdotTheta) + 1)
    

# Finds gradient of J(theta)
def gTheta(X, y, theta):
    thetaSz = len(theta)
    N = len(X)
    gTheta = []

    for row in range(thetaSz):
        total = 0
        for i in range(N):
            Xi = X[i].reshape(len(X[i]), 1)
            yi = y[i].reshape(1, 1)
            total += np.dot(Xi, sigmoid(np.dot(theta.T, Xi)) - yi)
        gTheta = [total/N]

    return np.array(gTheta)[0]
   
    
# Obtains Theta using Gradient Descent algorithm
def gradient_descent(X, y, theta, alpha = 0.000001, n_iter = 50):
    resultTheta = []
    
    for iteration in range(n_iter):
        gradient = gTheta(X, y, theta)
        for i in range(len(gradient)):
            gradient[i] = alpha * gradient[i]
        theta = theta - gradient
        resultTheta = theta.copy()
    
    return resultTheta
               
    
# Obtains Theta using Stochastic Gradient Descent algorithm
def stochastic_gradient_descent(X, y, theta, alpha = 0.000001, n_iter = 50):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    
    for iteration in range(n_iter):
        theta = theta - alpha * gTheta(sample_batch_X, sample_batch_y, theta)
        resultTheta = theta.copy()
    
    return resultTheta


# Obtains Theta using Stochastic Gradient Descent algorithm with momentum
def sgd_momentum(X, y, theta, alpha = 0.000001, n_iter = 20, eta = 0.9):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    velocity = 0
    
    for iteration in range(n_iter):
        velocity = eta * velocity - alpha * gTheta(sample_batch_X, sample_batch_y, theta)
        theta = theta + velocity
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using Stochastic Gradient Descent algorithm with Nesterov momentum
def sgd_nesterov_momentum(X, y, theta, alpha = 0.00001, n_iter = 10, eta = 0.9):
    resultTheta = []
    sample_batch_X, sample_batch_y = sampleBatches(X, y)
    velocity = 0
    
    for iteration in range(n_iter):
        velocity = eta * velocity - alpha * gTheta(sample_batch_X, sample_batch_y, theta + eta * velocity)
        theta = theta + velocity
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using AdaGrad algorithm
def ada_grad(X, y, theta, alpha = 0.00001, n_iter = 200):
    resultTheta = []
    r = 0
    
    for t in range(n_iter):
        r = r + gTheta(X, y, theta) * gTheta(X, y, theta)
        alpha_t = alpha / np.sqrt(r)
        theta   = theta - alpha_t * gTheta(X, y, theta)
        resultTheta = theta.copy()
   
    return resultTheta


# Obtains Theta using Adam algorithm
def adam(X, y, theta, alpha = 0.00001, n_iter = 200, rho_1 = 0.9, rho_2 = 0.999):
    resultTheta = []
    s = 0 # velocity variable in momentum
    r = 0 # stores exponentially delayed summation of squared gradients
    delta = 1e-8 # constant ~ 0
    
    for t in range(n_iter): # t refers to index of itereation
        s       = rho_1 * s + (1 - rho_1) * gTheta(X, y, theta)   # momentum step 
        r       = rho_2 * r + (1 - rho_2) * gTheta(X, y, theta) * gTheta(X, y, theta) # RMSProp step
        s_hat   = s / (1 - rho_1 ** (t+1))  # bias correction
        r_hat   = r / (1 - rho_2 ** (t+1))  # bias correction
        alpha_t = alpha / np.sqrt(r_hat + delta) # RMSProp step to calculate an adpative learning rate
        v       = -1 * alpha_t * s_hat  # calculate velocity
        theta   = theta + v   # update theta
        resultTheta = theta.copy()
        
    return resultTheta


# Processes the categorical data column Age
def process_Age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

    
# main function
if __name__=="__main__":
    # Ignores warnings
    warnings.filterwarnings("ignore")
    
    # Importing train and test data from csv files
    train = pd.read_csv('titanic_train.csv')
    test = pd.read_csv('titanic_test.csv')
    print("\nTrain data shape before pre-processing: ", train.shape)
    print("Test data shape before pre-processing: ", test.shape)

    # Preprocess categorical data - Age
    train['Age'] = train[['Age', 'Pclass']].apply(process_Age, axis = 1)
    test['Age'] = test[['Age', 'Pclass']].apply(process_Age, axis = 1)

    # Cabin: Lots of nulls. So, drop the column
    # Fare has 1 null value and so, we drop row corresponding to that value
    train.drop('Cabin',axis = 1,inplace = True)
    train.dropna(inplace = True)
    test.drop('Cabin',axis = 1,inplace = True)
    test.dropna(inplace = True)

    # We convert categorical features to use logistic regression with data
    # Conversion on train data
    # We are not converting categorical features in test data as we are not using it down here
    sex = pd.get_dummies(train['Sex'], drop_first = True)
    embark = pd.get_dummies(train['Embarked'], drop_first = True)
    train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace = True)
    train = pd.concat([train, sex, embark], axis = 1)

    # As the test data doesn't have info about 'Survived', we cannot club train & test sets to split into 80%-20%
    # So, we split train set itself into 80%-20% and can use the test data to make additional test if needed
    y = train['Survived'].to_numpy()
    y = y.reshape((y.shape[0], 1))
    train = train.drop('Survived', axis = 1)
    X = train.copy().to_numpy() 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 80)
    print("\n------ Train and Test sizes after split: ------")
    print("Train features: ", X_train.shape, "\t Train target: ", y_train.shape)
    print("Test features: ", X_test.shape, "\t Test target: ", y_test.shape)
    
    # Set theta values accordingly between 0 and 1
    thetaList = []
    for i in range(X.shape[1]):
        thetaList.append([0.005])
    theta = np.array(thetaList)
    
    # Gradient Descent on Train & Test sets
    gradientDescent_Train = gradient_descent(X_train, y_train, theta)
    gradientDescent_Test = gradient_descent(X_test, y_test, theta)
    
    # Stochastic Gradient Descent (SGD) on Train & Test sets
    stoc_gradientDescent_Train = stochastic_gradient_descent(X_train, y_train, theta)
    stoc_gradientDescent_Test = stochastic_gradient_descent(X_test, y_test, theta)
    
    # SGD with momentum on Train & Test sets
    SGD_Momentum_Train = sgd_momentum(X_train, y_train, theta)
    SGD_Momentum_Test = sgd_momentum(X_test, y_test, theta)
    
    # SGD with Nesterov momentum on Train & Test sets
    SGD_Nesterov_Train = sgd_nesterov_momentum(X_train, y_train, theta)
    SGD_Nesterov_Test = sgd_nesterov_momentum(X_test, y_test, theta)
    
    # AdaGrad on Train & Test sets
    adaGrad_Train = ada_grad(X_train, y_train, theta)
    adaGrad_Test = ada_grad(X_test, y_test, theta)
    
    # Adam on Train & Test sets
    adam_Train = adam(X_train, y_train, theta)
    adam_Test = adam(X_test, y_test, theta)
    
    
    # Now we find train & test errors for each algorithm using theta values obtained
    # Gradient Descent algorithm - Train set
    print("\n----- EVALUATION USING GRADIENT DESCENT -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(gradientDescent_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # Gradient Descent algorithm - Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(gradientDescent_Test, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))
    
    
    # Stochastic Gradient Descent algorithm - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(stoc_gradientDescent_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # Stochastic Gradient Descent algorithm - Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(stoc_gradientDescent_Test, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))
    
    
    # Stochastic Gradient Descent algorithm with momentum - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT WITH MOMENTUM -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(SGD_Momentum_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # Stochastic Gradient Descent algorithm with momentum - Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(SGD_Momentum_Test, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))
    
    
    # Stochastic Gradient Descent algorithm with Nesterov momentum - Train set
    print("\n----- EVALUATION USING STOCHASTIC GRADIENT DESCENT WITH NESTEROV MOMENTUM -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(SGD_Nesterov_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # Stochastic Gradient Descent algorithm with Nesterov momentum- Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(SGD_Nesterov_Train, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))
    
    
    # Stochastic Gradient Descent algorithm with Adagrad - Train set
    print("\n----- EVALUATION USING ADAGRAD -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(adaGrad_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # Stochastic Gradient Descent algorithm with Adagrad- Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(adaGrad_Train, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))
    
    
    # With Adam algorithm - Train set
    print("\n----- EVALUATION USING ADAM -----")
    print("For Train Set: \n")
    y_pred_for_train = []
    for i in range(len(X_train)):
        if findH(adam_Train, X_train[i]) > 0.7:
            y_pred_for_train.append(1)
        else:
            y_pred_for_train.append(0)
    y_pred_for_train = np.array(y_pred_for_train)
    print(classification_report(y_pred_for_train, y_train))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_train, y_train))
    
    # With Adam algorithm - Test set
    print("For Test Set: \n")
    y_pred_for_test = []
    for i in range(len(X_test)):
        if findH(adam_Train, X_test[i]) > 0.7:
            y_pred_for_test.append(1)
        else:
            y_pred_for_test.append(0)
    y_pred_for_test = np.array(y_pred_for_test)
    print(classification_report(y_pred_for_test, y_test))
    print("Overall Accuracy: ", accuracy_score(y_pred_for_test, y_test))