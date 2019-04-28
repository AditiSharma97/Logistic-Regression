import pandas as pd
import numpy as np


def sigmoid (z):
	return 1.0/(1.0 + np.exp(-z))

def predict_label (X, threshold = 0.5):
	return X >= threshold

def loss (y_predict, y_actual):
	return (-(y_actual* np.log(y_predict)) -((1-y_actual)*np.log(1-y_predict)))

file = "data_logistic.txt"
data = pd.read_csv(file, header = None)
inputDataSize = 1372
noOfTrainingExamples =  960
noOfTestingExamples = 412
x_train = data.iloc[206:206+noOfTrainingExamples, 0:4].values
y_train = data.iloc[206:206+noOfTrainingExamples, 4].values
x_test = data.iloc[0:206, 0:4].values
x_test_remaining = data.iloc[1166:inputDataSize, 0:4].values
x_test = np.concatenate((x_test, x_test_remaining), axis = 0)
y_test = data.iloc[0:206, 4].values
y_test_remaining = data.iloc[1166:inputDataSize, 4].values
y_test = np.concatenate((y_test, y_test_remaining), axis = 0)

intercept = np.ones((x_train.shape[0], 1))
x_train = np.concatenate ((intercept, x_train), axis = 1)
intercept = np.ones((x_test.shape[0], 1))
x_test = np.concatenate ((intercept, x_test), axis = 1)

learning_rate = 0.01
noOfEpochs = 100

w = np.zeros(5)
for i in range (noOfEpochs):
	z = np.dot (x_train, w)
	h = sigmoid(z)
	gradient = np.dot(x_train.T, (h - y_train))/y_train.size
	#gradient = np.dot(x_train.T, (h - y_train))	
	w -= learning_rate * gradient
	#print (w)
	#print ("loss")
	#print (loss (h, y_train))
print ("W")
print (w)
y_pred_by_algo = np.dot (x_train, w)
y_pred_by_algo = predict_label (y_pred_by_algo)
#print ("predictions")
#print (y_pred_by_algo)

count = 0
y_pred = np.dot (x_test, w)
y_pred = predict_label (y_pred)
for i in range (y_pred.size):
	if y_pred[i] != y_test[i]:
		count = count + 1

acc = (y_pred.size - count) / y_pred.size
print ("Accuracy")
print (acc) 

#Applying regularization
w_reg = np.zeros(5)
lamb = 0.01
for i in range (noOfEpochs):
	z = np.dot (x_train, w_reg)
	h = sigmoid(z)
	gradient_reg = np.dot(x_train.T, (h - y_train)) + (lamb*w_reg)
	w_reg -= learning_rate * gradient_reg

print ("W")
print (w_reg)
count_reg = 0
y_pred_reg = np.dot (x_test, w_reg)
y_pred_reg = predict_label (y_pred_reg)
for i in range (y_pred_reg.size):
	if y_pred_reg[i] != y_test[i]:
		count_reg = count_reg + 1

acc_reg = (y_pred_reg.size - count_reg) / y_pred_reg.size
print ("Accuracy")
print (acc_reg) 
