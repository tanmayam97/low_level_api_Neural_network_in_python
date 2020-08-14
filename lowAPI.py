import numpy as np
def standarize(tr):
    tr_std = (tr-np.mean(tr,axis=0)) / np.std(tr, axis = 0)
    return(tr_std)

def loss_cal(pred, actual):           # this function is used to calculate the loss
    mse = np.mean((pred - actual)**2)
    return(mse)

def forward_pass(samples , coefs , intercept):
    out = np.matmul(samples , coefs) + intercept
    return(out)
# y = mx + c similar to xiwi + b in NN
# intercept here is like a bias...this function replicates what happens during the forward propogation or you can say how
# the model is trained
# this is used for the prediction also.

def graidents(X , y , coefs , intercept):
    y_hat = forward_pass(X , coefs , intercept)
    err = (y_hat - y)
    coefs_grad = np.dot(X.T , err)  # this is like weights and X.T is transpose
    intercept_grad = np.sum(y_hat - y) # this is like bias
    return(coefs_grad , intercept_grad)

import pandas as pd

batch_size = 32 # defining the few hyperparameters
lr = 0.001
decay = 0.05
epochs = 10

lcn = pd.read_csv(r"C:\Users\Tanmay Ambatkar\Documents\DataSets\LungCapData.csv")
lcn = lcn.sample(frac = 1) #to shuffle the data and do the sampling. Instead of this, we can use train_test_split also

lcn.Gender.replace({"male":1 , "female":0}, inplace = True)
lcn.Smoke.replace({"no":0 , "yes":1}, inplace = True)
lcn.Caesarean.replace({"no":0,"yes":1}, inplace = True)

lcn_x = lcn.iloc[: , 1:6]
lcn_y = lcn.iloc[:, 0]

lcn_x = standarize(lcn_x)

lcn_x = np.array(lcn_x)
lcn_y = np.array(lcn_y)

dims = lcn_x.shape[1]
coefs = np.random.normal(size = [dims,1] , loc = 0) # defining some initial value of coefs
intercept = -0.001

for i in range(epochs):
    X_data = standarize(lcn_x)
    y_data = lcn_y
    n = X_data.shape[0] # n = number of rows
    train_loss = []
    val_loss = []
    for j in range(( n // batch_size)+1): # n is number of records, since few records will be left so we add 1
        start = j*batch_size
        end = min((j+1)*32 , n)
        X_train = X_data[start:end]
        y_train = y_data[start:end].reshape(end-start , -1)
        
        coefs_grad , intercept_grad = graidents(X_train , y_train , coefs, intercept)
        coefs -= lr*coefs_grad # you are updating the coefs.. and updation rule is coef = coef - learning rate*gradient
        intercept -= lr*intercept_grad #updating the bias
        
        y_hat_data = forward_pass(X_data , coefs, intercept)
        l = loss_cal(y_hat_data , y_data) # Calculating the Train loss
        train_loss.append(l)
        
    print("Epochs is..",i, 
         "Train loss is..", np.mean(train_loss))
    lr *= 1 -decay # learnign rate*(1-decay)

pred_value = forward_pass(lcn_x , coefs, intercept)
pred_value

pred_act_df = pd.DataFrame({"Predicted":pred_value[:,0], "Actual":lcn_y})
pred_act_df