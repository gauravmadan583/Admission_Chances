import math
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(X):
    X=np.exp(X)
    X=X/(X+1)
    return X

def costFunction(theta,X,y):
    m=X.shape[0]
    J=( y.transpose() @ np.log(sigmoid(X@theta)) + (1-y).transpose() @ np.log(1-sigmoid(X@theta)) )
    grad=X.transpose()@(sigmoid(X@theta)-y)/m
    return -J/m,grad

def gradientDescent(theta,X,y,iter):
    m=X.shape[0]
    J_history=np.zeros((iter))
    alpha=1
    for i in range(iter):
        J_history[i],grad=costFunction(theta,X,y)
        theta-=alpha*grad

    return theta,J_history

def plotpoints(X,y):
    X_pos=np.where(y==1)
    X_neg=np.where(y==0)
    for i in X_pos:
        plt.scatter(X[i,1:2],X[i,2:3],color="blue")
    for i in X_neg:
        plt.scatter(X[i,1:2],X[i,2:3],color="red")

    plt.show()
    return

# def plotBoundary(X,y):

def featureNormalize(X):
    X_norm=X
    m=X.shape[0]
    mu=np.sum(X,axis=0)/m
    diff=np.max(X,axis=0)-np.min(X,axis=0)
    X_norm=(X-mu)/diff
    return X_norm

f=open("ex2data1.txt","r")
f1=f.readlines()
count=0
for i in f1:
    count+=1

X=np.zeros((count,2))
y=np.zeros((count,1))
for i in range(count):
    X[i][0],X[i][1],y[i][0]=f1[i].split(",")


X=featureNormalize(X)

X=np.concatenate((np.ones((count,1)),X),axis=1)
plotpoints(X,y)
theta=np.zeros((X.shape[1],1))

theta,J_history=gradientDescent(theta,X,y,400)

print(costFunction(theta,X,y))
Y=sigmoid(X@theta)>0.5

# plotBoundary(X,Y)
plotpoints(X,Y)
plt.show()
plt.plot(J_history)
plt.show()
