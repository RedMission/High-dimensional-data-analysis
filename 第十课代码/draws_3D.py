#-*- coding:utf-8  -*-
'''
Created on 2020年4月3日

@author: Red Mission
'''
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

#参数选择的误差面
def draws_RMSECV(F,rmsecv_list):
    fig = plt.figure(figsize=(12,12))
    num = len(F)
    for i in np.arange(0,num):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_xlim(2,10)
        ax.set_ylim(0,16)

        ax.set_xlabel('interval')
        ax.set_ylabel('component')
        ax.set_zlabel('RMSECV')

        #The maximum number of latent variable of PLS is set to 15, 
        #the number of interval is set in the range [2, 12], and the interval is 2. 
        X = np.arange(2,12,2)
        Y = np.arange(1,16,1)

        X, Y = np.meshgrid(X, Y)#生成网格点坐标矩阵
        Z = np.array(rmsecv_list[i])

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        ax.set_zlim(0.00, 0.2)

        ax.set_title(F[i])

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax.view_init(elev=30,azim=30) #旋转视角
    plt.show()

def draws_pre_pharm(Y_test, Y_predict,y_train):
    plt.figure(figsize=(8, 8), facecolor='white')
    plt.subplot(221) 
    plt.title('oil')
    plt.plot([min(y_train[0]), max(y_train[0])], [min(y_train[0]), max(y_train[0])], 'black', label='y=x')
    plt.scatter(Y_test[0], Y_predict[0], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
    

    plt.subplot(222)
    plt.title('starch') 
    plt.plot([min(y_train[1]), max(y_train[1])], [min(y_train[1]), max(y_train[1])], 'black', label='y=x')
    #plt.scatter(y_train[1], y_trainPredict[1], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[1], Y_predict[1], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')

    
    plt.subplot(223) 
    plt.title('pro')
    plt.plot([min(y_train[2]), max(y_train[2])], [min(y_train[2]), max(y_train[2])], 'black', label='y=x')
    #plt.scatter(y_train[2], y_trainPredict[2], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[2], Y_predict[2], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
    
    
    plt.subplot(224) 
    plt.title('water')
    plt.plot([min(y_train[3]), max(y_train[3])], [min(y_train[3]), max(y_train[3])], 'black', label='y=x')
    #plt.scatter(y_train[3], y_trainPredict[3], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[3], Y_predict[3], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
   
    plt.tight_layout()
    plt.show()
