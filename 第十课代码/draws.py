#-*- coding:utf-8  -*-
'''
Created on 2020年4月3日

@author: Red Mission
'''
import numpy as np
import matplotlib.pyplot as plt
#实际值与有关预测值进行比较
def draws_pre_pharm(Y_test, Y_predict, y_trainPredict, y_train):
 
    plt.figure(figsize=(8, 8), facecolor='white')
    plt.subplot(221) 
    plt.title('oil')
    plt.plot([min(y_train[0]), max(y_train[0])], [min(y_train[0]), max(y_train[0])], 'black', label='y=x')
    plt.scatter(y_train[0], y_trainPredict[0], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[0], Y_predict[0], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
    

    plt.subplot(222)
    plt.title('starch') 
    plt.plot([min(y_train[1]), max(y_train[1])], [min(y_train[1]), max(y_train[1])], 'black', label='y=x')
    plt.scatter(y_train[1], y_trainPredict[1], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[1], Y_predict[1], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')

    
    plt.subplot(223) 
    plt.title('pro')
    plt.plot([min(y_train[2]), max(y_train[2])], [min(y_train[2]), max(y_train[2])], 'black', label='y=x')
    plt.scatter(y_train[2], y_trainPredict[2], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[2], Y_predict[2], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
    
    
    plt.subplot(224) 
    plt.title('water')
    plt.plot([min(y_train[3]), max(y_train[3])], [min(y_train[3]), max(y_train[3])], 'black', label='y=x')
    plt.scatter(y_train[3], y_trainPredict[3], s=20, c='r', marker='o', label='calibration set')
    plt.scatter(Y_test[3], Y_predict[3], s=30, c='b', marker='o', label='test set')
    plt.xlabel('Measured value')
    plt.ylabel(' Predicted value')
   
    plt.tight_layout()
    plt.show()

#从PLS模型中选择最佳潜在变量数的过程
#'oil', 'starch', 'pro', 'water'
def rmsecv_comp_line_pharm(max_components, rmsecv_list):
    
    plt.figure(figsize=(8, 8), facecolor='white')
    plt.subplot(221) 
    plt.title('oil')
    plt.plot(range(1, max_components + 1), rmsecv_list[0], '-o')
    plt.xlabel('num_components')
    plt.ylabel('RMSECV')

    plt.subplot(222)
    plt.title('starch') 
    plt.plot(range(1, max_components + 1), rmsecv_list[1], '-o')
    plt.xlabel('num_components')
    plt.ylabel('RMSECV')
    
    plt.subplot(223) 
    plt.title('pro')
    plt.plot(range(1, max_components + 1), rmsecv_list[2], '-o')
    plt.xlabel('num_components')
    plt.ylabel('RMSECV')   
    
    plt.subplot(224) 
    plt.title('water')
    plt.plot(range(1, max_components + 1), rmsecv_list[3], '-o')
    plt.xlabel('num_components')
    plt.ylabel('RMSECV')
   
    plt.tight_layout()
    plt.show()

def conventional_PCR(title,pc,msecv_list,msec_list):
    plt.figure(figsize=(10,10), facecolor='white')
    num = len(title) #属性个数
    for i in np.arange(0,num):
        pos = 221+i
        plt.subplot(pos) 
        plt.title(title[i])
        plt.plot(pc, np.sqrt(msecv_list[i][:]), '-ob', label = "RMSECV")
        plt.plot(pc, np.sqrt(msec_list[i][:]), '-or', label = "RMSEC")
        plt.xlabel('Measured value')
        plt.ylabel('Predicted value')
        plt.xlabel("Principal Components")
        plt.ylabel("RMSE")
        plt.legend()
   
        plt.tight_layout()
    plt.show()
        
def PCR_pre_pharm(title,y_list,z_list,predicted_list,r2cv_list):
    plt.figure(figsize=(10,10), facecolor='white')
    num = len(title) #属性个数
    for i in np.arange(0,num):
        pos = 221+i
        fig = plt.subplot(pos) 
        plt.scatter(y_list[i], predicted_list[i], c='red', edgecolors='k')
        plt.plot(y_list[i], z_list[i][1]+z_list[i][0]*y_list[i], c='blue', linewidth=1)
        plt.plot(y_list[i], y_list[i], color='green', linewidth=1)
        plt.title('{0} Conventional PCR -- $R^2$ (CV): {1:.2f}'.format(title[i],r2cv_list[i]))
        plt.xlabel('Measured '+title[i])
        plt.ylabel('Predicted '+title[i])

        plt.tight_layout()
    plt.show()
    
def IPLS_bar(a,title,intervals,mse_list):
    plt.figure(figsize=(8,8), facecolor='white')
    num = len(title) #属性个数
    for i in np.arange(0,num):
        pos = 221+i
        fig = plt.subplot(pos)
        plt.bar(np.arange(1, intervals + 1), mse_list[i], color='bgry')
        plt.xlabel(title[i]+" intervals")
        plt.ylabel("mse")
        plt.title(a)
        plt.tight_layout()
    plt.show()
    
    
    
    