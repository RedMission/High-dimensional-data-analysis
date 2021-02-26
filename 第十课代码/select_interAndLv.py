from pls_split import pls_split
import numpy as np
from NIPALS import _NIPALS
from function_cvAndWeight import cv, Weight
 
class select_interAndLv():
    def __init__(self, x_cal, y_cal, start, end, intervals):
        self.x_cal = x_cal
        self.y_cal = y_cal
        self.start = start
        self.end = end
        self.intervals = intervals

    
    def select_interAndLv(self, components, folds): 
 
        x_train, x_test, y_train, y_test = cv(folds, self.x_cal, self.y_cal)  
        
        length = (self.end - self.start) / self.intervals  #  
        if (self.end - self.start) % self.intervals != 0 :
            length = length + 1

        error = np.zeros((components, length))
        
        
        for k in range(folds):
            i2 = 0
            for i in range(self.start, self.end, self.intervals): 
                      
                demo = pls_split(x_train[k], y_train[k])
                split_list, intervals = demo.split(i)                     
                for j in range(0, components):
                    
                    rmsecv_list = []
                    Y_predict = np.zeros((y_test[k].shape[0], i))
                    
                    for i1 in range(i): 
                        better_components = j + 1
                        xTrain = split_list[i1]
                        Intervals = intervals[i1]
                        xTest = x_test[k][:, Intervals[0]:Intervals[1]]
                        
                        xtrMean = np.mean(xTrain, axis=0)
                        ytrMean = np.mean(y_train[k], axis=0)
                
                        better_demo = _NIPALS(better_components)
                        coef_list = better_demo.fit(xTrain, y_train[k], better_components)
                        coef_B = coef_list[better_components - 1]
                        yte_predict = better_demo.predict(xTest, coef_B, xtrMean, ytrMean)

                        Y_predict[:, i1] = yte_predict.ravel()
                        rmsecv = self.rmsecv(yte_predict, y_test[k])
                        rmsecv_list.append(rmsecv)
  
                    W_mat = Weight(rmsecv_list, i)    
 
                    y_predict = 0
                    for i1 in range(i):
                        y_predict = np.add(y_predict, W_mat[i1] * Y_predict[:, i1])
                        
                    y_predict = y_predict.reshape(-1, 1)    
                    RMSECV = self.rmsecv(y_predict, y_test[k]) 
                    
                    error[j, i2] = error[j, i2] + RMSECV
                i2 = i2 + 1
      
        component_op, interval_op = self.get_cv_parameter(error)
        component_op = component_op + 1
        interval_op = self.start + interval_op * self.intervals

        return component_op, interval_op
        
    def rmsecv(self, y_predict, y_measure):
        '''  calculate   RMSECV'''
        PRESS = np.square(np.subtract(y_predict, y_measure))
        all_PRESS = np.sum(PRESS, axis=0)
        RMSECV = np.sqrt(all_PRESS / y_measure.shape[0])
        return  RMSECV


    def get_cv_parameter(self, error):
        min_value = np.amin(error) 
        component, interval = error.shape
 
        for i in range(component):
            for j in range(interval):
                if min_value == error[i, j]:
                    component_op, interval_op = i, j
        
        return component_op, interval_op







