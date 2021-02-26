import numpy as np
from select_interAndLv import select_interAndLv
from pls_split import pls_split
from NIPALS import _NIPALS
from function_cvAndWeight import cv, Weight

class Stack_pls():
    def __init__(self, x_train, x_test, y_train, y_test, components, folds):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test 
        self.components = components 
        self.folds = folds 

    def stackPls(self, start, end, intervals):
        
        '''Choosing the optimal parameters, the optimal interval and the optimal latent variable '''
        demo = select_interAndLv(self.x_train, self.y_train, start, end, intervals)
        
        better_components, better_intervals = demo.select_interAndLv(self.components, self.folds)
        
        #print 'better_components:', better_components,"better_intervals:", better_intervals
        
        ''' Calculate cross validation error based on optimal parameters'''
        
        RMSECV_list = self.get_rmsecv(better_components, better_intervals)

        ''' Find the weight based on the cross validation error'''
        W_mat = Weight(RMSECV_list, better_intervals)     
 
        '''The final RMSEP is calculated''' 
        better_intervals_demo = pls_split(self.x_train, self.y_train)
        
        better_split_list, better_intervals_list = better_intervals_demo.split(better_intervals)
        y_predict = 0
        
        for j in range(better_intervals):
            
            xTrain = better_split_list[j]
            Intervals = better_intervals_list[j]
            xTest = self.x_test[:, Intervals[0]:Intervals[1]]
            
            xtrMean = np.mean(xTrain, axis=0)
            ytrMean = np.mean(self.y_train, axis=0)
            
            better_demo = _NIPALS(better_components)
            coef_list = better_demo.fit(xTrain, self.y_train, better_components)
            coef_B = coef_list[better_components - 1]
            yte_predict = better_demo.predict(xTest, coef_B, xtrMean, ytrMean)
            y_predict = np.add(y_predict, W_mat[j] * yte_predict)
      
        press = np.square(np.subtract(self.y_test, y_predict))
        all_press = np.sum(press, axis=0)
        RMSEP = np.sqrt(all_press / self.x_test.shape[0])
        
        return RMSEP,RMSECV_list,y_predict
          
    
    def get_rmsecv(self, better_components, better_intervals):
        
        rmsecv_list = []
        
        x_train, x_test, y_train, y_test = cv(self.folds, self.x_train, self.y_train)  
        
        for j in range(better_intervals):  
            
            RMSECV = 0                   
            for i in range(self.folds):
                
                better_intervals_demo = pls_split(x_train[i], y_train[i])
                better_split_list, better_intervals_list = better_intervals_demo.split(better_intervals)

                xTrain = better_split_list[j]
                Intervals = better_intervals_list[j]
                xTest = x_test[i][:, Intervals[0]:Intervals[1]]
                
                xtrMean = np.mean(xTrain, axis=0)
                ytrMean = np.mean(y_train[i], axis=0)
                
                better_demo = _NIPALS(better_components)
                coef_list = better_demo.fit(xTrain, y_train[i], better_components)
                coef_B = coef_list[better_components - 1]
                yte_predict = better_demo.predict(xTest, coef_B, xtrMean, ytrMean)
               
                PRESS = np.square(np.subtract(yte_predict, y_test[i]))
                all_PRESS = np.sum(PRESS, axis=0)
                rmsecv = np.sqrt(all_PRESS / y_test[i].shape[0])
                
                RMSECV = RMSECV + rmsecv
            rmsecv_list.append(RMSECV / 10)
        
        
        return rmsecv_list


