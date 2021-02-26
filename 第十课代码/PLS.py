from cross_validation import Cross_Validation
from NIPALS import _NIPALS
import numpy as np
 
class PLS():
    def __init__(self, x_train, y_train, x_test, y_test, n_fold=10, max_components=10):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_fold = n_fold
        self.max_components = max_components
                                            
    def pls(self):
        # Select the optimal principal component number
        pls_cv = Cross_Validation(self.x_train, self.y_train, self.n_fold, self.max_components)
        y_allPredict, y_measure = pls_cv.predict_cv()
        RMSECV, min_RMSECV, comp_best = pls_cv.mse_cv(y_allPredict, y_measure)
        #  Modeling by optimal principal component number
        pls = _NIPALS(comp_best)
        List_coef_B = pls.fit(self.x_train, self.y_train, comp_best)
        coef_B = List_coef_B[comp_best - 1] 
        
        x_trainMean = np.mean(self.x_train, axis=0)
        y_trainMean = np.mean(self.y_train, axis=0)
        y_trainPredict = pls.predict(self.x_train, coef_B, x_trainMean, y_trainMean)
        # compute RMSEC
        press = np.square(np.subtract(self.y_train, y_trainPredict))
        all_press = np.sum(press, axis=0)
        RMSEC = np.sqrt(all_press / self.x_train.shape[0])
         # compute RMSEP
        y_predict = pls.predict(self.x_test, coef_B, x_trainMean, y_trainMean)
        press = np.square(np.subtract(self.y_test, y_predict))
        all_press = np.sum(press, axis=0)
        RMSEP = np.sqrt(all_press / self.x_test.shape[0])
        
        return RMSECV, min_RMSECV, comp_best, RMSEC, RMSEP, y_trainPredict, y_predict