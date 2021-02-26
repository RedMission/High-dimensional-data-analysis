import numpy as np
from sklearn import cross_validation
from NIPALS import _NIPALS

class Cross_Validation():  # Variable initialization

    def __init__(self, x, y, n_fold, max_components):
        self.x = x
        self.y = y 
        self.n = x.shape[0]
        self.n_fold = n_fold
        self.max_components = max_components

    def cv(self):  # Divide training sets and test sets
        kf = cross_validation.KFold(self.n, self.n_fold)
        x_train = []
        y_train = []
        x_test = [] 
        y_test = []

        for train_index, test_index in kf:
            xtr, ytr = self.x[train_index], self.y[train_index]
            xte, yte = self.x[test_index], self.y[test_index]
            x_train.append(xtr)
            y_train.append(ytr)
            x_test.append(xte)
            y_test.append(yte)

        return x_train, x_test, y_train, y_test

    def predict_cv(self):
        x_train, x_test, y_train, y_test = self.cv()
        y_allPredict = np.ones((1, self.max_components))
        pls = _NIPALS(self.max_components)

        for i in range(self.n_fold):
            y_predict = np.zeros((y_test[i].shape[0], self.max_components))
            x_trainMean = np.mean(x_train[i], axis=0)
            y_trainMean = np.mean(y_train[i], axis=0)
            x_testCenter = np.subtract(x_test[i], x_trainMean)
            list_coef_B = pls.fit(x_train[i], y_train[i], self.max_components)
            for j in range(self.max_components):
                y_pre = np.dot(x_testCenter, list_coef_B[j])
                y_pre = y_pre + y_trainMean
                y_predict[:, j] = y_pre.ravel()
            y_allPredict = np.vstack((y_allPredict, y_predict))
        y_allPredict = y_allPredict[1:]
                                                           
        return y_allPredict, self.y

    def mse_cv(self, y_allPredict, y_measure):

            PRESS = np.square(np.subtract(y_allPredict, y_measure))
            all_PRESS = np.sum(PRESS, axis=0)

            RMSECV = np.sqrt(all_PRESS / self.n)
            min_RMSECV = min(RMSECV)
            comp_array = RMSECV.argsort()
            comp_best = comp_array[0] + 1  

            return RMSECV, min_RMSECV, comp_best
