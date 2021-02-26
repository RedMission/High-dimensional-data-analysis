import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import cross_validation

def Weight(RMSECV, INTERVAL):
    s = []
    sum_s = 0
    for i in range(INTERVAL):
        s.append(1.0 / RMSECV[i])
        sum_s = sum_s + s[i] * s[i]
 
    Weight_mat = np.zeros((INTERVAL, 1))
    for i in range(INTERVAL):
        w = s[i] * s[i] / sum_s
        Weight_mat[i] = w.ravel()
    Weight_minmax = MinMaxScaler((0, 1)).fit_transform(Weight_mat)  #
    Weight_minmax_sum = np.sum(Weight_minmax)
    W_mat = Weight_minmax / Weight_minmax_sum
    return W_mat


def cv(n_fold, x, y):  #  
        kf = cross_validation.KFold(x.shape[0], n_fold)
     
        x_train = []
        y_train = []
        x_test = [] 
        y_test = []
        for train_index, test_index in kf:
            xtr, ytr = x[train_index], y[train_index]
            xte, yte = x[test_index], y[test_index]
            x_train.append(xtr)
            y_train.append(ytr)
            x_test.append(xte)
            y_test.append(yte)
        return x_train, x_test, y_train, y_test







