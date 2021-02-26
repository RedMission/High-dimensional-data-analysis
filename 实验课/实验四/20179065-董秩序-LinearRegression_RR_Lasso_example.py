import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from scipy.io import loadmat,savemat    
from sklearn.cross_validation import train_test_split 

if __name__ == "__main__":     
 
    D = loadmat('E:\Documents\DAY\cornmat.mat')
    # print D.keys() 
    X = ('m5', 'mp5','mp6')
    Y = ('oil', 'starch', 'pro', 'water')
    
    for i in X:
        plt.figure(figsize=(9, 7))
        x = D[i]
        print "Measured values versus predicted values of",i,"instrument in corn datase"
        for j in range(4): 
            print "Component:",Y[j]
            y = D[Y[j]][:, 0:1]  

            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

            clf_lr = LinearRegression()
            clf_lr.fit(x_train, y_train)
            y_lr = clf_lr.predict(x_test)
            RMSEP_lr=np.sqrt(np.sum(np.square(np.subtract(y_lr,y_test)),axis=0)/y_test.shape[0])
            print "RMSEP_lr:",RMSEP_lr

            clf_Ridge = RidgeCV()
            clf_Ridge.fit(x_train, y_train)
            y_Ridge = clf_Ridge.predict(x_test)
            RMSEP_Ridge=np.sqrt(np.sum(np.square(np.subtract(y_Ridge,y_test)),axis=0)/y_test.shape[0])
            print "clf_Ridge.alpha_:",clf_Ridge.alpha_," RMSEP_Ridge[0]:", RMSEP_Ridge[0]

            clf_Lasso = LassoCV()
            clf_Lasso.fit(x_train, y_train.ravel())
            y_Lasso = clf_Lasso.predict(x_test).reshape(-1,1)
            RMSEP_Lasso=np.sqrt(np.sum(np.square(np.subtract(y_Lasso,y_test)),axis=0)/y_test.shape[0])
            print "clf_Lasso.alpha_:",clf_Lasso.alpha_," RMSEP_Lasso:", RMSEP_Lasso

            plt.subplot(2, 2, j+1)
            plt.plot(y_test, y_lr, 'bo', label='LinearRegression')
            plt.plot(y_test, y_Ridge, 'ro', label='Ridge')
            plt.plot(y_test, y_Lasso, 'go', label='Lasso')
            plt.plot([min(y_test),max(y_test)], [min(y_test),max(y_test)], color='black')

            plt.xlabel('Measured values')
            plt.ylabel('Predicted values')
            plt.legend(loc='upper left')
            plt.title(Y[j])

        plt.subplots_adjust(hspace = 0.3, wspace = 0.3)
        plt.show()

