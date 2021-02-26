from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict


def simple_regression(X,y):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Fit
    regr.fit(X, y)
    # Calibration校准
    y_c = regr.predict(X)
    # Cross-validation交叉验证
    y_cv = cross_val_predict(regr, X, y, cv=10)
    # Calculate scores for calibration and cross-validation
    #计算分数以进行校准和交叉验证
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculate mean square error for calibration and cross validation
    #计算均方误差以进行校准和交叉验证
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    return(y_cv, score_c, score_cv, mse_c, mse_cv)