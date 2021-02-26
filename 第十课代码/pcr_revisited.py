from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from PCR_simple_regression import simple_regression
import numpy as np
def pcr_revisited(X,y, pc):
    ''' Step 1: PCA on input data'''
    # Define the PCA object
    pca = PCA()
    # Preprocess (2) Standardize features by removing the mean and scaling to unit variance
    #预处理（2）通过去除均值并缩放到单位方差来标准化特征
    Xstd = StandardScaler().fit_transform(X)
    # Run PCA producing the reduced variable Xreg and select the first pc components
    #运行PCA产生减少的变量Xred并选择第一个pc组件
    Xpca = pca.fit_transform(Xstd)
    
    # Define a correlation array
    # 定义一个相关数组
    corr = np.zeros(Xpca.shape[1])
    # Calculate the absolute value of the correlation coefficients for each PC
    # 计算每个PC的相关系数的绝对值
    for i in range(Xpca.shape[1]):
        corr[i] = np.abs(np.corrcoef(Xpca[:,i], y)[0, 1])
    # Sort the array based on the corr values and select the last pc values 
    # 根据corr值对数组进行排序，然后选择最后一个pc值
    Xreg = (Xpca[:,np.argsort(corr)])[:,-pc:]
    
    ''' Step 2: regression on selected principal components'''
    y_cv, score_c, score_cv, mse_c, mse_cv = simple_regression(Xreg, y)
    return(y_cv, score_c, score_cv, mse_c, mse_cv)