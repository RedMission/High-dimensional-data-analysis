import numpy as np
def _nipals_twoblocks_inner_loop(X, Y, max_iter=500, tol=1e-06,):  
    y_score = Y[:, [0]] 
    x_weights_old = 0
    ite = 1
    while True:
        # 1.1 Update u: the X weights
        # regress each X column on y_score
# w=X.T*Y[:,0]/||Y[:,0]||
        x_weights = np.dot(X.T, y_score) / np.dot(y_score.T, y_score) 
        # 1.2 Normalize u
# w=w/||w||
        x_weights /= np.sqrt(np.dot(x_weights.T, x_weights)) 
 # 1.3 Update x_score: the X latent scores
       # t=X*w
        x_score = np.dot(X, x_weights) 
        # 2.1  regress each Y column on x_score
# q=Y*t/(t.T*t)
        y_weights = np.dot(Y.T, x_score) / np.dot(x_score.T, x_score) 

        # 2.2 Update y_score: the Y latent scores
        # u=Y*q/(q.T,q)
        y_score = np.dot(Y, y_weights) / np.dot(y_weights.T, y_weights) 
        x_weights_diff = x_weights - x_weights_old
        if np.dot(x_weights_diff.T, x_weights_diff) < tol :
            break

        if ite == max_iter:
            warnings.warn('Maximum number of iterations reached')
            break
        x_weights_old = x_weights
        ite += 1

    return x_weights, y_weights 
def _center_xy(X, Y):      

    # center
    x_mean = X.mean(axis=0)
    X_center = np.subtract(X, x_mean)
    y_mean = Y.mean(axis=0)
    Y_center = np.subtract(Y, y_mean)

    return X_center, Y_center, x_mean, y_mean

class _NIPALS():      
    def __init__(self, n_components, max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components 
        self.max_iter = max_iter         
        self.tol = tol
        self.copy = copy

    def fit(self, X, Y, n_components):    
        n = X.shape[0]
        p = X.shape[1]
        q = Y.shape[1]

        if n != Y.shape[0]:    
                'Incompatible shapes: X has %s samples, while Y '
                'has %s' % (X.shape[0], Y.shape[0])
        if self.n_components < 1 or self.n_components > p: 
            raise ValueError('invalid number of components')

        Xcenter, Ycenter, self.x_mean_, self.y_mean_ = _center_xy(X, Y) 
        # Residuals (deflated) matrices
        Xk = Xcenter
        Yk = Ycenter
        # Results matrices
        self.x_scores_ = np.zeros((n, self.n_components))  
        self.y_scores_ = np.zeros((n, self.n_components))  
        self.x_weights_ = np.zeros((p, self.n_components)) 
        self.y_weights_ = np.zeros((q, self.n_components)) 
        self.x_loadings_ = np.zeros((p, self.n_components)) 
        self.y_loadings_ = np.zeros((q, self.n_components))
 
        # NIPALS algo: outer loop, over components
        for k in range(self.n_components):
            x_weights, y_weights = _nipals_twoblocks_inner_loop(
                    X=Xk, Y=Yk, max_iter=self.max_iter, tol=self.tol,)
            # compute scores
            x_scores = np.dot(Xk, x_weights) 
            y_ss = np.dot(y_weights.T, y_weights)
            y_scores = np.dot(Yk, y_weights) / y_ss  
            x_loadings = np.dot(Xk.T, x_scores) / np.dot(x_scores.T, x_scores)
            # - substract rank-one approximations to obtain remainder matrix
            Xk -= np.dot(x_scores, x_loadings.T)

            y_loadings = (np.dot(Yk.T, x_scores) / np.dot(x_scores.T, x_scores))
            Yk -= np.dot(x_scores, y_loadings.T)
            self.x_scores_[:, k] = x_scores.ravel()  # T    
            self.y_scores_[:, k] = y_scores.ravel()  # U   
            self.x_weights_[:, k] = x_weights.ravel()  # W   
            self.y_weights_[:, k] = y_weights.ravel()  # C   
            self.x_loadings_[:, k] = x_loadings.ravel()  # P 
            self.y_loadings_[:, k] = y_loadings.ravel()  # Q 
           
        lists_coefs = []              
        for i in range(n_components):   
            self.x_rotations_ = np.dot(self.x_weights_[:, :i + 1], 
                                       np.linalg.inv(np.dot(self.x_loadings_[:, :i + 1].T, self.x_weights_[:, :i + 1])))
            self.coefs = np.dot(self.x_rotations_, self.y_loadings_[:, :i + 1].T)
             
            lists_coefs.append(self.coefs)
        
        return lists_coefs 

    def predict(self, x_test, coefs_B, xtr_mean, ytr_mean):

        xte_center = np.subtract(x_test, xtr_mean)
        y_pre = np.dot(xte_center, coefs_B)
        y_predict = np.add(y_pre, ytr_mean)          
 
        return y_predict




























