# Copyright (c) 2012-2014, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPy.inference.latent_function_inference.posterior import PosteriorExact as Posterior
from GPy.util.linalg import pdinv, dpotrs, tdot
from GPy.util import diag
import numpy as np
from GPy.inference.latent_function_inference import LatentFunctionInference
log_2_pi = np.log(2*np.pi)

class ExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        return input_dict

    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)

        if variance is None:
            variance = likelihood.gaussian_variance(Y_metadata)

        YYT_factor = Y-m

        if K is None:
            K = kern.K(X)

        Ky = K.copy()
        diag.add(Ky, variance+1e-8)

        Wi, LW, LWi, W_logdet = pdinv(Ky)

        alpha, _ = dpotrs(LW, YYT_factor, lower=1)
#         print('Ky: ', Ky)
#         print('Y.size: ', Y.size)
#         print('Y.shape[1]: ', Y.shape[1])
#         print('W_logdet: ', W_logdet)
#         print('alpha: ', alpha)
#         print('YYT_factor: ', YYT_factor)
#         print('LW: ', LW)
        log_marginal =  0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor))

        if Z_tilde is not None:
            # This is a correction term for the log marginal likelihood
            # In EP this is log Z_tilde, which is the difference between the
            # Gaussian marginal and Z_EP
            log_marginal += Z_tilde

        dL_dK = 0.5 * (tdot(alpha) - Y.shape[1] * Wi)

        dL_dthetaL = likelihood.exact_inference_gradients(np.diag(dL_dK), Y_metadata)

        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL, 'dL_dm':alpha}

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO
    
class myExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        return input_dict
    
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)
            
        kernel = kern.copy()

        if variance is None:
            variance = likelihood.variance * likelihood.common_variance

        YYT_factor = Y-m
        
        from autograd import grad, elementwise_grad
        import autograd.numpy as np
        import autograd.scipy as sc
        
        per = 1/likelihood.variance.copy()**0.5
        per /= np.sum(per)
        
        def jitchol(A):
            diagA = np.diag(A)
            jitter = diagA.mean() * 1e-6
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        
        def make_cov(X , v):
            ret = np.zeros((len(X), len(X)))
            for i in range(len(X)):
                for j in range(i, len(X)):
                    ret[i, j] = np.sqrt(np.sum(np.abs((X[i]-X[j])**2)))
                    ret[j, i] = ret[i, j].copy()
            return v[0]*np.exp(-0.5*(ret/v[1])**2)+v[2]
        
        def pdinv(A):
            L = jitchol(A)
            logdet = 2.*np.sum(np.log(np.diag(L)))
            return L, logdet
        
        def dpotrs(A, B, lower=1):
            rtn = sc.linalg.solve_triangular(A, B, lower=lower)
            rtn = sc.linalg.solve_triangular(A.T, rtn, lower=not(lower))
            return rtn
        
        def object_func_for_auto_grad(v):
            Ky = make_cov(X.reshape(-1), v[1:])+(np.diag(np.squeeze(likelihood.variance))*np.diag(np.squeeze(np.full(len(likelihood.variance), v[0]))))+np.diag(np.full(len(likelihood.variance), 1e-5))
            #diag.add(Ky, (v[0]*variance)+1e-8)
            LW, W_logdet = pdinv(Ky)
            alpha = dpotrs(LW, YYT_factor, lower=1)
            log_marginal = 0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor*per))
            
            return log_marginal
        
        def getValues(v):
            K = make_cov(X.reshape(-1), v[1:])
            Ky = K.copy()+(np.diag(np.squeeze(likelihood.variance))*np.diag(np.squeeze(np.full(len(likelihood.variance), v[0]))))+np.diag(np.full(len(likelihood.variance), 1e-5))
            diag.add(Ky, (v[0]*variance)+1e-8)
            LW, W_logdet = pdinv(Ky)
            alpha = dpotrs(LW, YYT_factor, lower=1)
            return LW, alpha, K
        
        v = np.array([likelihood.common_variance, kernel[0], kernel[1], kernel[2]])
        #print(v)
        
        LW, alpha, K = getValues(v)

        log_marginal =  object_func_for_auto_grad(v)
        #print(log_marginal)
        
        grad_object_func = grad(object_func_for_auto_grad)
        
        dv = grad_object_func(v)
        #print(dv)
            
        dL_dthetaL = dv[0]
        
        dL_dK = dv[1:]
        #print(dL_dK, dL_dthetaL)
        import numpy as np
        import scipy as sc

        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO
    
    
    
    
    
class yourExactGaussianInference(LatentFunctionInference):
    """
    An object for inference when the likelihood is Gaussian.

    The function self.inference returns a Posterior object, which summarizes
    the posterior.

    For efficiency, we sometimes work with the cholesky of Y*Y.T. To save repeatedly recomputing this, we cache it.

    """
    def __init__(self):
        pass#self._YYTfactor_cache = caching.cache()

    def to_dict(self):
        """
        Convert the object into a json serializable dictionary.

        Note: It uses the private method _save_to_input_dict of the parent.

        :return dict: json serializable dictionary containing the needed information to instantiate the object
        """

        input_dict = super(ExactGaussianInference, self)._save_to_input_dict()
        input_dict["class"] = "GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference"
        return input_dict
    
    def inference(self, kern, X, likelihood, Y, mean_function=None, Y_metadata=None, K=None, variance=None, Z_tilde=None):
        """
        Returns a Posterior class containing essential quantities of the posterior
        """

        if mean_function is None:
            m = 0
        else:
            m = mean_function.f(X)
            
        kernel = kern.copy()

        if variance is None:
            variance = likelihood.variance * likelihood.common_variance

        YYT_factor = Y-m
        
        from autograd import grad, elementwise_grad
        import autograd.numpy as np
        import autograd.scipy as sc
        
        per = 1/likelihood.variance.copy()**0.5
        per /= np.sum(per)
        
        def jitchol(A):
            diagA = np.diag(A)
            jitter = diagA.mean() * 1e-6
            L = np.linalg.cholesky(A + np.eye(A.shape[0]) * jitter)
            return L
        
        def make_cov(X , v):
            ret = np.zeros((len(X), len(X)), dtype=np.float)
            for i in range(len(X)):
                for j in range(i, len(X)):
                    ret[i, j] = np.sqrt(np.sum(np.abs((X[i]-X[j])**2)))
                    ret[j, i] = ret[i, j].copy()
            return v[0]*np.exp(-0.5*(ret/v[1])**2)+v[2]
        
        def pdinv(A):
            L = jitchol(A)
            logdet = 2.*np.sum(np.log(np.diag(L)))
            return L, logdet
        
        def dpotrs(A, B, lower=1):
            rtn = sc.linalg.solve_triangular(A, B, lower=lower)
            rtn = sc.linalg.solve_triangular(A.T, rtn, lower=not(lower))
            return rtn
        
        def object_func_for_auto_grad(v):
            lh = np.zeros(len(v[:-3]))
            print(lh[0])
            print(v[0]._value)
            lh[0] = v[0]
            for i in range(1, len(lh)):
                lh[i] = v[i]+lh[i-1]
            Ky = make_cov(X.reshape(-1), v[1:])+(np.diag(np.squeeze(lh))+np.diag(np.full(len(likelihood.variance), 1e-5)))
            #diag.add(Ky, (v[0]*variance)+1e-8)
            LW, W_logdet = pdinv(Ky)
            alpha = dpotrs(LW, YYT_factor, lower=1)
            log_marginal = 0.5*(-Y.size * log_2_pi - Y.shape[1] * W_logdet - np.sum(alpha * YYT_factor*per))
            
            return log_marginal
                                                 
        
        def getValues(v):
            lh = np.zeros(len(v[:-3]))
            lh[0] = v[0]
            for i in range(1, len(lh)):
                lh[i] = v[i]+lh[i-1]
            K = make_cov(X.reshape(-1), v[1:])
            Ky = K.copy()+(np.diag(np.squeeze(lh))+np.diag(np.full(len(likelihood.variance), 1e-5)))
            #diag.add(Ky, (v[0]*variance)+1e-8)
            LW, W_logdet = pdinv(Ky)
            alpha = dpotrs(LW, YYT_factor, lower=1)
            return LW, alpha, K
        
                           
        v = []
        for i in range(len(likelihood.variance)):
            v.append(likelihood.variance[i])
        v.append(kernel[0])
        v.append(kernel[1])
        v.append(kernel[2])
        v = np.array(v)
        #print(v)
        
        LW, alpha, K = getValues(v)

        log_marginal =  object_func_for_auto_grad(v)
        #print(log_marginal)
        
        grad_object_func = grad(object_func_for_auto_grad)
        
        dv = grad_object_func(v)
        #print(dv)
            
        dL_dthetaL = dv[:-3]
        
        dL_dK = dv[-3:]
        #print(dL_dK, dL_dthetaL)
        import numpy as np
        import scipy as sc

        return Posterior(woodbury_chol=LW, woodbury_vector=alpha, K=K), log_marginal, {'dL_dK':dL_dK, 'dL_dthetaL':dL_dthetaL}

    def LOO(self, kern, X, Y, likelihood, posterior, Y_metadata=None, K=None):
        """
        Leave one out error as found in
        "Bayesian leave-one-out cross-validation approximations for Gaussian latent variable models"
        Vehtari et al. 2014.
        """
        g = posterior.woodbury_vector
        c = posterior.woodbury_inv
        c_diag = np.diag(c)[:, None]
        neg_log_marginal_LOO = 0.5*np.log(2*np.pi) - 0.5*np.log(c_diag) + 0.5*(g**2)/c_diag
        #believe from Predictive Approaches for Choosing Hyperparameters in Gaussian Processes
        #this is the negative marginal LOO
        return -neg_log_marginal_LOO