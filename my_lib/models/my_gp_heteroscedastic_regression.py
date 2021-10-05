# Copyright (c) 2012 - 2014 the GPy Austhors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np
from GPy.core import GP
from ..inference.latent_function_inference import exact_gaussian_inference
from .. import likelihoods
from GPy import kern
from GPy import util

class myGPHeteroscedasticRegression(GP):
    """
    Gaussian Process model for heteroscedastic regression

    This is a thin wrapper around the models.GP class, with a set of sensible defaults

    :param X: input observations
    :param Y: observed values
    :param kernel: a GPy kernel, defaults to rbf

    NB: This model does not make inference on the noise outside the training set
    """
    def __init__(self, X, Y, kernel=None, Y_metadata=None):

        Ny = Y.shape[0]

        if Y_metadata is None:
            Y_metadata = {'output_index':np.arange(Ny)[:,None]}
        else:
            assert Y_metadata['output_index'].shape[0] == Ny

        if kernel is None:
            kernel = kern.RBF(X.shape[1]) + kern.Bias(X.shape[1])

        #Likelihood
        likelihood = likelihoods.myHeteroscedasticGaussian(Y_metadata)

        super(myGPHeteroscedasticRegression, self).__init__(X,Y,kernel,likelihood, inference_method=exact_gaussian_inference.myExactGaussianInference(), Y_metadata=Y_metadata)
        
        
    def parameters_changed(self):
        """
        Method that is called upon any changes to :class:`~GPy.core.parameterization.param.Param` variables within the model.
        In particular in the GP class this method re-performs inference, recalculating the posterior and log marginal likelihood and gradients of the model

        .. warning::
            This method is not designed to be called manually, the framework is set up to automatically call this method upon changes to parameters, if you call
            this method yourself, there may be unexpected consequences.
        """
        self.posterior, self._log_marginal_likelihood, self.grad_dict = self.inference_method.inference(self.kern, self.X, self.likelihood, self.Y_normalized, self.mean_function, self.Y_metadata)
        self.likelihood.update_gradients(self.grad_dict['dL_dthetaL'])
        self.kern.rbf.update_gradients_direct(self.grad_dict['dL_dK'][0], self.grad_dict['dL_dK'][1])
        self.kern.bias.update_gradients_full(np.array([self.grad_dict['dL_dK'][2]]), self.X)
        if self.mean_function is not None:
            self.mean_function.update_gradients(self.grad_dict['dL_dm'], self.X)