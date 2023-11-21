import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        
        return 1.0 / (1.0 + np.exp(-x))
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def fit(self, X, y, lr, tol=1e-7, max_iter=1e5):
        """
        Fit the regression coefficients via gradient descent or other methods 
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        - losses: list, a list of loss values at each iteration.        
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.matrix(np.zeros(X.shape[1])).T
        
        # List to store loss values at each iteration
        losses = []

        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient 
        # 2. Apply the update rule
        # 3. Check for convergence
        L = 100
        for i in range(int(max_iter)):
            L_before = L

            p = self.sigmoid(X.dot(self.coef_))
            grad = np.dot((p - y).T, X / X.shape[0])

            if self.penalty == 'l1':
                grad = grad + self.gamma * np.sign(self.coef_.T) / X.shape[1]
            elif self.penalty == 'l2':
                grad = grad + self.gamma * self.coef_.T

            grad = grad * lr

            self.coef_ -= grad.T    

            p = self.sigmoid(X.dot(self.coef_))
            L = -np.sum(y.T @ np.log(p) + (1 - y.T) @ np.log(1 - p)) / X.shape[0]

            if self.penalty == 'l1':
                L += self.gamma * np.sum(np.abs(self.coef_)) / X.shape[1]
            elif self.penalty == 'l2':
                L += self.gamma * np.sum(np.square(self.coef_)) / 2

            losses.append(L)

            if np.abs(L_before - L) <= tol:
                break                   

        return losses

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        
        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)
        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################

        return self.sigmoid(linear_output)
        
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
