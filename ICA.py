import warnings
import numpy as np


class ICA:
    def __init__(self, n_components):
        self.n_components = n_components

    def separate(self, X, method):
        if method not in ["fastica", "joint_likelihood"]:
            warnings.warn("Method must be one of \"fastica\" or \"joint_likelihood\"")
            method = "fastica"
        if X.shape[1] < self.n_components:
            warnings.warn(
                "Warning: X.shape[1] must be >= n_components (at least as many observed signals as independent sources).")

        X = self.preprocess(X)

        if method == "fastica":  # based on maximization of negentropy
            S = self.__fastICA(X)
            return S

        if method == "joint_likelihood":
            S = self.__joint_likelihood(X)
            return S

    def preprocess(self, X):
        X = X - np.mean(X, axis=0)
        cov = np.einsum("ni,nj->ij", X, X) / X.shape[0]
        eigenvalues, E = np.linalg.eig(cov)
        D_inv_half = np.diag(eigenvalues ** (-0.5))
        X_tilde = np.einsum('ij,kj->ik', np.dot(E, D_inv_half).dot(E.T), X).T
        return X_tilde

    def __fastICA(self, X):
        N = X.shape[1]  # number of observed signals
        W = np.zeros((X.shape[1], self.n_components)).T
        for p in range(self.n_components):
            converged = False
            wp = np.random.rand(N)[:, np.newaxis]
            wp = wp / np.linalg.norm(wp)
            while not converged:
                wp_old = wp
                wp = (np.tanh(X @ wp).T @ X / X.shape[0]).T - np.mean(
                    (1 - (np.tanh(wp.T @ X.T) ** 2))) * wp
                w_sum = np.zeros_like(wp)
                j = 0
                while j < p:
                    w_sum += ((wp.T @ W[j]) * W[j])[:, np.newaxis]
                    j += 1
                wp = wp - w_sum
                wp = wp / np.linalg.norm(wp)
                lim = np.abs(np.abs(np.dot(wp.squeeze(1), wp_old.squeeze(1))) - 1)
                if lim < 0.001:
                    converged = True
            W[p] = np.squeeze(wp)
        return W @ X.T

    def __joint_likelihood(self, X):
        # not implemented yet
        W = np.zeros((X.shape[1], self.n_components)).T
        return W @ X.T
