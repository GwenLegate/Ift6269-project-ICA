from numpy.linalg import LinAlgError
from scipy.io import wavfile
import numpy as np
import utils

class ICA_MLE:

    def __init__(self):
        self.max_iter = 1000
        self.stop_criteria = 1E-6 #0.007 I used this stop criteria to get the image separation to converge

    def sigmoid(self, z):
        # Numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def tanh(self, z):
        return np.tanh(z)

    def whiten(self, X):
        X_mean = X - X.mean(axis=1).reshape((-1, 1))
        cov = np.dot(X_mean, X_mean.T) / X_mean.shape[1]
        eigenvals, eigenvects = np.linalg.eigh(cov)
        D = np.diag(np.sqrt(eigenvals))
        D_inv = np.linalg.inv(D)

        # whitening array
        whiten = D_inv.dot(eigenvects.T)
        whitened_X = whiten.dot(X_mean)

        # array to un-whiten data after ICA
        undo_whiten = np.dot(D, eigenvects)
        return whitened_X, undo_whiten

    def log_likelihood(self, W, X):
        # compute log-likelihood

        # avoid negative value in logarithm
        det_W = np.linalg.det(W)

        if det_W >= 0:
            return (np.log(self.sigmoid(W @ X) @ (1 - self.sigmoid(W @ X).T)) + np.log(det_W)).sum()
        else:
            return (np.log(self.sigmoid(W @ X) @ (1 - self.sigmoid(W @ X).T))).sum()

    def sgd(self, X):

        components, samples = X.shape
        W = np.eye(components)

        X_tilde, _ = self.whiten(X)

        ll_old = 0
        ll_new = 0
        old_loss = 0
        new_loss = 0

        anneal = [0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005, 0.0002, 0.0002, 0.0001, 0.0001,
                  0.00005, 0.00005, 0.00002, 0.00002, 0.00001, 0.00001]
        print('Separating components ...')

        converged = False
        epoch = 0

        while not converged and epoch < self.max_iter:
            W_old = W
            old_loss = new_loss

            for alpha in anneal:
                # avoid singular matrix
                try:
                    W = W + alpha * (((1 - 2 * self.sigmoid(W @ X_tilde)) @ X_tilde.T) + np.linalg.inv(W.T))
                except LinAlgError:
                    W = W + alpha * (((1 - 2 * self.sigmoid(W @ X_tilde)) @ X_tilde.T) + np.linalg.pinv(W.T))

            ll_old = ll_new
            ll_new = self.log_likelihood(W, X_tilde)
            new_loss = abs(ll_new - ll_old)

            diff = abs(new_loss - old_loss)
            lim = np.linalg.norm(W - W_old)

            if diff < self.stop_criteria:
                converged = True

            epoch += 1
            #print("epoch {} \t loss: {}".format(epoch, lim))

        return W @ X_tilde