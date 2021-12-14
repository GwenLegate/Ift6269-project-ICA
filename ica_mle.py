from numpy.linalg import LinAlgError
from scipy.io import wavfile
import numpy as np
import utils

class ICA_MLE:

    def __init__(self):
        self.max_iter = 500
        self.stop_criteria = 1E-7
        self.lambda_min = 0.01

    def sigmoid(self, z):
        # Numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def tanh(self, z):
        return np.tanh(z / 2)

    def tanh_grad(self, z):
        return (1. - (z ** 2)) / 2.

    def newton_loss(self, Y, W):

        samples = Y[1].shape
        log_det = np.linalg.slogdet(W)[1]
        smooth_approx = np.sum(np.abs(Y) - np.log(1. + np.abs(Y)))

        return - log_det + (smooth_approx / samples)

    def newton_grad(self, Y, f_Y):

        components, samples = Y.shape

        return (np.inner(f_Y, Y) / samples) - np.eye(components)

    def compute_hessian(self, Y, df_Y):

        samples = Y[1].shape

        return np.inner(df_Y, Y ** 2) / samples

    def hessian_reg(self, h, lambda_min):

        # Regularizes the hessian approximation h using the constant lambda_min.

        # eigenvalues of the Hessian
        eigenvalues = 0.5 * (h + h.T - np.sqrt((h - h.T) ** 2 + 4.))

        # Regularize
        problematic_locs = eigenvalues < lambda_min
        np.fill_diagonal(problematic_locs, False)
        i, j = np.where(problematic_locs)
        h[i, j] += lambda_min - eigenvalues[i, j]

        return h

    def compute_direction(self, G, h):

        # H^-1 G

        return ((G * h.T) - G.T) / (h * h.T - 1.)

    def linesearch(self, Y, W, direction, initial_loss):

        # Performs a simple backtracking linesearch in the hessian direction.

        components = Y.shape[0]
        W_proj = np.dot(direction, W)
        step = 1.
        success = False
        ls_iteration = 10

        for n in range(ls_iteration):
            new_Y = np.dot(np.eye(components) + step * direction, Y)
            new_W = W + step * W_proj
            new_loss = self.newton_loss(new_Y, new_W)

            if new_loss < initial_loss:
                success = True
                break
            step /= 2.

        return success, new_Y, new_W, new_loss

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
            print("epoch {} \t loss: {}".format(epoch, lim))

        return W @ X_tilde

    def quasi_newton_ica(self, X):

        Y, _ = self.whiten(X)
        N, T = Y.shape
        W = np.eye(N)
        current_loss = self.newton_loss(Y, W)

        for n in range(self.max_iter):

            # Compute the PDF and its derivative
            tanh_Y = self.tanh(Y)
            f_grad = self.tanh_grad(tanh_Y)

            # Compute gradient
            G = self.newton_grad(Y, tanh_Y)

            # Stopping criterion
            gradient_norm = np.linalg.norm(G.ravel(), ord=np.inf)
            if gradient_norm < self.stop_criteria:
                break

            # Compute the approximation
            H = self.compute_hessian(Y, f_grad)

            # Regularize H
            H = self.hessian_reg(H, self.lambda_min)

            # Compute the descent direction
            direction = - self.compute_direction(G, H)

            # Do a line_search in that direction
            success, new_Y, new_W, new_loss = self.linesearch(Y, W, direction, current_loss)

            # If the line search failed, fall back to the gradient
            if not success:
                direction = - G
                _, new_Y, new_W, new_loss = self.linesearch(Y, W, direction, current_loss)

            # Update
            Y = new_Y
            W = new_W
            current_loss = new_loss

        return Y