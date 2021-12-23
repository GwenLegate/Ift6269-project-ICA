import warnings
import numpy as np


class ICA:
    def __init__(self, n_components):
        self.n_components = n_components

    def separate(self, X, method="fastica"):
        if method not in ["fastica", "joint_likelihood"]:
            warnings.warn("Method must be one of \"fastica\" or \"joint_likelihood\". defaulting to fastica...")
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

        # select n_components
        sorted_idx = (np.argsort(eigenvalues).tolist())[::-1]
        truncated = np.zeros((E.shape[0], self.n_components))
        truncated_eivalues = np.zeros(self.n_components)
        for k in range(self.n_components):
            truncated[:, k] = E[:, sorted_idx[k]]
            truncated_eivalues[k] = eigenvalues[sorted_idx[k]]

        D_inv_half = np.diag(truncated_eivalues ** (-0.5))
        X_tilde = ((D_inv_half @ truncated.T) @ X.T).T

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

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __joint_likelihood(self, X):
        # not working yet
        batch_size = 64
        lr = 0.001
        W = np.random.rand(X.shape[1], self.n_components).T
        rng = np.random.default_rng()
        converged = False
        epoch = 0
        while not converged and epoch < 10000:
            W_old = W
            indices = rng.choice(X.shape[0], batch_size, replace=False)
            W_sum = np.zeros_like(W)
            for idx in indices:
                v = 1 - 2.0 * self.sigmoid(W @ X[idx])
                v = np.outer(v, X[idx]) + np.linalg.pinv(W.T)
                W_sum += v
            W = W + lr * (W_sum / batch_size)
            lim = np.linalg.norm(W - W_old)
            if lim < 0.00001:
                converged = True
            epoch += 1
            print("epoch {} \t loss: {}".format(epoch, lim))
        return W @ X.T
import utils
if __name__ == "__main__":
    img1 = utils.load_img('images/mix1.png', False).ravel()
    img2 = utils.load_img('images/mix2.png', False).ravel()
    img3 = utils.load_img('images/mix3.png', False).ravel()
    X = np.row_stack((img1, img2, img3))
    print(X.shape)

    C = 3
    ica = ICA(C)
    S = ica.separate(X.T, "fastica")
    print(S.shape)

    utils.show_img(S[0].reshape((500, 500, 3)))
    utils.show_img(S[1].reshape((500, 500, 3)))
    utils.show_img(S[2].reshape((500, 500, 3)))