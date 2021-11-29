from scipy.io import wavfile
import numpy as np
import scipy
import evaluate
import utils


class Infomax_ICA:
    def __init__(self, X, n_components):
        self.n_components = n_components
        self.X = X
        self.max_iter = 500
        self.stop_criteria = 1E-6

    # decorrelates the data --> make covariance 0 between signals.  Returns whitened data and a matrix to undo the whitening after the ICA
    def whiten(self):
        X_mean = self.X - self.X.mean(axis=1).reshape((-1, 1))
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

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def update_W(self, W, whitened_X, bias, lrate):
        r, c = whitened_X.shape
        block1 = int(np.floor(np.sqrt(c / 3)))
        permute = np.random.permutation(c)
        for i in range(0, c, block1):
            if i + block1 < c:
                tt2 = i + block1
            else:
                tt2 = c
                block1 = c - i
            # unmix a signal chunk from i:tt2
            unmixed = np.dot(W, whitened_X[:, permute[i:tt2]]) + bias
            logit = 1-2*self.sigmoid(unmixed)
            #logit = 1 - (2 / (1 + np.exp(-unmixed)))
            W = W + lrate * np.dot(block1 * np.eye(r) + np.dot(logit, unmixed.T), W)
            bias = bias + lrate * logit.sum(axis=1).reshape(bias.shape)
            # Checking if W blows up
            if (np.isnan(W)).any() or np.max(np.abs(W)) > 1E8:
                print("Numeric error! restarting with lower learning rate")
                lrate = lrate * 0.9
                W = np.eye(r)
                bias = np.zeros((r, 1))
                error = 1

                if lrate > 1e-6 and np.linalg.matrix_rank(whitened_X) < r:
                    print("Data 1 is rank defficient"
                          ". I cannot compute " +
                          str(r) + " components.")
                    return (None, None, None, 1)

                if lrate < 1e-6:
                    print("Weight matrix may"
                          " not be invertible...")
                    return None, None, None, 1
                break
            else:
                error = 0

        return (W, bias, lrate, error)

    # whitened_X=AS.  Returns S, the independent signals
    def infomax(self, whitened_X):
        # init values
        r, _ = whitened_X.shape
        W = np.eye(r)
        W_old = np.eye(r)
        bias = np.zeros((r, 1))
        W_change = np.zeros(r)
        W_change_old = np.zeros(r)
        lrate = 0.001 / np.log(r)
        change = 1
        angle_delta = 0
        step = 1
        error = 0

        while step < self.max_iter and change > self.stop_criteria:
            W, bias, lrate, error = self.update_W(W, whitened_X, bias, lrate)
            #print(f'step: {step}, change: {change}')
            W_change = W - W_old
            change = np.linalg.norm(W_change.ravel()) ** 2

            if step > 2:
                rad_angle_delta = np.arccos(np.sum(W_change * W_change_old)
                                        / (np.linalg.norm(W_change.ravel()) * np.linalg.norm(W_change_old.ravel()) + 1e-8))
                angle_delta = rad_angle_delta * 180 / np.pi

            W_old = np.copy(W)

            if angle_delta > 60:
                lrate = lrate * 0.9
            W_change_old = np.copy(W_change)

            step += 1

        # extracted signal components
        return np.dot(W, whitened_X)

    def ica(self):
        whitened_X, unwhiten_X = self.whiten()
        sources = self.infomax(whitened_X)
        scale = sources.std(axis=1).reshape((-1, 1))
        sources = sources / scale

        return sources

if __name__ == "__main__":
    sample_rate1, data1 = wavfile.read('demos/mix1.wav')
    sample_rate2, data2 = wavfile.read('demos/mix2.wav')
    data = np.row_stack((data1, data2))
    sample_rates = np.column_stack((sample_rate1, sample_rate2))
    #utils.plot_signal(data1, sample_rate1, signal_name='mix1')
    infomax_ica = Infomax_ICA(data, data.shape[0])

    S = infomax_ica.ica()
    print(S[0, :10])
    print(evaluate.correlation(S))
    #print(np.sum(S[0]*S[1]) / np.sqrt(np.sum(S[0]**2) * np.sum(S[1]**2)))

    #utils.plot_signal(whitened_X.T[0], sample_rate1, signal_name='mix1 whitened')

