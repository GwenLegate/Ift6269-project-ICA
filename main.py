import evaluate
import utils
from ICA import ICA
from ica_mle import ICA_MLE
from infomax_ica import Infomax_ICA
import numpy as np
from scipy.io import wavfile
import time

# sound files
sample_rate1, data1 = wavfile.read('demos/mix1.wav')
sample_rate2, data2 = wavfile.read('demos/mix2.wav')
data_sound = np.row_stack((data1, data2))
sample_rates = np.row_stack((sample_rate1, sample_rate2))

# separate sound files infomax ICA
'''infomax = Infomax_ICA(data_sound, data_sound.shape[0])
S = infomax.ica()
print(S[0, :10])
print(evaluate.correlation(S))'''

# load mixed images and stack them
img1 = utils.load_img('images/mix1.png', False).ravel()
img2 = utils.load_img('images/mix2.png', False).ravel()
img3 = utils.load_img('images/mix3.png', False).ravel()
X = np.row_stack((img1, img2, img3))
print(X.shape)

# infomax ICA image separate
'''start_time = time.time()
infomax = Infomax_ICA(X, X.shape[0])
S = infomax.ica()
print(f'execution time: {time.time() - start_time}')'''
#print(evaluate.correlation(S))

# fast ica image separate
'''C = 3
start_time = time.time()
ica = ICA(C)
S = ica.separate(X.T, "fastica")
print(f'execution time: {time.time() - start_time}')'''


# mle ica image separate

start_time = time.time()
mle_ica = ICA_MLE()
S = mle_ica.sgd(X)
print(f'execution time: {time.time() - start_time}')


# show separated images
utils.show_img(S[0].reshape((500, 500, 3)))
utils.show_img(S[1].reshape((500, 500, 3)))
utils.show_img(S[2].reshape((500, 500, 3)))
