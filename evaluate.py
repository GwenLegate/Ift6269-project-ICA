import numpy as np

# normalized correlation of each signal combo
def correlation(S):
    r, _ = S.shape
    for i in range(r - 1):
        for j in range(i + 1, r):
            correlation = np.sum(S[i] * S[j]) / np.sqrt(np.sum(S[i] ** 2) * np.sum(S[j] ** 2))
            print(f'correlation between components {i + 1} and {j + 1} is {correlation}')