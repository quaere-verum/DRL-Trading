import pandas as pd
import numpy as np
try:
    from MathUtils import inverted_factorial
    #MathUtils is a personal C++ library which contains functions not provided by numpy. In particular, there is no
    #vectorized implementation of math.factorial.
except:
    import math
    def inverted_factorial(k: np.ndarray):
        return np.vectorize(math.factorial)(1/k)

def fractional_diff(s, d=1, expansion_limit=25):
    if not isinstance(d, (list, tuple, np.ndarray)):
        d = [d]
    ran = np.arange(expansion_limit+1)
    factors = np.ones(shape=(len(d), expansion_limit+1, expansion_limit))
    for k in range(1, expansion_limit+1):
        factors[:, k, :][:, 0:k] = np.array(d).reshape(-1, 1) - np.arange(k)
    weights = ((((-1)**ran)*inverted_factorial(ran)).reshape(1, -1)*np.prod(factors, axis=-1)).reshape(len(d), expansion_limit+1).T
    if not isinstance(s, pd.DataFrame):
        s = pd.DataFrame(s)
    shifts = np.flip(s.shift(np.arange(expansion_limit+1)).values.reshape(-1, expansion_limit+1, s.shape[1]), axis=1)
    return pd.DataFrame(np.sum(shifts*weights, axis=1), index=s.index, columns=s.columns)