import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin
from pdb import set_trace

class Perceptron(BaseEstimator, ClassifierMixin):
    """Perceptron classifier"""
    def __init__(self, max_iter=50, lern_rate=0.2, random_state=None):
        super(Perceptron, self).__init__()
        self.max_iter = max_iter
        self.lern_rate = lern_rate
        self.random_state = random_state

    def fit(self, X, y):
        set_trace()
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        self.w = random_state.rand(n_features)

        for i in range(self.max_iter):
            idx = random_state.randint(0, n_samples)
            x, expected = X[idx,:], y[idx]
            d = np.dot(x, self.w)
            error = expected - self.step_func(d)
            #if the error is zero (1-1 or 0-0), then there is no update in w
            #if the error is one (1-0), it means that the prediction was zero
            #if the erros is minus one (0-1), it means that the prediction was one
            self.w += self.lern_rate * error * x
            print(error)
        return self

    @classmethod
    def step_func(self, d):
        offset = 0 if d < 0 else 1
        return offset

    def predict(self, X):
        v = np.dot(X, self.w)
        y = np.array([self.step_func(d) for d in v])
        return y

if __name__ == '__main__':
    X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
    y = np.array([0,1,1,1])

    cls = Perceptron()
    cls.fit(X, y)
    print(cls.predict(X))