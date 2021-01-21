import numpy as np


class FuzzyCluster:
    def predict(self, X, y=None):
        """
        predict membership grad using fuzzy rules
        :param X: [n_samples, n_features]
        :param y: None
        :return: Mem [n_samples, n_clusters]
        """
        variance_ = self.scale * self.variance_
        X = np.array(X, dtype=np.float64)
        if y is not None:
            y = np.array(y, dtype=np.float64)

        assert hasattr(self, 'variance_'), "Model not fitted yet."
        assert hasattr(self, 'center_'), "Model not fitted yet."
        d = -(np.expand_dims(X, axis=2) - np.expand_dims(self.center_.T, axis=0)) ** 2 / (2 * variance_.T)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, np.finfo(np.float64).eps)
        return d / np.sum(d, axis=1, keepdims=True)

    def fit(self, X, y=None):
        pass

    def get_params(self):
        pass

    def set_params(self, **kwargs):
        for p in kwargs:
            setattr(self, p, kwargs[p])

    def x2xp(self, X, y=None, order=1, scale=1.):
        self.scale = scale
        if order == 0:
            return self.predict(X)
        else:
            N = X.shape[0]
            mem = np.expand_dims(self.predict(X), axis=1)
            X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
            X = np.repeat(X, repeats=self.n_cluster, axis=2)
            xp = X * mem
            xp = xp.reshape([N, -1])
            return xp

    def _euclidean_distance(self, X, Y=None):
        """
        return the element-wise euclidean distance between X and Y
        :param X: [n_samples_X, n_features]
        :param Y: if None, return the element-wise distance between X and X, else [n_samples_Y, n_features]
        :return: [n_samples_X, n_samples_Y]
        """
        if Y is None:
            Y = X.copy()
        Y = np.expand_dims(np.transpose(Y), 0)
        X = np.expand_dims(X, 2)
        D = np.sum((X - Y)**2, axis=1)
        return np.sqrt(D)


class FuzzyCMeans(FuzzyCluster):
    def __init__(self, n_cluster, scale=1., m='auto', error=1e-5, tol_iter=200, verbose=0):
        """
        Implantation of fuzzy c-means
        :param n_cluster: number of clusters
        :param m: fuzzy index
        :param error: max error for u_old - u_new to break the iteration
        :param tol_iter: total iteration number
        :param verbose: whether to print loss infomation during iteration
        """
        self.error = error
        self.tol_iter = tol_iter
        self.n_dim = None
        self.verbose = verbose
        self.fitted = False
        self.n_cluster = n_cluster
        self.m = m
        self.scale = scale

        super(FuzzyCMeans, self).__init__()

    def get_params(self, deep=True):
        return {
            'n_cluster': self.n_cluster,
            'error': self.error,
            'tol_iter': self.tol_iter,
            'scale': self.scale,
            'm': self.m,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for p, v in params.items():
            setattr(self, p, v)
        return self

    def fit(self, X, y=None):
        X = np.array(X, dtype=np.float64)
        if y is not None:
            y = np.array(y, dtype=np.float64)

        if self.m == 'auto':
            if min(X.shape[0], X.shape[1]-1) >= 3:
                self.m = min(X.shape[0], X.shape[1]-1) / (min(X.shape[0], X.shape[1]-1) - 2)
            else:
                self.m = 2

        N = X.shape[0]
        self.n_dim = X.shape[1]

        # init U
        U = np.random.rand(self.n_cluster, N)

        self.loss_hist = []
        for t in range(self.tol_iter):
            U, V, loss, signal = self._cmean_update(X, U)
            self.loss_hist.append(loss)
            if self.verbose > 0:
                print('[FCM Iter {}] Loss: {:.4f}'.format(t, loss))
            if signal:
                break
        self.fitted = True
        self.center_ = V
        self.train_u = U
        self.variance_ = np.zeros(self.center_.shape)  # n_clusters * n_dim
        for i in range(self.n_dim):
            self.variance_[:, i] = np.sum(
                U * ((X[:, i][:, np.newaxis] - self.center_[:, i].transpose())**2).T, axis=1
            ) / np.sum(U, axis=1)
        self.variance_ = np.fmax(self.variance_, np.finfo(np.float64).eps)
        return self

    def _cmean_update(self, X, U):
        old_U = U.copy()
        old_U = np.fmax(old_U, np.finfo(np.float64).eps)
        old_U_unexp = old_U.copy()
        old_U = self.normalize_column(old_U)**self.m

        # compute V
        V = np.dot(old_U, X) / old_U.sum(axis=1, keepdims=True)

        # compute U
        dist = self._euclidean_distance(X, V).T  # n_clusters * n_samples
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        loss = (old_U * dist ** 2).sum()
        dist = dist ** (2/(1-self.m))
        dist = np.fmax(dist, np.finfo(np.float64).eps)
        U = self.normalize_column(dist)
        if np.linalg.norm(U - old_U_unexp) < self.error:
            signal = True
        else:
            signal = False
        return U, V, loss, signal

    def normalize_column(self, U):
        return U/np.sum(U, axis=0, keepdims=True)

    def __str__(self):
        return "FCM"

    def fs_complexity(self):
        return self.n_cluster * self.n_dim


