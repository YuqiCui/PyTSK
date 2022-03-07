import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array


class BaseFuzzyClustering(object):
    def set_params(self, **params):
        """
        Setting attributes. Implemented to adapt the API of scikit-learn.
        """
        for p, v in params.items():
            setattr(self, p, v)
        return self


def __get_fuzzy_index__(m, N, D):
    if m == 'auto' and min(N, D - 1) >= 3:
        m_ = min(N, D - 1) / (min(N, D - 1) - 2)
    elif isinstance(m, float) or isinstance(m, int):
        m_ = float(m)
    else:
        m_ = 2
        print("Warning: auto set does not satisfied min(N, D - 1) >= 3, "
              "min(N, D - 1) = {}, setting m to 2".format(min(N, D - 1)))
    return m_


def __normalize_column__(X):
    X = np.fmax(X, np.finfo(np.float64).eps)
    return X / np.sum(X, axis=0, keepdims=True)


def __fcm_update__(data, u_old, m, dist="euclidean"):
    u_old = __normalize_column__(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    um = u_old ** m

    cntr = np.dot(um, data) / um.sum(axis=1, keepdims=True)
    d = cdist(data, cntr, metric=dist).T
    d = np.fmax(d, np.finfo(np.float64).eps)

    loss = np.sum(um * d ** 2)
    u = __normalize_column__(d ** (2 / (1 - m)))
    return cntr, u, loss, d


def __fcm_predict__(data, cntr, m, dist="euclidean"):
    d = cdist(data, cntr, metric=dist).T
    d = np.fmax(d, np.finfo(np.float64).eps)
    u = __normalize_column__(d ** (2 / (1 - m)))
    return u


def x2xp(X, U, order):
    """
    Convert matrix :math:`X` and :math:`U` into the consequent input matrix :math:`X_p`

    :param numpy.array x: size: :math:`[N,D]`. Input features.
    :param numpy.array u: size: :math:`[N,R]`. Corresponding membership degree matrix.
    :param int order: 0 or 1. The order of TSK models.
    :return: If :code:`order=0`, return :math:`U` directly, elif :code:`order=1`,
        return the matrix :math:`X_p` with the size of :math:`[N, (D+1)\times R]`.
        Details can be found at [2].
    [2] Wang S, Chung K F L, Zhaohong D, et al. Robust fuzzy clustering neural network
        based on ɛ-insensitive loss function[J]. Applied Soft Computing, 2007, 7(2): 577-584.
    """
    assert order == 0 or order == 1, "Order can only be 0 or 1."
    R = U.shape[1]
    if order == 1:
        N = X.shape[0]
        mem = np.expand_dims(U, axis=1)
        X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
        X = np.repeat(X, repeats=R, axis=2)
        xp = X * mem
        xp = xp.reshape([N, -1])
        return xp
    else:
        return U


def compute_variance(X, U, V):
    """

    Convert the feature matrix :math:`X` and the membership degree matrix :math:`U`
    into the consequent input matrix :math:`X_p`

    Each row in :math:`X\in \mathbb{R}^{N,D}` represents a :math:`D`-dimension input
    vector. Suppose vector :math:`\mathbf{x}` is one row, and then the consequent input
     matrix :math:`P` is computed as [5] for a first-order TSK:

    .. math::
        &\mathbf{x}_e = (1, \mathbf{x}),\\
        &\tilde{\mathbf{x}}_r = u_r \mathbf{x}_e,\\
        &\mathbf{p} = (\tilde{\mathbf{x}}_1, \tilde{\mathbf{x}}_2, ...,\tilde{\mathbf{x}}_R),

    where :math:`\mathbf{p}` is the corresponding row in :math:`P`, which is a
    :math:`(D+1)\times R`-dimension vector. Then the consequent parameters of TSK can
    be optimized by any linear regression algorithms.


    :param numpy.array x: size: :math:`[N,D]`. Input features.
    :param numpy.array u: size: :math:`[N,R]`. Corresponding membership degree matrix.
    :param int order: 0 or 1. The order of TSK models.
    :return: If :code:`order=0`, return :math:`U` directly, else if :code:`order=1`,
        return the matrix :math:`X_p` with the size of :math:`[N, (D+1)\times R]`.
        Details can be found at [2].

    [2] `Wang S, Chung K F L, Zhaohong D, et al. Robust fuzzy clustering neural network based
        on ɛ-insensitive loss function[J]. Applied Soft Computing, 2007, 7(2): 577-584.
        <https://www.sciencedirect.com/science/article/pii/S1568494606000469>`_
    """
    R = U.shape[0]
    D = X.shape[1]
    variance = np.zeros([R, D])
    for i in range(D):
        variance[:, i] = np.sum(
            U * ((X[:, i][:, np.newaxis] - V[:, i].T) ** 2).T, axis=1
        ) / np.sum(U, axis=1)
    return variance


class FuzzyCMeans(BaseFuzzyClustering, BaseEstimator, TransformerMixin):
    """
    The fuzzy c-means (FCM) clustering algorithm [1]. This implementation is adopted
    from the `scikit-fuzzy <https://pythonhosted.org/scikit-fuzzy/overview.html>`_ package.
    When constructing a TSK fuzzy system, a fuzzy clustering algorithm is usually used to
    compute the antecedent parameters, after that, the consequent parameters can be computed
    by least-squared error algorithms, such as Ridge regression [2]. How to use this class
     can be found at `Quick start <quick_start.html#training-with-fuzzy-clustering>`_.

    The objective function of the FCM is:

    .. math::
        &J = \sum_{i=1}^{N}\sum_{j=1}^{C} U_{i,j}^m\|\mathbf{x}_i - \mathbf{v}_j\|_2^2\\
        &s.t. \sum_{j=1}^{C}\mu_{i,j} = 1, i = 1,...,N,

    :param int n_cluster: Number of clusters, equal to the number of rules :math:`R` of a TSK model.
    :param float/str fuzzy_index: Fuzzy index of the FCM algorithm, default `auto`. If
        :code:`fuzzy_index=auto`, then the fuzzy index is computed as :math:`\min(N, D-1) / (\min(N, D-1)-2)`
        (If :math:`\min(N, D-1)<3`, fuzzy index will be set to 2), according to [3]. Otherwise the given
        float value is used.
    :param float/str sigma_scale: The scale parameter :math:`h` to adjust the actual standard deviation
        :math:`\sigma` of the Gaussian membership function in TSK antecedent part. If :code:`sigma_scale=auto`,
        :code:`sigma_scale` will be set as :math:`\sqrt{D}`, where :math:`D` is the input dimension [4].
        Otherwise the given float value is used.
    :param str/np.array init: The initialization strategy of the membership grid matrix :math:`U`.
        Support "random" or numpy array with the size of :math:`[R, N]`, where :math:`R` is the number of
        clusters/rules, :math:`N` is the number of training samples. If :code:`init="random"`, the initial
        membership grid matrix will be randomly initialized, otherwise the given matrix will be used.
    :param int tol_iter: The total iteration of the FCM algorithm.
    :param float error: The maximum error that will stop the iteration before maximum iteration is reached.
    :param str dist: The distance type for the :func:`scipy.spatial.distance.cdist` function, default
        "euclidean". The distance function can also be "braycurtis", "canberra", "chebyshev", "cityblock",
        "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski",
        "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
        "sokalmichener", "sokalsneath", "sqeuclidean", "yule".
    :param int verbose: If > 0, it will show the loss of the FCM objective function during iterations.
    :param int order: 0 or 1. Decide whether to construct a zero-order TSK or a first-order TSK.
    """

    def __init__(self, n_cluster, fuzzy_index="auto", sigma_scale="auto",
                 init="random", tol_iter=100, error=1e-6, dist="euclidean",
                 verbose=0, order=1):
        self.n_cluster = n_cluster
        self.fuzzy_index = fuzzy_index
        self.sigma_scale = sigma_scale
        self.init = init
        self.tol_iter = tol_iter
        assert self.tol_iter > 1, "tol_iter must > 1."
        self.error = error
        self.dist = dist
        self.verbose = verbose
        self.order = order

    def get_params(self, deep=True):
        return {
            "n_cluster": self.n_cluster,
            "fuzzy_index": self.fuzzy_index,
            "sigma_scale": self.sigma_scale,
            "init": self.init,
            "tol_iter": self.tol_iter,
            "error": self.error,
            "dist": self.dist,
            "verbose": self.verbose,
            "order": self.order,
        }

    def fit(self, X, y=None):
        """
        Run the FCM algorithm.

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number
            of training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.
        """
        check_array(X, ensure_2d=True)
        N, D = X.shape[0], X.shape[1]

        self.m_ = __get_fuzzy_index__(self.fuzzy_index, N, D)
        self.scale_ = np.sqrt(D) if self.sigma_scale == "auto" else self.sigma_scale
        self.n_features = D

        if self.init == "random":
            u = np.random.rand(self.n_cluster, N)
            u = __normalize_column__(u)
        elif isinstance(self.init, np.ndarray):
            u = __normalize_column__(self.init)
        else:
            raise ValueError("Unsupported init param, must be \"random\" or "
                             "numpy.ndarray of size [n_clusters, n_samples]")
        self.iter_cnt = 0
        cntr = None
        self.loss_hist = []
        for t in range(self.tol_iter):
            uold = u.copy()
            cntr, u, loss, d = __fcm_update__(X, uold, self.m_, self.dist)
            change = np.linalg.norm(uold - u, ord=2)
            self.loss_hist.append(loss)
            if self.verbose > 0:
                print('[FCM Iter {}] Loss: {:.4f}, change: {:.4f}'.format(t, loss, change))
            if change < self.error:
                break
            self.iter_cnt += 1
        self.fitted = True
        self.cluster_centers_ = cntr
        self.membership_degrees_ = u
        self.variance_ = compute_variance(X, self.membership_degrees_, self.cluster_centers_) * self.scale_

    def predict(self, X, y=None):
        """
        Predict the membership degrees of :code:`X` on each cluster.

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number
            of training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.
        :return: return the membership degree matrix :math:`U` with the size of :math:`[N, R]`, where :math:`N`
            is the number of samples of :code:`X`, and :math:`R` is the number of clusters/rules. :math:`U_{i,j}`
            represents the membership degree of the :math:`i`-th sample on the :math:`r`-th cluster.
        """
        u = __fcm_predict__(X, self.cluster_centers_, self.m_, self.dist)
        return u.T

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Compute the membership degree matrix :math:`U`, and use :math:`X` and :math:`U` to get the consequent
        input matrix :math:`P` using function :func:`x2xp(x, u, order) <x2xp>`

        :param numpy.array X: Input array with the size of :math:`[N, D]`, where :math:`N` is the number of
            training samples, and :math:`D` is number of features.
        :param numpy.array y: Not used. Pass None.
        :return: return the consequent input :math:`P` with the size of :math:`[N, (D+1)\times R]`, where
            :math:`N` is the number of test samples, :math:`D` is number of features, :math:`R` is the number
            of clusters/rules.
        """
        d = -(np.expand_dims(X, axis=2) - np.expand_dims(self.cluster_centers_.T, axis=0)) ** 2 \
            / (2 * self.variance_.T + 1e-12)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, 1e-16)
        u = d / np.sum(d, axis=1, keepdims=True)
        return x2xp(X, u, self.order)
