from sklearn.metrics import accuracy_score


class Callback:
    """
    Similar as the callback class in Keras, our package provides a simplified version
    of callback, which allow users to monitor metrics during the training.
    We strongly recommend uses to custom their callbacks, here we provide two
    examples, :func:`EvaluateAcc <EvaluateAcc>` and :func:`EarlyStoppingACC <EarlyStoppingACC>`.
    """
    def on_batch_begin(self, wrapper):
        pass

    def on_batch_end(self, wrapper):
        pass

    def on_epoch_begin(self, wrapper):
        pass

    def on_epoch_end(self, wrapper):
        pass


class EvaluateAcc(Callback):
    """

    Evaluate the accuracy during training.

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    """
    def __init__(self, X, y, verbose=0):
        super(EvaluateAcc, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.logs = []

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X).argmax(axis=1)
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["acc"] = acc
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] Test ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"]))


class EarlyStoppingACC(Callback):
    """

    Early-stopping by classification accuracy.

    :param numpy.array X: Feature matrix with the size of :math:`[N, D]`.
    :param numpy.array y: Label matrix with the size of :math:`[N, 1]`.
    :param int patience: Number of epochs with no improvement after which training will be stopped.
    :param int verbose: verbosity mode.
    :param str save_path: If :code:`save_path=None`, do not save models, else save the model
        with the best accuracy to the given path.
    """
    def __init__(self, X, y, patience=1, verbose=0, save_path=None):
        super(EarlyStoppingACC, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.patience = patience
        self.best_acc = 0
        self.cnt = 0
        self.logs = []
        self.save_path = save_path

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X).argmax(axis=1)
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)

        if acc > self.best_acc:
            self.best_acc = acc
            self.cnt = 0
            if self.save_path is not None:
                wrapper.save(self.save_path)
        else:
            self.cnt += 1
            if self.cnt > self.patience:
                wrapper.stop_training = True
        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["acc"] = acc
        cur_log["best_acc"] = self.best_acc
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] EarlyStopping Callback ACC: {:.4f}, Best ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"], cur_log["best_acc"]))

