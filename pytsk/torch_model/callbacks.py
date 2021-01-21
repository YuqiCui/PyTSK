from .eval import eval_func
import math
from .loader import auto_assign_data_loader


def assign_metric_func(metrics):
    """
    assign metric_func by metric name
    :param metrics: metric name. support: auc, acc, mse, rmse
    :return:
    """
    if metrics in eval_func:
        metrics_func = eval_func[metrics.lower()]
    elif callable(metrics):
        metrics_func = metrics
    else:
        raise ValueError("Unsupported metrics: {}, only support: {}".format(metrics, list(eval_func.keys())))
    return metrics_func


class Callback:
    def __init__(self):
        pass

    def register(self, trainer):
        """
        Register Trainer
        :param trainer: Trainer, need to register first to communicate with trainer.
        :return:
        """
        self.trainer = trainer
        self.device = self.trainer.device

    def __repr__(self):
        return "Callback()"

    def get_data_loader(self):
        """
        Automatically recognize input data format and transfer it to pytorch data loader.
        Must be called after this callback is registered.
        :return:
        """
        if not hasattr(self, "data_loader"):
            if isinstance(self.data, list) or isinstance(self.data, tuple):
                self.data_loader = auto_assign_data_loader(*self.data, batch_size=self.trainer.batch_size)
            else:
                self.data_loader = self.data

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, logs=None):
        pass

    def on_epoch_end(self, logs=None):
        pass

    def on_batch_begin(self, logs=None):
        pass

    def on_batch_end(self, logs=None):
        pass


class CheckPerformance(Callback):
    def __init__(self, data=None, name="Test", metrics="acc"):
        """

        :param data:
        :param name:
        :param metrics:
        """
        super(CheckPerformance, self).__init__()
        self.data = data
        self.metrics = metrics
        self.name = name
        self.metric_func = assign_metric_func(self.metrics)
        self.hist = []

    def on_epoch_end(self, logs=None):
        """

        :param logs:
        :return:
        """
        self.get_data_loader()
        metric = self.metric_func(self.trainer.model, self.data_loader, self.device)
        self.trainer.logs[self.name + " " + str(self.metrics).upper()] = metric
        self.hist.append(metric)


class EarlyStopping(Callback):
    def __init__(self, data=None, metrics="acc", patience=1, larger_is_better=True, eps=1e-12, save_path=None, only_save_best=True):
        """

        :param data:
        :param metrics:
        :param patience:
        :param larger_is_better:
        :param eps:
        :param save_path:
        :param only_save_best:
        """
        super(EarlyStopping, self).__init__()
        self.data = data
        self.metrics = metrics
        self.patience = patience
        self.save_path = save_path
        self.only_save_best = only_save_best
        self.metric_func = assign_metric_func(self.metrics)

        self.count = 0
        self.best_metric = -math.inf if larger_is_better else math.inf
        if larger_is_better:
            self.compare = lambda a, b: a > b + eps
        else:
            self.compare = lambda a, b: a < b - eps

        self.hist = []

    def __repr__(self):
        return "[EarlyStopping] metric={}, patience={}".format(self.metrics, self.patience)

    def on_epoch_end(self, logs=None):
        """

        :param logs:
        :return:
        """
        self.get_data_loader()
        metric = self.metric_func(self.trainer.model, self.data_loader, self.device)
        self.hist.append(metric)
        if self.compare(metric, self.best_metric):
            self.count = 0
            self.best_metric = metric
            self.best_epoch = logs["epoch"]
            if self.save_path is not None and self.only_save_best:
                self.trainer.model.save(self.save_path)
        else:
            self.count += 1
        if self.save_path is not None and not self.only_save_best:
            self.trainer.model.save(self.save_path)
        if self.count > self.patience:
            self.trainer.end_training = True
        self.trainer.logs["Val " + str(self.metrics).upper()] = metric
        self.trainer.logs["Best Val " + str(self.metrics).upper()] = self.best_metric
