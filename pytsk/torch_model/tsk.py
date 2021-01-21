"""
from pytsk.torch_model import TSK
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
import numpy as np
from .antecedents import AnteGaussianAndHTSK, AnteGaussianAndLogTSK, AnteGaussianAndTSK, GaussianAntecedent


class O0Consequent(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules):
        super(O0Consequent, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.cons = nn.Linear(n_rules, out_dim)

    def forward(self, x, frs):
        return self.cons(frs)


class O1Consequent(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules):
        super(O1Consequent, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.cons = nn.ModuleList([nn.Linear(self.in_dim, self.out_dim) for _ in range(self.n_rules)])  # consequents

    def forward(self, x, frs):
        cons = torch.cat([cons(x).unsqueeze(1) for cons in self.cons], dim=1)
        out = torch.sum(frs.unsqueeze(2) * cons, dim=1)
        return out


class TSK(nn.Module):
    def __init__(self, in_dim, out_dim, n_rules, order=1,
                 antecedent="tsk", bn=False, droprule=False):
        """

        :param in_dim: input dimension.
        :param out_dim: output dimension. C for a $C$-class classification problem, 1 for a single output regression
                    problem
        :param n_rules: number of rules.
        :param mf: type of membership function. Support: ["gaussian"]
        :param tnorm: type of t-norm. Support: ["and", "or"]. "and" means using Prod t-norm. "or" means using
                    Min t-norm.
        :param defuzzy: defuzzy type. Support: ["tsk", "htsk", "log"]
                    "tsk": weighted average, $y=\sum_r^R \frac{f_ry_r}{\sum_r^R f_r}$
                    "htsk": htsk defuzzy in [1].
                    "log": Log defuzzy in [1],[2].
        """
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rules = n_rules
        self.antecedent = antecedent
        self.order = order
        self.bn = bn
        self.droprule = droprule

        self._build_model()

    def _build_model(self):
        self.cons = O1Consequent(self.in_dim, self.out_dim, self.n_rules) if self.order == 1 else \
            O0Consequent(self.in_dim, self.out_dim, self.n_rules)

        if self.antecedent == "tsk":
            self.firing_level = AnteGaussianAndTSK(self.in_dim, self.n_rules)
        elif self.antecedent == "htsk":
            self.firing_level = AnteGaussianAndHTSK(self.in_dim, self.n_rules)
        elif self.antecedent == "logtsk":
            self.firing_level = AnteGaussianAndLogTSK(self.in_dim, self.n_rules)
        elif callable(self.antecedent):
            self.firing_level = self.antecedent(self.in_dim, self.n_rules)
        else:
            raise ValueError("Unsupported firing level type")

        if self.bn:
            self.bn_layer = nn.BatchNorm1d(self.in_dim)
        if self.droprule:
            self.droprule_layer = nn.Dropout(p=0.25)

    def init_model(self, X, y=None, scale=1., std=0.2, method="cluster", sigma=None, cluster_kwargs=None, eps=1e-8):
        self.firing_level.init_model(X, y, scale=scale, std=std, method=method, sigma=sigma, cluster_kwargs=cluster_kwargs, eps=eps)

    def forward(self, X, **kwargs):
        frs = self.firing_level(X)
        if self.droprule:
            frs = self.droprule_layer(frs)
        if self.bn:
            X = self.bn_layer(X)
        out = self.cons(X, frs)
        if kwargs.pop("frs", False):
            return out, frs
        return out

    def save(self, path=None):
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        self.load_state_dict(torch.load(path))
        self.firing_level.inited = True

    def antecedent_params(self, name=True):
        if name:
            return self.firing_level.named_parameters()
        else:
            return self.firing_level.parameters()

    def predict(self, X):
        my_tensor = next(self.parameters())
        cuda_check = my_tensor.is_cuda
        if cuda_check:
            device = my_tensor.get_device()
        else:
            device = "cpu"

        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(device)
            out = self(X)
            return out.detach().cpu().numpy()
        else:
            outs = []
            for s, (inputs, _) in enumerate(X):
                inputs = inputs.to(device)
                out = self(inputs)
                outs.append(out.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)

    def predict_score(self, X):
        my_tensor = next(self.parameters())
        cuda_check = my_tensor.is_cuda
        if cuda_check:
            device = my_tensor.get_device()
        else:
            device = "cpu"

        if isinstance(X, np.ndarray):
            X = torch.as_tensor(X).float().to(device)
            out = self(X)
            out = F.softmax(out, dim=1)
            return out.detach().cpu().numpy()
        else:
            outs = []
            for s, (inputs, _) in enumerate(X):
                inputs = inputs.to(device)
                out = self(inputs)
                out = F.softmax(out, dim=1)
                outs.append(out.detach().cpu().numpy())
            return np.concatenate(outs, axis=0)
