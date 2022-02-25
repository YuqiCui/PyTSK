import torch
import torch.nn as nn

from .utils import reset_params


class TSK(nn.Module):
    """

    Parent: :code:`torch.nn.Module`

    This module define the consequent part of the TSK model and combines it with a pre-defined
     antecedent module. The input of this module is the raw feature matrix, and output
     the final prediction of a TSK model.

    :param int in_dim: Number of features :math:`D`.
    :param int out_dim: Number of output dimension :math:`C`.
    :param int n_rule: Number of rules :math:`R`, must equal to the :code:`n_rule` of
        the :code:`Antecedent()`.
    :param torch.Module antecedent: An antecedent module, whose output dimension should be
        equal to the number of rules :math:`R`.
    :param int order: 0 or 1. The order of TSK. If 0, zero-oder TSK, else, first-order TSK.
    :param float eps: A constant to avoid the division zero error.
    :param torch.nn.Module consbn: If none, the raw feature will be used as the consequent input;
        If a pytorch module, then the consequent input will be the output of the given module.
        If you wish to use the BN technique we mentioned in
        `Models & Technique <../models.html#batch-normalization>`_,  you can set
        :code:`precons=nn.BatchNorm1d(in_dim)`.
    """
    def __init__(self, in_dim, out_dim, n_rule, antecedent, order=1, eps=1e-8, precons=None):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rule = n_rule
        self.antecedent = antecedent
        self.precons = precons

        self.order = order
        assert self.order == 0 or self.order == 1, "Order can only be 0 or 1."
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        if self.order == 0:
            self.cons = nn.Linear(self.n_rule, self.out_dim, bias=True)
        else:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rule, self.out_dim)

    def reset_parameters(self):
        """
        Re-initialize all parameters, including both consequent and antecedent parts.

        :return:
        """
        reset_params(self.antecedent)
        self.cons.reset_parameters()

        if self.precons is not None:
            self.precons.reset_parameters()

    def forward(self, X, get_frs=False):
        """

        :param torch.tensor X: Input matrix with the size of :math:`[N, D]`,
            where :math:`N` is the number of samples.
        :param bool get_frs: If true, the firing levels (the output of the antecedent)
            will also be returned.

        :return: If :code:`get_frs=True`, return the TSK output :math:`Y\in \mathbb{R}^{N,C}`
            and the antecedent output :math:`U\in \mathbb{R}^{N,R}`. If :code:`get_frs=False`,
            only return the TSK output :math:`Y`.
        """
        frs = self.antecedent(X)

        if self.precons is not None:
            X = self.precons(X)

        if self.order == 0:
            cons_input = frs
        else:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rule, X.size(1)])  # [n_batch, n_rule, in_dim]
            X = X * frs.unsqueeze(dim=2)
            X = X.view([X.size(0), -1])
            cons_input = torch.cat([X, frs], dim=1)

        output = self.cons(cons_input)
        if get_frs:
            return output, frs
        return output



