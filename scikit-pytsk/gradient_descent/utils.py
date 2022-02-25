import torch


def check_tensor(tensor, dtype):
    """
    Convert :code:`tensor` into a :code:`dtype` torch.Tensor.

    :param numpy.array/torch.tensor tensor: Input data.
    :param str dtype: PyTorch dtype string.
    :return: A :code:`dtype` torch.Tensor.
    """
    return torch.tensor(tensor, dtype=dtype)


def reset_params(model):
    """
    Reset all parameters in :code:`model`.
    :param torch.nn.Module model: Pytorch model.
    """
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for layer in model.children():
            reset_params(layer)


class NumpyDataLoader:
    """
    Convert numpy arrays into a dataloader.

    :param numpy.array *inputs: Numpy arrays.
    """
    def __init__(self, *inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)

    def __len__(self):
        return self.inputs[0].shape[0]

    def __getitem__(self, item):
        if self.n_inputs == 1:
            return self.inputs[0][item]
        else:
            return [array[item] for array in self.inputs]