import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.multiclass import type_of_target


def tensor2loader(*args, batch_size=32, shuffle=False):
    train_loader = DataLoader(
        dataset=TensorDataset(*args),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader


def classification_loader(*args, batch_size, shuffle=False):
    all_args = []
    for arg in args[:-1]:
        all_args.append(torch.as_tensor(arg).float())
    all_args.append(torch.as_tensor(args[-1]).long())
    train_loader = DataLoader(
        dataset=TensorDataset(*all_args),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader


def continuous_loader(*args, batch_size, shuffle=False):
    all_args = []
    for arg in args:
        all_args.append(torch.as_tensor(arg).float())
    train_loader = DataLoader(
        dataset=TensorDataset(*all_args),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return train_loader


def auto_assign_data_loader(*args, batch_size, shuffle=False):
    type_y = type_of_target(args[-1])
    if "continuous" in type_y:
        return continuous_loader(*args, batch_size=batch_size, shuffle=shuffle)
    else:
        return classification_loader(*args, batch_size=batch_size, shuffle=shuffle)