import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import LabelEncoder

def _stratified_sample(label: np.ndarray, n: int, random_state: int=None) -> list:
    """

    :param label: The classification label for stratified sample
    :param n: The number of index need to sample
    :return:
    """
    all_index = np.arange(label.shape[0])
    unique_label = np.unique(label)
    n_classes = len(unique_label)
    sample_index = []
    for y in unique_label:
        sample_index.extend(
            all_index[label == y][
                sample_without_replacement(np.sum(label == y), n_samples=n // n_classes, random_state=random_state)
            ]
        )
    return sample_index


def _nan_inter(a, b):
    if len(set(a) & set(b)) == 0:
        return True
    else:
        return False


def train_val_test_split(*arrays, val_size=None, test_size=None, stratified=False, random_state=None, verbose=0):
    assert all(arrays[0].shape[0] == arr.shape[0] for arr in arrays), "arrays must have equal length, got: {}".format(
        "\t".join([str(arr.shape) for arr in arrays])
    )
    assert test_size > 0, "test size must be larger than 0, got {}".format(test_size)
    assert val_size > 0, "val size must be larger than 0, got {}".format(val_size)
    N = arrays[0].shape[0]

    if test_size < 1:
        test_size = int(N * test_size)
    if val_size < 1:
        val_size = int(N * val_size)
    assert test_size + val_size < N, "test size + val size must < number of X, got: {}, N={} ".format(
        test_size + val_size, N
    )
    if stratified:
        assert "continuous" not in type_of_target(arrays[-1]), \
            "The last array must be classification label when stratified=True"
    if verbose > 0:
        print("sample val: {}, sample test: {}".format(val_size, test_size))
    if stratified:
        label = arrays[-1]
        test_index = _stratified_sample(label, test_size, random_state)
        train_val_index = np.setdiff1d(np.arange(N), test_index)
        val_index = train_val_index[_stratified_sample(arrays[-1][train_val_index], val_size, random_state)]
        train_index = np.setdiff1d(train_val_index, val_index)
    else:
        test_index = sample_without_replacement(N, n_samples=test_size, random_state=random_state)
        train_val_index = np.setdiff1d(np.arange(N), test_index)
        val_index = train_val_index[sample_without_replacement(len(train_val_index), n_samples=val_size, random_state=random_state)]
        train_index = np.setdiff1d(train_val_index, val_index)

    assert _nan_inter(train_index, test_index), "Train and test index contain intersection."
    assert _nan_inter(train_index, val_index), "Train and val index contain intersection."
    assert _nan_inter(test_index, val_index), "Test and val index contain intersection."


    ret_args = []
    for arr in arrays:
        ret_args.append(arr[train_index])
    for arr in arrays:
        ret_args.append(arr[val_index])
    for arr in arrays:
        ret_args.append(arr[test_index])
    return ret_args


# def load_libsvm_format(path, return_X_y=True, dropna=True):
#     df = pd.read_csv(path, sep=" ", encoding="utf-8", header=None)
#     cols = list(df.columns)
#     cols[0] = "label"
#     cols[1:] = ["f{}".format(i) for i in range(len(cols) - 1)]
#     df.columns = cols
#     for col in cols[1:]:
#         df[col] = df[col].map(lambda x: float(x.split(":")[1]) if not pd.isna(x) else None)
#     if dropna:
#         df.dropna(axis=0, inplace=True, how="any")
#     df['label'] = LabelEncoder().fit_transform(df['label'].values)
#     if return_X_y:
#         return df[cols[1:]].values, df["label"].values.ravel()
#     return df

def load_libsvm_format(path, dropna=True):
    with open(path, "r") as fr:
        lines = fr.read().splitlines()
    data = []
    label = []
    for idx, aline in enumerate(lines):
        aline = aline.strip().split(" ")
        y = float(aline[0])
        x = aline[1:]
        xs = []
        contain_nan = False
        for n in x:
            tmp = n.split(":")[-1]
            if len(tmp) == 0:
                tmp = float("nan")
                contain_nan = True
            else:
                tmp = float(tmp)
            xs.append(tmp)
        x = xs
        if idx == 0:
            N0 = len(x)
        else:
            assert len(x) == N0, "Row {} have diff length {}, except: {}".format(idx, N0, len(x))
        if contain_nan:
            if not dropna:
                label.append(y)
                data.append(x)
        else:
            label.append(y)
            data.append(x)

    label = np.array(label)
    data = np.array(data)
    if len(label) == 0 and dropna:
        print("Warning: All samples contain nan")
    # print(data[0])
    label = LabelEncoder().fit_transform(label)
    return data, label



