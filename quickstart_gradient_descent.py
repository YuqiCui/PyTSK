import numpy as np
import torch
import torch.nn as nn
from pytsk.gradient_descent.antecedent import AntecedentGMF, AntecedentShareGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK
from pmlb import fetch_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW

# Define random seed
torch.manual_seed(1447)
np.random.seed(1447)

# Prepare dataset by the PMLB package
X, y = fetch_data('segmentation', return_X_y=True, local_cache_dir='./data/')
n_class = len(np.unique(y))  # Num. of class

# split train-test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
))

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
    x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
))

# Z-score
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# Define TSK model parameters
n_rule = 4  # Num. of rules
lr = 0.01  # learning rate
weight_decay = 1e-8
consbn = True
order = 1

# --------- Define antecedent ------------
init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
gmf = nn.Sequential(
        AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
        # nn.LayerNorm(n_rule),
        # nn.ReLU()
    )
# set high_dim=True is highly recommended.

# --------- Define full TSK model ------------
model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)

# ----------------- optimizer ----------------------------
ante_param, other_param = [], []
for n, p in model.named_parameters():
    if "center" in n or "sigma" in n:
        ante_param.append(p)
    else:
        other_param.append(p)
optimizer = AdamW(
    [{'params': ante_param, "weight_decay": 0},
    {'params': other_param, "weight_decay": weight_decay},],
    lr=lr
)
# ----------------- split 10% data for earlystopping -----------------
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# ----------------- define the earlystopping callback -----------------
EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=40, save_path="tmp.pkl")

wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
              epochs=300, callbacks=[EACC], ur=0, ur_tau=1/n_class)
wrapper.fit(x_train, y_train)
wrapper.load("tmp.pkl")

y_pred = wrapper.predict(x_test).argmax(axis=1)
print("[TSK] ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

