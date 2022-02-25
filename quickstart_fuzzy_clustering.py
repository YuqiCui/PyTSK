import numpy as np
from pmlb import fetch_data
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from pytsk.cluster import FuzzyCMeans
from sklearn.pipeline import Pipeline

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

# --------------- Fit and predict ---------------
n_rule = 20
model = Pipeline(
    steps=[
        ("GaussianAntecedent", FuzzyCMeans(n_rule, sigma_scale="auto", fuzzy_index="auto")),
        ("Consequent", RidgeClassifier())
    ]
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

# ---------------- get the input of consequent part for further analysis-----------------
antecedent = model.named_steps['GaussianAntecedent']
consequent_input = antecedent.transform(x_test)
print(x_test.shape)
print(consequent_input.shape)

# --------------- Grid search all important parameters ---------------
param_grid = {
    "Consequent__alpha": [0.01, 0.1, 1, 10, 100],
    "GaussianAntecedent__n_rule": [10, 20, 30, 40],
    "GaussianAntecedent__sigma_scale": [0.01, 0.1, 1, 10, 100],
    "GaussianAntecedent__fuzzy_index": ["auto", 1.8, 2, 2.2],
}
search = GridSearchCV(model, param_grid, n_jobs=2, cv=5, verbose=10)
search.fit(x_train, y_train)
y_pred = search.predict(x_test)
print("ACC: {:.4f}".format(accuracy_score(y_test, y_pred)))

