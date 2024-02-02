import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



class IrisRegression:

    def __init__(self):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.2, random_state=42
    )
        self.params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }


    def train_model(self):
        ("k_means_iris_3", KMeans(n_clusters=3, n_init="auto")),
        lr = LogisticRegression(**self.params)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return(self.params, accuracy,lr)
