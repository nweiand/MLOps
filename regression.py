import numpy as  np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,  mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns


class IrisRegression:

    def __init__(self, random_state =42):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.2, random_state=random_state
    )


    def train_model(self):
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)

        return(y_pred)

    def calc_metrics(self):
        y_pred = self.train_model()
        return({'MEA':mean_absolute_error(self.y_test,y_pred),
                'MSE':mean_squared_error(self.y_test, y_pred),
                'MRSE': np.sqrt(mean_squared_error(self.y_test,y_pred))}
        )

if __name__ =="__main__":
    irreg = IrisRegression()
    print(irreg.calc_metrics())