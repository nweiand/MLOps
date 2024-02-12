import numpy as np
from sklearn  import datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split
import matplotlib.pyplot as  plt



class IrisSVM:

    def __init__(self, random_state =42):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.2, random_state=random_state
    )

    def SVM_Grid_Search(self):   
        param_grid = {
                'kernel': ['linear','poly', 'rbf', 'sigmoid'],
                'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                }
        clf = GridSearchCV(
            SVC(),
            param_grid = param_grid)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        return(y_pred)
    
    def plot_SVC(self):
        y_pred = self.SVM_Grid_Search()
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = y_pred


        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', marker='o', s=80, label='Data')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('SVC Decision Boundaries on Iris Dataset')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    svc = IrisSVM()
    svc.plot_SVC()
