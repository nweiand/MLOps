from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,train_test_split

class IrisSVM:

    def __init__(self, random_state =42):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.2, random_state=random_state
    )

    def SVM_Grid_Search(self):   
        param_grid = {
                'kernel': ['linear','poly', 'rbf', 'sigmoid','precomputed'],
                'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                },
        clf = GridSearchCV(
            SVC(param_grid = param_grid))
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        return(y_pred)
    
    def plot_SVC(self)
        y_pred = self.SVM_Grid_Search()

        