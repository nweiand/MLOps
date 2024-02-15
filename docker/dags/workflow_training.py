from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import random
import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Tracking:

    def __init__(self, model, name ):
        self.model = model
        self.name = name
# Erstellen der IrisRegressionsklasse

    def track_run(self, loss_metric_name:str, tag1:str, tag2:str):
        model_for_run = self.model
        params, loss_metric, lr = model_for_run.train_model()

        # Wo ist mlflow zu finden?
        mlflow.set_tracking_uri(uri="http://127.0.0.1:8000")

        # Erstellen eines neuen MLFlow Experimentes
        mlflow.set_experiment(self.name)

        # Erstellen eines neuen MLFlow Laufes
        with mlflow.start_run():
            # hyperparameter
            mlflow.log_params(params)

            # loss metric
            mlflow.log_metric(loss_metric_name, loss_metric)

            # Tag, mit dem das Experiment beschrieben werden kann
            mlflow.set_tag(tag1, tag2)

            # Signatur des Experiments
            signature = infer_signature(model_for_run.X_train, lr.predict(model_for_run.X_train))

            # Loggen des Modells
            model_info = mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="iris_model",
                signature=signature,
                input_example=model_for_run.X_train,
                registered_model_name="tracking-quickstart",
            )

class IrisRegression:

    def __init__(self, random_state = 8888):
        self.X, self.y = datasets.load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        self.X, self.y, test_size=0.2, random_state=42
    )
        self.params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": random_state,
        }

    def train_model(self):
        lr = LogisticRegression(**self.params)
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return(self.params, accuracy,lr)


def train_model():
    name = "MLflow IrisRegression"
    random_state = random.randint((0, 9999))
    model = IrisRegression(random_state=random_state)
    time = str(datetime.now())
    tracking = Tracking(model, name)
    tracking.track_run(loss_metric_name="accuracy", tag1="Training Run", tag2 = f"Training @{time}")

default_args = {
    'owner' : 'airflow',
    'start_date' : datetime(2024, 2, 12)
}

dag = DAG('ml_pipeline', default_args=default_args, schedule='@daily')

train_op = PythonOperator(
    task_id = 'model_training',
    python_callable=train_model,
    dag=dag
)

if __name__ =="__main__":
    reg = IrisRegression()
    model, accuracy, lr = reg.train_model()
    print(accuracy)