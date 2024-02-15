import mlflow
from mlflow.models import infer_signature
from regression import IrisRegression

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

if __name__ == "__main__":
    track = Tracking(IrisRegression(random_state=9876), name = "MLflow IrisRegression")
    track.track_run(loss_metric_name="accuracy", tag1="Training Info", tag2="Training Run 2")