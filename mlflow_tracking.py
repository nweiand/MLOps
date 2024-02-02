import mlflow
from mlflow.models import infer_signature
from regression import IrisRegression


irreg = IrisRegression()
params, accuracy, lr = irreg.train_model()

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow IrisRegression")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "First linear regression")

    # Infer the model signature
    signature = infer_signature(irreg.X_train, lr.predict(irreg.X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=irreg.X_train,
        registered_model_name="tracking-quickstart",
    )
