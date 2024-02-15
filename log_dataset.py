import mlflow.data
import pandas as pd
import numpy as np
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.datasets import load_iris

# Konstruieren eines Datnsatzes mit dem Iris-Datensatz
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
dataset: PandasDataset = mlflow.data.from_pandas(df)

with mlflow.start_run():
    # Loggen des Datensatzes. Der Datensatz wird zum Trainieren benutzt.
    mlflow.log_input(dataset, context="training")

run = mlflow.get_run(mlflow.last_active_run().info.run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
print(f"Dataset name: {dataset_info.name}")
print(f"Dataset digest: {dataset_info.digest}")
print(f"Dataset profile: {dataset_info.profile}")
print(f"Dataset schema: {dataset_info.schema}")
print(run.inputs.dataset_inputs)

