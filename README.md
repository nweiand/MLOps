# MLOps
This ML Project is an example of a (mostly) complete MLOps Framework. It uses
To install the dependencies run the command:

``` pip install -r requirements.txt```

## Components

The components of the paper https://arxiv.org/abs/2205.02302 are implemented in this GitHub Repository.

### CI/CD Component

The CI/CD Component in this case is GitHub Actions. It uses a .yaml file to automate 
workflows on specific triggers, like when a new commit is pushed to the repository. GitHub creates a complete
environment which is configurable in the .yaml file.

### Code Repository

I used GitHub as an  example for a source code repository

### Workflow Orchestration Component

Using Apache Airflow as a Workflow  Orchestration Component allows for automating tasks, such as model training. We have created
a DAG (directed acyclical graph) fot this. It loads the data, creates a train-test-split and then trains the model. 
Apache Airflow has to be started via two CLI commands, ```apache webserver --port 8080``` and ```apache scheduler``` respectively.

### Feature Store System

There is no specific feature store system used in this example. The feature store is the sklearn package, which contains
the Iris dataset which we use in our code.

### Model Training Infrastructure

In this case, our models are so small, that the only infrastructure that is needed is a single computer. 
Kubernetes for example is not needed, as there is no need for distributed computation.

### Model Registry

We use MLflow as our Model Registry. For all our test-runs we used MLflow to track the models and their parameters. MLflow is started with ```mlflow server --host 127.0.0.1 --port 8000```

### Model Metadata Store

As mentioned above, we use MLflow for our Metadata Store, to track our performance metrics, the data used and the
metadata of the  training itself, e.g. date and duration.

### Model Serving Component

To serve the model we choose one of the trained models and serve it via ```mlflow models serve --model-uri runs:/caf16f1fe3c949ae9ba534367fd14c17/model ```

After serving the model, we can access it via REST Api.

### Monitoring Component

For tracking the success or failure of our model in applications, we use MLflow, in the same vein as with our Model Metadata Store.
The performance metrics in our MLflow application 


## Explanation

At first, code is pushed to our source code repository. There, it triggers the CI/CD component, which runs the tests
we provided in our code.
Each day, a training run is done with different data via Apache Airflow. (Here it is different random states of the Iris dataset, akin to a bootstrap)
The data comes from the sklearn dataset module.

The training runs are saved and stored in MLflow,  
