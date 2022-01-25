# ClearML Demo
A small demo of ClearML modelling the titanic dataset using scikit-learn and Keras.
## Setup

To begin install the project requirements:
**pip**
```bash
pip install -r requirements.txt
```

**poetry**
```bash
poetry install
poetry shell
```

To setup clearml, run the following command:

```bash
clearml login
```

If you are running a custom server manually override your `~/clearml.conf` with the proper credentials. To create new credentials go to:

```
Settings / Workspaces / App Credentials
```

and click on `Create new credentials`, copy the credentials and paste them in your `~/clearml.conf`.

## Usage

To run the demo execute the following command:
```
python main.py
```

You can use the following options to configure the experiment:
```
Options:
  --project-name TEXT             Name of the project  [default: titanic-demo]
  --train-size FLOAT              Size of the training set  [default: 0.9]
  --epochs INTEGER                Number of epochs  [default: 100]
  --batch-size INTEGER            Batch size  [default: 32]
  --n-layer INTEGER               Number of layers  [default: 2]
  --n-units INTEGER               Number of units  [default: 32]
  ```